"""
services/paper_summary.py
-------------------------
A one-sentence description of a dataset, written by an LLM from the abstract
of the dataset's primary paper (scripts/generate_paper_summaries.py stores it
in bids_datasets.paper_summary; routers/query.py shows it, labelled as
generated, with a link to the paper).

Most OpenNeuro datasets have no description of their own, and README
openings are often boilerplate. The abstract says what was studied; the
sentence keeps only what data was collected, never the findings.

Guards, since the sentence is shown as a description:
  - every number in it must appear in the paper title, abstract or dataset
    name (no invented sample sizes);
  - no run of COPY_WORDS or more words copied from the abstract (a summary,
    not a quote: abstracts are mostly publisher-copyrighted);
  - a length range.
A sentence that fails a guard is not stored.

About 10% of "primary" papers belong to a related study by the same lab
(crawlers/paper_enrichment_plan.md, role check), so the card says the text
describes the linked paper.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import IO, Any, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

PROMPT_VERSION = "1"
MIN_CHARS = 40
MAX_CHARS = 320
MAX_WORDS = 45
COPY_WORDS = 9

SYSTEM = (
    "You write one-sentence descriptions of open neuroimaging datasets for a "
    "dataset search engine. You describe data, never findings."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")


@dataclass(frozen=True)
class SummarySource:
    dataset_name: str
    accession: str
    paper_key: str
    title: str
    abstract: str

    @property
    def abstract_sha(self) -> str:
        return hashlib.sha256(self.abstract.encode("utf-8")).hexdigest()[:16]


def build_prompt(source: SummarySource) -> str:
    return (
        f"Dataset: {source.dataset_name} (OpenNeuro {source.accession})\n"
        f"Paper title: {source.title}\n"
        f"Paper abstract:\n{source.abstract}\n\n"
        "Write one sentence of at most 35 words describing the data this study "
        "collected: who took part (numbers and groups, if the abstract gives "
        "them), the imaging or recording method, and the task or condition.\n"
        "- Describe the data, not the findings: no results, conclusions or "
        "interpretations.\n"
        "- Use only facts stated above; leave out anything the abstract does not say.\n"
        "- Paraphrase; do not copy phrases longer than a few words.\n"
        "- Start with the content, e.g. \"fMRI from 48 adults performing ...\", "
        "not with \"This study\" or \"The dataset\".\n"
        'Answer as JSON: {"summary": "..."}'
    )


def parse_answer(raw: str) -> Optional[str]:
    """The sentence from the model's JSON answer, or None."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1]
    try:
        summary = json.loads(raw).get("summary")
    except (ValueError, AttributeError):
        return None
    if not isinstance(summary, str):
        return None
    summary = " ".join(summary.split()).strip().strip('"“”').strip()
    return summary or None


def _numbers(text: str) -> set:
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(text)}


def _words(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def check(summary: str, source: SummarySource) -> Optional[str]:
    """Why *summary* must not be stored, or None when it passes."""
    if not (MIN_CHARS <= len(summary) <= MAX_CHARS):
        return f"length {len(summary)}"
    if len(summary.split()) > MAX_WORDS:
        return "too many words"
    allowed = _numbers(" ".join((source.title, source.abstract, source.dataset_name)))
    invented = _numbers(summary) - allowed
    if invented:
        return f"numbers not in the source: {sorted(invented)}"
    abstract_words = _words(source.abstract)
    copied = {tuple(abstract_words[i:i + COPY_WORDS])
              for i in range(len(abstract_words) - COPY_WORDS + 1)}
    words = _words(summary)
    for i in range(len(words) - COPY_WORDS + 1):
        if tuple(words[i:i + COPY_WORDS]) in copied:
            return f"copies {COPY_WORDS}+ words of the abstract"
    return None


def paper_url(doi: Optional[str], pmid: Optional[str], pmcid: Optional[str],
              arxiv_id: Optional[str]) -> Optional[str]:
    """A link to the paper: DOI first, then PubMed, PMC, arXiv."""
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


# ── Running it over the database ──────────────────────────────────────────────
# Shared by scripts/generate_paper_summaries.py and the crawler's post-crawl
# step (crawlers/openneuro-crawler/crawler_service.py).

DEFAULT_MODEL = os.getenv("PAPER_SUMMARY_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = "gemini-2.5-flash-lite"

# Each dataset's most trusted primary paper that has an abstract.
_SELECT_SQL = """
    SELECT DISTINCT ON (d.id)
           d.id, d.accession_id, d.name, d.paper_summary_meta,
           p.paper_key, p.title, p.abstract
    FROM bids_datasets d
    JOIN bids_dataset_papers dp ON dp.dataset_id = d.id AND dp.role = 'primary'
    JOIN bids_papers p ON p.id = dp.paper_id AND NULLIF(btrim(p.abstract), '') IS NOT NULL
    {where}
    ORDER BY d.id, dp.position, p.id
"""

# Summaries whose paper is no longer a primary paper of the dataset.
_PRUNE_SQL = """
    UPDATE bids_datasets d
    SET paper_summary = NULL, paper_summary_meta = NULL
    WHERE d.paper_summary IS NOT NULL {scope}
      AND NOT EXISTS (
          SELECT 1 FROM bids_dataset_papers dp JOIN bids_papers p ON p.id = dp.paper_id
          WHERE dp.dataset_id = d.id AND dp.role = 'primary'
            AND p.paper_key = d.paper_summary_meta->>'paper_key')
"""

_WRITE_SQL = """
    UPDATE bids_datasets SET paper_summary = :summary, paper_summary_meta = CAST(:meta AS JSONB)
    WHERE id = :id
"""


def _up_to_date(meta: Any, source: SummarySource) -> bool:
    if isinstance(meta, str):
        meta = json.loads(meta)
    return bool(meta) and (
        meta.get("paper_key") == source.paper_key
        and meta.get("abstract_sha") == source.abstract_sha
        and meta.get("prompt_version") == PROMPT_VERSION
    )


def summarise_one(source: SummarySource, model: str = DEFAULT_MODEL) -> Tuple[SummarySource, Optional[str], str]:
    """(source, checked summary or None, why not / the error). Blocking."""
    from llm_client import llm_generate

    try:
        raw = llm_generate(
            build_prompt(source),
            system=SYSTEM,
            temperature=0.2,
            max_tokens=400,
            models=[model, FALLBACK_MODEL],
            json_schema=JSON_SCHEMA,
            call_timeout=60,
            budget_seconds=180,
            use_fallback=False,  # abstracts go to Gemini only
        )
    except Exception as exc:  # LLMAllFailedError or a network error
        return source, None, f"model call failed: {str(exc)[:200]}"
    summary = parse_answer(raw)
    if summary is None:
        return source, None, f"unreadable answer: {raw[:120]!r}"
    problem = check(summary, source)
    return (source, None, f"rejected ({problem}): {summary}") if problem else (source, summary, "")


async def has_columns(conn: Any) -> bool:
    """paper_summary exists and the paper tables are there."""
    from sqlalchemy import text

    found = (await conn.execute(text(
        "SELECT (SELECT count(*) FROM information_schema.columns "
        "        WHERE table_name = 'bids_datasets' AND column_name = 'paper_summary') > 0 "
        "   AND to_regclass('bids_dataset_papers') IS NOT NULL"))).scalar()
    return bool(found)


async def run(
    engine: Any,
    accessions: Optional[Sequence[str]] = None,
    *,
    model: str = DEFAULT_MODEL,
    workers: int = 4,
    force: bool = False,
    dry_run: bool = False,
    limit: int = 0,
    out: Optional[IO[str]] = None,
) -> Counter:
    """Summarise the datasets that need it (default: all with a primary-paper
    abstract). Commits after every write, so a stopped run keeps its work and
    holds no lock while the model answers. Returns counts:
    done, rejected, failed, pruned, skipped (up to date)."""
    from sqlalchemy import text

    stats: Counter = Counter()
    accessions = [a for a in (accessions or []) if a]
    where, params = "", {}
    if accessions:
        where, params = "WHERE d.accession_id = ANY(:acc)", {"acc": list(accessions)}
    async with engine.connect() as conn:
        if not await has_columns(conn):
            raise RuntimeError("paper_summary or the paper tables are missing: run "
                               "crawlers/migrate_add_paper_search.sql and "
                               "crawlers/migrate_add_paper_summaries.sql first")
        rows = (await conn.execute(text(_SELECT_SQL.format(where=where)), params)).mappings().all()

    ids = {}
    todo = []
    for row in rows:
        source = SummarySource(
            dataset_name=row["name"] or "", accession=row["accession_id"] or str(row["id"]),
            paper_key=row["paper_key"], title=row["title"] or "", abstract=row["abstract"])
        ids[source.accession] = row["id"]
        if force or not _up_to_date(row["paper_summary_meta"], source):
            todo.append(source)
        else:
            stats["skipped"] += 1
    if limit:
        todo = todo[:limit]
    log.info("paper summaries: %d datasets with a primary-paper abstract, %d to write (model %s)",
             len(rows), len(todo), model)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [loop.run_in_executor(pool, summarise_one, s, model) for s in todo]
        async with engine.connect() as conn:
            if not dry_run:
                scope, scope_params = ("AND d.accession_id = ANY(:acc)", params) if accessions else ("", {})
                stats["pruned"] = (await conn.execute(
                    text(_PRUNE_SQL.format(scope=scope)), scope_params)).rowcount or 0
                await conn.commit()
            for done in asyncio.as_completed(futures):
                source, summary, problem = await done
                if out is not None:
                    out.write(json.dumps({
                        "accession": source.accession, "dataset": source.dataset_name,
                        "paper_key": source.paper_key, "paper_title": source.title,
                        "summary": summary, "problem": problem or None,
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                if summary is None:
                    stats["failed" if problem.startswith("model call") else "rejected"] += 1
                    log.warning("paper summary %s: %s", source.accession, problem[:300])
                    continue
                stats["done"] += 1
                if dry_run:
                    continue
                meta = {
                    "paper_key": source.paper_key,
                    "abstract_sha": source.abstract_sha,
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                await conn.execute(text(_WRITE_SQL), {
                    "id": ids[source.accession], "summary": summary, "meta": json.dumps(meta)})
                await conn.commit()
                answered = stats["done"] + stats["rejected"] + stats["failed"]
                if answered % 50 == 0:
                    log.info("paper summaries: %d / %d answered", answered, len(todo))
    return stats
