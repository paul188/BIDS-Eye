"""
services/indexing/dataset_description.py
----------------------------------------
Two things the result card shows besides the README passage, both written by
an LLM from labelled sources and checked before they are stored:

1. paper_relation, for every dataset with a primary paper that has an
   abstract. How the paper relates to the dataset:

       reports         the paper reports this dataset
       larger_sample   the paper uses a larger sample that includes it
       related_study   an earlier or related study by the same lab
       tool            a tool or method paper
       unclear

   plus mismatches (sample size, modality, population) and evidence quotes.
   Each quote must occur in the source it names (readme_profile's
   quote_in_source). The card shows "Linked paper: <title> (<year>), <relation>"
   only for reports / larger_sample with at least one verified quote; the
   other relations are kept for review (a wrong link can be demoted from
   primary to related in bids_dataset_papers). The abstract is never shown.

2. dataset_summary, only where the README and description together are
   shorter than THIN_CHARS: two or three sentences on what the dataset holds,
   from the README and catalogue fields first. The paper is used only when
   its relation is shown, and only in sentences that start with "According
   to the linked paper,". The check enforces that framing: a number found
   only in the paper is allowed only in such a sentence.

Results carry a hash of their inputs and the prompt version, so a rerun redoes
only what changed. scripts/describe_datasets.py runs this over the database;
the crawler runs it for the datasets it indexed (post_crawl.py).
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple

from services.indexing.readme_profile import quote_in_source
from services.schema.tables import dataset_papers as _dataset_papers, datasets as _datasets

log = logging.getLogger(__name__)

RELATION_PROMPT_VERSION = "1"
SUMMARY_PROMPT_VERSION = "2"
DEFAULT_MODEL = os.getenv("DATASET_DESCRIPTION_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = "gemini-2.5-flash-lite"

THIN_CHARS = 500           # README + description shorter than this: write a summary
README_CHARS = 6000        # README text given to the model
PAPER_PREFIX = "According to the linked paper,"
COPY_WORDS = 9             # no run this long copied from the abstract
SUMMARY_MIN_CHARS, SUMMARY_MAX_CHARS = 40, 560

RELATIONS = ("reports", "larger_sample", "related_study", "tool", "unclear")
SHOWN_RELATIONS = ("reports", "larger_sample")
RELATION_LABELS = {
    "reports": "reports this dataset",
    "larger_sample": "uses a larger sample that includes this dataset",
    "related_study": "an earlier or related study by the same lab",
    "tool": "a tool or method paper",
    "unclear": "relation unclear",
}
SOURCES = ("README", "CATALOGUE", "PAPER")

_MODALITIES = {
    "functional_mri": "fMRI", "anatomical_mri": "structural MRI", "diffusion_mri": "diffusion MRI",
    "electroencephalography": "EEG", "magnetoencephalography": "MEG", "intracranial_eeg": "iEEG",
    "positron_emission_tomography": "PET", "fnirs": "fNIRS", "perfusion_asl": "ASL perfusion",
    "magnetic_resonance_spectroscopy": "MR spectroscopy", "motion_capture": "motion capture",
    "behavioural_data": "behavioural data", "field_maps": "field maps",
}
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"])")
# A summary describes the data, not the prompt's sources or their gaps.
_SOURCE_TALK_RE = re.compile(r"\b(?:README|CATALOGUE|catalogue|OpenNeuro dataset)\b")
_MISSING_TALK_RE = re.compile(
    r"\b(?:not (?:available|provided|specified|described|given|stated|mentioned)|"
    r"no (?:further |additional )?(?:details|information))\b", re.I)


# ── Sources ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Paper:
    paper_key: str
    title: str
    abstract: str
    year: Optional[int] = None
    role_reason: str = ""


@dataclass(frozen=True)
class DatasetSource:
    accession: str
    name: str
    readme: str = ""
    description: str = ""
    catalogue: Tuple[Tuple[str, str], ...] = ()   # ("Subjects", "46"), …
    paper: Optional[Paper] = None

    @property
    def thin(self) -> bool:
        return len(self.readme.strip()) + len(self.description.strip()) < THIN_CHARS

    def catalogue_text(self) -> str:
        lines = [f"Name: {self.name}"]
        if self.description.strip():
            lines.append(f"Description: {self.description.strip()}")
        lines += [f"{label}: {value}" for label, value in self.catalogue if value]
        return "\n".join(lines)

    def paper_text(self) -> str:
        return f"{self.paper.title}\n{self.paper.abstract}" if self.paper else ""

    def readme_text(self) -> str:
        return self.readme.strip()[:README_CHARS]

    def source_text(self, label: str) -> str:
        return {"README": self.readme_text(), "CATALOGUE": self.catalogue_text(),
                "PAPER": self.paper_text()}.get(label, "")


def catalogue_fields(row: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Catalogue lines from a database row (see _SELECT_SQL)."""
    def join(values, limit=12):
        values = [v for v in (values or []) if v]
        shown = ", ".join(values[:limit])
        return shown + (f" (+{len(values) - limit} more)" if len(values) > limit else "")

    modalities = [_MODALITIES.get(v, v.replace("_", " ")) for v in sorted(row.get("datatypes") or [])]
    tasks = [t.replace("_", " ") for t in sorted(row.get("tasks") or [])]
    diagnoses = row.get("diagnoses") or {}
    if isinstance(diagnoses, str):
        diagnoses = json.loads(diagnoses)
    dx = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in
                   sorted(diagnoses.items(), key=lambda kv: -kv[1])[:8])
    ages = ""
    if row.get("age_min") is not None or row.get("age_max") is not None:
        lo, hi = row.get("age_min"), row.get("age_max")
        ages = f"{lo:g}–{hi:g}" if lo is not None and hi is not None else f"{lo or hi:g}"
    subjects = row.get("subject_count")
    return tuple((label, value) for label, value in (
        ("Subjects", str(subjects) if subjects else ""),
        ("Modalities", join(modalities)),
        ("Tasks", join(tasks)),
        ("Participant diagnoses (count)", dx),
        ("Ages", ages),
        ("Study domain", (row.get("study_domain") or "").strip()),
        ("Study design", (row.get("study_design") or "").strip()),
        ("Diagnosis status", (row.get("dx_status") or "").strip()),
    ) if value)


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def relation_input_sha(source: DatasetSource) -> str:
    paper = source.paper
    return _sha(RELATION_PROMPT_VERSION, source.readme_text(), source.catalogue_text(),
                paper.paper_key if paper else "", source.paper_text(),
                paper.role_reason if paper else "")


def summary_input_sha(source: DatasetSource, paper_used: bool) -> str:
    return _sha(SUMMARY_PROMPT_VERSION, source.readme_text(), source.catalogue_text(),
                source.paper_text() if paper_used else "")


# ── 1. How the paper relates to the dataset ───────────────────────────────────

RELATION_SYSTEM = (
    "You check whether a scientific paper describes a given open neuroimaging "
    "dataset. You answer only from the labelled sources and quote them exactly."
)

RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": list(RELATIONS)},
        "mismatches": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "aspect": {"type": "string", "enum": ["sample_size", "modality", "population", "other"]},
                "note": {"type": "string"},
            },
            "required": ["aspect", "note"],
        }},
        "evidence": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": list(SOURCES)},
                "quote": {"type": "string"},
            },
            "required": ["source", "quote"],
        }},
    },
    "required": ["relation", "mismatches", "evidence"],
}


def relation_prompt(source: DatasetSource) -> str:
    paper = source.paper
    assert paper is not None, "asked only for a dataset with a linked paper"
    return (
        f"[README] of OpenNeuro dataset {source.accession}\n"
        f"{source.readme_text() or '(none)'}\n\n"
        f"[CATALOGUE] fields recorded for the dataset\n{source.catalogue_text()}\n\n"
        f"[PAPER] linked to the dataset\nTitle: {paper.title}\nAbstract: {paper.abstract}\n\n"
        f"[WHY LINKED] {paper.role_reason or '(not recorded)'}\n\n"
        "How does the PAPER relate to the dataset described by README and CATALOGUE?\n"
        "- reports: the paper reports this dataset (same study, same data)\n"
        "- larger_sample: the paper analyses a larger sample that includes this dataset\n"
        "- related_study: an earlier or related study by the same lab, not this data\n"
        "- tool: a tool, method or software paper\n"
        "- unclear: the sources do not settle it\n"
        "List mismatches between dataset and paper in sample size, modality or population "
        "(empty if none). Give 1 to 3 evidence quotes for your answer, each copied word "
        "for word from one source, labelled README, CATALOGUE or PAPER."
    )


@dataclass
class Relation:
    relation: str
    mismatches: List[Dict[str, str]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def verified_quotes(self) -> int:
        return sum(bool(e.get("verified")) for e in self.evidence)

    @property
    def shown(self) -> bool:
        return self.relation in SHOWN_RELATIONS and self.verified_quotes > 0


def _load_json(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1]
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_relation(raw: str, source: DatasetSource) -> Optional[Relation]:
    """The model's answer with each quote checked against its source; None if unreadable."""
    answer = _load_json(raw)
    if answer is None or answer.get("relation") not in RELATIONS:
        return None
    mismatches = [
        {"aspect": str(m.get("aspect", "other")), "note": str(m.get("note", ""))[:300]}
        for m in answer.get("mismatches") or [] if isinstance(m, dict) and m.get("note")
    ][:5]
    evidence = []
    for item in (answer.get("evidence") or [])[:5]:
        if not isinstance(item, dict):
            continue
        label, quote = str(item.get("source", "")), str(item.get("quote", "")).strip()
        if not quote:
            continue
        evidence.append({"source": label, "quote": quote[:400],
                         "verified": quote_in_source(quote, source.source_text(label))})
    return Relation(answer["relation"], mismatches, evidence)


# ── 2. A summary where the README is thin ─────────────────────────────────────

SUMMARY_SYSTEM = (
    "You describe open neuroimaging datasets for a dataset search engine, in plain "
    "factual sentences. You describe data, never findings, and invent nothing."
)

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def summary_prompt(source: DatasetSource, use_paper: bool) -> str:
    parts = [
        f"[README] of OpenNeuro dataset {source.accession}\n{source.readme_text() or '(none)'}",
        f"[CATALOGUE] fields recorded for the dataset\n{source.catalogue_text()}",
    ]
    if use_paper:
        assert source.paper is not None, "use_paper needs a linked paper"
        parts.append(f"[PAPER] that reports this dataset\nTitle: {source.paper.title}\n"
                     f"Abstract: {source.paper.abstract}")
    rules = [
        "Write 1 to 3 sentences (at most 70 words) saying what this dataset contains: "
        "who took part, the imaging or recording methods, and the tasks or conditions. "
        "If the sources say little, write one short sentence with what is known.",
        "Start with the content, e.g. \"fMRI and structural MRI from 28 adults performing "
        "a theory-of-mind task.\" Do not mention README, CATALOGUE, PAPER, the dataset's "
        "name or its accession number, and never say what is missing or unknown.",
        "Use README and CATALOGUE first.",
        (f"You may add what only the PAPER says, but only in a sentence that starts with "
         f"\"{PAPER_PREFIX}\", and such a sentence describes what the paper's study did "
         f"(\"{PAPER_PREFIX} the study also recorded …\"), never what this dataset "
         f"contains. Never mix paper facts into the other sentences."
         if use_paper else "There is no paper; use only README and CATALOGUE."),
        "Describe the data, not results or conclusions. Leave out what the sources do "
        "not say. Paraphrase; do not copy long phrases.",
        'Answer as JSON: {"summary": "..."}',
    ]
    return "\n\n".join(parts) + "\n\n" + "\n".join(f"- {r}" for r in rules)


def parse_summary(raw: str) -> Optional[str]:
    answer = _load_json(raw)
    summary = answer.get("summary") if answer else None
    if not isinstance(summary, str):
        return None
    summary = " ".join(summary.split()).strip().strip('"“”').strip()
    return summary or None


def _numbers(text: str) -> set:
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(text or "")}


def _words(text: str) -> list:
    return _WORD_RE.findall((text or "").lower())


def check_summary(summary: str, source: DatasetSource, paper_used: bool) -> Optional[str]:
    """Why *summary* must not be stored, or None when it passes."""
    if not (SUMMARY_MIN_CHARS <= len(summary) <= SUMMARY_MAX_CHARS):
        return f"length {len(summary)}"
    if PAPER_PREFIX.lower() in summary.lower() and not paper_used:
        return "cites a paper it was not given"
    talk = _SOURCE_TALK_RE.search(summary)
    if talk or (source.accession and source.accession.lower() in summary.lower()):
        return f"talks about its sources ({talk.group(0) if talk else source.accession})"
    if _MISSING_TALK_RE.search(summary):
        return "says what is missing"
    dataset_numbers = _numbers(" ".join((source.readme_text(), source.catalogue_text())))
    paper_numbers = _numbers(source.paper_text()) if paper_used else set()
    for sentence in _SENTENCE_RE.split(summary):
        from_paper = sentence.strip().lower().startswith(PAPER_PREFIX.lower())
        allowed = dataset_numbers | (paper_numbers if from_paper else set())
        invented = _numbers(sentence) - allowed
        if invented:
            where = "the sources" if from_paper or not paper_numbers & invented else \
                "README/catalogue (paper facts need the 'According to the linked paper' framing)"
            return f"numbers not in {where}: {sorted(invented)}"
    if paper_used:
        assert source.paper is not None, "paper_used needs a linked paper"
        abstract = _words(source.paper.abstract)
        copied = {tuple(abstract[i:i + COPY_WORDS]) for i in range(len(abstract) - COPY_WORDS + 1)}
        words = _words(summary)
        if any(tuple(words[i:i + COPY_WORDS]) in copied for i in range(len(words) - COPY_WORDS + 1)):
            return f"copies {COPY_WORDS}+ words of the abstract"
    return None


def worth_summarising(source: DatasetSource) -> bool:
    """A thin dataset with something to say beyond its name."""
    return source.thin and (bool(source.readme.strip() or source.description.strip())
                            or bool(source.catalogue) or source.paper is not None)


# ── Model calls (blocking; run in threads) ────────────────────────────────────

def _ask(prompt: str, system: str, schema: dict, model: str) -> str:
    from llm_client import llm_generate

    return llm_generate(
        prompt, system=system, temperature=0.1,
        # Gemini 2.5 spends output tokens on thinking first; small caps cut answers off.
        max_tokens=4096, models=[model, FALLBACK_MODEL], json_schema=schema,
        call_timeout=90, budget_seconds=240,
        use_fallback=False,  # dataset and paper text go to Gemini only
    )


@dataclass
class Outcome:
    source: DatasetSource
    relation: Optional[Relation] = None       # new relation, when one was asked
    relation_error: str = ""
    summary: Optional[str] = None
    summary_paper_used: bool = False   # the paper was offered to the summary prompt
    summary_error: str = ""
    summary_asked: bool = False


def describe_one(source: DatasetSource, need_relation: bool, current_relation: Optional[dict],
                 need_summary: bool, model: str = DEFAULT_MODEL) -> Outcome:
    """Ask for the relation and/or the summary of one dataset.

    A model call that fails on every tier (LLMAllFailedError) is recorded in
    the outcome and the batch goes on. Anything else raised is a bug and
    stops the batch: it used to be recorded as "model call failed" for every
    dataset, and the batch "completed" with all of them failed.
    """
    from llm_client import LLMAllFailedError

    outcome = Outcome(source)
    shown = bool(current_relation and current_relation.get("shown"))
    if need_relation and source.paper:
        try:
            outcome.relation = parse_relation(
                _ask(relation_prompt(source), RELATION_SYSTEM, RELATION_SCHEMA, model), source)
            if outcome.relation is None:
                outcome.relation_error = "unreadable relation answer"
        except LLMAllFailedError as exc:
            outcome.relation_error = f"model call failed: {str(exc)[:200]}"
        if outcome.relation is not None:
            shown = outcome.relation.shown
        elif need_summary:
            need_summary = False  # the summary depends on the relation; next run
    if need_summary:
        outcome.summary_asked = True
        use_paper = bool(source.paper) and shown
        outcome.summary_paper_used = use_paper
        try:
            summary = parse_summary(_ask(summary_prompt(source, use_paper), SUMMARY_SYSTEM,
                                         SUMMARY_SCHEMA, model))
        except LLMAllFailedError as exc:
            outcome.summary_error = f"model call failed: {str(exc)[:200]}"
            return outcome
        if summary is None:
            outcome.summary_error = "unreadable summary answer"
            return outcome
        problem = check_summary(summary, source, use_paper)
        if problem:
            outcome.summary_error = f"rejected ({problem}): {summary}"
        else:
            outcome.summary = summary
    return outcome


# ── Running it over the database ──────────────────────────────────────────────

# Every dataset with a primary paper that has an abstract, or a thin README.
_SELECT_SQL = """
    SELECT d.id, d.accession_id, d.name, d.readme_text, d.description_text,
           d.study_domain, d.study_design, d.dx_status, d.age_min, d.age_max,
           d.paper_relation, d.dataset_summary_meta,
           (SELECT count(DISTINCT o.subject) FROM bids_objects o
             WHERE o.dataset_id = d.id AND o.subject IS NOT NULL)            AS subject_count,
           (SELECT array_agg(DISTINCT o.datatype) FROM bids_objects o
             WHERE o.dataset_id = d.id AND o.datatype IS NOT NULL)           AS datatypes,
           (SELECT array_agg(DISTINCT o.task) FROM bids_objects o
             WHERE o.dataset_id = d.id AND o.task IS NOT NULL)               AS tasks,
           (SELECT jsonb_object_agg(g.dx, g.n) FROM (
                SELECT p.diagnosis AS dx, count(*) AS n FROM bids_participants p
                WHERE p.dataset_id = d.id AND NULLIF(btrim(p.diagnosis), '') IS NOT NULL
                GROUP BY 1) g)                                               AS diagnoses,
           pp.paper_key, pp.title, pp.abstract, pp.year, pp.role_reason
    FROM bids_datasets d
    LEFT JOIN LATERAL (
        SELECT p.paper_key, p.title, p.abstract, p.year, dp.role_reason
        FROM bids_dataset_papers dp JOIN bids_papers p ON p.id = dp.paper_id
        WHERE dp.dataset_id = d.id AND dp.role = 'primary'
          AND NULLIF(btrim(p.abstract), '') IS NOT NULL
        ORDER BY dp.position, p.id LIMIT 1) pp ON true
    WHERE (pp.paper_key IS NOT NULL
           OR length(coalesce(d.readme_text, '')) + length(coalesce(d.description_text, '')) < :thin)
      {scope}
"""

# Relations whose paper is no longer a primary paper; summaries of datasets
# that are no longer thin.
_PRUNE_SQL = (
    """
    UPDATE bids_datasets d SET paper_relation = NULL
    WHERE d.paper_relation IS NOT NULL {scope}
      AND NOT EXISTS (
          SELECT 1 FROM bids_dataset_papers dp JOIN bids_papers p ON p.id = dp.paper_id
          WHERE dp.dataset_id = d.id AND dp.role = 'primary'
            AND p.paper_key = d.paper_relation->>'paper_key')
    """,
    """
    UPDATE bids_datasets d SET dataset_summary = NULL, dataset_summary_meta = NULL
    WHERE d.dataset_summary_meta IS NOT NULL {scope}
      AND length(coalesce(d.readme_text, '')) + length(coalesce(d.description_text, '')) >= :thin
    """,
)

_WRITE_RELATION_SQL = f"UPDATE {_datasets.name} SET paper_relation = CAST(:value AS JSONB) WHERE id = :id"
_WRITE_SUMMARY_SQL = (f"UPDATE {_datasets.name} SET dataset_summary = :summary, "
                      "dataset_summary_meta = CAST(:meta AS JSONB) WHERE id = :id")


def source_from_row(row: Dict[str, Any]) -> DatasetSource:
    paper = None
    if row.get("paper_key"):
        paper = Paper(paper_key=row["paper_key"], title=row.get("title") or "",
                      abstract=row.get("abstract") or "", year=row.get("year"),
                      role_reason=row.get("role_reason") or "")
    return DatasetSource(
        accession=row.get("accession_id") or str(row["id"]), name=row.get("name") or "",
        readme=row.get("readme_text") or "", description=row.get("description_text") or "",
        catalogue=catalogue_fields(row), paper=paper)


def _as_dict(value: Any) -> Optional[dict]:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else None


def plan_work(row: Dict[str, Any], source: DatasetSource, force: bool) -> Tuple[bool, bool]:
    """(needs a relation, needs a summary) for one dataset."""
    relation = _as_dict(row.get("paper_relation"))
    need_relation = source.paper is not None and (
        force or not relation or relation.get("input_sha") != relation_input_sha(source))
    if not worth_summarising(source):
        return need_relation, False
    if need_relation:
        return True, True  # the summary may use the paper only once the relation is known
    meta = _as_dict(row.get("dataset_summary_meta"))
    paper_offered = source.paper is not None and bool(relation and relation.get("shown"))
    need_summary = force or not meta or meta.get("input_sha") != summary_input_sha(source, paper_offered)
    return False, need_summary


async def has_columns(conn: Any) -> bool:
    from sqlalchemy import text

    found = (await conn.execute(text(
        f"SELECT (SELECT count(*) FROM information_schema.columns WHERE table_name = '{_datasets.name}' "
        "        AND column_name IN ('paper_relation', 'dataset_summary', 'dataset_summary_meta')) = 3 "
        f"   AND to_regclass('{_dataset_papers.name}') IS NOT NULL"))).scalar()
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
    """Write relations and thin-README summaries that are missing or stale
    (default: every dataset). Commits after each dataset, so a stopped run keeps
    its work and holds no lock while the model answers. Returns counts:
    relation:<name>, relation_shown, summary_done, summary_rejected, failed,
    pruned, skipped."""
    from sqlalchemy import text

    stats: Counter = Counter()
    accessions = [a for a in (accessions or []) if a]
    scope = ""
    params: dict[str, Any] = {"thin": THIN_CHARS}
    if accessions:
        scope, params = "AND d.accession_id = ANY(:acc)", {**params, "acc": list(accessions)}
    async with engine.connect() as conn:
        if not await has_columns(conn):
            raise RuntimeError("paper_relation / dataset_summary or the paper tables are missing: "
                               "run crawlers/migrate_add_paper_search.sql and "
                               "crawlers/migrate_add_dataset_descriptions.sql first")
        rows = (await conn.execute(text(_SELECT_SQL.format(scope=scope)), params)).mappings().all()

    jobs = []
    for row in rows:
        source = source_from_row(dict(row))
        need_relation, need_summary = plan_work(dict(row), source, force)
        if need_relation or need_summary:
            jobs.append((row["id"], source, need_relation, _as_dict(row.get("paper_relation")),
                         need_summary))
        else:
            stats["skipped"] += 1
    if limit:
        jobs = jobs[:limit]
    log.info("dataset descriptions: %d candidates, %d to do (model %s)", len(rows), len(jobs), model)

    def job(dataset_id, source, need_relation, current, need_summary):
        return dataset_id, describe_one(source, need_relation, current, need_summary, model)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [loop.run_in_executor(pool, job, *args) for args in jobs]
        async with engine.connect() as conn:
            if not dry_run:
                for statement in _PRUNE_SQL:
                    stats["pruned"] += (await conn.execute(
                        text(statement.format(scope=scope)), params)).rowcount or 0
                await conn.commit()
            for done in asyncio.as_completed(futures):
                dataset_id, outcome = await done
                await _record(conn, dataset_id, outcome, model, stats, out, dry_run)
    return stats


async def _record(conn, dataset_id, outcome: Outcome, model: str, stats: Counter,
                  out: Optional[IO[str]], dry_run: bool) -> None:
    from sqlalchemy import text

    source = outcome.source
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if outcome.relation_error:
        stats["failed"] += 1
        log.warning("relation %s: %s", source.accession, outcome.relation_error)
    if outcome.summary_error:
        stats["failed" if outcome.summary_error.startswith(("model call", "unreadable")) else
              "summary_rejected"] += 1
        log.warning("summary %s: %s", source.accession, outcome.summary_error[:300])
    relation = outcome.relation
    if relation is not None:
        stats[f"relation:{relation.relation}"] += 1
        stats["relation_shown"] += relation.shown
    if outcome.summary:
        stats["summary_done"] += 1
    if out is not None:
        out.write(json.dumps({
            "accession": source.accession, "dataset": source.name,
            "paper": source.paper.title if source.paper else None,
            "relation": relation.relation if relation else None,
            "shown": relation.shown if relation else None,
            "mismatches": relation.mismatches if relation else None,
            "evidence": relation.evidence if relation else None,
            "summary": outcome.summary, "summary_paper_used": outcome.summary_paper_used,
            "errors": [e for e in (outcome.relation_error, outcome.summary_error) if e] or None,
        }, ensure_ascii=False) + "\n")
        out.flush()
    if dry_run:
        return
    if relation is not None:
        assert source.paper is not None, "a relation needs a linked paper"
        value = {
            "paper_key": source.paper.paper_key, "relation": relation.relation,
            "shown": relation.shown, "mismatches": relation.mismatches,
            "evidence": relation.evidence, "input_sha": relation_input_sha(source),
            "prompt_version": RELATION_PROMPT_VERSION, "model": model, "checked_at": now,
        }
        await conn.execute(text(_WRITE_RELATION_SQL), {"id": dataset_id, "value": json.dumps(value)})
    if outcome.summary_asked and (outcome.summary or outcome.summary_error.startswith("rejected")):
        # A rejected summary is recorded too (without text), so the same input
        # is not asked again; a changed README or relation asks again.
        meta = {"input_sha": summary_input_sha(source, outcome.summary_paper_used),
                "paper_used": outcome.summary_paper_used, "prompt_version": SUMMARY_PROMPT_VERSION,
                "model": model, "generated_at": now, "rejected": not outcome.summary}
        await conn.execute(text(_WRITE_SUMMARY_SQL), {
            "id": dataset_id, "summary": outcome.summary, "meta": json.dumps(meta)})
    await conn.commit()
