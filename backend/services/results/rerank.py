"""
services/results/rerank.py
--------------------------
LLM re-ranking of the first result candidates.

The SQL search finds candidates; this step reads them. For a new question the
top RERANK_CANDIDATES datasets (in the order the SQL ranking gives) are shown
to an LLM together with the question, each as its profile
(services/indexing/readme_profile.py), or name, study fields and README opening when no
profile exists yet, plus its modalities, tasks and size. The model scores each
one:
    3  matches every requirement of the question
    2  matches the main requirement; a secondary one is not stated
    1  related, but misses a requirement
    0  does not match

The result order then becomes
    judged 3, 2, 1 (best first; ties keep the SQL order)
    → datasets beyond the candidate window, in SQL order
    → judged 0
Nothing is removed: a wrong judgement can cost a position, never a result.
The judgement is made once per question and cached with the query, so later
pages reuse it.

Re-ranking is skipped when the user asked for an order or a top N, when the
question returned the fallback sample, when there is nothing to reorder, and
whenever the model fails or is slower than RERANK_TIMEOUT_SECONDS.

Off unless RERANK_ENABLED is "true": the step sends candidate descriptions
(public catalogue metadata) to Gemini, which needs an explicit decision.

Config (env):
  RERANK_ENABLED          "true" turns it on (default off)
  RERANK_MODEL            models, comma-separated, in order (default: profile
                          rerank in backend/llm/llm_config.yaml)
  RERANK_THINKING         thinking level (default: that profile's, low)
  RERANK_CANDIDATES       how many datasets are judged (default 40)
  RERANK_BATCH            datasets per model call; calls run in parallel (default 20)
  RERANK_TIMEOUT_SECONDS  wall-clock limit for the whole step (default 20)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from services.database_stats.database_stats import database_stats
from services.schema.tables import datasets as _datasets
from services.switches import Switch
from services.translate import optional_columns

log = logging.getLogger(__name__)

PROFILE_COLUMN = "llm_profile"
README_EXCERPT_CHARS = 700

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "score": {"type": "integer", "enum": [0, 1, 2, 3]},
                },
                "required": ["n", "score"],
            },
        }
    },
    "required": ["judgments"],
}

SYSTEM_PROMPT = """\
You judge search results in a catalogue of open neuroimaging datasets. You are
strict about what the question requires and fair about what a short dataset
description can show."""

_INSTRUCTIONS = """\
Question: {question}
{interpretation}
Score every dataset below against the question:
  3  matches every requirement of the question
  2  matches the main requirement; a secondary requirement is not stated either way
  1  related to the question, but clearly misses a requirement
  0  does not match the question
The main requirement is what the question is about (a condition, a topic, a
task, a population). Modalities, tasks and sizes listed for a dataset are
facts from its files; the description is what the uploader wrote, summarised.
A condition that a dataset only excluded, measured with a questionnaire, or
mentions as background does not make it a dataset about that condition.

Answer with JSON: {{"judgments": [{{"n": 1, "score": 3}}, ...]}}, one entry per dataset.

Datasets
========
{candidates}"""


@dataclass
class RerankResult:
    ordered_ids: List[str] = field(default_factory=list)   # judged 3, 2, 1
    demoted_ids: List[str] = field(default_factory=list)   # judged 0
    scores: Dict[str, int] = field(default_factory=dict)


def settings() -> Dict[str, Any]:
    # Models, thinking level and per-call timeouts are in profile rerank of
    # llm_config.yaml; RERANK_MODEL / RERANK_THINKING override them.
    models = os.getenv("RERANK_MODEL", "")
    return {
        "enabled": Switch.RERANK_ENABLED.is_on(),
        "models": [m.strip() for m in models.split(",") if m.strip()],
        "thinking": os.getenv("RERANK_THINKING") or None,
        "candidates": max(2, int(os.getenv("RERANK_CANDIDATES", "40"))),
        "batch": max(1, int(os.getenv("RERANK_BATCH", "20"))),
        "timeout": float(os.getenv("RERANK_TIMEOUT_SECONDS", "20")),
    }


def should_rerank(apply_relevance: bool, total: int, is_fallback: bool) -> bool:
    if not settings()["enabled"] or is_fallback or not apply_relevance or total < 2:
        return False
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


# ── Candidate descriptions ────────────────────────────────────────────────────

def _details_sql() -> str:
    cols = ["d.id", "d.accession_id", "d.name"]
    for column in ("dx_status", "study_domain", "study_design", "age_min", "age_max", "size_bytes"):
        if optional_columns.usable(database_stats.coverage, column):
            cols.append(f"d.{column}")
    cols.append(f"d.{PROFILE_COLUMN}")
    if optional_columns.usable(database_stats.coverage, "readme_text"):
        cols.append(f"left(d.readme_text, {README_EXCERPT_CHARS}) AS readme_excerpt")
    return f"SELECT {', '.join(cols)} FROM {_datasets.name} d WHERE d.id = ANY(:ids)"


_FILES_SQL = text("""
    SELECT o.dataset_id,
           array_agg(DISTINCT o.datatype) FILTER (WHERE o.datatype IS NOT NULL) AS datatypes,
           (array_agg(DISTINCT o.task) FILTER (WHERE o.task IS NOT NULL))[1:8] AS tasks,
           count(DISTINCT o.subject) AS subjects
    FROM bids_objects o
    WHERE o.dataset_id = ANY(:ids)
    GROUP BY o.dataset_id
""")

_SPACE_RE = re.compile(r"\s+")


def _clip(value: Any, limit: int) -> str:
    text_value = _SPACE_RE.sub(" ", str(value or "")).strip()
    return text_value if len(text_value) <= limit else text_value[: limit - 1] + "…"


def describe(row: Dict[str, Any], files: Optional[Dict[str, Any]]) -> str:
    """One candidate as the model sees it."""
    lines = [f"Name: {_clip(row.get('name'), 160)}"]
    profile = row.get(PROFILE_COLUMN)
    if profile:
        lines.append(f"Description: {_clip(profile, 900)}")
    else:
        for key, label in (("dx_status", "Diagnosis field"), ("study_domain", "Study domain"),
                           ("study_design", "Study design")):
            if row.get(key):
                lines.append(f"{label}: {_clip(row[key], 160)}")
        if row.get("readme_excerpt"):
            lines.append(f"README opening: {_clip(row['readme_excerpt'], README_EXCERPT_CHARS)}")
    if files:
        if files.get("datatypes"):
            lines.append("Modalities: " + ", ".join(sorted(files["datatypes"])))
        if files.get("tasks"):
            lines.append("Tasks: " + ", ".join(files["tasks"]))
        if files.get("subjects"):
            lines.append(f"Subjects: {files['subjects']}")
    if row.get("age_min") is not None and row.get("age_max") is not None:
        lines.append(f"Ages: {row['age_min']:g}–{row['age_max']:g}")
    return "\n".join(lines)


def build_prompt(question: str, interpretation: Optional[str], descriptions: Sequence[str]) -> str:
    blocks = [f"[{n}]\n{desc}" for n, desc in enumerate(descriptions, start=1)]
    return _INSTRUCTIONS.format(
        question=question.strip(),
        interpretation=(f"Read as: {interpretation.strip()}\n" if interpretation else ""),
        candidates="\n\n".join(blocks),
    )


def parse_scores(raw: str, count: int) -> Dict[int, int]:
    """{position (1-based): score}; entries out of range or malformed are dropped."""
    text_value = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    data = json.loads(text_value)
    items = data.get("judgments") if isinstance(data, dict) else data
    scores: Dict[int, int] = {}
    for item in items or []:
        try:
            n, score = int(item["n"]), int(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= n <= count and 0 <= score <= 3:
            scores[n] = score
    return scores


def order_from_scores(candidate_ids: Sequence[str], scores: Dict[str, int]) -> RerankResult:
    """Judged 3, 2, 1 first (SQL order breaks ties); judged 0 to the end."""
    position = {cid: i for i, cid in enumerate(candidate_ids)}
    judged = [cid for cid in candidate_ids if cid in scores]
    ordered = sorted((cid for cid in judged if scores[cid] > 0),
                     key=lambda cid: (-scores[cid], position[cid]))
    demoted = [cid for cid in judged if scores[cid] == 0]
    return RerankResult(ordered_ids=ordered, demoted_ids=demoted,
                        scores={cid: scores[cid] for cid in judged})


# ── The step ──────────────────────────────────────────────────────────────────

def _judge_batch(question: str, interpretation: Optional[str], descriptions: List[str],
                 cfg: Dict[str, Any]) -> Dict[int, int]:
    from llm_client import llm_generate

    raw = llm_generate(
        build_prompt(question, interpretation, descriptions),
        system=SYSTEM_PROMPT,
        profile="rerank",
        models=cfg["models"],
        json_schema=RESPONSE_SCHEMA,
        thinking_level=cfg["thinking"],
    )
    return parse_scores(raw, len(descriptions))


async def rerank_candidates(
    session: Any,
    candidate_ids: Sequence[Any],
    question: str,
    interpretation: Optional[str] = None,
) -> Optional[RerankResult]:
    """Judge *candidate_ids* (already in SQL order); None when the step could not run."""
    cfg = settings()
    ids = [str(cid) for cid in candidate_ids][: cfg["candidates"]]
    if len(ids) < 2:
        return None
    uuids = [UUID(cid) for cid in ids]
    # In a savepoint: a failed read must not abort the search's transaction
    # (a rollback of all of it would also undo its limits).
    try:
        async with session.begin_nested():
            rows = (await session.execute(text(_details_sql()), {"ids": uuids})).mappings().all()
            files = (await session.execute(_FILES_SQL, {"ids": uuids})).mappings().all()
    except SQLAlchemyError as exc:
        log.warning("Re-ranking skipped: could not load candidates (%s)", exc)
        return None
    by_id = {str(r["id"]): dict(r) for r in rows}
    files_by_id = {str(f["dataset_id"]): dict(f) for f in files}
    ids = [cid for cid in ids if cid in by_id]
    descriptions = [describe(by_id[cid], files_by_id.get(cid)) for cid in ids]

    batches = [list(range(i, min(i + cfg["batch"], len(ids)))) for i in range(0, len(ids), cfg["batch"])]

    async def run(batch: List[int]) -> Dict[str, int]:
        scores = await asyncio.to_thread(
            _judge_batch, question, interpretation, [descriptions[i] for i in batch], cfg
        )
        return {ids[batch[n - 1]]: score for n, score in scores.items()}

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(run(b) for b in batches), return_exceptions=True),
            timeout=cfg["timeout"],
        )
    except asyncio.TimeoutError:
        log.warning("Re-ranking skipped: no answer within %.0f s", cfg["timeout"])
        return None
    from llm_client import LLMAllFailedError

    scores: Dict[str, int] = {}
    for result in results:
        if isinstance(result, LLMAllFailedError):  # the model could not be asked
            log.warning("Re-ranking batch failed: %s", str(result)[:300])
            continue
        if isinstance(result, BaseException):  # a bug, or a cancelled batch
            raise result
        scores.update(result)
    if not scores:
        return None
    ranked = order_from_scores(ids, scores)
    log.info("Re-ranked %d of %d candidates: %s", len(scores), len(ids),
             dict(sorted(_count(scores.values()).items(), reverse=True)))
    return ranked


def _count(values) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out
