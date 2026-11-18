"""
services/stores/interpretation_store.py
---------------------------------------
Durable record of how each question was translated into SQL.

Why: the translation is made by language models and is not repeatable. The
same question gave different SQL, and different results, for 19 of 69 requests
in two runs of the same code (benchmarks/demand, 2026-11-15); one request went
from 255 results to none. A stored translation is reused for the same question,
so a user sees the same answer every time, and the rows can be checked later,
offline, against the database (review_* columns).

What is stored is the SQL, not the result: it is run again on every request, so
new datasets show up. A translation is stored only when it ran as written and
found a real result (services/search.py _worth_storing): an empty, relaxed or
repaired search must not become permanent. A row belongs to one version of the
translating code (pipeline_version, a hash of its source files), so a deploy
that changes the pipeline starts with a fresh cache and keeps the old rows for
comparison.

Off unless INTERPRETATION_CACHE=true. The benchmark runner never reads it: it
calls text_to_sql() directly, and only the router looks here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parents[2]
# Every code and vocabulary file of these folders decides what a question is
# translated into, so all of them count, new ones included (a hand-kept list
# missed the compiler, criteria.py, cohort_count.py, senses.yaml and
# word_forms.yaml). Not recursive: __pycache__ stays out.
_VERSION_DIRS = (_BACKEND.parent / "LLM_preprocessor", _BACKEND.parent / "RAG")
_VERSION_SUFFIXES = (".py", ".yaml", ".yml", ".json")
# The backend modules the translation imports (text_to_sql.py and everything
# it reaches; test_interpretation_store.py checks this list against the
# imports), and the few-shot examples. models.py and schemas.py are left out:
# they define the shape of rows and responses, not the translation.
_VERSION_FILES = (
    _BACKEND / "constants.py",
    _BACKEND / "few_shot_examples.json",
    _BACKEND / "services" / "catalogue" / "catalogue_values.py",
    _BACKEND / "services" / "translate" / "concept_keys.py",
    _BACKEND / "services" / "translate" / "participant_query_compiler.py",
    _BACKEND / "services" / "query_guard.py",
    _BACKEND / "services" / "catalogue" / "schema_capabilities.py",
    _BACKEND / "services" / "translate" / "sql_rewriter.py",
    _BACKEND / "services" / "fulltext" / "text_search.py",
    _BACKEND / "services" / "translate" / "text_to_sql.py",
    _BACKEND / "services" / "stores" / "translation_store.py",
    _BACKEND / "services" / "fulltext" / "uncertain_tier.py",
    _BACKEND / "services" / "catalogue" / "word_form_store.py",
)
_VERSION_ENV = ("UNCERTAIN_TIER", "UNCERTAIN_POSSIBLE", "UNCERTAIN_MAX_UNSURE",
                "TEXT_SEARCH_PAPERS", "TEXT_SEARCH_EXACT", "GEMINI_MODEL_CASCADE",
                "ANTHROPIC_FALLBACK_MODEL")
_version: Optional[str] = None


def version_paths() -> list:
    """The files pipeline_version() reads, in a fixed order."""
    paths = list(_VERSION_FILES)
    for directory in _VERSION_DIRS:
        if directory.is_dir():
            paths.extend(sorted(p for p in directory.iterdir()
                                if p.is_file() and p.suffix in _VERSION_SUFFIXES))
    return paths


def enabled() -> bool:
    return os.getenv("INTERPRETATION_CACHE", "").strip().lower() in ("1", "true", "yes", "on")


def pipeline_version() -> str:
    """Hash of the translating code and vocabulary, plus the settings that change its output."""
    global _version
    if _version is None:
        digest = hashlib.sha256()
        for path in version_paths():
            # The name too: a renamed or added file is a different pipeline.
            digest.update(f"{path.parent.name}/{path.name}\0".encode())
            try:
                digest.update(path.read_bytes())
            except OSError:
                digest.update(b"missing")
        for name in _VERSION_ENV:
            digest.update(f"{name}={os.getenv(name, '')};".encode())
        _version = digest.hexdigest()[:16]
    return _version


def normalize(question: str) -> str:
    """The form under which a question is stored: case, spacing and end punctuation do not matter."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    return text.rstrip(" ?.!")


def _key(question: str) -> str:
    return hashlib.sha256(f"{pipeline_version()}\n{normalize(question)}".encode()).hexdigest()


async def get(session: AsyncSession, question: str) -> Optional[Dict[str, Any]]:
    """The stored translation of *question* for the running code version, or None."""
    if not enabled():
        return None
    from models import QueryInterpretation
    try:
        row = (await session.execute(
            select(QueryInterpretation).where(QueryInterpretation.key == _key(question))
        )).scalar_one_or_none()
        if row is None or row.review_verdict == "wrong":
            return None
        await session.execute(
            update(QueryInterpretation).where(QueryInterpretation.key == row.key)
            .values(hits=QueryInterpretation.hits + 1, last_used_at=func.now())
        )
        await session.commit()
        return json.loads(row.payload)
    except Exception:
        log.exception("interpretation cache: lookup failed")
        await session.rollback()
        return None


async def log_query(session: AsyncSession, question: str, payload: Dict[str, Any], total: int,
                    *, from_cache: bool, relaxed: bool, self_corrected: bool, fallback: bool) -> None:
    """Write one row to query_log for a search that was run. Always on; never raises."""
    from models import QueryLog
    try:
        # A savepoint keeps a failed insert from breaking the response; the
        # commit makes the row last (get_db never commits).
        async with session.begin_nested():
            session.add(QueryLog(
                question=question, pipeline_version=pipeline_version(),
                payload=json.dumps(payload, ensure_ascii=False), total=total,
                from_cache=from_cache, relaxed=relaxed, self_corrected=self_corrected, fallback=fallback,
            ))
        await session.commit()
    except Exception:
        log.warning("query log: insert failed", exc_info=True)


async def put(session: AsyncSession, question: str, payload: Dict[str, Any], total: int) -> None:
    """Store a translation that found something. Never raises."""
    if not enabled() or total <= 0:
        return
    from models import QueryInterpretation
    try:
        key = _key(question)
        exists = (await session.execute(
            select(QueryInterpretation.key).where(QueryInterpretation.key == key)
        )).scalar_one_or_none()
        if exists is None:
            session.add(QueryInterpretation(
                key=key, pipeline_version=pipeline_version(), question=question,
                question_normalized=normalize(question),
                payload=json.dumps(payload, ensure_ascii=False), total_at_creation=total,
            ))
            await session.commit()
    except Exception:
        log.exception("interpretation cache: store failed")
        await session.rollback()
