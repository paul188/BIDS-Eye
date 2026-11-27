"""
services/stores/interpretations.py
----------------------------------
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
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select, update, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from services.switches import Switch

log = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parents[2]
# Every code and vocabulary file of these folders decides what a question is
# translated into, so all of them count, new ones included (a hand-kept list
# missed the compiler, criteria.py, cohort_count.py, senses.yaml and
# word_forms.yaml). Not recursive: __pycache__ stays out.
_VERSION_DIRS = (_BACKEND.parent / "LLM_preprocessor", _BACKEND.parent / "LLM_preprocessor" / "preprocess",
                 _BACKEND.parent / "RAG")
_VERSION_SUFFIXES = (".py", ".yaml", ".yml", ".json")
# The backend modules the translation imports (text_to_sql.py and everything
# it reaches; test_interpretations.py checks this list against the
# imports), and the few-shot examples. models.py and schemas.py are left out:
# they define the shape of rows and responses, not the translation; so is
# exceptions/domain.py (query_guard's SearchBusyError), which only sets HTTP answers,
# and exceptions/expected.py, the base class of the errors.
_VERSION_FILES = (
    _BACKEND / "services" / "database_stats" / "database_stats.py",
    _BACKEND / "services" / "schema" / "check.py",
    _BACKEND / "services" / "schema" / "coded_fields.py",
    _BACKEND / "services" / "schema" / "tables.py",
    _BACKEND / "services" / "translate" / "concept_keys.py",
    _BACKEND / "services" / "translate" / "optional_columns.py",
    _BACKEND / "services" / "translate" / "prompt" / "few_shot_examples.json",
    _BACKEND / "services" / "translate" / "prompt" / "schema_ddl.py",
    _BACKEND / "services" / "translate" / "participant_query_compiler.py",
    _BACKEND / "services" / "query_guard.py",
    _BACKEND / "services" / "translate" / "sql_rewriter.py",
    _BACKEND / "services" / "fulltext" / "text_search.py",
    _BACKEND / "services" / "fulltext" / "tsquery.py",
    _BACKEND / "services" / "fulltext" / "ranking.py",
    _BACKEND / "services" / "fulltext" / "vectors.py",
    _BACKEND / "services" / "fulltext" / "word_frequencies.py",
    _BACKEND / "services" / "results" / "rankers.py",
    _BACKEND / "services" / "translate" / "text_to_sql.py",
    _BACKEND / "services" / "translate" / "value_check.py",
    _BACKEND / "services" / "sql_tree.py",
    _BACKEND / "services" / "switches.py",
    _BACKEND / "services" / "uncertain" / "tier.py",
    _BACKEND / "services" / "uncertain" / "classify.py",
    _BACKEND / "services" / "uncertain" / "conditions.py",
    _BACKEND / "services" / "evidence" / "__init__.py",
    _BACKEND / "services" / "evidence" / "base.py",
    _BACKEND / "services" / "evidence" / "branches.py",
    _BACKEND / "services" / "evidence" / "confirmed.py",
    _BACKEND / "services" / "evidence" / "contradicted.py",
    _BACKEND / "services" / "evidence" / "possible.py",
    _BACKEND / "services" / "evidence" / "probable.py",
    _BACKEND / "services" / "uncertain" / "scanner.py",
    _BACKEND / "services" / "uncertain" / "fields" / "__init__.py",
    _BACKEND / "services" / "uncertain" / "fields" / "base.py",
    _BACKEND / "services" / "uncertain" / "fields" / "age.py",
    _BACKEND / "services" / "uncertain" / "fields" / "diagnosis.py",
    _BACKEND / "services" / "uncertain" / "fields" / "sex.py",
    _BACKEND / "services" / "uncertain" / "text.py",
    _BACKEND / "services" / "fulltext" / "word_form_counts.py",
    _BACKEND / "worker_db.py",
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
    return Switch.INTERPRETATION_CACHE.is_on()


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


_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-"})


def normalize(question: str) -> str:
    """The text two questions must share to count as the same question.

    Case, spacing, the kind of quote marks and dashes, invisible characters
    (zero-width spaces, soft hyphens, as pasted from PDFs) and punctuation at
    the end are ignored. Every word and number stays: "with" and "without",
    "at least 40" and "at most 40" must never share a key. Two questions that
    share a key get the same stored translation for good, so this is an
    equality of text, not of meaning: no stopwords, stems or synonyms.

    Also the key under which identical running searches are merged
    (services/search.py ask). A change here needs no new pipeline_version: a
    stored row is the translation of its normalised text, whatever form the
    question came in.
    """
    text = unicodedata.normalize("NFKC", question or "").translate(_QUOTES)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".?! ").strip()


def _key(question: str) -> str:
    return hashlib.sha256(f"{pipeline_version()}\n{normalize(question)}".encode()).hexdigest()


async def get(session: AsyncSession, question: str) -> Optional[Dict[str, Any]]:
    """The stored translation of *question* for the running code version, or None.

    A database error reading it gives None (the question is translated anew);
    one counting the hit only costs the count: the translation is still
    returned, since translating anew is what the store exists to avoid."""
    if not enabled():
        return None
    from models import QueryInterpretation
    try:
        row = (await session.execute(
            select(QueryInterpretation).where(QueryInterpretation.key == _key(question))
        )).scalar_one_or_none()
    except SQLAlchemyError:
        log.warning("interpretation cache: lookup failed", exc_info=True)
        await session.rollback()
        return None
    if row is None or row.review_verdict == "wrong":
        return None
    key = row.key
    try:
        payload = json.loads(row.payload)
    except ValueError:  # JSONDecodeError; written by put(), so never expected
        log.error("interpretation cache: stored payload of %s is not JSON", key)
        return None
    try:
        await session.execute(
            update(QueryInterpretation).where(QueryInterpretation.key == key)
            .values(hits=QueryInterpretation.hits + 1, last_used_at=func.now())
        )
        await session.commit()
    except SQLAlchemyError:
        log.warning("interpretation cache: hit of %s not counted", key, exc_info=True)
        await session.rollback()
    return payload


async def log_query(session: AsyncSession, question: str, payload: Dict[str, Any], total: int,
                    *, from_cache: bool, relaxed: bool, self_corrected: bool, fallback: bool) -> None:
    """Write one row to query_log for a search that was run. Always on.

    A database error is logged and the search answers anyway; anything else
    (a payload json cannot encode) is a bug and is raised.
    """
    from models import QueryLog
    row = QueryLog(
        question=question, pipeline_version=pipeline_version(),
        payload=json.dumps(payload, ensure_ascii=False), total=total,
        from_cache=from_cache, relaxed=relaxed, self_corrected=self_corrected, fallback=fallback,
    )
    try:
        # A savepoint keeps a failed insert from breaking the response; the
        # commit makes the row last (get_db never commits).
        async with session.begin_nested():
            session.add(row)
        await session.commit()
    except SQLAlchemyError:
        log.warning("query log: insert failed", exc_info=True)
        await session.rollback()  # a failed commit leaves the session unusable until then


async def put(session: AsyncSession, question: str, payload: Dict[str, Any], total: int) -> None:
    """Store a translation that found something.

    The first translation of a question stays: when two workers store the
    same question at once, the second insert does nothing (ON CONFLICT), where
    it used to fail on the primary key and log a traceback. A database error
    is logged and the search answers anyway; anything else is a bug and is
    raised.
    """
    if not enabled() or total <= 0:
        return
    from sqlalchemy.dialects.postgresql import insert

    from models import QueryInterpretation
    statement = insert(QueryInterpretation).values(
        key=_key(question), pipeline_version=pipeline_version(), question=question,
        question_normalized=normalize(question),
        payload=json.dumps(payload, ensure_ascii=False), total_at_creation=total,
    ).on_conflict_do_nothing(index_elements=[QueryInterpretation.key])
    try:
        await session.execute(statement)
        await session.commit()
    except SQLAlchemyError:
        log.warning("interpretation cache: store failed", exc_info=True)
        await session.rollback()
