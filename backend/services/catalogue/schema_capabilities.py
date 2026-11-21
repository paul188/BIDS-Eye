"""
services/catalogue/schema_capabilities.py
-----------------------------------------
Which optional bids_datasets columns can the SQL model rely on?

Columns such as readme_text or institutions are added by migrations and filled
by backfill scripts, so a given database may lack them or hold them empty.
Telling the SQL model about such a column produces SQL that fails (column
missing) or silently returns nothing (column empty).

At startup the backend looks at the database once and records which optional
columns exist and are populated. Prompt building then
  - renders only those columns into the schema (constants.build_schema_ddl),
  - drops few-shot examples that use any other optional column,
  - widens the disease text fallback with dx_status / study_domain when present.

The same inspection records whether the full-text column search_tsv exists
(services/fulltext/text_search.py). Without it text search falls back to ILIKE. It
also records search_tsv_papers, the same vector plus the text of the papers a
dataset names (crawlers/migrate_add_paper_search.sql), which text search uses
when TEXT_SEARCH_PAPERS is on.

Likewise search_tsv_exact and search_tsv_exact_papers, the same vectors
without stemming, and the table bids_word_forms
(crawlers/migrate_add_exact_search.sql). Text search uses them when
TEXT_SEARCH_EXACT is on, for words that must match in the form asked for.

After a migration or backfill, restart the backend to pick up the new state.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any, Collection, FrozenSet, List

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from constants import OPTIONAL_DATASET_COLUMNS

log = logging.getLogger(__name__)

# A column counts as populated when at least this share of datasets has a
# value. Below that, a filter on it would discard almost every dataset.
MIN_COVERAGE = 0.05

_available: FrozenSet[str] = frozenset()

# Columns the API reads directly (not through generated SQL). For these only
# existence matters, not how many rows are filled.
# llm_profile is read by the re-ranker (services/results/rerank.py); readme_text gives
# each result its README excerpt; paper_relation and dataset_summary its paper
# line and generated summary (services/results/readme_snippets.py, services/results/dataset_cards.py).
DISPLAY_COLUMNS = ("deprecated_reason", "llm_profile", "readme_text", "paper_relation",
                   "dataset_summary")
_present_display: FrozenSet[str] = frozenset()


# Stored tsvector behind text_match(); see crawlers/migrate_add_search_tsv.sql.
FTS_COLUMN = "search_tsv"
_has_fts: bool = False


def has_fts() -> bool:
    """True if bids_datasets has the full-text column in the running database."""
    return _has_fts


# search_tsv plus the dataset's paper text; see crawlers/migrate_add_paper_search.sql.
PAPER_FTS_COLUMN = "search_tsv_papers"
_has_paper_fts: bool = False


def has_paper_fts() -> bool:
    """True if bids_datasets has the full-text column that includes paper text."""
    return _has_paper_fts


def set_paper_fts(available: bool) -> None:
    """Override the detected paper full-text capability (tests)."""
    global _has_paper_fts
    _has_paper_fts = bool(available)


def set_fts(available: bool) -> None:
    """Override the detected full-text capability (tests)."""
    global _has_fts
    _has_fts = bool(available)


# Per-dataset summaries of bids_objects (crawlers/migrate_add_dataset_facets.sql).
# The executed SQL reads them instead of the file table (sql_rewriter._use_summaries).
FACETS_TABLE = "bids_dataset_facets"
SUBJECT_COUNT_COLUMN = "n_subjects"
_has_facets: bool = False
_has_subject_counts: bool = False


def has_dataset_facets() -> bool:
    """True if bids_dataset_facets exists and covers every dataset that has files."""
    return _has_facets


def has_subject_counts() -> bool:
    """True if bids_datasets.n_subjects exists and is filled."""
    return _has_subject_counts


def set_summaries(facets: bool, subject_counts: bool) -> None:
    """Override the detected summary tables (tests)."""
    global _has_facets, _has_subject_counts
    _has_facets, _has_subject_counts = bool(facets), bool(subject_counts)


async def _detect_summaries(connection: Any, present: Collection[str]) -> None:
    """Record whether the per-dataset summaries can replace the file table.

    A summary that misses a dataset would hide it from the search, so the
    facets table counts only when no dataset with files lacks rows in it.

    The reads run in a savepoint: *connection* goes on to the other checks
    of detect_available_columns, and a failed statement outside one would
    abort it and make them fail too.
    """
    global _has_facets, _has_subject_counts
    _has_facets = _has_subject_counts = False
    try:
        async with connection.begin_nested():
            if SUBJECT_COUNT_COLUMN in present:
                filled = (await connection.execute(text(
                    f"SELECT count({SUBJECT_COUNT_COLUMN}) FROM bids_datasets"))).scalar()
                _has_subject_counts = bool(filled)
            exists = (await connection.execute(text(
                f"SELECT to_regclass('{FACETS_TABLE}') IS NOT NULL"))).scalar()
            if exists:
                uncovered = (await connection.execute(text(
                    "SELECT count(*) FROM bids_datasets d "
                    f"WHERE NOT EXISTS (SELECT 1 FROM {FACETS_TABLE} f WHERE f.dataset_id = d.id) "
                    "AND EXISTS (SELECT 1 FROM bids_objects o WHERE o.dataset_id = d.id)"))).scalar()
                _has_facets = not uncovered
                if uncovered:
                    log.warning("%s misses %d datasets that have files: not used. Run "
                                "remote/dataset_facets.py --refresh-all and restart.", FACETS_TABLE, uncovered)
    except SQLAlchemyError as exc:  # a database without these: search as before
        log.info("Dataset summaries not usable: %s", exc)
        _has_facets = _has_subject_counts = False
    # The values generated SQL is checked against (services/catalogue/catalogue_values.py).
    from services.catalogue import catalogue_values
    catalogue_values.clear()
    if _has_facets:
        await catalogue_values.try_load(connection)
    log.info("Dataset summaries: facets table %s, stored subject count %s",
             "used" if _has_facets else "not used", "used" if _has_subject_counts else "not used")
# The two vectors without stemming, and the catalogue's word forms; see
# crawlers/migrate_add_exact_search.sql.
EXACT_FTS_COLUMN = "search_tsv_exact"
EXACT_PAPER_FTS_COLUMN = "search_tsv_exact_papers"
WORD_FORMS_TABLE = "bids_word_forms"
_has_exact_fts: bool = False
_has_exact_paper_fts: bool = False
_has_word_forms: bool = False


def has_exact_fts() -> bool:
    """True if bids_datasets has the unstemmed full-text column."""
    return _has_exact_fts


def has_exact_paper_fts() -> bool:
    """True if bids_datasets has the unstemmed column that includes paper text."""
    return _has_exact_paper_fts


def has_word_forms() -> bool:
    """True if the table of the catalogue's word forms exists."""
    return _has_word_forms


def set_exact_fts(available: bool, papers: bool = False, word_forms: bool = False) -> None:
    """Override the detected exact-search capabilities (tests)."""
    global _has_exact_fts, _has_exact_paper_fts, _has_word_forms
    _has_exact_fts = bool(available)
    _has_exact_paper_fts = bool(available and papers)
    _has_word_forms = bool(word_forms)


def has_display_column(name: str) -> bool:
    """True if bids_datasets has this display column in the running database."""
    return name in _present_display


def set_display_columns(columns: Collection[str]) -> None:
    """Override the detected display columns (tests)."""
    global _present_display
    _present_display = frozenset(columns)


def available_columns() -> FrozenSet[str]:
    """Optional bids_datasets columns that exist and are populated."""
    return _available


def set_available_columns(columns: Collection[str]) -> None:
    """Override the detected state (tests, or an operator who knows better)."""
    global _available
    unknown = set(columns) - set(OPTIONAL_DATASET_COLUMNS)
    if unknown:
        raise ValueError(f"not optional dataset columns: {sorted(unknown)}")
    _available = frozenset(columns)


def unavailable_columns() -> FrozenSet[str]:
    return frozenset(OPTIONAL_DATASET_COLUMNS) - _available


def uses_unavailable_column(sql: str) -> bool:
    """True if *sql* references an optional column the database cannot serve."""
    missing = unavailable_columns()
    if not missing:
        return False
    pattern = r"\b(?:" + "|".join(sorted(map(re.escape, missing))) + r")\b"
    return re.search(pattern, sql) is not None


def disease_text_columns() -> List[str]:
    """Short free-text columns for substring search, most specific first.

    Used when the database has no full-text column: text_match() then expands
    to ILIKE on these. dx_status and study_domain are short fields about the
    cohort and the topic, so a match there is strong evidence. README text is
    left out on purpose: a plain substring match on it is noisy ("bipolar
    channels", "Beck Depression Inventory").
    """
    columns = ["name", "description_text"]
    columns += [c for c in ("dx_status", "study_domain") if c in _available]
    return columns


def _non_empty_expression(column: str, sql_type: str) -> str:
    if sql_type.endswith("[]"):
        return f"count(*) FILTER (WHERE cardinality({column}) > 0)"
    if sql_type == "TEXT":
        return f"count(*) FILTER (WHERE NULLIF(btrim({column}), '') IS NOT NULL)"
    return f"count({column})"


async def _table_exists(connection: Any, table: str) -> bool:
    # to_regclass answers NULL for a missing table; it fails only when the
    # connection does, and then so does every other check.
    result = await connection.execute(
        text("SELECT to_regclass(:table) IS NOT NULL"), {"table": table})
    return bool(result.scalar_one())


async def detect_available_columns(connection: Any) -> FrozenSet[str]:
    """Inspect bids_datasets and return the optional columns that hold data.

    *connection* is an AsyncConnection or AsyncSession. Column names come from
    OPTIONAL_DATASET_COLUMNS, never from user input.
    """
    rows = await connection.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'bids_datasets'"
        )
    )
    present = {row[0] for row in rows}
    global _present_display, _has_fts, _has_paper_fts
    global _has_exact_fts, _has_exact_paper_fts, _has_word_forms
    _present_display = frozenset(name for name in DISPLAY_COLUMNS if name in present)
    _has_fts = FTS_COLUMN in present
    _has_paper_fts = _has_fts and PAPER_FTS_COLUMN in present
    await _detect_summaries(connection, present)
    _has_exact_fts = _has_fts and EXACT_FTS_COLUMN in present
    _has_exact_paper_fts = (_has_exact_fts and _has_paper_fts
                            and EXACT_PAPER_FTS_COLUMN in present)
    _has_word_forms = _has_exact_fts and await _table_exists(connection, WORD_FORMS_TABLE)
    if _has_exact_fts:
        log.info("Unstemmed full-text column %s found%s%s (used when TEXT_SEARCH_EXACT is on)",
                 EXACT_FTS_COLUMN,
                 ", with paper text" if _has_exact_paper_fts else "",
                 ", word forms table present" if _has_word_forms else ", no word forms table")
    if _has_paper_fts:
        log.info("Full-text column %s found (used when TEXT_SEARCH_PAPERS is on)", PAPER_FTS_COLUMN)
    log.info(
        "Full-text column %s %s", FTS_COLUMN,
        "found → text_match() uses full-text search and ranking" if _has_fts
        else "missing → text_match() falls back to ILIKE on short text columns",
    )
    candidates = [name for name in OPTIONAL_DATASET_COLUMNS if name in present]
    absent = [name for name in OPTIONAL_DATASET_COLUMNS if name not in present]
    if absent:
        log.info("Optional dataset columns not in this database: %s", ", ".join(absent))
    if not candidates:
        return frozenset()

    counts_sql = ", ".join(
        f"{_non_empty_expression(name, OPTIONAL_DATASET_COLUMNS[name][0])} AS {name}"
        for name in candidates
    )
    row = (
        await connection.execute(text(f"SELECT count(*) AS total, {counts_sql} FROM bids_datasets"))
    ).mappings().one()
    total = int(row["total"] or 0)
    needed = max(1, math.ceil(MIN_COVERAGE * total))

    usable = set()
    for name in candidates:
        filled = int(row[name] or 0)
        enabled = total > 0 and filled >= needed
        log.info(
            "Optional dataset column %-13s %5d / %d datasets → %s",
            name, filled, total, "shown to the SQL model" if enabled else "hidden (too sparse)",
        )
        if enabled:
            usable.add(name)
    return frozenset(usable)


async def refresh_available_columns(engine: Any) -> FrozenSet[str]:
    """Detect and store the state. When the database cannot be read, assume
    no optional column, no full-text column and no summary table. A bug in
    the detection is raised: startup fails instead of running degraded."""
    global _available, _present_display, _has_fts, _has_paper_fts
    global _has_exact_fts, _has_exact_paper_fts, _has_word_forms
    try:
        async with engine.connect() as connection:
            _available = await detect_available_columns(connection)
    except (SQLAlchemyError, OSError, asyncio.TimeoutError) as exc:
        log.warning("Could not inspect optional dataset columns (%s) — hiding all of them", exc)
        from services.catalogue import catalogue_values
        set_summaries(False, False)
        catalogue_values.clear()
        _available = frozenset()
        _present_display = frozenset()
        _has_fts = False
        _has_paper_fts = False
        _has_exact_fts = False
        _has_exact_paper_fts = False
        _has_word_forms = False
    return _available
