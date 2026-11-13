"""
services/schema_capabilities.py
-------------------------------
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
(services/text_search.py). Without it text search falls back to ILIKE.

After a migration or backfill, restart the backend to pick up the new state.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Collection, FrozenSet, List

from sqlalchemy import text

from constants import OPTIONAL_DATASET_COLUMNS

log = logging.getLogger(__name__)

# A column counts as populated when at least this share of datasets has a
# value. Below that, a filter on it would discard almost every dataset.
MIN_COVERAGE = 0.05

_available: FrozenSet[str] = frozenset()

# Columns the API reads directly (not through generated SQL). For these only
# existence matters, not how many rows are filled.
# llm_profile is read by the re-ranker (services/rerank.py).
DISPLAY_COLUMNS = ("deprecated_reason", "llm_profile")
_present_display: FrozenSet[str] = frozenset()


# Stored tsvector behind text_match(); see crawlers/migrate_add_search_tsv.sql.
FTS_COLUMN = "search_tsv"
_has_fts: bool = False


def has_fts() -> bool:
    """True if bids_datasets has the full-text column in the running database."""
    return _has_fts


def set_fts(available: bool) -> None:
    """Override the detected full-text capability (tests)."""
    global _has_fts
    _has_fts = bool(available)


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
    global _present_display, _has_fts
    _present_display = frozenset(name for name in DISPLAY_COLUMNS if name in present)
    _has_fts = FTS_COLUMN in present
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
    """Detect and store the state. On any error, assume no optional column."""
    global _available, _present_display, _has_fts
    try:
        async with engine.connect() as connection:
            _available = await detect_available_columns(connection)
    except Exception as exc:
        log.warning("Could not inspect optional dataset columns (%s) — hiding all of them", exc)
        _available = frozenset()
        _present_display = frozenset()
        _has_fts = False
    return _available
