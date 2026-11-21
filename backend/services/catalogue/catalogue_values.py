"""
services/catalogue/catalogue_values.py
--------------------------------------
Caches, for each coded field, the values that occur in the catalogue and how
many datasets have each. A value is known when at least one dataset has it.

  datatype, suffix, task         from bids_dataset_facets
                                 (crawlers/migrate_add_dataset_facets.sql)
  diagnosis, sex, handedness     from bids_participants

Users:
  services/translate/value_check.py   checks the datatype, suffix and task
                                      values in generated SQL
  LLM_preprocessor/preprocess         drops vocabulary codes no dataset has
                                      (the DB MISS warning to the SQL model)

The cache grows with the number of distinct codes, not with the datasets:
about 5,500 values on 2026-11-21 (3,389 suffixes, 1,709 tasks, 199
diagnoses). Free text (dx_status) and the optional participant columns are
left out on purpose: they would grow with every dataset.

The values are read at startup (schema_capabilities) and again before a
search once the cached copy is REFRESH_SECONDS old, one query per table. A
field is cached only after its table was read: while the facets table is
missing or incomplete, the facet fields are not cached and the value check
is off.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from services.catalogue import schema_capabilities

log = logging.getLogger(__name__)

FACETS_TABLE = "bids_dataset_facets"
PARTICIPANTS_TABLE = "bids_participants"
FACET_FIELDS = ("datatype", "suffix", "task")
PARTICIPANT_FIELDS = ("diagnosis", "sex", "handedness")
FIELDS = FACET_FIELDS + PARTICIPANT_FIELDS
# The facets table is refreshed after each crawl (post_crawl.py), so a value
# added by a crawl is wrongly rejected for at most 10 minutes.
REFRESH_SECONDS = 600


# ── The cached values ─────────────────────────────────────────────────────────

# Per field: each value and the number of datasets that have it. A field is
# missing until its table was read.
_counts: dict[str, dict[str, int]] = {}


def loaded(field: str) -> bool:
    """True when the values of *field* are cached."""
    return field in _counts


def known_values(field: str) -> Collection[str]:
    """The values of *field* that occur in some dataset; empty when not cached."""
    return _counts.get(field, {}).keys()


def dataset_count(field: str, value: str) -> int:
    """How many datasets have *value* in *field*; 0 when unknown or not cached."""
    return _counts.get(field, {}).get(value, 0)


def clear() -> None:
    """Drop every cached value, which switches the value check off."""
    global _counts
    _counts = {}


# ── Loading them ──────────────────────────────────────────────────────────────

# time.monotonic() of the last load(), whether it succeeded or not.
_last_attempt: float | None = None


def _count_sql(table: str, fields: tuple[str, ...]) -> str:
    """One statement for every field of *table*: rows of (field, value, datasets)."""
    # Not an injection risk: table and field names are constants of this module.
    return " UNION ALL ".join(
        f"SELECT '{field}', {field}, count(DISTINCT dataset_id) FROM {table} "
        f"WHERE {field} IS NOT NULL GROUP BY {field}"
        for field in fields)


async def _read(connection: AsyncConnection | AsyncSession, table: str,
                fields: tuple[str, ...]) -> dict[str, dict[str, int]] | None:
    """The counts of *fields* from *table*, or None when the read failed.

    The read runs in a savepoint, so a failure leaves the caller's
    transaction usable.
    """
    try:
        async with connection.begin_nested():
            rows = (await connection.execute(text(_count_sql(table, fields)))).all()
    except SQLAlchemyError as exc:
        log.warning("Reading %s from %s failed (%s); %s", ", ".join(fields), table, exc,
                    "keeping the old values" if loaded(fields[0]) else "not checked for now")
        return None
    counts: dict[str, dict[str, int]] = {field: {} for field in fields}
    for field, value, datasets in rows:
        counts[field][value] = int(datasets)
    return counts


async def load(connection: AsyncConnection | AsyncSession) -> None:
    """Read the values and their dataset counts, one query per table, and cache them.

    The facet fields are read only while schema_capabilities finds the facets
    table complete; otherwise they are dropped from the cache. A table whose
    read fails keeps its old values, or stays uncached if there were none:
    the cache only serves optional checks, so a failed read must not fail
    startup or a search. refresh_if_stale() tries again after REFRESH_SECONDS.
    """
    global _counts, _last_attempt
    tables = [(PARTICIPANTS_TABLE, PARTICIPANT_FIELDS)]
    counts = dict(_counts)
    if schema_capabilities.has_dataset_facets():
        tables.insert(0, (FACETS_TABLE, FACET_FIELDS))
    else:
        for field in FACET_FIELDS:
            counts.pop(field, None)
    for table, fields in tables:
        read = await _read(connection, table, fields)
        if read is not None:
            counts.update(read)
    # One assignment, so a search never sees half of a reload.
    _counts = counts
    log.info("Catalogue values: %s",
             ", ".join(f"{len(_counts[field])} {field}" for field in FIELDS if field in _counts)
             or "none")
    # Set only after the reads succeeded or failed with a database error. A
    # bug (any other exception) skips this line, so every search fails until
    # it is fixed instead of the bug going quiet for REFRESH_SECONDS. Two
    # searches arriving together may both reload; that is only a duplicate
    # query.
    _last_attempt = time.monotonic()


async def refresh_if_stale(connection: AsyncConnection | AsyncSession) -> None:
    """load() when the last load is REFRESH_SECONDS old."""
    if _last_attempt is not None and time.monotonic() - _last_attempt < REFRESH_SECONDS:
        return
    await load(connection)
