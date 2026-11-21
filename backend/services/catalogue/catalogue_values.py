"""
services/catalogue/catalogue_values.py
--------------------------------------
Caches the datatype, suffix and task values that occur in the catalogue.
services/translate/value_check.py uses them to validate the values in
generated SQL.

The values are read from bids_dataset_facets (crawlers/migrate_add_dataset_facets.sql)
at startup (schema_capabilities) and again before a search once the cached
copy is REFRESH_SECONDS old. When that table is missing, nothing is cached
and the value check is switched off.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from exceptions.expected import ExpectedError
from services.catalogue import schema_capabilities

log = logging.getLogger(__name__)

FIELDS = ("datatype", "suffix", "task")
FACETS_TABLE = "bids_dataset_facets"
# The facets table is refreshed after each crawl (post_crawl.py), so a value
# added by a crawl is wrongly rejected for at most 10 minutes.
REFRESH_SECONDS = 600


class CatalogueLoadError(ExpectedError):
    """The values could not be read from the facets table."""


# ── The cached values ─────────────────────────────────────────────────────────

# The cached values per field. None means nothing is cached and the value
# check is off.
_known: dict[str, frozenset[str]] | None = None


def loaded() -> bool:
    """True when values are cached. Without them the value check is off."""
    return _known is not None


def known_values(field: str) -> frozenset[str]:
    """The values of *field* that occur in some dataset."""
    return (_known or {}).get(field, frozenset())


def is_known(field: str, value: str) -> bool:
    return value in known_values(field)


def set_known(values: Mapping[str, Iterable[str]]) -> None:
    """Replace the cached values. Called by load() and by tests."""
    global _known
    _known = {field: frozenset(values.get(field, ())) for field in FIELDS}


def clear() -> None:
    """Drop the cached values, which switches the value check off."""
    global _known
    _known = None


# ── Loading them from the facets table ────────────────────────────────────────

# time.monotonic() of the last try_load(), whether it succeeded or not.
_last_attempt: float | None = None


async def load(connection: AsyncConnection | AsyncSession) -> None:
    """Read the values from the facets table and cache them.

    The reads run in a savepoint, so the caller's transaction stays usable
    when they fail. On failure this raises CatalogueLoadError and the cached
    values stay as they were.
    """
    values: dict[str, list[str]] = {}
    try:
        async with connection.begin_nested():
            for field in FIELDS:
                # Not an injection risk: field and table name are constants of this module.
                result = await connection.execute(text(
                    f"SELECT DISTINCT {field} FROM {FACETS_TABLE} WHERE {field} IS NOT NULL"))
                values[field] = list(result.scalars().all())
    except SQLAlchemyError as exc:
        raise CatalogueLoadError(
            f"reading {', '.join(FIELDS)} from {FACETS_TABLE} failed: {exc}") from exc
    set_known(values)
    log.info("Catalogue values: %s",
             ", ".join(f"{len(field_values)} {field}" for field, field_values in values.items()))


async def try_load(connection: AsyncConnection | AsyncSession) -> None:
    """load(), but never raises CatalogueLoadError.

    If the read fails, the old values stay cached, or the value check stays
    off if there were none. The value check is optional: it only catches
    wrong values in generated SQL, so a failed read must not fail startup or
    a search. refresh_if_stale() tries again after REFRESH_SECONDS.
    """
    global _last_attempt
    try:
        await load(connection)
    except CatalogueLoadError as exc:
        note = "keeping the old values" if loaded() else "value check off for now"
        log.warning("%s; %s", exc, note)
    # Set only after load() returns or raises CatalogueLoadError. A bug in
    # load() (any other exception) skips this line, so every search fails
    # until it is fixed instead of the bug going quiet for REFRESH_SECONDS.
    # Two searches arriving together may both reload; that is only a
    # duplicate query.
    _last_attempt = time.monotonic()


async def refresh_if_stale(connection: AsyncConnection | AsyncSession) -> None:
    """try_load() when the facets table is in use and the last try is REFRESH_SECONDS old."""
    if not schema_capabilities.has_dataset_facets():
        return
    if _last_attempt is not None and time.monotonic() - _last_attempt < REFRESH_SECONDS:
        return
    await try_load(connection)
