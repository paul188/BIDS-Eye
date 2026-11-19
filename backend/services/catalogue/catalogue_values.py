"""
services/catalogue/catalogue_values.py
--------------------------------------
The datatype, suffix and task values that occur in some dataset, for the
check of generated SQL (services/translate/value_check.py).

They come from bids_dataset_facets (one row per distinct dataset, datatype,
suffix and task; crawlers/migrate_add_dataset_facets.sql): three DISTINCT
reads of about 21,000 rows. Without that table nothing is loaded and the
check is off.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

from exceptions.expected import ExpectedError

log = logging.getLogger(__name__)

FIELDS = ("datatype", "suffix", "task")
FACETS_TABLE = "bids_dataset_facets"
# The facets are refreshed after each crawl (post_crawl.py); a new task code
# must not be dropped for long because this copy is older.
REFRESH_SECONDS = 600

_known: Dict[str, FrozenSet[str]] = {}
_loaded_at: Optional[float] = None  # when set_known() last ran or a reload failed


# ── Loading the known values ──────────────────────────────────────────────────
# At startup (schema_capabilities) and, at most every REFRESH_SECONDS, before
# a search (refresh_if_stale).

def set_known(values: Mapping[str, Iterable[str]]) -> None:
    """Replace the known values (load(), tests). Empty: the check is off."""
    global _known, _loaded_at
    _known = {field: frozenset(values.get(field, ())) for field in FIELDS} if values else {}
    _loaded_at = time.monotonic()


_LOAD_SQL = " UNION ALL ".join(
    f"SELECT '{field}' AS field, {field} AS value FROM {FACETS_TABLE} "
    f"WHERE {field} IS NOT NULL GROUP BY {field}"
    for field in FIELDS
)


class CatalogueLoadError(ExpectedError):
    """The known values could not be read from the facets table."""


async def load(connection: Any) -> None:
    """Read the known values from the facets table (AsyncConnection or AsyncSession).

    The read runs in a savepoint: the caller's transaction (the startup
    detection, a search) stays usable when it fails. Raises CatalogueLoadError
    then; the known values are unchanged.
    """
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    try:
        async with connection.begin_nested():
            rows = (await connection.execute(text(_LOAD_SQL))).all()
    except SQLAlchemyError as exc:
        raise CatalogueLoadError(
            f"reading {', '.join(FIELDS)} from {FACETS_TABLE} failed: {exc}") from exc
    values: Dict[str, set] = {field: set() for field in FIELDS}
    for field, value in rows:
        values[field].add(value)
    set_known(values)
    log.info("Catalogue values: %s", ", ".join(f"{len(v)} {f}" for f, v in values.items()))


async def refresh_if_stale(session: Any) -> None:
    """Reload the known values when they are older than REFRESH_SECONDS.

    Only while the facets table is in use. A failed reload keeps the old
    values (none: the check stays off) and is tried again after
    REFRESH_SECONDS, not on every search.
    """
    global _loaded_at
    from services.catalogue import schema_capabilities

    if not schema_capabilities.has_dataset_facets():
        return
    if _loaded_at is not None and time.monotonic() - _loaded_at < REFRESH_SECONDS:
        return
    try:
        await load(session)
    except CatalogueLoadError as exc:  # the check only removes errors; never fail a search for it
        log.warning("Could not reload catalogue values (%s); %s",
                    exc, "keeping the old ones" if _known else "the check stays off")
        _loaded_at = time.monotonic()


# ── Looking a value up ────────────────────────────────────────────────────────

def loaded() -> bool:
    """True when known values are loaded; the check is off without them."""
    return bool(_known)


def known(field: str) -> FrozenSet[str]:
    """The values of *field* that occur in some dataset."""
    return _known.get(field, frozenset())


def is_known(field: str, value: str) -> bool:
    return value in known(field)
