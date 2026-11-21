"""
services/fulltext/word_frequencies.py
-------------------------------------
How many datasets match a tsquery, counted on search_tsv (never on paper text).

The counts decide which words of a 'loose' term a dataset must contain
(tsquery._loose_core), check search terms a language model proposed
(tsquery.count_matches) and pick the parts of the uncertain tier
(services/fulltext/uncertain_tier.py). A count may also require an unstemmed
guard on search_tsv_exact (exact word forms, text_search.py).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional, Sequence

from services.catalogue import off_loop_query, schema_capabilities
from services.fulltext.vectors import EXACT_FTS_COLUMN, EXACT_TS_CONFIG, FTS_COLUMN, TS_CONFIG

log = logging.getLogger(__name__)

# Given tsquery strings, returns how many datasets match each one, or None
# when the answer is not available. A second argument, when given, holds a
# guard per query (or None): a tsquery the unstemmed vector must match too.
FrequencyLookup = Callable[..., Optional[List[int]]]

# Until something else is installed, the lookup opens its own short-lived
# connection (_own_connection_lookup), so every entry point gets frequencies:
# the API (main.py installs a pooled lookup at startup), one-off containers,
# the recall benchmark. set_frequency_lookup(None) switches lookups off.
_OWN_CONNECTION = object()
_frequency_lookup: Any = _OWN_CONNECTION
FREQUENCY_TIMEOUT_SECONDS = 3.0

_COUNT_SQL = (
    f"SELECT (SELECT count(*) FROM bids_datasets d "
    f"        WHERE d.{FTS_COLUMN} @@ to_tsquery('{TS_CONFIG}', q.tsq)) "
    "FROM unnest(CAST(:queries AS text[])) WITH ORDINALITY AS q(tsq, ord) "
    "ORDER BY q.ord"
)
_GUARDED_COUNT_SQL = (
    f"SELECT (SELECT count(*) FROM bids_datasets d "
    f"        WHERE d.{FTS_COLUMN} @@ to_tsquery('{TS_CONFIG}', q.tsq) "
    f"          AND (q.guard IS NULL OR "
    f"               d.{EXACT_FTS_COLUMN} @@ to_tsquery('{EXACT_TS_CONFIG}', q.guard))) "
    "FROM unnest(CAST(:queries AS text[]), CAST(:guards AS text[])) "
    "     WITH ORDINALITY AS q(tsq, guard, ord) "
    "ORDER BY q.ord"
)


def _count_statement(queries: List[str], guards: Optional[List[Optional[str]]]):
    """(SQL, parameters) for the counts; the guarded form only when needed."""
    if guards and any(guards):
        return _GUARDED_COUNT_SQL, {"queries": queries, "guards": list(guards)}
    return _COUNT_SQL, {"queries": queries}


def set_frequency_lookup(lookup: Optional[FrequencyLookup]) -> None:
    global _frequency_lookup
    _frequency_lookup = lookup


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _own_connection_lookup(queries: List[str],
                           guards: Optional[List[Optional[str]]] = None) -> Optional[List[int]]:
    """Count with a connection of its own; for callers that installed nothing.

    Runs only off the event loop (the translation pipeline runs in a worker
    thread) and only when the database has search_tsv.
    """
    if _on_event_loop() or not schema_capabilities.has_fts():
        return None
    from sqlalchemy import text

    async def _count(connection: Any) -> List[int]:
        sql, params = _count_statement(queries, guards)
        rows = await connection.execute(text(sql), params)
        return [int(row[0]) for row in rows]

    return off_loop_query.run_with_own_connection(_count, FREQUENCY_TIMEOUT_SECONDS)


def make_frequency_lookup(engine: Any, loop: asyncio.AbstractEventLoop,
                          timeout: float = FREQUENCY_TIMEOUT_SECONDS) -> FrequencyLookup:
    """Pooled frequency lookup for code running in a worker thread.

    The translation pipeline runs in asyncio.to_thread, while the database
    engine belongs to the event loop; the count is scheduled there.
    """
    from sqlalchemy import text

    async def _count(queries: List[str], guards: Optional[List[Optional[str]]]) -> List[int]:
        async with engine.connect() as connection:
            sql, params = _count_statement(queries, guards)
            rows = await connection.execute(text(sql), params)
            return [int(row[0]) for row in rows]

    def lookup(queries: List[str],
               guards: Optional[List[Optional[str]]] = None) -> Optional[List[int]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return None  # called on an event loop: blocking here would deadlock
        return off_loop_query.run_on_loop(_count(queries, guards), loop, timeout)

    return lookup


def _lookup_counts(queries: List[str],
                   guards: Optional[Sequence[Optional[str]]] = None) -> Optional[List[int]]:
    """Datasets matching each query; with *guards*, also its unstemmed guard.

    The lookup gets the guards only when there is one, so a lookup that knows
    nothing of them (tests, older callers) keeps working. A lookup raises
    CatalogueLookupError when the database cannot answer; the search then
    goes on without the counts. Anything else it raises is a bug.
    """
    lookup = _own_connection_lookup if _frequency_lookup is _OWN_CONNECTION else _frequency_lookup
    if lookup is None:
        return None
    try:
        if guards and any(guards):
            counts = lookup(queries, list(guards))
        else:
            counts = lookup(queries)
    except off_loop_query.CatalogueLookupError as exc:
        log.warning("Word frequency lookup failed (%s) — using the longest word", exc)
        return None
    if not counts or len(counts) != len(queries):
        return None
    return counts


def guarded_counts(queries: List[str],
                   guards: Sequence[Optional[str]]) -> Optional[List[int]]:
    """_lookup_counts, passing the guards on only when there is one."""
    if any(guards):
        return _lookup_counts(queries, list(guards))
    return _lookup_counts(queries)
