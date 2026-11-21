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

import logging
from typing import Any, Callable, List, Optional, Sequence

import worker_db
from services.fulltext.vectors import EXACT_FTS_COLUMN, EXACT_TS_CONFIG, FTS_COLUMN, TS_CONFIG

log = logging.getLogger(__name__)

# Given tsquery strings, returns how many datasets match each one, or None
# when the answer is not available. A second argument, when given, holds a
# guard per query (or None): a tsquery the unstemmed vector must match too.
FrequencyLookup = Callable[..., Optional[List[int]]]

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


def _database_counts(queries: List[str],
                     guards: Optional[List[Optional[str]]] = None) -> Optional[List[int]]:
    """The counts from the database (worker_db.current()).

    Called from the translation worker thread, which cannot use the app's
    engine itself.
    """
    from sqlalchemy import text

    sql, params = _count_statement(queries, guards)

    async def count(connection: Any) -> List[int]:
        rows = await connection.execute(text(sql), params)
        return [int(row[0]) for row in rows]

    return worker_db.current().run(count, FREQUENCY_TIMEOUT_SECONDS)


_frequency_lookup: Optional[FrequencyLookup] = _database_counts


def set_frequency_lookup(lookup: Optional[FrequencyLookup]) -> None:
    """Install another lookup (tests); None switches lookups off."""
    global _frequency_lookup
    _frequency_lookup = lookup


def reset_frequency_lookup() -> None:
    """Back to asking the database."""
    set_frequency_lookup(_database_counts)


def _lookup_counts(queries: List[str],
                   guards: Optional[Sequence[Optional[str]]] = None) -> Optional[List[int]]:
    """Datasets matching each query; with *guards*, also its unstemmed guard.

    The lookup gets the guards only when there is one, so a lookup that knows
    nothing of them (tests, older callers) keeps working. A lookup raises
    worker_db.WorkerDbError when the database cannot answer; the search then
    goes on without the counts. Anything else it raises is a bug.
    """
    lookup = _frequency_lookup
    if lookup is None:
        return None
    try:
        if guards and any(guards):
            counts = lookup(queries, list(guards))
        else:
            counts = lookup(queries)
    except worker_db.WorkerDbError as exc:
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
