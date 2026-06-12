"""
services/query_cache.py
-----------------------
In-memory store for translated queries, keyed by an opaque ``query_id``.

The Text-To-SQL pipeline (Gemini intent + RAG + SQL generation) is expensive and
non-deterministic, so we run it once per question and cache the resulting base
SELECT plus its ranking inputs. Pagination then reuses the cached SQL — only the
relevance ordering + LIMIT/OFFSET are re-applied per page (see
``services.sql_rewriter.build_page_sql``). The client never sends SQL back; it
sends a ``query_id``, so the only SQL ever executed is server-generated.

This is a process-local dict with TTL + LRU eviction. It is correct because the
backend runs a *single* uvicorn worker (see backend/Dockerfile — the CMD has no
``--workers``; dev uses ``--reload``). If the backend is ever scaled to multiple
workers/replicas, a query_id created on one worker won't resolve on another —
replace this with a shared store (Redis, already used by DaRe, or the DB).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Tunables
_TTL_SECONDS = 30 * 60  # query_ids expire 30 min after creation
_MAX_ENTRIES = 500      # hard cap; oldest entries evicted first (LRU-ish)


@dataclass
class CachedQuery:
    """Everything needed to re-run any page of a translated query."""
    base_sql: str
    params: Dict[str, Any]
    scored_filters: List[Dict[str, Any]]
    apply_relevance: bool
    explanation: Optional[str] = None
    self_corrected: bool = False
    total: Optional[int] = None
    created_at: float = field(default_factory=time.monotonic)


class _QueryCache:
    def __init__(self, ttl: float = _TTL_SECONDS, max_entries: int = _MAX_ENTRIES):
        self._ttl = ttl
        self._max = max_entries
        self._store: "Dict[str, CachedQuery]" = {}
        self._lock = threading.Lock()

    def _expired(self, entry: CachedQuery, now: float) -> bool:
        return (now - entry.created_at) > self._ttl

    def _evict(self, now: float) -> None:
        # Drop expired entries first, then trim to the size cap (oldest first).
        for key in [k for k, v in self._store.items() if self._expired(v, now)]:
            self._store.pop(key, None)
        if len(self._store) > self._max:
            for key in sorted(self._store, key=lambda k: self._store[k].created_at)[
                : len(self._store) - self._max
            ]:
                self._store.pop(key, None)

    def put(self, entry: CachedQuery) -> str:
        query_id = uuid4().hex
        with self._lock:
            self._store[query_id] = entry
            self._evict(time.monotonic())
        return query_id

    def get(self, query_id: str) -> Optional[CachedQuery]:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(query_id)
            if entry is None:
                return None
            if self._expired(entry, now):
                self._store.pop(query_id, None)
                return None
            return entry

    def update(self, query_id: str, entry: CachedQuery) -> None:
        """Persist an in-place change (e.g. a self-corrected base SQL)."""
        with self._lock:
            if query_id in self._store:
                self._store[query_id] = entry


# Module-level singleton used by the query router.
query_cache = _QueryCache()
