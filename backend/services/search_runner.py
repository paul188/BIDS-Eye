"""
services/search_runner.py
-------------------------
Runs a translated search against the catalogue, one page at a time.

The first page COUNTs the cached base SQL behind the guards of
services/query_guard.py. That validates it: a query the database rejects is
repaired once by the model (or replaced by FALLBACK_SQL), and a query that
finds fewer than MIN_RESULTS datasets is re-run without its free-text-only
conditions. The first page also asks the re-ranker once. Every page then
applies ordering and LIMIT/OFFSET (services/translate/sql_rewriter.build_page_sql) and
builds the result cards (services/results/dataset_cards.py, readme_snippets.py).
Changes to the query are written back to the query cache, so later pages
reuse them.
"""

from __future__ import annotations

import asyncio
import logging
import os

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions.expected import ExpectedError
from routers.schemas import DatasetSchema
from services.results import dataset_cards, match_reasons, query_filters, readme_snippets
from services import query_guard
from services.stores.translated_queries import CachedQuery, query_cache
from services.results.rerank import rerank_candidates, settings as rerank_settings, should_rerank
from services.translate.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL, build_count_sql, build_page_sql
from services.translate.text_to_sql import _correct_sql_with_gemini

log = logging.getLogger(__name__)

# Capped here as well as in the cache so a page can't request an unbounded slice.
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20
# Below this many results a free-text-only condition is dropped (see
# count_with_base_correction) ...
MIN_RESULTS = 3
# ... and only when what is left without it still narrows the catalogue.
MAX_RELAXED_RESULTS = 300


def clamp_page(page: int, page_size: int) -> tuple[int, int]:
    page = max(1, page)
    page_size = max(1, min(page_size or DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE))
    return page, page_size


class GeneratedQueryError(ExpectedError):
    """A generated query the guards refused or the database rejected. The
    message is the reason, as self-correction is told it; the session's
    transaction may be aborted."""


async def guarded_count(session: AsyncSession, base_sql: str, params: dict) -> int:
    """COUNT of a generated query behind the guards of services/query_guard.py:
    per-transaction limits, then the static and plan checks.

    Raises GeneratedQueryError when the query is refused or fails. A lost
    connection is not the query's fault and is raised as it is."""
    await query_guard.apply_limits(session)
    count_sql = build_count_sql(base_sql)
    try:
        if base_sql != _FALLBACK_SQL:
            await query_guard.vet(session, base_sql, count_sql, params)
        result = await session.execute(text(count_sql), params)
    except query_guard.QueryRefusedError as exc:
        raise GeneratedQueryError(str(exc)) from exc
    except DBAPIError as exc:
        if exc.connection_invalidated:
            raise
        raise GeneratedQueryError(str(exc)) from exc
    return int(result.scalar() or 0)


async def count_with_base_correction(
    session: AsyncSession, query_id: str, cached: CachedQuery, question: str
) -> int:
    """Count total matching datasets, self-correcting the cached *base* SQL once
    via Gemini if it raises a DB error.

    Running the COUNT validates the base SQL before any page is fetched. A fix is
    persisted back to the cache so later page turns reuse the corrected base.
    """
    try:
        total = await guarded_count(session, cached.base_sql, cached.params)
        if total < MIN_RESULTS and cached.relaxed_sql and not cached.relaxed:
            # An empty or nearly empty page is the worst answer: search once
            # more without the free-text-only conditions and say so. This is
            # the check of a free-text phrase against the datasets that match
            # the other filters: a phrase that leaves fewer than MIN_RESULTS
            # of them is not used as a filter. The text rank still puts the
            # datasets that do contain the words first.
            # In a savepoint: a refused or failing fallback must not undo the
            # count above or the transaction's limits.
            try:
                async with session.begin_nested():
                    relaxed_total = await guarded_count(session, cached.relaxed_sql, cached.params)
            except GeneratedQueryError as exc:
                log.info("Search without the free-text conditions not used: %s", exc)
                relaxed_total = 0
            # Without the phrase the search must still be a search: 990 or
            # 1,400 datasets in no useful order (0 correct answers among the
            # first 20 in every measured case) are no better than none.
            if total < relaxed_total <= MAX_RELAXED_RESULTS:
                cached.base_sql = cached.relaxed_sql
                cached.relaxed = True
                cached.filter_sql = None
                cached.explanation = ((cached.explanation or "").rstrip() +
                                      " Fewer than three datasets matched every condition, so the"
                                      " free-text conditions only order the result.").strip()
                query_cache.update(query_id, cached)
                return relaxed_total
        return total
    except GeneratedQueryError as exc:
        log.warning("Generated SQL refused or failed, asking for a correction: %s", exc)
        # Roll back the aborted transaction before any retry — without this,
        # every subsequent execute on the same session fails with
        # InFailedSQLTransactionError.
        await session.rollback()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not api_key:
            raise
        corrected = await asyncio.to_thread(
            _correct_sql_with_gemini, cached.base_sql, str(exc), question, api_key
        )
        if corrected == cached.base_sql:
            cached.base_sql = _FALLBACK_SQL
            cached.params = {}
            cached.self_corrected = True
            query_cache.update(query_id, cached)
            return await guarded_count(session, _FALLBACK_SQL, {})
        cached.base_sql = corrected
        cached.self_corrected = True
        query_cache.update(query_id, cached)
        try:
            return await guarded_count(session, cached.base_sql, cached.params)
        except GeneratedQueryError as exc:
            log.warning("Corrected SQL refused or failed too, using the fallback sample: %s", exc)
            await session.rollback()
            cached.base_sql = _FALLBACK_SQL
            cached.params = {}
            query_cache.update(query_id, cached)
            return await guarded_count(session, _FALLBACK_SQL, {})


async def rerank_once(
    session: AsyncSession, query_id: str, cached: CachedQuery, question: str, total: int
) -> None:
    """Let an LLM judge the first candidates and store the new order (once per query)."""
    cached.rerank_done = True
    if not question or not should_rerank(
        cached.apply_relevance, total, cached.base_sql == _FALLBACK_SQL
    ):
        query_cache.update(query_id, cached)
        return
    candidate_sql, params = build_page_sql(
        cached.base_sql, cached.scored_filters, True,
        rerank_settings()["candidates"], 0, rank_query=cached.rank_query,
    )
    # In a savepoint: a failed read only skips the re-ranking.
    try:
        async with session.begin_nested():
            rows = await session.execute(text(candidate_sql), {**cached.params, **params})
            candidate_ids = [row["id"] for row in rows.mappings().all()]
    except SQLAlchemyError as exc:
        log.warning("Re-ranking skipped: candidates not read (%s)", exc)
        query_cache.update(query_id, cached)
        return
    result = await rerank_candidates(session, candidate_ids, question, cached.explanation)
    if result is not None:
        cached.rerank_ordered = result.ordered_ids
        cached.rerank_demoted = result.demoted_ids
    query_cache.update(query_id, cached)


def _highlight_for(cached: CachedQuery) -> list[tuple]:
    if cached.self_corrected:
        return []
    return query_filters.highlight_terms(cached.filter_sql)


def _strengths_for(cached: CachedQuery) -> list[float]:
    """Field strengths (tesla) to mark in the README excerpts."""
    if cached.self_corrected:
        return []
    return query_filters.highlight_field_strengths(cached.filter_sql)


def _exact_words_for(cached: CachedQuery) -> frozenset:
    """The highlight words that only count as written (exact word forms)."""
    if cached.self_corrected:
        return frozenset()
    return query_filters.highlight_exact_words(cached.filter_sql)


async def run_page(
    session: AsyncSession,
    query_id: str,
    cached: CachedQuery,
    question: str,
    page: int,
    page_size: int,
) -> tuple[list[DatasetSchema], int]:
    """_run_page_unguarded in one of the few slots for generated SQL.

    Raises query_guard.SearchBusyError when no slot frees up in time.
    """
    async with query_guard.slot():
        return await _run_page_unguarded(session, query_id, cached, question, page, page_size)


async def _run_page_unguarded(
    session: AsyncSession,
    query_id: str,
    cached: CachedQuery,
    question: str,
    page: int,
    page_size: int,
) -> tuple[list[DatasetSchema], int]:
    """Fetch one page for a cached query: total count + the page's datasets.

    The COUNT runs first (validating / self-correcting the base SQL); the page
    query then reuses the now-validated base, so it only applies ordering +
    LIMIT/OFFSET.
    """
    if cached.total is None:
        total = await count_with_base_correction(session, query_id, cached, question)
        cached.total = total
        query_cache.update(query_id, cached)
    else:
        total = cached.total

    if not cached.rerank_done:
        await rerank_once(session, query_id, cached, question, total)

    offset = (page - 1) * page_size

    def page_query(ranked: bool) -> tuple[str, dict]:
        page_sql, extra_params = build_page_sql(
            cached.base_sql,
            cached.scored_filters,
            cached.apply_relevance and ranked,
            page_size,
            offset,
            rank_query=cached.rank_query if ranked else None,
            rerank_ordered=cached.rerank_ordered,
            rerank_demoted=cached.rerank_demoted,
        )
        return page_sql, {**cached.params, **extra_params}

    await query_guard.apply_limits(session)
    rows = None
    if cached.apply_relevance and not cached.ranking_skipped:
        # The relevance ranking checks every matched code per dataset and took
        # 98 s for "MRI datasets" and 174 s for "resting-state datasets"
        # (2026-11-14) where the unranked page takes 4-9 s. A ranked page gets
        # a short timeout of its own; past it the user gets the unranked page.
        page_sql, params = page_query(ranked=True)
        try:
            async with session.begin_nested():
                await session.execute(text(
                    f"SET LOCAL statement_timeout = '{query_guard.ranked_page_timeout()}'"))
                rows = (await session.execute(text(page_sql), params)).mappings().all()
            await query_guard.apply_limits(session)  # the page's timeout outlives its savepoint
        except DBAPIError as exc:
            if not query_guard.is_timeout(exc):
                raise
            log.warning("Ranked page too slow; serving query %s unranked", query_id)
            cached.ranking_skipped = True
            query_cache.update(query_id, cached)
    if rows is None:
        page_sql, params = page_query(ranked=False)
        rows = (await session.execute(text(page_sql), params)).mappings().all()
    datasets = await dataset_cards.rows_to_datasets(session, rows)
    await readme_snippets.attach_snippets(session, datasets, _highlight_for(cached),
                                          _exact_words_for(cached), _strengths_for(cached))
    await dataset_cards.attach_descriptions(session, datasets)
    await match_reasons.attach_match_reasons(
        session, datasets, None if cached.self_corrected else cached.filter_sql)
    return datasets, total
