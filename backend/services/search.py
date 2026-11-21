"""
services/search.py
------------------
A natural-language search from question to result page.

ask() is the entry point: identical questions running at the same time share
one search, whoever asked them. answer_question is that one search. It translates the question once: a stored translation of the
same question (services/stores/interpretation_store.py) is reused, otherwise the
Text-To-SQL layer writes one (services/translate/text_to_sql.py). The resulting base
SELECT is cached under an opaque ``query_id`` (services/stores/query_cache.py), the
first page is run (services/search_runner.py), and the search is logged and,
when it worked, stored for the next time. next_page fetches further pages of
a cached query without another language-model call.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions.domain import DomainError
from schemas import QueryFilterSchema, QueryResponse, TextToSQLResult
from single_flight import SingleFlight
from services.catalogue.catalogue_values import catalogue_values
from services.stores import interpretation_store
from services.results import query_filters
from services import search_runner
from services.stores.query_cache import CachedQuery, query_cache
from services.translate.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL
from services.translate.text_to_sql import text_to_sql


def _found_message(total: int) -> str:
    return f"Found {total} dataset{'' if total == 1 else 's'} matching your query."


def _filters_for(cached: CachedQuery) -> list[QueryFilterSchema]:
    """The query's conditions as chips; none once the SQL was self-corrected,
    since the corrected query no longer matches the conditions read earlier."""
    if cached.self_corrected or not cached.filter_sql:
        return []
    return [QueryFilterSchema(**f.to_dict()) for f in query_filters.describe(cached.filter_sql)]


def _worth_storing(cached: CachedQuery, total: int) -> bool:
    """Whether a new translation becomes the stored answer to its question.

    Only one that ran as written and found a real result. The stored SQL is
    reused for good, so these get a fresh attempt next time instead:
    - an emergency fallback or a model-repaired SQL (self_corrected);
    - a relaxed re-run: it depends on how many datasets matched that day, and
      stored it would never again try the full conditions;
    - fewer than search_runner.MIN_RESULTS results, the case the relaxed re-run is for.
    """
    return (total >= search_runner.MIN_RESULTS
            and cached.base_sql != _FALLBACK_SQL
            and not cached.self_corrected
            and not cached.relaxed)


# Searches running now, by normalised question. The answer does not depend on
# who asks (the stored translations in interpretation_store are shared the same
# way), so a double submit, a retry, a second tab or another user asking the
# same thing gets the running search's answer without another language-model
# call. One uvicorn process serves the API (backend/Dockerfile;
# tests/test_single_process.py keeps it so), so it is kept in memory.
_in_flight: SingleFlight[QueryResponse] = SingleFlight()


async def ask(session: AsyncSession, question: str) -> QueryResponse:
    """The answer to *question*, from the search for it already running if
    there is one.

    Who asked is not stored anywhere: the only record of a search is its
    anonymous query_log row (models.QueryLog), as the site's privacy notice
    says (frontend/src/components/CookieBanner.vue).

    Raises query_guard.SearchBusyError when no slot for generated SQL frees up.
    """
    return await _in_flight.run(
        interpretation_store.normalize(question), lambda: answer_question(session, question)
    )


async def answer_question(session: AsyncSession, question: str) -> QueryResponse:
    """One search: translation, first page, logging. The same for every user.

    Raises query_guard.SearchBusyError when no slot for generated SQL frees up.
    """
    page, page_size = search_runner.clamp_page(1, search_runner.DEFAULT_PAGE_SIZE)
    # The same question gets the same translation every time (the models that
    # write it are not repeatable): reuse a stored one when there is one.
    await catalogue_values.refresh_if_stale(session)
    stored = await interpretation_store.get(session, question)
    if stored is not None:
        translation = TextToSQLResult(
            sql=stored["sql"], explanation=stored.get("explanation") or "",
            scored_filters=stored.get("scored_filters") or [],
            apply_relevance=bool(stored.get("apply_relevance")),
            params=stored.get("params") or {}, rank_tsquery=stored.get("rank_tsquery"),
            filter_sql=stored.get("filter_sql"), relaxed_sql=stored.get("relaxed_sql"),
        )
    else:
        translation = await text_to_sql(question)

    cached = CachedQuery(
        base_sql=translation.sql,
        params=translation.params,
        scored_filters=translation.scored_filters,
        apply_relevance=translation.apply_relevance,
        rank_tsquery=translation.rank_tsquery,
        explanation=translation.explanation,
        self_corrected=translation.self_corrected,
        filter_sql=translation.filter_sql,
        relaxed_sql=translation.relaxed_sql,
        relaxed=bool(stored and stored.get("relaxed")),
    )
    query_id = query_cache.put(cached)

    datasets, total = await search_runner.run_page(
        session, query_id, cached, question, page, page_size
    )
    # Surface any base-SQL self-correction that happened during the count.
    reranked = len(cached.rerank_ordered) + len(cached.rerank_demoted)
    translation = translation.model_copy(
        update={
            "sql": cached.base_sql,
            "self_corrected": cached.self_corrected,
            "reranked": reranked or None,
        }
    )

    payload = {
        "sql": cached.base_sql, "params": cached.params,
        "scored_filters": cached.scored_filters, "apply_relevance": cached.apply_relevance,
        "rank_tsquery": cached.rank_tsquery, "explanation": cached.explanation,
        "filter_sql": cached.filter_sql,
        "relaxed_sql": None if cached.relaxed else cached.relaxed_sql,
        "relaxed": cached.relaxed,
    }
    # Every search is recorded for good, whatever it found.
    await interpretation_store.log_query(
        session, question, payload, total, from_cache=stored is not None,
        relaxed=cached.relaxed, self_corrected=bool(cached.self_corrected),
        fallback=cached.base_sql == _FALLBACK_SQL,
    )
    if stored is None and _worth_storing(cached, total):
        await interpretation_store.put(session, question, payload, total)

    return QueryResponse(
        message=_found_message(total),
        translation=translation,
        filters=_filters_for(cached),
        datasets=datasets,
        query_id=query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


class SearchExpiredError(
    DomainError, status_code=410, detail="This search has expired. Please run the query again."
):
    """The query_id is no longer in the query cache."""


async def next_page(session: AsyncSession, query_id: str, page: int, page_size: int) -> QueryResponse:
    """Another page of an already-translated query, reusing its cached base
    SQL (no new LLM call).

    Raises SearchExpiredError when the query left the cache, and
    query_guard.SearchBusyError when no slot for generated SQL frees up.
    """
    cached = query_cache.get(query_id)
    if cached is None:
        raise SearchExpiredError(query_id)

    page, page_size = search_runner.clamp_page(page, page_size)
    # The question is only needed for self-correction, which already happened on
    # the first page; an empty string is fine here.
    datasets, total = await search_runner.run_page(
        session, query_id, cached, "", page, page_size
    )

    return QueryResponse(
        message=_found_message(total),
        translation=None,
        datasets=datasets,
        query_id=query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )
