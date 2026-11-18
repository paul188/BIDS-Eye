"""
services/search.py
------------------
A natural-language search from question to result page.

answer_question translates the question once: a stored translation of the
same question (services/stores/interpretation_store.py) is reused, otherwise the
Text-To-SQL layer writes one (services/translate/text_to_sql.py). The resulting base
SELECT is cached under an opaque ``query_id`` (services/stores/query_cache.py), the
first page is run (services/search_runner.py), and the search is logged and,
when it worked, stored for the next time. next_page fetches further pages of
a cached query without another language-model call.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions.domain import DomainError
from schemas import QueryFilterSchema, QueryResponse, TextToSQLResult
from services.catalogue import catalogue_values
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


async def answer_question(session: AsyncSession, question: str,
                          user: Optional[str]) -> QueryResponse:
    """One search: translation, first page, logging. *user* is the login the
    search is recorded under in user_queries (none: not recorded).

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

    await _record_user_query(session, user, question)

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


async def _record_user_query(session: AsyncSession, user: Optional[str], question: str) -> None:
    if user is None:
        return
    from models import UserQuery
    try:
        async with session.begin_nested():
            session.add(UserQuery(github_login=user, question=question))
        # The savepoint only protects the response from a failed insert. Without
        # this commit the row is rolled back when the session closes (get_db
        # never commits), which left user_queries empty from June 2026 on.
        await session.commit()
    except Exception:
        pass  # stale/legacy JWT sub not in users table — don't crash the response


class SearchExpiredError(DomainError):
    """The query_id is no longer in the query cache."""

    status_code = 410
    detail = "This search has expired. Please run the query again."


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
