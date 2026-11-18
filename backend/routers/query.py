"""
routers/query.py
----------------
Natural-language dataset query endpoint.

Accepts a plain-text question, passes it through the Text-To-SQL layer, executes
the resulting SQL against the BIDS database, and returns matching datasets.

The expensive translation (Gemini intent + RAG + SQL generation) runs once per
question; the resulting base SELECT is cached under an opaque ``query_id`` so
subsequent pages reuse it. Relevance ordering/weighting and LIMIT/OFFSET are
applied per page in postprocessing (see services.sql_rewriter.build_page_sql).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_db
from rate_limits import Category, rate_limit
from routers.auth import require_auth
from schemas import (
    DatasetSchema,
    QueryFilterSchema,
    QueryPageRequest,
    QueryRequest,
    QueryResponse,
    TextToSQLResult,
)
from services import catalogue_values, interpretation_store, query_filters
from services import query_guard, search_runner
from services.query_cache import CachedQuery, query_cache
from services import translation_store
from services.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL
from services.text_to_sql import text_to_sql

router = APIRouter(prefix="/query", tags=["query"])
log = logging.getLogger(__name__)


def _found_message(total: int) -> str:
    return f"Found {total} dataset{'' if total == 1 else 's'} matching your query."


def _filters_for(cached: CachedQuery) -> list[QueryFilterSchema]:
    """The query's conditions as chips; none once the SQL was self-corrected,
    since the corrected query no longer matches the conditions read earlier."""
    if cached.self_corrected or not cached.filter_sql:
        return []
    return [QueryFilterSchema(**f.to_dict()) for f in query_filters.describe(cached.filter_sql)]


async def _run_page(
    session: AsyncSession,
    query_id: str,
    cached: CachedQuery,
    question: str,
    page: int,
    page_size: int,
) -> tuple[list[DatasetSchema], int]:
    try:
        return await search_runner.run_page(session, query_id, cached, question, page, page_size)
    except query_guard.SearchBusy:
        raise HTTPException(status_code=503, detail="The search is busy. Please try again in a minute.")


# Searches running now, by (user, normalised question). The frontend sends a
# question only once while it runs; this is the same rule on the server, for
# double submits, retries and second tabs. One uvicorn process serves the API
# (backend/Dockerfile), so a dict in memory sees every request.
_in_flight: dict[tuple[str, str], asyncio.Future] = {}


@router.post("", response_model=QueryResponse, dependencies=[rate_limit(Category.SEARCH)])
async def query_datasets(
    body: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[dict, Depends(require_auth)],
):
    """
    Submit a natural-language question and receive the first page of matching
    BIDS datasets, plus a ``query_id`` for paging through the rest.

    The same question from the same user while it is still running does not
    start a second search: it gets the running search's answer (or error),
    without another language-model call. The rate limit (rate_limits.py,
    SEARCH) still counts it, since it ran before this function.
    """
    key = (_auth.get("sub", ""), interpretation_store.normalize(body.question))
    running = _in_flight.get(key)
    if running is not None:
        try:
            return await asyncio.shield(running)
        except asyncio.CancelledError:
            if running.cancelled():
                raise HTTPException(status_code=503, detail="The same search was interrupted. Please try again.")
            raise

    answer: asyncio.Future = asyncio.get_running_loop().create_future()
    # Marks an error as seen when no second request was waiting for it.
    answer.add_done_callback(lambda f: f.cancelled() or f.exception())
    _in_flight[key] = answer
    try:
        response = await _answer_question(body, session, _auth)
    except asyncio.CancelledError:
        answer.cancel()
        raise
    except BaseException as exc:
        answer.set_exception(exc)
        raise
    else:
        answer.set_result(response)
        return response
    finally:
        _in_flight.pop(key, None)


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


async def _answer_question(body: QueryRequest, session: AsyncSession, _auth: dict) -> QueryResponse:
    """One search: translation, first page, logging."""
    page, page_size = search_runner.clamp_page(1, search_runner.DEFAULT_PAGE_SIZE)
    # The same question gets the same translation every time (the models that
    # write it are not repeatable): reuse a stored one when there is one.
    await catalogue_values.refresh_if_stale(session)
    stored = await interpretation_store.get(session, body.question)
    if stored is not None:
        translation = TextToSQLResult(
            sql=stored["sql"], explanation=stored.get("explanation") or "",
            scored_filters=stored.get("scored_filters") or [],
            apply_relevance=bool(stored.get("apply_relevance")),
            params=stored.get("params") or {}, rank_tsquery=stored.get("rank_tsquery"),
            filter_sql=stored.get("filter_sql"), relaxed_sql=stored.get("relaxed_sql"),
        )
    else:
        # The exact-match store of services/translation_store.py (off unless
        # TRANSLATION_STORE is set) is asked inside text_to_sql.
        await translation_store.refresh(session)
        translation = await text_to_sql(body.question)

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

    datasets, total = await _run_page(
        session, query_id, cached, body.question, page, page_size
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
        session, body.question, payload, total, from_cache=stored is not None,
        relaxed=cached.relaxed, self_corrected=bool(cached.self_corrected),
        fallback=cached.base_sql == _FALLBACK_SQL,
    )
    if stored is None and _worth_storing(cached, total):
        await interpretation_store.put(session, body.question, payload, total)
    # Keep the translation for the next time this question is asked, but only
    # one that ran as written and found something: a corrected or empty search
    # should get a fresh attempt.
    if (translation.store_entry is not None and not cached.self_corrected
            and cached.base_sql != _FALLBACK_SQL and total > 0):
        translation.store_entry.final_sql = cached.base_sql
        translation.store_entry.result_total = total
        await translation_store.remember(session, translation.store_entry)
    elif translation.from_store:
        await translation_store.count_hit(session, body.question)

    from models import UserQuery
    try:
        async with session.begin_nested():
            session.add(UserQuery(github_login=_auth["sub"], question=body.question))
        # The savepoint only protects the response from a failed insert. Without
        # this commit the row is rolled back when the session closes (get_db
        # never commits), which left user_queries empty from June 2026 on.
        await session.commit()
    except Exception:
        pass  # stale/legacy JWT sub not in users table — don't crash the response

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


@router.post("/page", response_model=QueryResponse, dependencies=[rate_limit(Category.PAGE)])
async def query_page(
    body: QueryPageRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
):
    """Fetch another page of an already-translated query, reusing its cached base
    SQL (no new LLM call)."""
    cached = query_cache.get(body.query_id)
    if cached is None:
        raise HTTPException(
            status_code=410,
            detail="This search has expired. Please run the query again.",
        )

    page, page_size = search_runner.clamp_page(body.page, body.page_size)
    # The question is only needed for self-correction, which already happened on
    # the first page; an empty string is fine here.
    datasets, total = await _run_page(
        session, body.query_id, cached, "", page, page_size
    )

    return QueryResponse(
        message=_found_message(total),
        translation=None,
        datasets=datasets,
        query_id=body.query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )
