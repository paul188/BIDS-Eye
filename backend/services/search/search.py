"""
services/search/search.py
-------------------------
A natural-language search from question to result page.

ask() is the entry point: identical questions running at the same time share
one search, whoever asked them. answer_question() is that one search: it reuses
a stored translation of the question (services/stores/interpretations.py) or has
the Text-To-SQL layer write one (services/translate/text_to_sql.py), keeps the
translated query under an opaque ``query_id``
(services/stores/translated_queries.py), runs the first page
(services/search/search_runner.py), logs the search, and stores a new translation when
it is worth storing. Further pages are services/search/pagination.py.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from routers.schemas import QueryResponse, TextToSQLResult
from single_flight import SingleFlight
from services.database_stats.database_stats import database_stats
from services.stores import interpretations
from services.results.display import chips, result_page
from services.search import search_runner
from services.stores.translated_queries import CachedQuery, query_cache
from services.translate import optional_columns
from services.translate.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL
from services.translate.text_to_sql import text_to_sql


# Searches running now, by normalised question. The answer does not depend on
# who asks (the stored translations in interpretations are shared the same
# way), so a double submit, a retry, a second tab or another user asking the
# same thing gets the running search's answer without another language-model
# call. One uvicorn process serves the API (backend/Dockerfile;
# tests/test_single_process.py keeps it so), so it is kept in memory.
_in_flight: SingleFlight[QueryResponse] = SingleFlight()


async def ask(session: AsyncSession, question: str) -> QueryResponse:
    """The answer to *question*, from the search for it already running if
    there is one.

    Who asked is not known here: the search's record is its anonymous
    query_log row (models.QueryLog), as the site's privacy notice says
    (frontend/src/components/CookieBanner.vue). The answer is saved under a
    user's account only by the router, and only for a user who turned their
    history on (services/stores/search_history.py).

    Raises query_guard.SearchBusyError when no slot for generated SQL frees up.
    """
    return await _in_flight.run(
        interpretations.normalize(question), lambda: answer_question(session, question)
    )


async def answer_question(session: AsyncSession, question: str) -> QueryResponse:
    """One search: translation, first page, logging. The same for every user.

    Raises query_guard.SearchBusyError when no slot for generated SQL frees up.
    """
    page, page_size = search_runner.clamp_page(1, search_runner.DEFAULT_PAGE_SIZE)
    await database_stats.refresh_if_stale(session)
    optional_columns.log_changes(database_stats.coverage)
    # The same question gets the same translation every time (the models that
    # write it are not repeatable): reuse a stored one when there is one.
    stored = await interpretations.get(session, question)
    if stored is not None:
        translation = TextToSQLResult(
            sql=stored.sql, explanation=stored.explanation or "",
            scored_filters=stored.scored_filters,
            apply_relevance=stored.apply_relevance,
            params=stored.params, rank_query=stored.rank_query,
            filter_sql=stored.filter_sql, relaxed_sql=stored.relaxed_sql,
            sql_path=stored.sql_path,
        )
    else:
        translation = await text_to_sql(question)

    cached = CachedQuery(
        base_sql=translation.sql,
        params=translation.params,
        scored_filters=translation.scored_filters,
        apply_relevance=translation.apply_relevance,
        rank_query=translation.rank_query,
        explanation=translation.explanation,
        self_corrected=translation.self_corrected,
        filter_sql=translation.filter_sql,
        relaxed_sql=translation.relaxed_sql,
        relaxed=stored is not None and stored.relaxed,
    )
    query_id = query_cache.put(cached)

    datasets, total = await search_runner.run_page(
        session, query_id, cached, question, page, page_size
    )
    # run_page may have repaired or relaxed the SQL and re-ranked the results.
    reranked = len(cached.rerank_ordered) + len(cached.rerank_demoted)
    translation = translation.model_copy(
        update={
            "sql": cached.base_sql,
            "self_corrected": cached.self_corrected,
            "reranked": reranked or None,
        }
    )

    payload = interpretations.StoredTranslation(
        sql=cached.base_sql, params=cached.params,
        scored_filters=cached.scored_filters, apply_relevance=cached.apply_relevance,
        rank_query=cached.rank_query, explanation=cached.explanation,
        filter_sql=cached.filter_sql,
        relaxed_sql=None if cached.relaxed else cached.relaxed_sql,
        relaxed=cached.relaxed,
        sql_path=translation.sql_path,
    )
    # Every search that ran is logged, whatever it found.
    await interpretations.log_query(
        session, question, payload, total, from_cache=stored is not None,
        relaxed=cached.relaxed, self_corrected=bool(cached.self_corrected),
        fallback=cached.base_sql == _FALLBACK_SQL,
    )
    if stored is None:
        await interpretations.put(session, question, payload, cached, total)

    return result_page.response(
        datasets, total, query_id, page, page_size,
        translation=translation, filters=chips.for_sql(cached.explainable_sql),
    )
