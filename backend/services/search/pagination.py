"""
services/search/pagination.py
-----------------------------
Further pages of a search that services/search/search.py has already answered, from
its cached query (services/stores/translated_queries.py) and without another
language-model call.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions.domain import DomainError
from routers.schemas import QueryResponse
from services.search import search_runner
from services.results.display import result_page
from services.stores.translated_queries import query_cache


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

    return result_page.response(datasets, total, query_id, page, page_size)
