"""
routers/search.py
-----------------
The search endpoints. POST /query answers a natural-language question with
the first page of matching datasets and a ``query_id``; GET /query/{query_id}
fetches further pages of that query. The search itself is services/search/search.py,
the further pages services/search/pagination.py;
this module does the HTTP side: authentication, rate limits, and saving the
answer for a user who turned their history on
(services/stores/search_history.py). A busy or expired search raises a
DomainError (exceptions/domain.py), which main.py answers; nothing here
catches it.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from deps import AsyncSessionDep, GithubIdDep
from security.rate_limits import Category, rate_limit
from routers.schemas import QueryPageRequest, QueryRequest, QueryResponse
from services.search import pagination, search
from services.stores import search_history

search_router = APIRouter(prefix="/query", tags=["query"])

@search_router.post("", dependencies=[rate_limit(Category.SEARCH)])
async def query_datasets(
    body: QueryRequest,
    session: AsyncSessionDep,
    github_id: GithubIdDep,
) -> QueryResponse:
    """
    Submit a natural-language question and receive the first page of matching
    BIDS datasets, plus a ``query_id`` for paging through the rest.

    A question that is already being searched for, by anyone, gets that
    search's answer (or error) without another language-model call
    (services/search/search.py ask). The rate limit (security/rate_limits.py, SEARCH) still
    counts it, since it ran before this function.

    The search itself does not know who asked. Only here, and only for a user
    who turned their history on, is the answer saved under their account;
    ``history_id`` then names the saved entry.
    """
    response = await search.ask(session, body.question)
    history_id = await search_history.record(session, github_id, body.question, response)
    # A copy: identical questions asked at once share one response object.
    return response.model_copy(update={"history_id": history_id}) if history_id else response


@search_router.get("/{query_id}", dependencies=[rate_limit(Category.PAGE)])
async def query_page(
    query_id: str,
    session: AsyncSessionDep,
    http_response: Response,
    page: int = 1,
    page_size: int = 20,
) -> QueryResponse:
    """Fetch another page of an already-translated query, reusing its cached base
    SQL (no new LLM call)."""
    answer = await pagination.next_page(session, query_id, page, page_size)
    http_response.headers["Cache-Control"] = "private"
    return answer


@search_router.post("/page", deprecated=True, dependencies=[rate_limit(Category.PAGE)])
async def query_page_old(
    body: QueryPageRequest,
    session: AsyncSessionDep,
) -> QueryResponse:
    """The former address of query_page, for tabs that still run the old
    frontend. Remove it once the new frontend has been live for a while."""
    return await pagination.next_page(session, body.query_id, body.page, body.page_size)
