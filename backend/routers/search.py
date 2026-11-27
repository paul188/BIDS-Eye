"""
routers/search.py
-----------------
The search endpoints. POST /query answers a natural-language question with
the first page of matching datasets and a ``query_id``; POST /query/page
fetches further pages of that query. The search itself is services/search.py;
this module does the HTTP side: authentication, rate limits, and saving the
answer for a user who turned their history on
(services/stores/search_history.py). A busy or expired search raises a
DomainError (exceptions/domain.py), which main.py answers; nothing here
catches it.
"""

from __future__ import annotations

from fastapi import APIRouter

from deps import AsyncSessionDep
from security.rate_limits import Category, rate_limit
from routers.auth import CurrentUserDep, user_id
from schemas import QueryPageRequest, QueryRequest, QueryResponse
from services import search
from services.stores import search_history

search_router = APIRouter(prefix="/query", tags=["query"])

@search_router.post("", dependencies=[rate_limit(Category.SEARCH)])
async def query_datasets(
    body: QueryRequest,
    session: AsyncSessionDep,
    user: CurrentUserDep,
) -> QueryResponse:
    """
    Submit a natural-language question and receive the first page of matching
    BIDS datasets, plus a ``query_id`` for paging through the rest.

    A question that is already being searched for, by anyone, gets that
    search's answer (or error) without another language-model call
    (services/search.py ask). The rate limit (security/rate_limits.py, SEARCH) still
    counts it, since it ran before this function.

    The search itself does not know who asked. Only here, and only for a user
    who turned their history on, is the answer saved under their account;
    ``history_id`` then names the saved entry.
    """
    response = await search.ask(session, body.question)
    history_id = await search_history.record(session, user_id(user), body.question, response)
    # A copy: identical questions asked at once share one response object.
    return response.model_copy(update={"history_id": history_id}) if history_id else response


@search_router.post("/page", dependencies=[rate_limit(Category.PAGE)])
async def query_page(
    body: QueryPageRequest,
    session: AsyncSessionDep,
) -> QueryResponse:
    """Fetch another page of an already-translated query, reusing its cached base
    SQL (no new LLM call)."""
    return await search.next_page(session, body.query_id, body.page, body.page_size)
