"""
routers/query.py
----------------
The search endpoints. POST /query answers a natural-language question with
the first page of matching datasets and a ``query_id``; POST /query/page
fetches further pages of that query. The search itself is services/search.py;
this module does the HTTP side: authentication, rate limits and one running
search per user and question. A busy or expired search raises a
DomainError (exceptions/domain.py), which main.py answers; nothing here
catches it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_db
from rate_limits import Category, rate_limit
from routers.auth import require_auth
from schemas import QueryPageRequest, QueryRequest, QueryResponse
from single_flight import SingleFlight
from services.stores import interpretation_store
from services import search

router = APIRouter(prefix="/query", tags=["query"])

# Searches running now, by (user, normalised question). The frontend sends a
# question only once while it runs; this is the same rule on the server, for
# double submits, retries and second tabs. One uvicorn process serves the API
# (backend/Dockerfile; tests/test_single_process.py keeps it so), so it is
# kept in memory (single_flight.py).
_in_flight: SingleFlight[QueryResponse] = SingleFlight()


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
    return await _in_flight.run(
        key, lambda: search.answer_question(session, body.question, _auth.get("sub"))
    )


@router.post("/page", response_model=QueryResponse, dependencies=[rate_limit(Category.PAGE)])
async def query_page(
    body: QueryPageRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
):
    """Fetch another page of an already-translated query, reusing its cached base
    SQL (no new LLM call)."""
    return await search.next_page(session, body.query_id, body.page, body.page_size)
