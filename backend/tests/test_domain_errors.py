"""Every expected failure (a DomainError: busy or expired search, login,
rate limit) reaches the client as its own status and text through the one
handler (exceptions/handlers.py); any other exception gets the handlers'
plain 500 without its message; and identical questions sent at the same time
over HTTP still share one search (services/search.py ``_in_flight``).

Needs the backend image (routers.search imports the db package)."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

import rate_limits
import routers.auth as auth
import routers.search as query_router
import sessions
from deps import get_db, require_auth
from exceptions.domain import DomainError
from exceptions.handlers import UNEXPECTED_DETAIL, add_error_handlers
from schemas import QueryResponse
from services import query_guard, search


def _app() -> FastAPI:
    app = FastAPI()
    add_error_handlers(app)
    app.include_router(query_router.search_router, prefix="/api")

    async def no_db():
        yield None

    app.dependency_overrides[get_db] = no_db
    app.dependency_overrides[require_auth] = lambda: {"sub": "a"}
    return app


@pytest.fixture(autouse=True)
def _fresh_limits():
    rate_limits.reset()
    yield
    rate_limits.reset()


async def _post(path: str, json: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=json)


def _response(total: int) -> QueryResponse:
    return QueryResponse(message=f"{total}", translation=None, datasets=[], query_id="q",
                         total=total, page=1, page_size=20, has_more=False)


def test_an_unknown_query_id_is_410_from_the_real_service():
    reply = asyncio.run(_post("/api/query/page", {"query_id": "no-such-query"}))
    assert reply.status_code == 410
    assert reply.json() == {"detail": "This search has expired. Please run the query again."}


@pytest.mark.parametrize(
    "error, status, detail",
    [
        (query_guard.SearchBusyError("no search slot free within 30s"), 503,
         "The search is busy. Please try again in a minute."),
        (search.SearchExpiredError("q"), 410, "This search has expired. Please run the query again."),
    ],
)
def test_each_error_from_the_search_gets_its_answer(monkeypatch, error, status, detail):
    async def fails(session, question):
        raise error

    monkeypatch.setattr(search, "answer_question", fails)
    reply = asyncio.run(_post("/api/query", {"question": "art"}))
    assert reply.status_code == status
    # Only the public text; the internal message ("within 30s") stays in logs.
    assert reply.json() == {"detail": detail}


def test_busy_on_a_later_page_is_503(monkeypatch):
    async def busy(session, query_id, page, page_size):
        raise query_guard.SearchBusyError("no search slot free within 30s")

    monkeypatch.setattr(search, "next_page", busy)
    reply = asyncio.run(_post("/api/query/page", {"query_id": "q"}))
    assert reply.status_code == 503


def test_every_planned_error_is_a_domain_error_with_its_own_answer():
    errors = (
        query_guard.SearchBusyError, search.SearchExpiredError, rate_limits.RateLimitError,
        sessions.NotSignedInError, auth.OAuthStateError, auth.GitHubTokenError,
        auth.GitHubLoginMissingError, auth.NotInGitHubOrgError,
    )
    for error in errors:
        assert issubclass(error, DomainError)
        assert isinstance(error.status_code, int) and error.detail


def test_an_error_class_without_its_answer_is_refused_when_defined():
    with pytest.raises(TypeError):
        class NoAnswerError(DomainError, status_code=400):
            pass


def test_an_unexpected_error_is_a_plain_500_without_its_message(monkeypatch):
    async def crashes(session, question):
        raise RuntimeError("password=hunter2 in the connection string")

    monkeypatch.setattr(search, "answer_question", crashes)

    async def post():
        # raise_app_exceptions=False: Starlette re-raises after answering, so
        # the server can log the traceback; the client only sees the answer.
        transport = httpx.ASGITransport(app=_app(), raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/query", json={"question": "art"})

    reply = asyncio.run(post())
    assert reply.status_code == 500
    assert reply.json() == {"detail": UNEXPECTED_DETAIL}


def test_an_error_raised_with_its_own_detail_and_headers_sends_them():
    app = FastAPI()
    add_error_handlers(app)

    @app.get("/limited")
    def limited():
        raise rate_limits.RateLimitError("internal", detail={"code": "rate_limit_exceeded", "limit": 3},
                                      headers={"Retry-After": "7"})

    async def get():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/limited")

    reply = asyncio.run(get())
    assert reply.status_code == 429
    assert reply.json() == {"detail": {"code": "rate_limit_exceeded", "limit": 3}}
    assert reply.headers["retry-after"] == "7"
    # The class default is untouched by the instance's detail.
    assert rate_limits.RateLimitError.detail == {"code": "rate_limit_exceeded"}


def test_identical_questions_over_http_share_one_search(monkeypatch):
    runs: list[str] = []

    async def slow(session, question):
        runs.append(question)
        await asyncio.sleep(0.05)
        return _response(len(runs))

    monkeypatch.setattr(search, "answer_question", slow)

    async def scenario():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(
                client.post("/api/query", json={"question": "drawings or art"}),
                client.post("/api/query", json={"question": "Drawings or art?"}),
            )

    first, second = asyncio.run(scenario())
    assert runs == ["drawings or art"]
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert len(search._in_flight) == 0
