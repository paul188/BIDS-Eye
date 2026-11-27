"""The server-side twins of the search page's guards: one search per question
while it runs, whoever asks it (services/search.py ``_in_flight``), and no
blank questions (schemas.QueryRequest). Who asked is not recorded:
tests/test_query_logging.py."""
from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

import routers.search as query_router
from security.rate_limits import RateLimitError
from routers.schemas import QueryRequest, QueryResponse
from services import search


def _response(total: int) -> QueryResponse:
    return QueryResponse(message=f"{total}", translation=None, datasets=[], query_id="q",
                         total=total, page=1, page_size=20, has_more=False)


def _counting_search(monkeypatch, *, fail_with: Exception | None = None) -> list[str]:
    """Replace one search with a slow fake that counts its runs."""
    runs: list[str] = []

    async def fake(session, question):
        runs.append(question)
        await asyncio.sleep(0.05)
        if fail_with is not None:
            raise fail_with
        return _response(len(runs))

    monkeypatch.setattr(search, "answer_question", fake)
    return runs


def test_the_same_question_while_it_runs_is_answered_by_the_running_search(monkeypatch):
    runs = _counting_search(monkeypatch)

    async def scenario():
        return await asyncio.gather(
            query_router.query_datasets(QueryRequest(question="drawings or art"), None, None),
            # Case, spacing and end punctuation do not make it another question.
            query_router.query_datasets(QueryRequest(question="  Drawings  or art?"), None, None),
            query_router.query_datasets(QueryRequest(question="drawings or art"), None, None),
        )

    first, second, third = asyncio.run(scenario())
    assert runs == ["drawings or art"]          # one search, one daily search charged
    assert first is second is third
    assert len(search._in_flight) == 0


def test_another_user_asking_the_same_question_shares_the_search(monkeypatch):
    runs = _counting_search(monkeypatch)

    async def scenario():
        return await asyncio.gather(
            query_router.query_datasets(QueryRequest(question="art"), None, None),
            query_router.query_datasets(QueryRequest(question="Art?"), None, None),
        )

    first, second = asyncio.run(scenario())
    assert runs == ["art"]
    assert first is second
    assert len(search._in_flight) == 0


def test_other_questions_run_their_own_search(monkeypatch):
    runs = _counting_search(monkeypatch)

    async def scenario():
        await asyncio.gather(
            query_router.query_datasets(QueryRequest(question="art"), None, None),
            query_router.query_datasets(QueryRequest(question="music"), None, None),
        )

    asyncio.run(scenario())
    assert sorted(runs) == ["art", "music"]


def test_asking_again_after_the_search_finished_runs_it_again(monkeypatch):
    runs = _counting_search(monkeypatch)
    for _ in range(2):
        asyncio.run(query_router.query_datasets(QueryRequest(question="art"), None, None))
    assert runs == ["art", "art"]


def test_an_error_reaches_the_waiting_request_too(monkeypatch):
    limit = RateLimitError("search per minute used up")
    runs = _counting_search(monkeypatch, fail_with=limit)

    async def scenario():
        return await asyncio.gather(
            query_router.query_datasets(QueryRequest(question="art"), None, None),
            query_router.query_datasets(QueryRequest(question="art"), None, None),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    assert runs == ["art"]
    assert all(r is limit for r in results)
    assert len(search._in_flight) == 0


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_questions_are_refused(question):
    with pytest.raises(ValidationError):
        QueryRequest(question=question)


def test_questions_are_trimmed_and_capped():
    assert QueryRequest(question="  art  ").question == "art"
    QueryRequest(question="x" * 2000)
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 2001)
