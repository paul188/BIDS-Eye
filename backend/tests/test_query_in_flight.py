"""The server-side twins of the search page's guards: one search per question
while it runs (routers/query.py ``_in_flight``), and no blank questions
(schemas.QueryRequest)."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import routers.query as query_router
from schemas import QueryRequest, QueryResponse


def _response(total: int) -> QueryResponse:
    return QueryResponse(message=f"{total}", translation=None, datasets=[], query_id="q",
                         total=total, page=1, page_size=20, has_more=False)


def _counting_search(monkeypatch, *, fail_with: Exception | None = None):
    """Replace one search with a slow fake that counts its runs."""
    runs: list[str] = []

    async def fake(session, question, user):
        runs.append(question)
        await asyncio.sleep(0.05)
        if fail_with is not None:
            raise fail_with
        return _response(len(runs))

    monkeypatch.setattr(query_router.search, "answer_question", fake)
    return runs


def test_the_same_question_while_it_runs_is_answered_by_the_running_search(monkeypatch):
    runs = _counting_search(monkeypatch)

    async def scenario():
        return await asyncio.gather(
            query_router.query_datasets(QueryRequest(question="drawings or art"), None, {"sub": "a"}),
            # Case, spacing and end punctuation do not make it another question.
            query_router.query_datasets(QueryRequest(question="  Drawings  or art?"), None, {"sub": "a"}),
            query_router.query_datasets(QueryRequest(question="drawings or art"), None, {"sub": "a"}),
        )

    first, second, third = asyncio.run(scenario())
    assert runs == ["drawings or art"]          # one search, one daily search charged
    assert first is second is third
    assert len(query_router._in_flight) == 0


def test_other_users_and_other_questions_run_their_own_search(monkeypatch):
    runs = _counting_search(monkeypatch)

    async def scenario():
        await asyncio.gather(
            query_router.query_datasets(QueryRequest(question="art"), None, {"sub": "a"}),
            query_router.query_datasets(QueryRequest(question="art"), None, {"sub": "b"}),
            query_router.query_datasets(QueryRequest(question="music"), None, {"sub": "a"}),
        )

    asyncio.run(scenario())
    assert sorted(runs) == ["art", "art", "music"]


def test_asking_again_after_the_search_finished_runs_it_again(monkeypatch):
    runs = _counting_search(monkeypatch)
    for _ in range(2):
        asyncio.run(query_router.query_datasets(QueryRequest(question="art"), None, {"sub": "a"}))
    assert runs == ["art", "art"]


def test_an_error_reaches_the_waiting_request_too(monkeypatch):
    limit = HTTPException(status_code=429, detail={"code": "rate_limit_exceeded"})
    runs = _counting_search(monkeypatch, fail_with=limit)

    async def scenario():
        return await asyncio.gather(
            query_router.query_datasets(QueryRequest(question="art"), None, {"sub": "a"}),
            query_router.query_datasets(QueryRequest(question="art"), None, {"sub": "a"}),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    assert runs == ["art"]
    assert all(r is limit for r in results)
    assert len(query_router._in_flight) == 0


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_questions_are_refused(question):
    with pytest.raises(ValidationError):
        QueryRequest(question=question)


def test_questions_are_trimmed_and_capped():
    assert QueryRequest(question="  art  ").question == "art"
    QueryRequest(question="x" * 2000)
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 2001)
