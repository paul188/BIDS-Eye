"""Every search is recorded in user_queries, and the row is really committed.

From June to November 2026 the insert ran inside a savepoint that nothing
committed, so the table stayed empty while the code looked as if it logged.

Needs the BIDS-SQL submodule (routers.query imports the db package)."""

from __future__ import annotations

import asyncio

import routers.query as query_router
from schemas import QueryRequest, TextToSQLResult
from services import search, search_runner


class _Savepoint:
    def __init__(self, session: "_Session") -> None:
        self.session = session

    async def __aenter__(self) -> "_Savepoint":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.session.released.extend(self.session.pending)
        self.session.pending.clear()
        return False


class _Session:
    """Keeps apart what was added, what a savepoint released and what was committed."""

    def __init__(self, fail_insert: bool = False) -> None:
        self.pending: list = []
        self.released: list = []
        self.committed: list = []
        self.fail_insert = fail_insert

    def add(self, row) -> None:
        if self.fail_insert:
            raise RuntimeError("foreign key violation: login not in users")
        self.pending.append(row)

    def begin_nested(self) -> _Savepoint:
        return _Savepoint(self)

    async def commit(self) -> None:
        self.committed.extend(self.released)
        self.released.clear()


def _search(monkeypatch, session: _Session, question: str = "fMRI datasets with n-back"):
    async def translate(question_: str) -> TextToSQLResult:
        return TextToSQLResult(sql="SELECT 1", explanation="")

    async def empty_page(*args, **kwargs):
        return [], 0

    monkeypatch.setattr(search, "text_to_sql", translate)
    monkeypatch.setattr(search_runner, "run_page", empty_page)
    monkeypatch.setattr(search, "_filters_for", lambda cached: [])
    return asyncio.run(query_router.query_datasets(
        QueryRequest(question=question), session, {"sub": "some-login"}))


def test_a_search_is_recorded_and_committed(monkeypatch) -> None:
    session = _Session()
    _search(monkeypatch, session, "fMRI datasets with n-back")

    by_user = [row for row in session.committed if type(row).__name__ == "UserQuery"]
    assert [(row.github_login, row.question) for row in by_user] == [
        ("some-login", "fMRI datasets with n-back")]
    # The permanent log of every search (models.QueryLog) gets its row too.
    logged = [row for row in session.committed if type(row).__name__ == "QueryLog"]
    assert [(row.question, row.total, row.from_cache) for row in logged] == [
        ("fMRI datasets with n-back", 0, False)]
    assert not session.released, "the row was left in a savepoint and never committed"


def test_a_failed_insert_does_not_break_the_search(monkeypatch) -> None:
    session = _Session(fail_insert=True)
    response = _search(monkeypatch, session)

    assert response.total == 0
    assert not session.committed
