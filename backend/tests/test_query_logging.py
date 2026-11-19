"""Every search is recorded in query_log, anonymously, and the row is really committed.

The only record of a search is its query_log row: nothing stores who asked.
The site's privacy notice says so (frontend/src/components/CookieBanner.vue).
Until November 2026 a second table, user_queries, kept each question with the
asker's GitHub login; it was removed for that notice.

From June to November 2026 that insert ran inside a savepoint that nothing
committed; the checks on `released` below catch the same mistake for query_log.

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

    def __init__(self) -> None:
        self.pending: list = []
        self.released: list = []
        self.committed: list = []

    def add(self, row) -> None:
        self.pending.append(row)

    def begin_nested(self) -> _Savepoint:
        return _Savepoint(self)

    async def commit(self) -> None:
        self.committed.extend(self.released)
        self.released.clear()

    async def rollback(self) -> None:
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


def test_a_search_is_logged_and_committed(monkeypatch) -> None:
    session = _Session()
    _search(monkeypatch, session, "fMRI datasets with n-back")

    logged = [row for row in session.committed if type(row).__name__ == "QueryLog"]
    assert [(row.question, row.total, row.from_cache) for row in logged] == [
        ("fMRI datasets with n-back", 0, False)]
    assert not session.released, "the row was left in a savepoint and never committed"


def test_no_row_records_who_asked(monkeypatch) -> None:
    session = _Session()
    _search(monkeypatch, session)

    rows = session.committed + session.released + session.pending
    assert {type(row).__name__ for row in rows} == {"QueryLog"}
    for row in rows:
        assert "some-login" not in {str(value) for value in vars(row).values()}, row
