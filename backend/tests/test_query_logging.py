"""Every search is recorded in query_log, anonymously, and the row is really committed.

A search's query_log row never says who asked, and for a user who has not
turned on "Remember my searches and search results" it is the only record
(the saved searches are services/stores/search_history.py, checked in
tests/test_search_history.py). The site's privacy notice says so
(frontend/src/components/CookieBanner.vue).
Until November 2026 a second table, user_queries, kept each question with the
asker's GitHub login; it was removed for that notice.

From June to November 2026 that insert ran inside a savepoint that nothing
committed; the checks on `released` below catch the same mistake for query_log.

Needs the BIDS-SQL submodule (routers.search imports the db package)."""

from __future__ import annotations

import asyncio

import routers.search as query_router
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


def _search(
    monkeypatch, session: _Session, question: str = "fMRI datasets with n-back",
    github_id: int | None = None,
):
    async def translate(question_: str) -> TextToSQLResult:
        return TextToSQLResult(sql="SELECT 1", explanation="")

    async def empty_page(*args, **kwargs):
        return [], 0

    monkeypatch.setattr(search, "text_to_sql", translate)
    monkeypatch.setattr(search_runner, "run_page", empty_page)
    monkeypatch.setattr(search, "_filters_for", lambda cached: [])
    return asyncio.run(query_router.query_datasets(
        QueryRequest(question=question), session, github_id))


def test_a_search_is_logged_and_committed(monkeypatch) -> None:
    session = _Session()
    _search(monkeypatch, session, "fMRI datasets with n-back")

    logged = [row for row in session.committed if type(row).__name__ == "QueryLog"]
    assert [(row.question, row.total, row.from_cache) for row in logged] == [
        ("fMRI datasets with n-back", 0, False)]
    assert not session.released, "the row was left in a savepoint and never committed"


def test_no_row_records_who_asked(monkeypatch) -> None:
    # The route gets the GitHub account number (GithubIdDep), never the login.
    # With history on it goes into the history entry only (stubbed here).
    saved: list = []

    async def record(session_, github_id, question, response):
        saved.append(github_id)
        return None

    monkeypatch.setattr(query_router.search_history, "record", record)
    session = _Session()
    _search(monkeypatch, session, github_id=4242)

    assert saved == [4242]
    rows = session.committed + session.released + session.pending
    assert {type(row).__name__ for row in rows} == {"QueryLog"}
    for row in rows:
        assert "4242" not in {str(value) for value in vars(row).values()}, row
