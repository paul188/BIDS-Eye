"""A user's searches are saved only after they turned that on, only under
their own GitHub account number, for RETENTION_DAYS, and are gone when they
turn it off (services/stores/search_history.py, routers/history.py). The
privacy notice promises each of these (frontend/src/components/CookieBanner.vue).

The checks without a database run everywhere. The others need the scratch
database BIDSEYE_TEST_PG_DSN names (tests/postgres.py) and are skipped without
it. Needs the BIDS-SQL submodule (the db package)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from security import rate_limits
import routers.history as history_router
import routers.search as query_router
from deps import get_db, require_auth, user_id
from exceptions.handlers import add_error_handlers
from database.app_tables import QueryLog, SearchHistoryConsent, SearchHistoryEntry
from routers.schemas import QueryRequest, QueryResponse
from services.search import search
from services.stores import search_history

from tests.postgres import fresh_schema, needs_postgres

ANA, BEN = 101, 202
_TABLES = [SearchHistoryConsent.__table__, SearchHistoryEntry.__table__]


def _answer(total: int = 3) -> QueryResponse:
    return QueryResponse(message=f"Found {total} datasets matching your query.", translation=None,
                         datasets=[], query_id="q1", total=total, page=1, page_size=20)


@pytest.fixture(autouse=True)
def _fresh_limits():
    rate_limits.reset()
    yield
    rate_limits.reset()


def _with_session(test):
    """Run ``test(session)`` against empty history tables, in a schema of their own."""
    async def run():
        async with fresh_schema("history") as engine:
            async with engine.begin() as conn:
                await conn.run_sync(lambda sync: [t.create(sync) for t in _TABLES])
            async with AsyncSession(engine, expire_on_commit=False) as session:
                return await test(session)
    return asyncio.run(run())


async def _count(session: AsyncSession, model) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


# ── Without a database ────────────────────────────────────────────────────────

def test_the_account_number_is_an_integer_or_nothing() -> None:
    assert user_id({"sub": "ana", "uid": 101}) == 101
    for user in ({"sub": "ana"}, {"sub": "ana", "uid": "101"}, {"sub": "ana", "uid": True}):
        assert user_id(user) is None


def test_a_session_without_account_number_saves_nothing(monkeypatch) -> None:
    # session=None: any use of the database would fail.
    assert asyncio.run(search_history.record(None, None, "fMRI", _answer())) is None

    async def ask(session, question):
        return _answer()

    monkeypatch.setattr(search, "ask", ask)
    reply = asyncio.run(query_router.query_datasets(QueryRequest(question="fMRI"), None, None))
    assert reply.history_id is None


def _app(user: dict, session: AsyncSession | None = None) -> FastAPI:
    app = FastAPI()
    add_error_handlers(app)
    app.include_router(query_router.search_router, prefix="/api")
    app.include_router(history_router.history_router, prefix="/api")

    async def db():
        yield session

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[require_auth] = lambda: user
    return app


async def _call(app: FastAPI, method: str, path: str, json: dict | None = None) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, path, json=json)


def test_an_old_session_is_asked_to_sign_in_again() -> None:
    app = _app({"sub": "ana"})
    reply = asyncio.run(_call(app, "PUT", "/api/history/enabled", {"enabled": True}))
    assert reply.status_code == 409
    assert "sign in again" in reply.json()["detail"]
    listing = asyncio.run(_call(app, "GET", "/api/history")).json()
    assert listing == {"enabled": False, "retention_days": search_history.RETENTION_DAYS, "entries": []}


# ── With a database ───────────────────────────────────────────────────────────

@needs_postgres
def test_nothing_is_saved_before_the_user_turns_it_on() -> None:
    async def test(session):
        assert await search_history.record(session, ANA, "fMRI", _answer()) is None
        assert await search_history.list_entries(session, ANA) is None
        assert await _count(session, SearchHistoryEntry) == 0
        assert await _count(session, SearchHistoryConsent) == 0
    _with_session(test)


@needs_postgres
def test_a_saved_search_comes_back_as_it_was_answered() -> None:
    async def test(session):
        await search_history.enable(session, ANA)
        entry_id = await search_history.record(session, ANA, "fMRI in children", _answer(7))
        assert entry_id

        listed = await search_history.list_entries(session, ANA)
        assert [(e.id, e.question, e.total) for e in listed] == [(entry_id, "fMRI in children", 7)]

        entry = await search_history.get_entry(session, ANA, UUID(entry_id))
        assert entry.question == "fMRI in children"
        assert entry.answer.total == 7 and entry.answer.history_id == entry_id
        assert entry.answer.message == _answer(7).message
    _with_session(test)


@needs_postgres
def test_turning_it_on_twice_keeps_the_saved_searches() -> None:
    async def test(session):
        await search_history.enable(session, ANA)
        await search_history.record(session, ANA, "fMRI", _answer())
        await search_history.enable(session, ANA)
        assert len(await search_history.list_entries(session, ANA)) == 1
    _with_session(test)


@needs_postgres
def test_a_user_reaches_only_their_own_searches() -> None:
    async def test(session):
        await search_history.enable(session, ANA)
        await search_history.enable(session, BEN)
        entry_id = UUID(await search_history.record(session, ANA, "fMRI", _answer()))

        assert await search_history.list_entries(session, BEN) == []
        with pytest.raises(search_history.HistoryEntryNotFoundError):
            await search_history.get_entry(session, BEN, entry_id)
        await search_history.delete_entry(session, BEN, entry_id)
        assert await _count(session, SearchHistoryEntry) == 1

        await search_history.delete_entry(session, ANA, entry_id)
        assert await _count(session, SearchHistoryEntry) == 0
        with pytest.raises(search_history.HistoryEntryNotFoundError):
            await search_history.get_entry(session, ANA, entry_id)
    _with_session(test)


@needs_postgres
def test_turning_it_off_deletes_the_users_rows_and_no_one_elses() -> None:
    async def test(session):
        await search_history.enable(session, ANA)
        await search_history.enable(session, BEN)
        await search_history.record(session, ANA, "fMRI", _answer())
        await search_history.record(session, BEN, "EEG", _answer())

        await search_history.disable(session, ANA)

        assert await search_history.list_entries(session, ANA) is None
        assert await search_history.record(session, ANA, "MEG", _answer()) is None
        remaining = (await session.execute(select(SearchHistoryEntry.github_id))).scalars().all()
        assert remaining == [BEN]
        consents = (await session.execute(select(SearchHistoryConsent.github_id))).scalars().all()
        assert consents == [BEN]
    _with_session(test)


@needs_postgres
def test_a_search_is_deleted_after_the_storage_period() -> None:
    async def test(session):
        await search_history.enable(session, ANA)
        await search_history.enable(session, BEN)
        old = await search_history.record(session, ANA, "old", _answer())
        await search_history.record(session, ANA, "new", _answer())
        too_old = datetime.now(timezone.utc) - timedelta(days=search_history.RETENTION_DAYS + 1)
        await session.execute(
            update(SearchHistoryEntry).where(SearchHistoryEntry.question == "old")
            .values(asked_at=too_old))
        await session.commit()

        with pytest.raises(search_history.HistoryEntryNotFoundError):
            await search_history.get_entry(session, ANA, UUID(old))
        # Someone else using their history removes it for good.
        await search_history.list_entries(session, BEN)
        questions = (await session.execute(select(SearchHistoryEntry.question))).scalars().all()
        assert questions == ["new"]
    _with_session(test)


@needs_postgres
def test_an_unused_consent_ends_after_the_storage_period() -> None:
    async def test(session):
        await search_history.enable(session, ANA)
        await search_history.enable(session, BEN)
        await search_history.record(session, ANA, "fMRI", _answer())
        too_old = datetime.now(timezone.utc) - timedelta(days=search_history.RETENTION_DAYS + 1)
        await session.execute(
            update(SearchHistoryEntry).where(SearchHistoryEntry.github_id == ANA)
            .values(asked_at=too_old))
        await session.execute(
            update(SearchHistoryConsent).where(SearchHistoryConsent.github_id == ANA)
            .values(last_used_at=too_old))
        await session.commit()

        assert await search_history.list_entries(session, ANA) is None
        assert await search_history.record(session, ANA, "again", _answer()) is None
        assert await _count(session, SearchHistoryEntry) == 0
        consents = (await session.execute(select(SearchHistoryConsent.github_id))).scalars().all()
        assert consents == [BEN]
    _with_session(test)


@needs_postgres
def test_over_http_a_search_is_saved_listed_reopened_and_deleted(monkeypatch) -> None:
    shared = _answer(5)

    async def ask(session, question):
        return shared  # one object for every asker, as from the in-flight search

    monkeypatch.setattr(search, "ask", ask)

    async def test(session):
        ana = _app({"sub": "ana", "uid": ANA}, session)
        ben = _app({"sub": "ben", "uid": BEN}, session)

        before = (await _call(ana, "POST", "/api/query", {"question": "fMRI"})).json()
        assert before["history_id"] is None

        assert (await _call(ana, "PUT", "/api/history/enabled", {"enabled": True})).json() == {"enabled": True}
        saved = (await _call(ana, "POST", "/api/query", {"question": "fMRI in children"})).json()
        assert saved["history_id"]
        assert shared.history_id is None, "the shared response was changed"
        # Ben asks the same and has not turned it on.
        assert (await _call(ben, "POST", "/api/query", {"question": "fMRI in children"})).json()["history_id"] is None

        listing = (await _call(ana, "GET", "/api/history")).json()
        assert listing["enabled"] is True
        assert [(e["id"], e["question"], e["total"]) for e in listing["entries"]] == [
            (saved["history_id"], "fMRI in children", 5)]
        assert (await _call(ben, "GET", "/api/history")).json()["entries"] == []

        path = f"/api/history/{saved['history_id']}"
        assert (await _call(ben, "GET", path)).status_code == 404
        entry = (await _call(ana, "GET", path)).json()
        assert entry["answer"]["total"] == 5 and entry["question"] == "fMRI in children"

        assert (await _call(ana, "DELETE", path)).status_code == 204
        assert (await _call(ana, "GET", path)).status_code == 404

        await _call(ana, "POST", "/api/query", {"question": "EEG"})
        assert (await _call(ana, "PUT", "/api/history/enabled", {"enabled": False})).json() == {"enabled": False}
        assert await _count(session, SearchHistoryEntry) == 0
        assert (await _call(ana, "GET", "/api/history")).json()["enabled"] is False
        assert (await _call(ana, "GET", f"/api/history/{uuid4()}")).status_code == 404
    _with_session(test)


# Who asks → the session they hold. Ben has saving turned on too; the old
# session predates saved searches and has no account number.
_ASKERS = {
    "owner": {"sub": "ana", "uid": ANA},
    "other user": {"sub": "ben", "uid": BEN},
    "old session": {"sub": "ana"},
}
_SECRET = "fMRI in children"


@needs_postgres
@pytest.mark.parametrize("asker, method, path, body, status, left", [
    # The owner reaches the search and can remove it.
    ("owner", "GET", "/api/history", None, 200, 1),
    ("owner", "GET", "/api/history/{id}", None, 200, 1),
    ("owner", "DELETE", "/api/history/{id}", None, 204, 0),
    ("owner", "PUT", "/api/history/enabled", {"enabled": False}, 200, 0),
    # Anyone else neither reads nor removes it. A DELETE answers 204 either
    # way, so the answer does not tell whether the search exists.
    ("other user", "GET", "/api/history", None, 200, 1),
    ("other user", "GET", "/api/history/{id}", None, 404, 1),
    ("other user", "DELETE", "/api/history/{id}", None, 204, 1),
    ("other user", "PUT", "/api/history/enabled", {"enabled": False}, 200, 1),
    ("old session", "GET", "/api/history", None, 200, 1),
    ("old session", "GET", "/api/history/{id}", None, 404, 1),
    ("old session", "DELETE", "/api/history/{id}", None, 204, 1),
    ("old session", "PUT", "/api/history/enabled", {"enabled": False}, 200, 1),
    ("old session", "PUT", "/api/history/enabled", {"enabled": True}, 409, 1),
])
def test_each_history_address_for_each_asker(asker, method, path, body, status, left) -> None:
    """One saved search of Ana's: what each asker gets from each address, and
    whether the search is still there afterwards."""
    async def test(session):
        await search_history.enable(session, ANA)
        await search_history.enable(session, BEN)
        entry_id = await search_history.record(session, ANA, _SECRET, _answer())

        reply = await _call(_app(_ASKERS[asker], session), method, path.format(id=entry_id), body)

        assert reply.status_code == status
        assert (_SECRET in reply.text) == (asker == "owner" and method == "GET")
        session.expire_all()
        owned = select(func.count()).select_from(SearchHistoryEntry).where(SearchHistoryEntry.github_id == ANA)
        assert (await session.execute(owned)).scalar_one() == left
    _with_session(test)


def test_the_anonymous_search_log_has_no_user_column() -> None:
    columns = set(QueryLog.__table__.columns.keys())
    assert not columns & {"github_id", "github_login", "user", "user_id", "history_id"}
    assert not QueryLog.__table__.foreign_keys
    assert not any(fk.column.table is QueryLog.__table__ for t in _TABLES for fk in t.foreign_keys)
