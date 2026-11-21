"""Database queries from a worker thread (worker_db.py): the database failing
or being slow is WorkerDbError, a bug is not."""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.exc import OperationalError

import worker_db
from worker_db import OnEventLoopError, WorkerDbError


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


class FakeEngine:
    """connect() yields a marker instead of a connection."""

    @asynccontextmanager
    async def connect(self):
        yield "connection"


@pytest.fixture
def pooled(loop):
    return worker_db.PooledWorkerDb(FakeEngine(), loop)


def test_an_answer_comes_back(pooled) -> None:
    async def answer(connection):
        return [connection, 3, 4]
    assert pooled.run(answer, timeout=5) == ["connection", 3, 4]


def test_a_database_error_is_a_worker_db_error(pooled) -> None:
    async def fails(connection):
        raise OperationalError("SELECT", {}, Exception("connection reset"))
    with pytest.raises(WorkerDbError, match="connection reset"):
        pooled.run(fails, timeout=5)


def test_a_slow_database_is_a_worker_db_error(pooled) -> None:
    async def slow(connection):
        await asyncio.sleep(5)
    with pytest.raises(WorkerDbError, match="TimeoutError"):
        pooled.run(slow, timeout=0.05)


def test_a_bug_is_raised_as_it_is(pooled) -> None:
    async def buggy(connection):
        raise TypeError("a bug")
    with pytest.raises(TypeError):
        pooled.run(buggy, timeout=5)


@pytest.mark.parametrize("db", [worker_db.OwnConnectionWorkerDb(),
                                worker_db.PooledWorkerDb(engine=object(), loop=None)])
def test_a_call_on_an_event_loop_is_refused_before_it_can_block(db) -> None:
    async def query(connection):
        pytest.fail("ran on the event loop")

    async def on_loop():
        return db.run(query, timeout=5)

    with pytest.raises(OnEventLoopError):
        asyncio.run(on_loop())


def test_an_own_connection_to_an_unreachable_database_is_a_worker_db_error(monkeypatch) -> None:
    import db.db

    monkeypatch.setattr(db.db.engine, "url", "postgresql+asyncpg://x:y@127.0.0.1:1/none", raising=False)

    async def query(connection):
        return 1
    with pytest.raises(WorkerDbError) as caught:
        worker_db.OwnConnectionWorkerDb().run(query, timeout=5)
    assert isinstance(caught.value.__cause__, (OSError, OperationalError)), repr(caught.value.__cause__)


def test_entry_points_that_skip_startup_get_an_own_connection() -> None:
    """Benchmarks and one-off runs never call install(); they still reach the database."""
    pooled = worker_db.PooledWorkerDb(FakeEngine(), loop=None)
    worker_db.install(pooled)
    try:
        assert worker_db.current() is pooled
    finally:
        worker_db.reset()
    assert isinstance(worker_db.current(), worker_db.OwnConnectionWorkerDb)
