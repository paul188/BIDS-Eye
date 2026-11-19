"""Catalogue queries from a worker thread (services/catalogue/off_loop_query.py):
the database failing or being slow is CatalogueLookupError, a bug is not."""
from __future__ import annotations

import asyncio
import threading

import pytest
from sqlalchemy.exc import OperationalError

from services.catalogue import off_loop_query
from services.catalogue.off_loop_query import CatalogueLookupError


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_an_answer_comes_back(loop) -> None:
    async def answer():
        return [3, 4]
    assert off_loop_query.run_on_loop(answer(), loop, timeout=5) == [3, 4]


def test_a_database_error_is_a_lookup_error(loop) -> None:
    async def fails():
        raise OperationalError("SELECT", {}, Exception("connection reset"))
    with pytest.raises(CatalogueLookupError, match="connection reset"):
        off_loop_query.run_on_loop(fails(), loop, timeout=5)


def test_a_slow_database_is_a_lookup_error(loop) -> None:
    async def slow():
        await asyncio.sleep(5)
    with pytest.raises(CatalogueLookupError, match="TimeoutError"):
        off_loop_query.run_on_loop(slow(), loop, timeout=0.05)


def test_a_bug_is_raised_as_it_is(loop) -> None:
    async def buggy():
        raise TypeError("a bug")
    with pytest.raises(TypeError):
        off_loop_query.run_on_loop(buggy(), loop, timeout=5)


def test_an_own_connection_to_an_unreachable_database_is_a_lookup_error(monkeypatch) -> None:
    import db.db

    monkeypatch.setattr(db.db.engine, "url", "postgresql+asyncpg://x:y@127.0.0.1:1/none", raising=False)

    async def query(connection):
        return 1
    with pytest.raises(CatalogueLookupError) as caught:
        off_loop_query.run_with_own_connection(query, timeout=5)
    assert isinstance(caught.value.__cause__, (OSError, OperationalError)), repr(caught.value.__cause__)
