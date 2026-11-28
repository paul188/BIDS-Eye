"""
database/worker.py
------------
Database access for code that runs in a worker thread.

The translation pipeline runs in asyncio.to_thread. The app's database engine
belongs to the event loop of the main thread: an asyncio connection cannot be
used from another thread or another loop. Code in the worker thread that needs
the database (word counts in services/fulltext/text_search.py, word forms in
services/fulltext/word_form_counts.py) asks the installed WorkerDb instead:

    PooledWorkerDb         sends the query to the app's event loop, which runs
                           it with the app's pooled engine; main.py installs
                           it at startup
    OwnConnectionWorkerDb  opens a short-lived connection in an event loop of
                           its own; the default, for entry points that skip
                           main.py (one-off containers, scripts, benchmarks)

Every failure the database can cause (an error, a refused connection, a
timeout, the loop shutting down) is raised as WorkerDbError, and so is a call
from an event loop, where waiting would block that loop. Callers catch only
WorkerDbError and go on without the answer. Anything else raised is a bug and
is not caught.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, TypeVar

from sqlalchemy.exc import SQLAlchemyError

from exceptions.expected import ExpectedError

T = TypeVar("T")

# Takes an open connection (sqlalchemy AsyncConnection), returns the answer.
Query = Callable[[Any], Awaitable[T]]


class WorkerDbError(ExpectedError):
    """The database could not answer a query from a worker thread."""


class OnEventLoopError(WorkerDbError):
    """Called from an event loop: waiting for the answer would block it."""


# A database error; a refused connection (asyncpg raises ConnectionRefusedError,
# an OSError, unwrapped at connect); a timeout (asyncio's and concurrent's are
# separate classes before Python 3.11); the loop cancelling the task at shutdown.
_FAILURES = (SQLAlchemyError, OSError, asyncio.TimeoutError,
             concurrent.futures.TimeoutError, concurrent.futures.CancelledError)


def _error(exc: BaseException) -> WorkerDbError:
    return WorkerDbError(f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)


class WorkerDb(ABC):
    """Runs one query for code in a worker thread and waits for the answer."""

    def run(self, query: Query[T], timeout: float) -> T:
        """*query(connection)*'s result, waiting at most *timeout* seconds."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._run(query, timeout)
        raise OnEventLoopError("a worker-thread query was called on an event loop")

    @abstractmethod
    def _run(self, query: Query[T], timeout: float) -> T:
        """Like run(); called only off any event loop."""


class PooledWorkerDb(WorkerDb):
    """Queries on *loop* (the app's, in another thread) with its *engine*."""

    def __init__(self, engine: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._engine = engine
        self._loop = loop

    def _run(self, query: Query[T], timeout: float) -> T:
        async def with_connection() -> T:
            async with self._engine.connect() as connection:
                return await query(connection)

        future = asyncio.run_coroutine_threadsafe(with_connection(), self._loop)
        try:
            return future.result(timeout=timeout)
        except _FAILURES as exc:
            future.cancel()
            raise _error(exc) from exc


class OwnConnectionWorkerDb(WorkerDb):
    """Queries on a connection of its own, in a new event loop per query.

    Also raises WorkerDbError when the app's database settings cannot be
    loaded (a one-off container without METADATA_DB_*: importing db.db fails
    with pydantic's ValidationError).
    """

    def _run(self, query: Query[T], timeout: float) -> T:
        try:
            from db.db import engine as app_engine
        # pydantic.ValidationError of the settings is a ValueError.
        except (ImportError, ValueError) as exc:
            raise _error(exc) from exc
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        async def with_connection() -> T:
            # NullPool: the connection belongs to this short-lived event loop only.
            engine = create_async_engine(app_engine.url, poolclass=NullPool)
            try:
                async with engine.connect() as connection:
                    return await query(connection)
            finally:
                await engine.dispose()

        try:
            return asyncio.run(asyncio.wait_for(with_connection(), timeout))
        except _FAILURES as exc:
            raise _error(exc) from exc


_current: WorkerDb = OwnConnectionWorkerDb()


def install(db: WorkerDb) -> None:
    """Make *db* the one every worker-thread query uses (main.py at startup)."""
    global _current
    _current = db


def reset() -> None:
    """Back to a connection of its own per query (the state before startup)."""
    install(OwnConnectionWorkerDb())


def current() -> WorkerDb:
    return _current
