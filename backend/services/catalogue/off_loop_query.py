"""
services/catalogue/off_loop_query.py
------------------------------------
Catalogue queries for code that runs in a worker thread: the translation
pipeline runs in asyncio.to_thread and asks the catalogue for word
frequencies (services/fulltext/text_search.py) and word forms
(services/catalogue/word_form_store.py).

Two ways to run one, shared by both:
  run_on_loop              on the app's event loop with its pooled engine
                           (the lookups main.py installs at startup)
  run_with_own_connection  with a short-lived connection of its own, for
                           callers that installed nothing (scripts, benchmarks)

Both raise CatalogueLookupError when the database cannot answer in time;
the lookups' callers catch only that and search without the numbers.
Anything else raised is a bug and is not caught.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

from sqlalchemy.exc import SQLAlchemyError

from exceptions.expected import ExpectedError

T = TypeVar("T")


class CatalogueLookupError(ExpectedError):
    """A catalogue query from a worker thread could not be answered."""


# A database error; a refused connection (asyncpg raises ConnectionRefusedError,
# an OSError, unwrapped at connect); a timeout (asyncio's and concurrent's are
# separate classes before Python 3.11); the loop cancelling the task at shutdown.
_FAILURES = (SQLAlchemyError, OSError, asyncio.TimeoutError,
             concurrent.futures.TimeoutError, concurrent.futures.CancelledError)


def _error(exc: BaseException) -> CatalogueLookupError:
    return CatalogueLookupError(f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)


def run_on_loop(coroutine: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop,
                timeout: float) -> T:
    """Run *coroutine* on *loop* (another thread's) and wait at most *timeout* s."""
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    try:
        return future.result(timeout=timeout)
    except _FAILURES as exc:
        future.cancel()
        raise _error(exc) from exc


def run_with_own_connection(query: Callable[[Any], Awaitable[T]], timeout: float) -> T:
    """Run *query(connection)* on a connection of its own, in a new event loop.

    Only off the event loop. Also raises CatalogueLookupError when the app's
    database settings cannot be loaded (a one-off container without
    METADATA_DB_*: importing db.db fails with pydantic's ValidationError).
    """
    try:
        from db.db import engine as app_engine
    # pydantic.ValidationError of the settings is a ValueError.
    except (ImportError, ValueError) as exc:
        raise _error(exc) from exc
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    async def _run() -> T:
        # NullPool: the connection belongs to this short-lived event loop only.
        engine = create_async_engine(app_engine.url, poolclass=NullPool)
        try:
            async with engine.connect() as connection:
                return await query(connection)
        finally:
            await engine.dispose()

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout))
    except _FAILURES as exc:
        raise _error(exc) from exc
