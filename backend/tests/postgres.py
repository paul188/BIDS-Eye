"""The scratch Postgres of the tests that need real SQL.

BIDSEYE_TEST_PG_DSN names it (postgresql+asyncpg://user:password@host:port/db);
without it these tests are skipped. Each use works in a schema of its own and
drops it afterwards, so the database itself stays empty.

    docker run -d --rm --name bidseye-test-pg -e POSTGRES_PASSWORD=test \
        -p 127.0.0.1:55439:5432 postgres:16
    BIDSEYE_TEST_PG_DSN=postgresql+asyncpg://postgres:test@127.0.0.1:55439/postgres \
        python -m pytest tests

A test marks itself with needs_postgres and runs its coroutine with asyncio.run:

    async with fresh_schema("venues") as engine:
        async with engine.begin() as conn:
            await run_statements(conn, _SCHEMA_SQL)

A test of the real tables takes the fixture migrated_database (conftest.py):
the URL of a database the Alembic revisions built. It is shared by the whole
run, so such a test changes nothing in it, or rolls back (rolled_back).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

DSN_VARIABLE = "BIDSEYE_TEST_PG_DSN"
DSN = os.getenv(DSN_VARIABLE)

needs_postgres = pytest.mark.skipif(not DSN, reason=f"{DSN_VARIABLE} not set")


@asynccontextmanager
async def scratch_engine() -> AsyncIterator[AsyncEngine]:
    """An engine on the scratch database, in its default schema."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    assert DSN, f"{DSN_VARIABLE} not set: mark the test with needs_postgres"
    engine = create_async_engine(DSN, poolclass=NullPool)
    try:
        yield engine
    finally:
        await engine.dispose()


@asynccontextmanager
async def fresh_schema(prefix: str) -> AsyncIterator[AsyncEngine]:
    """An engine whose connections work in a new, empty schema named after
    *prefix*; the schema is dropped on the way out, also after a failure."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    schema = f"{prefix}_{uuid.uuid4().hex[:8]}"
    async with scratch_engine() as admin:
        async with admin.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA {schema}"))
        engine = create_async_engine(
            DSN, poolclass=NullPool, connect_args={"server_settings": {"search_path": schema}})
        try:
            yield engine
        finally:
            await engine.dispose()
            async with admin.begin() as conn:
                await conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))


async def run_statements(conn: AsyncConnection, sql: str) -> None:
    """Run the statements of *sql*, which are separated by semicolons."""
    from sqlalchemy import text

    for statement in sql.split(";"):
        if statement.strip():
            await conn.execute(text(statement))


@contextmanager
def new_database(prefix: str) -> Iterator[str]:
    """The URL of a new database without tables on the scratch server; the
    database is dropped on the way out."""
    from sqlalchemy.engine import make_url

    assert DSN, f"{DSN_VARIABLE} not set: mark the test with needs_postgres"
    name = f"{prefix}_{uuid.uuid4().hex[:8]}"
    asyncio.run(_outside_a_transaction(f"CREATE DATABASE {name}"))
    try:
        yield make_url(DSN).set(database=name).render_as_string(hide_password=False)
    finally:
        asyncio.run(_outside_a_transaction(f"DROP DATABASE {name} WITH (FORCE)"))


async def _outside_a_transaction(statement: str) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    assert DSN
    engine = create_async_engine(DSN, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.execute(text(statement))
    finally:
        await engine.dispose()


def run_on(url: str, work: Callable[[AsyncEngine], Awaitable[Any]]) -> Any:
    """What *work* returns for an engine on the database at *url*."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    async def run() -> Any:
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            return await work(engine)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def rolled_back(url: str, work: Callable[[AsyncConnection], Awaitable[Any]]) -> Any:
    """What *work* returns for a connection on the database at *url*, whose
    changes are then undone."""
    async def run(engine: AsyncEngine) -> Any:
        async with engine.connect() as conn:
            transaction = await conn.begin()
            try:
                return await work(conn)
            finally:
                await transaction.rollback()

    return run_on(url, run)
