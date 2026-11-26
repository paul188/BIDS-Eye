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
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

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
