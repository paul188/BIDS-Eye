"""The Alembic revisions (backend/migrations/) build the database the code
expects.

Skipped unless BIDSEYE_TEST_PG_DSN names a scratch server
(test_match_reasons_postgres.py says how to start one); each test builds a
database of its own there and drops it.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from services.schema import tables
from services.schema.check import check_schema
from services.schema.revision import (SchemaRevisionError, alembic_config, check_revision,
                                      newest_revisions)

DSN = os.getenv("BIDSEYE_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="BIDSEYE_TEST_PG_DSN not set")


async def _admin(statement: str) -> None:
    engine = create_async_engine(DSN, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.execute(text(statement))
    finally:
        await engine.dispose()


@pytest.fixture
def empty_database() -> Iterator[str]:
    """The URL of a new database without tables."""
    name = f"migrations_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f"CREATE DATABASE {name}"))
    try:
        yield make_url(DSN).set(database=name).render_as_string(hide_password=False)
    finally:
        asyncio.run(_admin(f"DROP DATABASE {name} WITH (FORCE)"))


@pytest.fixture
def migrated_database(empty_database: str) -> str:
    command.upgrade(alembic_config(empty_database), "head")
    return empty_database


def _on(url: str, work):
    async def run():
        engine = create_async_engine(url)
        try:
            return await work(engine)
        finally:
            await engine.dispose()
    return asyncio.run(run())


def test_there_is_one_newest_revision() -> None:
    assert len(newest_revisions()) == 1


def test_an_empty_database_is_refused_at_startup(empty_database: str) -> None:
    with pytest.raises(SchemaRevisionError, match="is at migration none"):
        _on(empty_database, check_revision)


def test_a_migrated_database_passes_the_startup_checks(migrated_database: str, monkeypatch) -> None:
    monkeypatch.setenv(tables.EXACT_FLAG, "true")
    monkeypatch.setenv(tables.PAPERS_FLAG, "true")

    async def checks(engine):
        await check_revision(engine)
        async with engine.connect() as conn:
            await check_schema(conn)

    _on(migrated_database, checks)


def test_the_revisions_create_what_the_models_declare(migrated_database: str) -> None:
    """A model changed without a revision shows here."""
    from db.base import Base

    def differences(connection):
        context = MigrationContext.configure(
            connection, opts={"compare_type": True, "include_object": _declared_only})
        return compare_metadata(context, Base.metadata)

    async def compare(engine):
        async with engine.connect() as conn:
            return await conn.run_sync(differences)

    assert _on(migrated_database, compare) == []


def test_the_statements_the_code_still_runs_change_nothing(migrated_database: str) -> None:
    """The crawler and the backend still run CREATE ... IF NOT EXISTS statements
    of their own. On a migrated database they must find everything in place."""
    from tests.test_schema_ddl_postgres import CODE, _definitions

    async def before_and_after(engine):
        before = await _definitions(engine, "public")
        async with engine.begin() as conn:
            for statement in CODE:
                await conn.exec_driver_sql(statement)
        return before, await _definitions(engine, "public")

    before, after = _on(migrated_database, before_and_after)
    assert after == before


def _declared_only(obj, name, type_, reflected, compare_to) -> bool:
    """As in migrations/env.py: what only the database has is left alone."""
    return not (reflected and compare_to is None)
