"""The table definitions agree: the migrations (crawlers/migrate_add_*.sql),
the CREATE and ALTER statements the crawler and the backend run themselves,
and services/schema/tables.py.

The statements exist twice because the crawler creates what a fresh database
lacks, and its image has neither the migrations nor all of backend/. Skipped
unless BIDSEYE_TEST_PG_DSN names a scratch database
(test_match_reasons_postgres.py says how to start one).
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

from services.fulltext import word_form_counts
from services.schema import tables
from services.schema.check import check_schema

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "crawlers" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

from remote import dataset_links, openneuro_metadata, paper_store, site_places  # noqa: E402

DSN = os.getenv("BIDSEYE_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="BIDSEYE_TEST_PG_DSN not set")

# In the order they were written; a later one reads columns of an earlier one.
MIGRATIONS = ("openneuro_metadata", "institutions", "readme_profiles", "search_tsv",
              "dataset_links", "site_places", "paper_search", "exact_search",
              "dataset_facets", "dataset_descriptions")
# What the crawler and the backend run, in an order a fresh database allows.
CODE = (*openneuro_metadata.ensure_columns_sql(), *openneuro_metadata.search_vector_sql(),
        *openneuro_metadata.exact_search_vector_sql(), *dataset_links.TABLES_SQL,
        *paper_store.DDL, *paper_store.DESCRIPTION_DDL, *site_places._DDL,
        *word_form_counts.TABLES_SQL)

_COLUMNS = """
    SELECT table_name || '.' || column_name,
           concat_ws(' | ', data_type, udt_name, is_nullable, column_default, generation_expression)
    FROM information_schema.columns WHERE table_schema = current_schema()"""
_INDEXES = "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = current_schema()"
_CONSTRAINTS = """
    SELECT conrelid::regclass || ' ' || conname, pg_get_constraintdef(oid)
    FROM pg_constraint WHERE connamespace = current_schema()::regnamespace"""


async def _in_fresh_schema(statements, check):
    """Run *check* on an engine whose schema has BIDS-SQL's tables and *statements*."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from db.base import Base
    import db.models  # noqa: F401  (fills Base.metadata)

    schema = f"schema_ddl_{uuid.uuid4().hex[:8]}"
    admin = create_async_engine(DSN)
    async with admin.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_async_engine(DSN, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            for statement in statements:
                await raw.execute(statement)   # a whole file, with its BEGIN and COMMIT
        return await check(engine, schema)
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        await admin.dispose()


async def _definitions(engine, schema):
    from sqlalchemy import text

    async with engine.connect() as conn:
        return {(kind, name.replace(f"{schema}.", "")): definition.replace(f"{schema}.", "")
                for kind, sql in (("column", _COLUMNS), ("index", _INDEXES), ("constraint", _CONSTRAINTS))
                for name, definition in await conn.execute(text(sql))}


def _migrations():
    return [(ROOT / "crawlers" / f"migrate_add_{name}.sql").read_text() for name in MIGRATIONS]


def test_every_migration_tables_py_names_is_run_here() -> None:
    named = {info[tables.MIGRATION] for table in tables.MIGRATED_TABLES
             for info in (table.info, *(column.info for column in table.columns))
             if tables.MIGRATION in info}
    assert named == {f"crawlers/migrate_add_{name}.sql" for name in MIGRATIONS}


def test_the_migrations_create_what_tables_py_declares(monkeypatch) -> None:
    monkeypatch.setenv(tables.EXACT_FLAG, "true")
    monkeypatch.setenv(tables.PAPERS_FLAG, "true")

    async def check(engine, schema):
        async with engine.connect() as conn:
            await check_schema(conn)

    asyncio.run(_in_fresh_schema(_migrations(), check))


def test_the_code_creates_the_same_as_the_migrations() -> None:
    migrated = asyncio.run(_in_fresh_schema(_migrations(), _definitions))
    created = asyncio.run(_in_fresh_schema(CODE, _definitions))
    different = {key: (definition, migrated.get(key)) for key, definition in created.items()
                 if migrated.get(key) != definition}
    assert different == {}
