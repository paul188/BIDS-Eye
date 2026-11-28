"""The Alembic revisions (backend/migrations/) build the database the code
expects.

Skipped unless BIDSEYE_TEST_PG_DSN names a scratch server (tests/postgres.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from config.settings import settings_obj
from services.schema import tables
from services.schema.check import check_schema
from services.schema.revision import (SchemaRevisionError, alembic_config, check_revision,
                                      declared_only, newest_revisions)

from tests.postgres import needs_postgres, rolled_back, run_on

_CRAWLER = Path(__file__).resolve().parents[3] / "ingest" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

from remote import openneuro_metadata, paper_store  # noqa: E402


def test_there_is_one_newest_revision() -> None:
    assert len(newest_revisions()) == 1


def test_every_revision_tables_py_names_exists() -> None:
    named = {info[tables.MIGRATION] for table in tables.MIGRATED_TABLES
             for info in (table.info, *(column.info for column in table.columns))
             if tables.MIGRATION in info}
    revisions = {script.revision for script in
                 ScriptDirectory.from_config(alembic_config()).walk_revisions()}
    assert named and named <= revisions


@needs_postgres
def test_an_empty_database_is_refused_at_startup(empty_database: str) -> None:
    with pytest.raises(SchemaRevisionError, match="is at migration none"):
        run_on(empty_database, check_revision)


@needs_postgres
def test_a_migrated_database_passes_the_startup_checks(migrated_database: str, monkeypatch) -> None:
    monkeypatch.setattr(settings_obj, tables.EXACT_FLAG, True)
    monkeypatch.setattr(settings_obj, tables.PAPERS_FLAG, True)

    async def checks(engine):
        await check_revision(engine)
        async with engine.connect() as conn:
            await check_schema(conn)

    run_on(migrated_database, checks)


@needs_postgres
def test_the_revisions_create_what_the_models_declare(migrated_database: str) -> None:
    """A model changed without a revision shows here."""
    from db.base import Base

    def differences(connection):
        context = MigrationContext.configure(
            connection, opts={"compare_type": True, "include_object": declared_only})
        return compare_metadata(context, Base.metadata)

    async def compare(engine):
        async with engine.connect() as conn:
            return await conn.run_sync(differences)

    assert run_on(migrated_database, compare) == []


# The expression each search vector of bids_datasets is generated from, as the
# code writes it. The benchmarks and the tests of the ranking read these.
VECTORS = {
    "search_tsv": openneuro_metadata.SEARCH_VECTOR_EXPRESSION,
    openneuro_metadata.EXACT_SEARCH_COLUMN: openneuro_metadata.EXACT_SEARCH_VECTOR_EXPRESSION,
    paper_store.PAPER_SEARCH_COLUMN: paper_store.PAPER_SEARCH_VECTOR_EXPRESSION,
    paper_store.PAPER_EXACT_SEARCH_COLUMN: paper_store.PAPER_EXACT_SEARCH_VECTOR_EXPRESSION,
}

_GENERATED_FROM = text(
    "SELECT generation_expression FROM information_schema.columns "
    "WHERE table_schema = current_schema() AND table_name = 'bids_datasets' "
    "AND column_name = :column")


def test_the_vectors_here_are_the_ones_tables_py_declares() -> None:
    assert set(VECTORS) == tables.SEARCH_VECTORS


@needs_postgres
@pytest.mark.parametrize("column", sorted(VECTORS))
def test_a_search_vector_is_generated_from_the_expression_in_the_code(
        migrated_database: str, column: str) -> None:
    """PostgreSQL stores an expression in its own spelling, so the code's one
    is given to a second column and the two stored spellings are compared."""
    async def both(conn):
        await conn.execute(text(
            f"ALTER TABLE bids_datasets ADD COLUMN probe tsvector "
            f"GENERATED ALWAYS AS ({VECTORS[column]}) STORED"))
        return [(await conn.execute(_GENERATED_FROM, {"column": name})).scalar_one()
                for name in (column, "probe")]

    in_database, from_code = rolled_back(migrated_database, both)
    assert in_database == from_code


@needs_postgres
def test_the_columns_the_crawler_writes_exist(migrated_database: str) -> None:
    async def columns(conn):
        return set((await conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'bids_datasets'"))).scalars())

    written = {"institutions", *openneuro_metadata.METADATA_COLUMNS,
               *openneuro_metadata.STATUS_COLUMNS, *openneuro_metadata.SNAPSHOT_COLUMNS}
    assert written <= rolled_back(migrated_database, columns)
