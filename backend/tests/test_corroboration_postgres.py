"""Corroboration of a confirmed clause on a real Postgres
(services/evidence/SCORING.md): which source groups confirm it.

Skipped unless BIDSEYE_TEST_PG_DSN names a scratch database (see
tests/test_match_reasons_postgres.py); works in a schema of its own.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from services.uncertain import tier as ut

DSN = os.getenv("BIDSEYE_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="BIDSEYE_TEST_PG_DSN not set")

_SCHEMA_SQL = """
CREATE TABLE bids_datasets (id UUID PRIMARY KEY, name TEXT, description_text TEXT, readme_text TEXT,
                            dx_status TEXT, search_tsv TSVECTOR);
CREATE TABLE bids_participants (dataset_id UUID, participant_id TEXT, diagnosis TEXT)
"""
# key → (README, diagnosis of its one participant)
_DATASETS = {
    "coded_and_named": ("Patients with epilepsy.", "epilepsy"),
    "coded": ("Patients.", "epilepsy"),
    "named": ("Patients with epilepsy.", "other"),
    "neither": ("Patients.", "other"),
}
CLAUSE = ("(EXISTS(SELECT 1 FROM bids_participants AS p WHERE p.dataset_id = d.id "
          "AND p.diagnosis = 'epilepsy') OR d.search_tsv @@ to_tsquery('english', 'epilepsy'))")


async def _corroboration() -> dict:
    """{dataset key: (level, corroboration)} of the tier query."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    schema = f"corroboration_{uuid.uuid4().hex[:8]}"
    admin = create_async_engine(DSN)
    async with admin.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_async_engine(DSN, connect_args={"server_settings": {"search_path": schema}})
    ids = {key: uuid.uuid4() for key in _DATASETS}
    try:
        async with engine.begin() as conn:
            for statement in _SCHEMA_SQL.split(";"):
                await conn.execute(text(statement))
            for key, (readme, code) in _DATASETS.items():
                await conn.execute(text(
                    "INSERT INTO bids_datasets VALUES (:id, :name, '', :readme, NULL, "
                    "to_tsvector('english', :readme))"), {"id": ids[key], "name": key, "readme": readme})
                await conn.execute(text("INSERT INTO bids_participants VALUES (:id, 'sub-1', :code)"),
                                   {"id": ids[key], "code": code})
        sql = ut.apply("SELECT d.id FROM bids_datasets AS d WHERE " + CLAUSE, papers=False)
        async with AsyncSession(engine) as session:
            rows = (await session.execute(text(sql))).mappings().all()
        by_id = {value: key for key, value in ids.items()}
        return {by_id[r["id"]]: (r[ut.LEVEL_COLUMN], r[ut.CORROBORATION_COLUMN]) for r in rows}
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        await admin.dispose()


def test_a_second_source_group_is_one_more() -> None:
    rows = asyncio.run(_corroboration())
    assert rows == {"coded_and_named": (0, 1), "coded": (0, 0), "named": (0, 0)}
