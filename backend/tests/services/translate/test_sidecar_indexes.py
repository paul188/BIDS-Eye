"""The sidecar indexes match what the SQL rewriter writes.

PostgreSQL uses an expression index only for the very expression it was built
on. The indexes on bids_objects (backend/migrations/README.md) are on the
guarded number cast of a sidecar value; if the rewriter wrote that cast
another way, searches for field strength or repetition time would silently
take 20 seconds again.

PostgreSQL stores an expression in its own spelling, so each test builds a
second index from what the rewriter writes and compares the two definitions
(needs the scratch server, tests/postgres.py).
"""

from __future__ import annotations

import re

import pytest
from sqlalchemy import text

from tests.postgres import needs_postgres, rolled_back

sqlglot = pytest.importorskip("sqlglot")

from services.translate.sql_rewriter import sql_rewrite_service  # noqa: E402

pytestmark = needs_postgres

# Sidecar key -> the index on its number cast.
NUMBER_INDEXES = {"MagneticFieldStrength": "ix_bids_objects_field_strength",
                  "RepetitionTime": "ix_bids_objects_repetition_time",
                  "SamplingFrequency": "ix_bids_objects_sampling_frequency"}
SCANNER_MODEL_INDEX = "ix_bids_objects_scanner_model_any_trgm"

_DEFINITION = text("SELECT indexdef FROM pg_indexes "
                   "WHERE schemaname = current_schema() AND indexname = :name")


def _definitions(url: str, index: str, probe_sql: str) -> tuple[str, str]:
    """The definition of *index* and of the index *probe_sql* creates, the
    second under the name of the first."""
    async def both(conn):
        await conn.execute(text(probe_sql))
        existing, probe = [(await conn.execute(_DEFINITION, {"name": name})).scalar_one()
                           for name in (index, "probe")]
        return existing, probe.replace("INDEX probe ", f"INDEX {index} ")

    return rolled_back(url, both)


@pytest.mark.parametrize("key", NUMBER_INDEXES)
@pytest.mark.parametrize("cast", ["CAST(o2.other_entities ->> '{key}' AS FLOAT)",
                                  "(o2.other_entities ->> '{key}')::float",
                                  "CAST((o2.other_entities ->> '{key}') AS DOUBLE PRECISION)"])
def test_the_rewriters_number_cast_is_the_indexed_expression(
        migrated_database: str, key: str, cast: str) -> None:
    sql = ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS o2 "
           f"WHERE o2.dataset_id = d.id AND {cast.format(key=key)} >= 7)")
    guarded = sql_rewrite_service._guard_json_number_casts(sql)
    written = re.search(r"CASE WHEN .*? END", guarded).group(0).replace("o2.", "")
    existing, probe = _definitions(
        migrated_database, NUMBER_INDEXES[key],
        f"CREATE INDEX probe ON bids_objects (dataset_id, ({written})) WHERE ({written}) IS NOT NULL")
    assert existing == probe


@pytest.mark.parametrize("key", ["ManufacturersModelName", "ManufacturerModelName"])
def test_the_rewriters_scanner_model_expression_is_the_indexed_one(
        migrated_database: str, key: str) -> None:
    sql = ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS o2 "
           f"WHERE o2.dataset_id = d.id AND o2.other_entities ->> '{key}' ILIKE '%prisma%')")
    unified = sql_rewrite_service._unify_scanner_model_key(sql)
    written = re.search(r"COALESCE\(.*?\)(?= ILIKE)", unified).group(0).replace("o2.", "")
    existing, probe = _definitions(
        migrated_database, SCANNER_MODEL_INDEX,
        f"CREATE INDEX probe ON bids_objects USING gin (({written}) gin_trgm_ops) "
        f"WHERE {written} IS NOT NULL")
    assert existing == probe
