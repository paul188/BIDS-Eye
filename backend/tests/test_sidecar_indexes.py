"""The sidecar indexes match what the SQL rewriter writes.

PostgreSQL uses an expression index only for the very expression it was built
on. crawlers/migrate_add_sidecar_indexes.sql indexes the guarded number cast
of a sidecar value; if the rewriter wrote that cast another way, searches for
field strength or repetition time would silently take 20 seconds again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

sqlglot = pytest.importorskip("sqlglot")

from services.translate.sql_rewriter import sql_rewrite_service  # noqa: E402

MIGRATION = Path(__file__).resolve().parents[2] / "crawlers" / "migrate_add_sidecar_indexes.sql"
KEYS = ("MagneticFieldStrength", "RepetitionTime", "SamplingFrequency")


def _flat(sql: str) -> str:
    return re.sub(r"\s+", " ", sql)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("cast", ["CAST(o2.other_entities ->> '{key}' AS FLOAT)",
                                  "(o2.other_entities ->> '{key}')::float",
                                  "CAST((o2.other_entities ->> '{key}') AS DOUBLE PRECISION)"])
def test_the_rewriters_number_cast_is_the_indexed_expression(key: str, cast: str) -> None:
    sql = ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS o2 "
           f"WHERE o2.dataset_id = d.id AND {cast.format(key=key)} >= 7)")
    guarded = sql_rewrite_service._guard_json_number_casts(sql)
    written = re.search(r"CASE WHEN .*? END", guarded).group(0).replace("o2.", "")
    assert _flat(written) in _flat(MIGRATION.read_text())
