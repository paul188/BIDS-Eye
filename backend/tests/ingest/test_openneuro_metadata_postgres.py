"""apply_fields against the real bids_datasets table: the UPDATE must run on
asyncpg (every parameter typed) and fill a name only where one is missing."""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

from tests.postgres import needs_postgres, rolled_back

_CRAWLER = Path(__file__).resolve().parents[3] / "ingest" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

from remote import openneuro_metadata as meta  # noqa: E402

_ROWS = {"ds900001": "", "ds900002": "Unnamed Dataset", "ds900003": "ds900003", "ds900004": "Kept title"}


@needs_postgres
def test_apply_fields_fills_only_missing_names(migrated_database: str) -> None:
    async def work(conn):
        for accession, name in _ROWS.items():
            await conn.execute(text(
                "INSERT INTO bids_datasets (id, root_path, description, name, bids_version, "
                "dataset_type, source_type, accession_id) VALUES (gen_random_uuid(), :root, "
                "'{}', :name, 'unknown', 'raw', 'openneuro', :acc)"),
                {"root": f"/test/{accession}", "name": name, "acc": accession})
        record = {"metadata": {"datasetName": "OpenNeuro title"}}
        for accession in _ROWS:
            assert await meta.apply_fields(conn, accession, meta.extract_fields(record)) == 1
        # No name from the API leaves the stored one alone.
        await meta.apply_fields(conn, "ds900001", meta.extract_fields({}))
        rows = await conn.execute(text(
            "SELECT accession_id, name FROM bids_datasets WHERE accession_id LIKE 'ds9000%'"))
        return dict(rows.all())

    assert rolled_back(migrated_database, work) == {
        "ds900001": "OpenNeuro title", "ds900002": "OpenNeuro title",
        "ds900003": "OpenNeuro title", "ds900004": "Kept title"}
