#!/usr/bin/env python3
"""
scripts/backfill_dataset_links.py
---------------------------------
Fill bids_derivatives and the 'dicom' rows of bids_dataset_sites for datasets
that are already indexed (new datasets get them from post_crawl.py):

  derivatives  OpenNeuro's preprocessing bucket (s3://openneuro-derivatives:
               fmriprep, mriqc, fitlins, xcpd), listed once, and each
               dataset's own derivatives/ folders via the OpenNeuro file tree
  sites        bids_datasets.institutions copied as source 'dicom'

The 'ethics_board' site rows come with the OpenNeuro metadata
(scripts/backfill_openneuro_metadata.py).

Run crawlers/migrate_add_dataset_links.sql first. Then:

    python scripts/backfill_dataset_links.py --dry-run --limit 20
    python scripts/backfill_dataset_links.py
    python scripts/backfill_dataset_links.py --accessions-file ids.txt --sql-out links.sql

--sql-out writes the statements to a file instead of a database, for a
database the script cannot reach (apply it with psql -v ON_ERROR_STOP=1 -f).
Without it the accession list comes from the database, or from
--accessions / --accessions-file.

Public sources only, no LLM. Database connection as in
scripts/backfill_openneuro_metadata.py (DATABASE_URL or METADATA_DB_*).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "crawlers" / "openneuro-crawler"))

from remote import dataset_links as links  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backfill_dataset_links")


def _q(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def sql_for(found: Dict[str, Dict[str, List[links.Derivative]]], accessions: List[str]) -> str:
    """The writes as one psql script: replace per (dataset, source), then dicom sites."""
    lines = ["BEGIN;"] + [s.rstrip() + ";" for s in links.TABLES_SQL]
    for source, per_dataset in found.items():
        for accession, derivatives in sorted(per_dataset.items()):
            lines.append(
                "DELETE FROM bids_derivatives x USING bids_datasets d WHERE x.dataset_id = d.id "
                f"AND d.accession_id = {_q(accession)} AND x.source = {_q(source)};")
            for d in derivatives:
                lines.append(
                    "INSERT INTO bids_derivatives (dataset_id, pipeline, pipeline_name, "
                    "pipeline_version, source, location, n_subjects) "
                    f"SELECT d.id, {_q(d.pipeline)}, {_q(d.pipeline_name)}, {_q(d.pipeline_version)}, "
                    f"{_q(source)}, {_q(d.location)}, {_q(d.n_subjects)} FROM bids_datasets d "
                    f"WHERE d.accession_id = {_q(accession)} "
                    "ON CONFLICT (dataset_id, source, location) DO NOTHING;")
    array = "ARRAY[" + ", ".join(_q(a) for a in accessions) + "]::text[]"
    lines += [
        "DELETE FROM bids_dataset_sites s USING bids_datasets d WHERE s.dataset_id = d.id "
        f"AND d.accession_id = ANY({array}) AND s.source = 'dicom';",
        "INSERT INTO bids_dataset_sites (dataset_id, name_raw, source) "
        "SELECT DISTINCT d.id, btrim(n), 'dicom' FROM bids_datasets d, unnest(d.institutions) n "
        f"WHERE d.accession_id = ANY({array}) AND btrim(n) <> '' "
        "ON CONFLICT (dataset_id, source, name_raw) DO NOTHING;",
        "COMMIT;",
    ]
    return "\n".join(lines) + "\n"


def fetch_all(accessions: List[str]) -> Dict[str, Dict[str, List[links.Derivative]]]:
    log.info("Listing s3://%s …", links.DERIVATIVES_BUCKET)
    repositories = links.list_bucket_repositories()
    log.info("  %d datasets have OpenNeuro derivatives", sum(1 for a in accessions if a in repositories))
    bucket = links.openneuro_derivatives(accessions, repositories=repositories)
    log.info("Reading derivatives/ folders of %d datasets via the OpenNeuro API …", len(accessions))
    own = links.in_dataset_derivatives(accessions)
    missing = len(accessions) - len(own)
    if missing:
        log.warning("  %d datasets not returned by the API: their in_dataset rows are left as they are",
                    missing)
    return {links.SOURCE_OPENNEURO_DERIVATIVES: bucket, links.SOURCE_IN_DATASET: own}


def report(found: Dict[str, Dict[str, List[links.Derivative]]]) -> None:
    for source, per_dataset in found.items():
        rows = [d for ds in per_dataset.values() for d in ds]
        with_any = sum(1 for ds in per_dataset.values() if ds)
        top = Counter(d.pipeline for d in rows).most_common(12)
        log.info("%s: %d rows in %d datasets; pipelines %s", source, len(rows), with_any, top)


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    try:
        return (f"postgresql+asyncpg://{os.environ['METADATA_DB_USER']}:"
                f"{os.environ['METADATA_DB_PASSWORD']}@{os.environ['METADATA_DB_HOST']}:"
                f"{os.getenv('METADATA_DB_PORT', '5432')}/{os.environ['METADATA_DB_NAME']}")
    except KeyError as exc:
        raise SystemExit(f"Database settings missing ({exc}). Set DATABASE_URL or METADATA_DB_*.")


async def _accessions_from_db(engine) -> List[str]:
    from sqlalchemy import text
    async with engine.connect() as conn:
        rows = await conn.execute(text(
            "SELECT DISTINCT accession_id FROM bids_datasets "
            "WHERE source_type LIKE 'openneuro%' AND accession_id IS NOT NULL ORDER BY 1"))
        return [row[0] for row in rows]


async def _write(engine, found, accessions: List[str]) -> None:
    async with engine.begin() as conn:
        await links.ensure_tables(conn)
        for source, per_dataset in found.items():
            for accession, derivatives in per_dataset.items():
                await links.replace_derivatives(conn, accession, source, derivatives)
        await links.refresh_dicom_sites(conn, accessions)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    parser.add_argument("--limit", type=int, help="only the first N datasets")
    parser.add_argument("--accessions", help="comma-separated accession ids")
    parser.add_argument("--accessions-file", type=Path, help="one accession id per line")
    parser.add_argument("--sql-out", type=Path, help="write a psql script instead of a database")
    args = parser.parse_args()

    accessions: List[str] = []
    if args.accessions:
        accessions = [a.strip() for a in args.accessions.split(",") if a.strip()]
    elif args.accessions_file:
        accessions = [l.strip() for l in args.accessions_file.read_text().splitlines() if l.strip()]

    if not accessions and args.sql_out:
        raise SystemExit("--sql-out needs --accessions or --accessions-file")
    engine = None
    if not accessions or not (args.dry_run or args.sql_out):
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(_database_url())
        if not accessions:
            accessions = await _accessions_from_db(engine)
    if args.limit:
        accessions = accessions[:args.limit]

    found = await asyncio.to_thread(fetch_all, accessions)
    report(found)
    if args.dry_run:
        log.info("Dry run: nothing written.")
    elif args.sql_out:
        args.sql_out.write_text(sql_for(found, accessions), encoding="utf-8")
        log.info("Wrote %s", args.sql_out)
    else:
        await _write(engine, found, accessions)
        log.info("Written.")
    if engine is not None:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
