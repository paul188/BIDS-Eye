"""
scripts/backfill_openneuro_metadata.py
--------------------------------------
Fill the OpenNeuro metadata columns for datasets that are ALREADY indexed,
without re-crawling them:

    readme_text, dx_status, study_domain, study_design, age_min, age_max, size_bytes

and keep deprecated_reason current: a dataset OpenNeuro no longer has, or one it
flags as deprecated, gets a reason; the mark is cleared when the dataset is fine.

No S3 access and no file indexing happens here. The script asks the OpenNeuro
GraphQL API for 12 datasets per request (about 150 requests and a few minutes
for the whole catalogue) and updates one row per dataset.

Run the migration first:

    psql ... -f crawlers/migrate_add_openneuro_metadata.sql

Then:

    python scripts/backfill_openneuro_metadata.py --dry-run --limit 24   # look first
    python scripts/backfill_openneuro_metadata.py                        # write

Options:
    --dry-run            fetch and report, write nothing
    --limit N            stop after N datasets
    --accessions A,B     only these accession ids
    --only-missing       skip datasets that already have any of the columns filled
    --save-jsonl PATH    also store the raw API records (one JSON object per line)
    --from-jsonl PATH    read records from such a file instead of calling the API

Safe to repeat: a value the API does not provide never overwrites a stored one.
deprecated_reason is the exception — it mirrors the current upstream status, and
is only written when the API gave a definite answer for that dataset.

Database connection: DATABASE_URL if set, otherwise METADATA_DB_USER,
METADATA_DB_PASSWORD, METADATA_DB_HOST, METADATA_DB_PORT and METADATA_DB_NAME
(the variables the backend container already has). Needs only SQLAlchemy and
asyncpg, so it runs inside the backend image.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "crawlers" / "openneuro-crawler"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from remote.openneuro_metadata import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    METADATA_COLUMNS,
    STATUS_COLUMNS,
    OpenNeuroAPIError,
    apply_fields,
    deprecation_reason,
    extract_fields,
    fetch_batch,
    set_deprecation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_openneuro_metadata")


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        # Accept a plain postgresql:// URL as well as an asyncpg one.
        if url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        return url
    try:
        return (
            f"postgresql+asyncpg://{os.environ['METADATA_DB_USER']}:"
            f"{os.environ['METADATA_DB_PASSWORD']}@"
            f"{os.environ['METADATA_DB_HOST']}:{os.getenv('METADATA_DB_PORT', '5432')}/"
            f"{os.environ['METADATA_DB_NAME']}"
        )
    except KeyError as exc:
        raise SystemExit(
            f"Database settings missing ({exc}). Set DATABASE_URL or the METADATA_DB_* variables."
        )


async def _missing_columns(conn) -> List[str]:
    rows = await conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'bids_datasets'"
        )
    )
    present = {row[0] for row in rows}
    return [name for name in (*METADATA_COLUMNS, *STATUS_COLUMNS) if name not in present]


async def _target_accessions(conn, only_missing: bool) -> List[str]:
    where = "source_type LIKE 'openneuro%' AND accession_id IS NOT NULL"
    if only_missing:
        where += " AND " + " AND ".join(f"{name} IS NULL" for name in METADATA_COLUMNS)
    rows = await conn.execute(
        text(f"SELECT DISTINCT accession_id FROM bids_datasets WHERE {where} ORDER BY 1")
    )
    return [row[0] for row in rows]


def _records_from_jsonl(path: Path) -> Tuple[Dict[str, Optional[dict]], Dict[str, str]]:
    """Read saved API answers: records, and reasons for unavailable datasets."""
    records: Dict[str, Optional[dict]] = {}
    unavailable: Dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
                records[entry["accession"]] = entry.get("dataset")
                if entry.get("unavailable"):
                    unavailable[entry["accession"]] = entry["unavailable"]
    return records, unavailable


def _batches(items: List[str], size: int) -> Iterator[List[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--accessions", default="")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--save-jsonl", type=Path)
    parser.add_argument("--from-jsonl", type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--pause", type=float, default=0.4,
                        help="seconds to wait between API requests")
    args = parser.parse_args()

    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as conn:
            missing = await _missing_columns(conn)
            if missing:
                log.error(
                    "bids_datasets lacks column(s) %s — run "
                    "crawlers/migrate_add_openneuro_metadata.sql first.",
                    ", ".join(missing),
                )
                return 2
            accessions = await _target_accessions(conn, args.only_missing)

        if args.accessions:
            wanted = {a.strip() for a in args.accessions.split(",") if a.strip()}
            unknown = sorted(wanted - set(accessions))
            if unknown:
                log.warning("Not indexed (or filtered out), skipping: %s", ", ".join(unknown))
            accessions = [a for a in accessions if a in wanted]
        if args.limit:
            accessions = accessions[: args.limit]

        offline, offline_unavailable = (
            _records_from_jsonl(args.from_jsonl) if args.from_jsonl else (None, {})
        )
        log.info(
            "%d dataset(s) to process%s%s",
            len(accessions),
            f" from {args.from_jsonl}" if offline is not None else " via the OpenNeuro API",
            "  [DRY RUN — nothing is written]" if args.dry_run else "",
        )

        filled: Counter = Counter()
        updated = failed = 0
        no_record: List[str] = []     # API returned nothing, reason unknown
        removed: List[str] = []       # API says: removed, deleted or made private
        flagged: List[str] = []       # exists, but OpenNeuro marks it deprecated
        reasons: Counter = Counter()
        save = open(args.save_jsonl, "w", encoding="utf-8") if args.save_jsonl else None
        try:
            for number, batch in enumerate(_batches(accessions, args.batch_size), start=1):
                if offline is not None:
                    records = {a: offline.get(a) for a in batch}
                    unavailable = {a: offline_unavailable[a] for a in batch if a in offline_unavailable}
                else:
                    try:
                        records, unavailable = await asyncio.to_thread(fetch_batch, batch)
                    except OpenNeuroAPIError as exc:
                        # Nothing is written for this batch; rerun with
                        # --only-missing to pick it up later.
                        failed += len(batch)
                        log.warning("Batch %d (%s …) failed: %s", number, batch[0], exc)
                        continue

                rows: List[Tuple[str, dict]] = []
                # accession → reason (None clears the mark). Only datasets the
                # API gave a definite answer for appear here.
                status: Dict[str, Optional[str]] = {}
                for accession_id in batch:
                    record = records.get(accession_id)
                    why_unavailable = unavailable.get(accession_id)
                    if save is not None:
                        save.write(json.dumps({
                            "accession": accession_id, "dataset": record,
                            "unavailable": why_unavailable,
                        }) + "\n")
                    if record is None:
                        if why_unavailable:
                            removed.append(accession_id)
                            status[accession_id] = why_unavailable
                            reasons[why_unavailable] += 1
                        else:
                            no_record.append(accession_id)
                        continue
                    status[accession_id] = deprecation_reason(record)
                    if status[accession_id]:
                        flagged.append(accession_id)
                        reasons[status[accession_id]] += 1
                    fields = extract_fields(record)
                    for name in METADATA_COLUMNS:
                        if fields.get(name) is not None:
                            filled[name] += 1
                    rows.append((accession_id, fields))

                if not args.dry_run and (rows or status):
                    # One transaction per batch: an interruption loses at most
                    # the batch in flight.
                    async with engine.begin() as conn:
                        for accession_id, fields in rows:
                            updated += await apply_fields(conn, accession_id, fields)
                        for accession_id, reason in status.items():
                            await set_deprecation(conn, accession_id, reason)
                elif args.dry_run:
                    for accession_id, fields in rows[:3]:
                        log.info(
                            "  %s: readme=%s chars, dx_status=%r, study_domain=%r, ages=%s–%s, size=%s",
                            accession_id,
                            len(fields["readme_text"] or ""),
                            fields["dx_status"], fields["study_domain"],
                            fields["age_min"], fields["age_max"], fields["size_bytes"],
                        )

                if number % 20 == 0:
                    log.info("  … %d / %d datasets", min(number * args.batch_size, len(accessions)), len(accessions))
                if offline is None and args.pause > 0:
                    await asyncio.sleep(args.pause)
        finally:
            if save is not None:
                save.close()

        log.info(
            "Done. datasets=%d  rows_updated=%d  unavailable_upstream=%d  flagged_deprecated=%d  "
            "no_api_record=%d  failed_requests=%d%s",
            len(accessions), updated, len(removed), len(flagged), len(no_record), failed,
            "  (DRY RUN)" if args.dry_run else "",
        )
        for reason, count in reasons.most_common():
            log.info("  marked deprecated  %3d × %s", count, reason)
        if removed:
            log.info("  Unavailable on OpenNeuro: %s", ", ".join(removed))
        if flagged:
            log.info("  Flagged deprecated by OpenNeuro: %s", ", ".join(flagged))
        if no_record:
            # No data and no definite answer from the API: status left untouched.
            log.info("  No record, reason unknown (left unchanged): %s", ", ".join(no_record))
        for name in METADATA_COLUMNS:
            log.info("  %-13s provided by the API for %d dataset(s)", name, filled[name])
        return 1 if failed else 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
