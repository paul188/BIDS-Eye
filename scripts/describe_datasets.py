#!/usr/bin/env python3
"""Check how each primary paper relates to its dataset, and summarise thin READMEs.

backend/services/indexing/dataset_description.py does the work; this runs it over the
database:

  - paper_relation for every dataset with a primary paper that has an
    abstract: reports / larger_sample / related_study / tool / unclear, with
    mismatches and evidence quotes checked against the sources. The card shows
    the paper line only for reports / larger_sample with a verified quote.
  - dataset_summary (2-3 sentences, shown as generated) only where README and
    description are under 500 characters, from README and catalogue first and
    from the paper only as "According to the linked paper, …".

A dataset is skipped when its inputs and the prompt version are unchanged, so
the script can be stopped and rerun at any time. The crawler runs the same
step for the datasets it indexed (crawlers/openneuro-crawler/post_crawl.py).

Usage (inside the backend container, which has the keys and the database):

    python scripts/describe_datasets.py --dry-run --limit 10 --save-jsonl /tmp/sample.jsonl
    python scripts/describe_datasets.py                     # everything that needs it
    python scripts/describe_datasets.py --accessions ds000171,ds004627 --force

Papers whose relation is not shown (related_study, tool, unclear):

    SELECT accession_id, paper_relation->>'relation', paper_relation->>'paper_key'
    FROM bids_datasets WHERE paper_relation->>'shown' = 'false';

Run crawlers/migrate_add_dataset_descriptions.sql first. Needs GEMINI_API_KEY;
README, catalogue and paper text go to Gemini only, never to the Claude fallback.
Database connection: DATABASE_URL, or the METADATA_DB_* variables.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO / "backend", _REPO / "LLM_preprocessor", _REPO / "RAG"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from services.indexing import dataset_description as dd  # noqa: E402

log = logging.getLogger("describe_datasets")


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        return url
    try:
        return (
            "postgresql+asyncpg://"
            f"{os.environ['METADATA_DB_USER']}:{os.environ['METADATA_DB_PASSWORD']}"
            f"@{os.environ['METADATA_DB_HOST']}:{os.environ.get('METADATA_DB_PORT', '5432')}"
            f"/{os.environ['METADATA_DB_NAME']}"
        )
    except KeyError as exc:
        raise SystemExit(
            f"Database settings missing ({exc}). Set DATABASE_URL or the METADATA_DB_* variables.")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="ask the model, write nothing")
    parser.add_argument("--limit", type=int, default=0, help="at most this many datasets")
    parser.add_argument("--accessions", default="", help="comma-separated accession ids")
    parser.add_argument("--force", action="store_true", help="redo datasets that are up to date")
    parser.add_argument("--model", default=dd.DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4, help="parallel datasets")
    parser.add_argument("--save-jsonl", type=Path, help="append every result here for review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        log.error("GEMINI_API_KEY is not set")
        return 2

    started = time.monotonic()
    engine = create_async_engine(_database_url())
    out = args.save_jsonl.open("a", encoding="utf-8") if args.save_jsonl else None
    try:
        stats = await dd.run(
            engine, [a.strip() for a in args.accessions.split(",")],
            model=args.model, workers=args.workers, force=args.force,
            dry_run=args.dry_run, limit=args.limit, out=out)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    finally:
        if out:
            out.close()
        await engine.dispose()

    relations = ", ".join(f"{k.split(':', 1)[1]} {v}" for k, v in sorted(stats.items())
                          if k.startswith("relation:"))
    log.info("Done in %.0f s%s", time.monotonic() - started,
             " (dry run, nothing written)" if args.dry_run else "")
    log.info("Relations: %s; shown %d", relations or "none asked", stats["relation_shown"])
    log.info("Summaries: %d written, %d rejected by the checks; %d failed calls; "
             "%d up to date; %d stale cleared",
             stats["summary_done"], stats["summary_rejected"], stats["failed"],
             stats["skipped"], stats["pruned"])
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
