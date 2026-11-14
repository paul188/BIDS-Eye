#!/usr/bin/env python3
"""Write a one-sentence description of each dataset from its primary paper's abstract.

For every dataset with a primary paper that has an abstract (bids_papers,
bids_dataset_papers; filled by scripts/backfill_papers.py or the crawler), an
LLM writes one sentence on what data was collected. The sentence is checked
(backend/services/paper_summary.py: no invented numbers, no copied phrases,
length) and stored in bids_datasets.paper_summary with its paper_key, model and
prompt version in paper_summary_meta. The result card shows it labelled as
generated, with a link to the paper.

A dataset is skipped when its summary was made from the same paper and
abstract with the same prompt version, so the script can be stopped and rerun
at any time. Summaries whose paper is no longer the dataset's primary paper
(after a new paper lookup) are cleared. The crawler runs the same step for the
datasets it indexed (crawlers/openneuro-crawler/crawler_service.py).

Usage (inside the backend container, which has the keys and the database):

    python scripts/generate_paper_summaries.py --dry-run --limit 10 --save-jsonl /tmp/sample.jsonl
    python scripts/generate_paper_summaries.py                 # all datasets that need one
    python scripts/generate_paper_summaries.py --accessions ds000171,ds004627 --force

Run crawlers/migrate_add_paper_summaries.sql first. Needs GEMINI_API_KEY; the
abstracts (public text) go to Gemini only, never to the Claude fallback.
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

from services import paper_summary as ps  # noqa: E402

log = logging.getLogger("generate_paper_summaries")


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
    parser.add_argument("--model", default=ps.DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4, help="parallel model calls")
    parser.add_argument("--save-jsonl", type=Path, help="append every result here for review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        log.error("GEMINI_API_KEY is not set")
        return 2

    started = time.monotonic()
    engine = create_async_engine(_database_url())
    out = args.save_jsonl.open("a", encoding="utf-8") if args.save_jsonl else None
    try:
        stats = await ps.run(
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

    log.info("Done: %d written, %d rejected by the checks, %d failed, %d up to date, "
             "%d stale cleared, %.0f s%s",
             stats["done"], stats["rejected"], stats["failed"], stats["skipped"],
             stats["pruned"], time.monotonic() - started,
             " (dry run, nothing written)" if args.dry_run else "")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
