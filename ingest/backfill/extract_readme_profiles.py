#!/usr/bin/env python3
"""Read every dataset's README with an LLM once and store what it says.

For each dataset the model returns a short profile, the conditions the text
names with their role (patients, model, controls, excluded, measured,
mentioned), species, age groups, topics and paradigms, each condition and
paradigm with a verbatim quote. Quotes are checked against the input; see
backend/services/indexing/readme_profile.py.

backend/services/indexing/readme_profile_runner.py does the work, the same code
the crawler runs after each cycle for the datasets it indexed
(ingest/openneuro-crawler/post_crawl.py). This script runs it over the whole
catalogue, or over a sample to review before writing.

Writes bids_datasets.llm_profile / llm_conditions / llm_profile_meta and
bids_dataset_facts. A dataset is skipped when its stored profile was made from
the same input with the same prompt version, so the script can be stopped and
rerun at any time, and a rerun after a README changed redoes only that one.

Usage (inside the backend container, which has the keys and the database):

    python ingest/backfill/extract_readme_profiles.py --dry-run --limit 20 --save-jsonl /out/sample.jsonl
    python ingest/backfill/extract_readme_profiles.py                     # all datasets that need it
    python ingest/backfill/extract_readme_profiles.py --accessions ds000171,ds004627 --force

Needs the database at the newest migration (backend/migrations/README.md). Needs GEMINI_API_KEY.
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

_REPO = Path(__file__).resolve().parents[2]
for _path in (_REPO / "backend", _REPO / "LLM_preprocessor", _REPO / "RAG"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from services.indexing import readme_profile_runner as runner  # noqa: E402

log = logging.getLogger("extract_readme_profiles")


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
            f"Database settings missing ({exc}). Set DATABASE_URL or the METADATA_DB_* variables."
        ) from None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="ask the model, write nothing")
    parser.add_argument("--limit", type=int, default=0, help="at most this many datasets")
    parser.add_argument("--accessions", default="", help="comma-separated accession ids")
    parser.add_argument("--force", action="store_true", help="redo datasets that are up to date")
    parser.add_argument("--model", default=runner.DEFAULT_MODEL)
    parser.add_argument("--thinking", default=runner.DEFAULT_THINKING,
                        help="Gemini 3+ thinking level: minimal, low, medium, high")
    parser.add_argument("--workers", type=int, default=6, help="parallel model calls")
    parser.add_argument("--no-codes", action="store_true", help="skip diagnosis code mapping")
    parser.add_argument("--save-jsonl", type=Path, help="append every result here for review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        log.error("GEMINI_API_KEY is not set")
        return 2

    started = time.monotonic()
    engine = create_async_engine(_database_url())
    out = args.save_jsonl.open("a") if args.save_jsonl else None
    try:
        stats = await runner.run(
            engine, [a.strip() for a in args.accessions.split(",")],
            model=args.model, thinking=args.thinking, workers=args.workers,
            force=args.force, dry_run=args.dry_run, limit=args.limit,
            codes=not args.no_codes, out=out)
    except runner.SchemaMissingError as exc:
        log.error("%s", exc)
        return 2
    except runner.ResolverUnavailableError as exc:
        log.error("Diagnosis resolver unavailable (%s). Fix it, or run with --no-codes "
                  "to store the facts without codes.", exc)
        return 2
    finally:
        if out:
            out.close()
        await engine.dispose()

    log.info("Done: %d profiled, %d failed, %d up to date, %d unverified facts, %.0f s%s",
             stats["done"], stats["failed"], stats["skipped"], stats["unverified"],
             time.monotonic() - started, " (dry run, nothing written)" if args.dry_run else "")
    log.info("Condition roles (role, quote verified): %s",
             ", ".join(f"{k[5:]}={n}" for k, n in sorted(stats.items()) if k.startswith("role:")))
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
