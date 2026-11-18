#!/usr/bin/env python3
"""
scripts/refresh_word_forms.py
-----------------------------
Rebuild bids_word_forms: every word form in the datasets' text, the stem the
search reduces it to, and the number of datasets that contain it
(backend/services/catalogue/word_form_store.py). The search uses the table to notice
words whose stem also matches unrelated words ("ageing" and "age").

The crawler rebuilds the table after each cycle (post_crawl.py). Run this
where no crawler runs, after datasets were loaded or their text changed.

Run crawlers/migrate_add_exact_search.sql first. Then:

    python scripts/refresh_word_forms.py --dry-run
    python scripts/refresh_word_forms.py

No LLM, no network: one statement over the database. --dry-run builds the
table inside a transaction, prints the counts and rolls back.

Database connection: DATABASE_URL if set, otherwise the METADATA_DB_* variables.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "backend"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from services.catalogue import word_form_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("refresh_word_forms")


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


class _Rollback(Exception):
    pass


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="build, report, roll back")
    args = parser.parse_args()

    engine = create_async_engine(_database_url())
    try:
        try:
            async with engine.begin() as conn:
                # The role may carry a statement timeout meant for searches.
                await conn.execute(text("SET LOCAL statement_timeout = 0"))
                stats = await word_form_store.refresh(conn)
                log.info("bids_word_forms: %(forms)d forms under %(stems)d stems", stats)
                if args.dry_run:
                    raise _Rollback
        except _Rollback:
            log.info("Dry run: rolled back, nothing written.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
