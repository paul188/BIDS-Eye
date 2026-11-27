#!/usr/bin/env python3
"""
RAG/tools/mine_word_forms.py
--------------------------
List the stems under which the catalogue has several word forms, as material
for RAG/word_forms.yaml.

The search matches a word by its stem. Where a stem merges words that mean
different things ("severe" and "several"), a search for one returns every
dataset with the other. This script prints the stems where that would cost
most: those with the most datasets in forms other than the commonest one.
It decides nothing. Read the list, and add to word_forms.yaml only the groups
that are plainly different words; most groups are ordinary inflections
("scan", "scans", "scanning") and need no entry.

Reads bids_word_forms (filled after each crawl, post_crawl.py). Then:

    python RAG/tools/mine_word_forms.py
    python RAG/tools/mine_word_forms.py --limit 400 --min-other 20

Database connection: DATABASE_URL if set, otherwise the METADATA_DB_* variables.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "RAG"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from word_forms import inflections  # noqa: E402

# Stems ranked by the datasets in forms that are not the commonest form's
# plural or singular; the regular inflections are folded in Python below.
STEMS_SQL = (
    "SELECT stem, array_agg(form || ':' || ndoc ORDER BY ndoc DESC, form) "
    "FROM bids_word_forms GROUP BY stem HAVING count(*) > 1"
)


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    try:
        return (f"postgresql+asyncpg://{os.environ['METADATA_DB_USER']}:"
                f"{os.environ['METADATA_DB_PASSWORD']}@{os.environ['METADATA_DB_HOST']}:"
                f"{os.getenv('METADATA_DB_PORT', '5432')}/{os.environ['METADATA_DB_NAME']}")
    except KeyError as exc:
        raise SystemExit(f"Database settings missing ({exc}). Set DATABASE_URL or METADATA_DB_*.") from None


def rank(rows, min_other: int):
    """(datasets in other forms, stem, forms) for stems worth a look."""
    found = []
    for stem, values in rows:
        forms = []
        for value in values:
            form, _, count = value.rpartition(":")
            forms.append((form, int(count)))
        own = inflections(forms[0][0])
        other = max((n for form, n in forms if form not in own), default=0)
        if other >= min_other:
            found.append((other, stem, forms))
    return sorted(found, key=lambda item: (-item[0], item[1]))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=200, help="stems to print")
    parser.add_argument("--min-other", type=int, default=10,
                        help="datasets the largest other form must have")
    parser.add_argument("--forms", type=int, default=8, help="forms shown per stem")
    args = parser.parse_args()

    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as conn:
            rows = list(await conn.execute(text(STEMS_SQL)))
    finally:
        await engine.dispose()
    for other, stem, forms in rank(rows, args.min_other)[:args.limit]:
        shown = ", ".join(f"{form} ({n})" for form, n in forms[:args.forms])
        print(f"{stem:<16} {other:>5}  {shown}")


if __name__ == "__main__":
    asyncio.run(main())
