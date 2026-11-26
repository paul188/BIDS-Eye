"""
remote/dataset_facets.py
------------------------
Keeps the per-dataset summaries of bids_objects current
(created by the migrations, backend/migrations/):

  bids_dataset_facets       distinct (dataset, datatype, suffix, task)
  bids_datasets.n_subjects  number of distinct subjects

The search reads these instead of the file table. They are copies: a dataset
whose files changed must be refreshed, or the search does not find it by its
new data types. post_crawl.py calls refresh() for the datasets of each crawl
cycle; check() reports every dataset whose summary differs from its files.

    python -m remote.dataset_facets --check        # list differences
    python -m remote.dataset_facets --refresh-all  # rebuild everything
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from sqlalchemy import text

_EXISTS = text(
    "SELECT to_regclass('bids_dataset_facets') IS NOT NULL AND EXISTS ("
    "SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() "
    "AND table_name = 'bids_datasets' AND column_name = 'n_subjects')"
)
_IDS = "SELECT id FROM bids_datasets WHERE accession_id = ANY(:accessions)"


async def available(conn) -> bool:
    """True when the migration has been applied to this database."""
    return bool((await conn.execute(_EXISTS)).scalar())


async def refresh(conn, accessions: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Rebuild the summaries for *accessions* (all datasets when None).

    Runs in the caller's transaction, so readers see the old rows until it
    commits. Does nothing when the migration has not been applied.
    """
    if not await available(conn):
        return {"skipped": 1}
    scope = "" if accessions is None else f" WHERE dataset_id IN ({_IDS})"
    params = {} if accessions is None else {"accessions": list(accessions)}
    await conn.execute(text("DELETE FROM bids_dataset_facets" + scope), params)
    inserted = await conn.execute(text(
        "INSERT INTO bids_dataset_facets (dataset_id, datatype, suffix, task) "
        "SELECT DISTINCT dataset_id, datatype, suffix, task FROM bids_objects" + scope), params)
    counted = await conn.execute(text(
        "UPDATE bids_datasets d SET n_subjects = ("
        "SELECT COUNT(DISTINCT o.subject) FROM bids_objects o "
        "WHERE o.dataset_id = d.id AND o.subject IS NOT NULL)"
        + ("" if accessions is None else f" WHERE d.id IN ({_IDS})")), params)
    return {"facets": inserted.rowcount, "datasets": counted.rowcount}


_CHECK = text("""
WITH truth AS (SELECT DISTINCT dataset_id, datatype, suffix, task FROM bids_objects),
     missing AS (SELECT dataset_id FROM (SELECT * FROM truth EXCEPT SELECT * FROM bids_dataset_facets) x),
     extra   AS (SELECT dataset_id FROM (SELECT * FROM bids_dataset_facets EXCEPT SELECT * FROM truth) x),
     counts  AS (SELECT d.id AS dataset_id FROM bids_datasets d
                 LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL
                 GROUP BY d.id, d.n_subjects
                 HAVING d.n_subjects IS DISTINCT FROM COUNT(DISTINCT o.subject))
SELECT d.accession_id FROM bids_datasets d
WHERE d.id IN (SELECT dataset_id FROM missing UNION SELECT dataset_id FROM extra
               UNION SELECT dataset_id FROM counts)
ORDER BY 1
""")


async def check(conn) -> List[str]:
    """Accessions whose summary differs from bids_objects (empty = current)."""
    return [row[0] for row in await conn.execute(_CHECK)]


if __name__ == "__main__":  # pragma: no cover
    import asyncio
    import sys

    async def _main() -> int:
        from db.db import engine
        if "--refresh-all" in sys.argv:
            async with engine.begin() as conn:
                print(await refresh(conn))
        async with engine.connect() as conn:
            stale = await check(conn)
        print(f"{len(stale)} datasets differ from their files" + (": " + ", ".join(stale[:50]) if stale else ""))
        return 1 if stale else 0

    sys.exit(asyncio.run(_main()))
