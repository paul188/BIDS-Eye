"""
scripts/backfill_institutions.py
---------------------------------
One-off script: populate bids_datasets.institutions for all already-indexed
datasets by aggregating InstitutionName from their sidecar JSONs.

Run once after the DB migration adds the institutions column:

    cd /path/to/BIDS-Eye
    python scripts/backfill_institutions.py

Requires the same environment as the backend (DATABASE_URL etc.).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make BIDS-SQL models importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "BIDS-SQL"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crawlers" / "openneuro-crawler"))

from sqlalchemy import text
from db.db import async_session_maker


async def main() -> None:
    async with async_session_maker() as session:
        result = await session.execute(text("""
            UPDATE bids_datasets d
            SET institutions = sub.institutions
            FROM (
                SELECT
                    dataset_id,
                    array_agg(DISTINCT other_entities->>'InstitutionName')
                        FILTER (WHERE other_entities->>'InstitutionName' IS NOT NULL
                                  AND other_entities->>'InstitutionName' != '')
                        AS institutions
                FROM bids_objects
                GROUP BY dataset_id
            ) sub
            WHERE d.id = sub.dataset_id
              AND sub.institutions IS NOT NULL
        """))
        await session.commit()
        print(f"Updated {result.rowcount} dataset(s).")


if __name__ == "__main__":
    asyncio.run(main())
