"""
scripts/demote_paper_links.py
-----------------------------
Demote paper links that the paper-relation check (backend/services/
dataset_description.py, column bids_datasets.paper_relation) found are not the
dataset's own paper: tool or method papers and related studies of the same
lab. A demoted link becomes 'related', stops counting as search evidence, and
the dataset's paper_text is rebuilt from its remaining primary papers.

The new role carries the prefix "override: " in role_reason, so later runs of
scripts/backfill_papers.py and the post-crawl step keep it
(remote/paper_store.py, apply_overrides).

    python scripts/demote_paper_links.py --dry-run     # list what would change
    python scripts/demote_paper_links.py               # write

Options:
    --relations tool,related_study   relations to demote (default)
    --dry-run                        report only

Database connection as scripts/backfill_papers.py (DATABASE_URL or METADATA_DB_*).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "crawlers" / "openneuro-crawler"))
sys.path.insert(0, str(_REPO / "scripts"))

from sqlalchemy import bindparam, text  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY  # noqa: E402
from sqlalchemy.types import String  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from backfill_papers import _database_url  # noqa: E402
from remote import paper_store  # noqa: E402

_CANDIDATES = text("""
    SELECT d.id, d.accession_id, d.paper_relation->>'relation', d.paper_relation->>'paper_key',
           dp.paper_id, dp.role
    FROM bids_datasets d
    JOIN bids_papers p ON p.paper_key = d.paper_relation->>'paper_key'
    JOIN bids_dataset_papers dp ON dp.dataset_id = d.id AND dp.paper_id = p.id
    WHERE d.paper_relation->>'relation' = ANY(:relations)
    ORDER BY d.accession_id
""").bindparams(bindparam("relations", type_=ARRAY(String)))

_PRIMARY_PAPERS = text("""
    SELECT p.title, p.abstract, p.mesh_terms, p.keywords
    FROM bids_dataset_papers dp JOIN bids_papers p ON p.id = dp.paper_id
    WHERE dp.dataset_id = :dataset_id AND dp.role = 'primary'
    ORDER BY dp.position
""")


async def _run(relations: list, dry_run: bool) -> int:
    engine = create_async_engine(_database_url())
    changed = 0
    try:
        async with engine.begin() as conn:
            rows = (await conn.execute(_CANDIDATES, {"relations": relations})).all()
            for dataset_id, accession, relation, key, paper_id, role in rows:
                if role != "primary":
                    print(f"{accession}  {key}  {relation}: already {role}")
                    continue
                print(f"{accession}  {key}  {relation}: primary -> related")
                changed += 1
                if dry_run:
                    continue
                await conn.execute(text(
                    "UPDATE bids_dataset_papers SET role = 'related', role_reason = :reason "
                    "WHERE dataset_id = :dataset_id AND paper_id = :paper_id"),
                    {"reason": f"{paper_store.OVERRIDE_PREFIX}paper-relation check: {relation}",
                     "dataset_id": dataset_id, "paper_id": paper_id})
                primary = [dict(r._mapping) for r in await conn.execute(_PRIMARY_PAPERS, {"dataset_id": dataset_id})]
                await conn.execute(text("UPDATE bids_datasets SET paper_text = :paper_text WHERE id = :dataset_id"),
                                   {"paper_text": paper_store.paper_text(primary), "dataset_id": dataset_id})
            if dry_run:
                await conn.rollback()
    finally:
        await engine.dispose()
    print(f"{changed} link(s) {'would be demoted (dry run)' if dry_run else 'demoted'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relations", default="tool,related_study")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    relations = [r.strip() for r in args.relations.split(",") if r.strip()]
    return asyncio.run(_run(relations, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
