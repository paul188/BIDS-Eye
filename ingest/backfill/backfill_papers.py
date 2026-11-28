"""
ingest/backfill/backfill_papers.py
--------------------------
Find the papers each indexed dataset names, look them up, and store them:

    bids_papers, bids_dataset_papers, bids_datasets.paper_text

(docs/plans/Paper enrichment plan.md explains why; the code is in
ingest/openneuro-crawler/remote/paper_links.py, paper_lookup.py and
paper_store.py.)

Per run: one OpenNeuro request per 60 datasets for the associated-paper
fields, then about one Europe PMC request per 15 papers. The whole catalogue
(about 1,200 papers) takes a few minutes. No key and no e-mail address is sent
to any service.

    python ingest/backfill/backfill_papers.py --dry-run --limit 50     # look first
    python ingest/backfill/backfill_papers.py                          # write

Options:
    --dry-run            look everything up and report, write nothing
    --limit N            only the first N datasets
    --accessions A,B     only these accession ids
    --parts a,b          what of a paper becomes searchable text
                         (default title,abstract,mesh,keywords)
    --save-json PATH     also store what was looked up (OpenNeuro fields, links,
                         papers), so a later run needs no network
    --from-json PATH     use such a file instead of calling the services
    --show A,B           print the papers and roles of these datasets

Safe to repeat: the links of each dataset are replaced, papers are updated in
place. The tables and columns are created if missing.

Database connection: DATABASE_URL if set, otherwise the METADATA_DB_* variables
the backend container already has. Needs only SQLAlchemy and asyncpg.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "ingest" / "openneuro-crawler"))

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from remote import paper_lookup, paper_store  # noqa: E402
from remote.paper_links import PaperId  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backfill_papers")


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
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


def _load_saved(path: Path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    key_of = {PaperId(*item.split(":", 1)): key for item, key in saved["key_of"].items()}
    return saved["paper_fields"], (key_of, saved["papers"])


async def _run(args: argparse.Namespace) -> int:
    accessions: Optional[List[str]] = None
    if args.accessions:
        accessions = [a.strip() for a in args.accessions.split(",") if a.strip()]
    parts = tuple(p.strip() for p in args.parts.split(",") if p.strip())
    unknown = set(parts) - set(paper_store.TEXT_PARTS)
    if unknown:
        raise SystemExit(f"Unknown --parts {sorted(unknown)}; choose from {paper_store.TEXT_PARTS}")

    engine = create_async_engine(_database_url())
    try:
        async with engine.begin() as conn:
            datasets = await paper_store.load_datasets(conn, accessions)
            if args.limit:
                datasets = datasets[:args.limit]
            wanted = [d["accession"] for d in datasets]
            log.info("%d datasets", len(wanted))

            if args.from_json:
                paper_fields, resolved = _load_saved(Path(args.from_json))
            else:
                paper_fields = paper_store.fetch_paper_fields(wanted)
                log.info("OpenNeuro answered for %d datasets", len(paper_fields))
                resolved = None

            # Look everything up once, so it can be saved, shown and written.
            entries, papers, key_of = paper_store.plan(
                [dict(d) for d in datasets], paper_fields, paper_lookup.http_get, resolved=resolved)
            resolved = (key_of, papers)
            if args.save_json:
                Path(args.save_json).write_text(json.dumps({
                    "paper_fields": paper_fields,
                    "key_of": {str(pid): key for pid, key in key_of.items()},
                    "papers": papers,
                }), encoding="utf-8")
                log.info("saved lookups to %s", args.save_json)
            for accession in filter(None, (a.strip() for a in (args.show or "").split(","))):
                for entry in entries.get(accession, []):
                    paper = papers[entry["key"]]
                    print(f"{accession}  {entry['role']:8s} {paper['title'][:90]} ({paper.get('year')})\n"
                          f"{'':10s}{entry['reason']}; from {entry['sources']}")

            stats = await paper_store.enrich(
                conn, wanted, paper_fields=paper_fields, resolved=resolved, parts=parts,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                await conn.rollback()
    finally:
        await engine.dispose()

    width = max(len(name) for name in stats)
    for name, value in stats.items():
        print(f"{name:<{width}}  {value}")
    print("dry run: nothing written" if args.dry_run else "written")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--accessions")
    parser.add_argument("--parts", default=",".join(paper_store.TEXT_PARTS))
    parser.add_argument("--save-json")
    parser.add_argument("--from-json")
    parser.add_argument("--show")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
