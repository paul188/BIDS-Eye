"""
scripts/backfill_readme.py
--------------------------
One-off script: populate bids_datasets.readme_text (and fill paper_references
as a fallback) for all already-indexed OpenNeuro datasets.

Strategy per dataset:
  1. Check the local mirror (~/.bids-sql/openneuro/{accession_id}/) for an
     existing README file — no S3 request needed if already cached.
  2. If not on disk, fetch from S3 anonymously.
  3. Sanitize (strip HTML / markdown image blobs, collapse whitespace).
  4. Store a tail-biased 8 000-char slice in readme_text.
  5. Extract all URLs from the full text; if paper_references is currently
     NULL for the dataset, write those URLs as the fallback value.

Run once after the DB migration adds the readme_text column:

    cd /path/to/BIDS-Eye
    python scripts/backfill_readme.py

Options (env vars):
    BACKFILL_DRY_RUN=1   print what would be updated without writing to DB
    BACKFILL_LIMIT=50    stop after N datasets (useful for a test run)
    BIDS_SQL_CACHE=/...  override the mirror cache root (default: ~/.bids-sql)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "BIDS-SQL"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crawlers" / "openneuro-crawler"))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from sqlalchemy import select, update

from db.db import async_session_maker
from db.models import BIDSDataset

# ── constants ─────────────────────────────────────────────────────────────────
OPENNEURO_BUCKET = "openneuro.org"
_CACHE_DIR = Path(os.environ.get("BIDS_SQL_CACHE", Path.home() / ".bids-sql")) / "openneuro"
_README_NAMES = ("README", "README.md", "README.txt", "README.rst")
_URL_RE = re.compile(r"https?://\S+")
_HALF = 4000  # tail-biased slice: keep first + last _HALF chars

DRY_RUN = os.environ.get("BACKFILL_DRY_RUN", "").strip() == "1"
LIMIT = int(os.environ.get("BACKFILL_LIMIT", "0") or 0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── S3 client (lazy singleton) ─────────────────────────────────────────────────
_s3 = None

def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(signature_version=UNSIGNED),
        )
    return _s3


# ── README helpers ─────────────────────────────────────────────────────────────

def _read_mirror_readme(accession_id: str) -> bytes | None:
    """Return raw bytes of the first README variant found in the local mirror."""
    mirror = _CACHE_DIR / accession_id
    for name in _README_NAMES:
        p = mirror / name
        if p.exists() and p.stat().st_size > 0:
            return p.read_bytes()
    return None


def _fetch_s3_readme(accession_id: str) -> bytes | None:
    """
    Fetch the first README variant from S3. Returns None if the dataset is
    restricted (AccessDenied) or has no README.
    """
    client = _s3_client()
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=OPENNEURO_BUCKET, Prefix=f"{accession_id}/"
        ):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                filename = key[len(accession_id) + 1:]
                if filename in _README_NAMES:
                    resp = client.get_object(Bucket=OPENNEURO_BUCKET, Key=key)
                    raw = resp["Body"].read()
                    # Cache to mirror so future re-runs don't re-fetch
                    dest = _CACHE_DIR / accession_id / filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
                    return raw
    except Exception as exc:
        if "AccessDenied" in str(exc):
            log.debug("  %s: access denied (restricted dataset)", accession_id)
        else:
            log.warning("  %s: S3 fetch failed — %s", accession_id, exc)
    return None


def _process_readme(raw: bytes) -> tuple[str | None, list[str]]:
    """
    Sanitize and truncate README bytes.

    Returns:
        (readme_text, url_list)
        readme_text — tail-biased 8 000-char slice, or None if empty after cleaning
        url_list    — up to 20 URLs extracted from the full text (before truncation)
    """
    # Detect UTF-16 (BOM or dense null bytes) and re-decode accordingly
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    elif len(raw) > 20 and raw[1:20:2] == b"\x00" * 9:
        # UTF-16 LE without BOM: every odd byte is 0x00
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    # Strip any remaining null bytes (breaks Postgres UTF-8 validation)
    text = text.replace("\x00", "")

    # Extract URLs from the FULL text before any truncation — references section
    # is usually at the bottom and would otherwise be cut off.
    urls = list(dict.fromkeys(_URL_RE.findall(text)))[:20]

    # Strip HTML tags and markdown image blobs (base64 src can be enormous)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return None, urls

    # Tail-biased slice: keep head + tail so both the intro and the
    # references/acknowledgments at the bottom are searchable
    if len(text) > _HALF * 2:
        text = text[:_HALF] + "\n…\n" + text[-_HALF:]

    return text or None, urls


# ── main backfill loop ─────────────────────────────────────────────────────────

async def main() -> None:
    if DRY_RUN:
        log.info("DRY RUN — no DB writes will be made")

    async with async_session_maker() as session:
        # Fetch all OpenNeuro datasets that still have readme_text = NULL
        rows = (
            await session.execute(
                select(BIDSDataset.id, BIDSDataset.accession_id, BIDSDataset.paper_references)
                .where(
                    BIDSDataset.source_type == "openneuro",
                    BIDSDataset.accession_id.isnot(None),
                    BIDSDataset.readme_text.is_(None),
                )
                .order_by(BIDSDataset.accession_id)
            )
        ).all()

    total = len(rows)
    if LIMIT:
        rows = rows[:LIMIT]
    log.info(
        "Found %d dataset(s) with readme_text=NULL%s",
        total,
        f" — capped at {LIMIT}" if LIMIT else "",
    )

    updated = skipped = s3_fetched = mirror_hit = 0

    for dataset_id, accession_id, existing_refs in rows:
        log.info("Processing %s …", accession_id)

        # 1. Try local mirror first
        raw = _read_mirror_readme(accession_id)
        if raw is not None:
            mirror_hit += 1
            log.info("  %s: README found in mirror cache", accession_id)
        else:
            # 2. Fall back to S3
            raw = _fetch_s3_readme(accession_id)
            if raw is not None:
                s3_fetched += 1
                log.info("  %s: README fetched from S3", accession_id)
            else:
                log.info("  %s: no README found — skipping", accession_id)
                skipped += 1
                continue

        readme_text, urls = _process_readme(raw)
        if not readme_text:
            log.info("  %s: README empty after sanitization — skipping", accession_id)
            skipped += 1
            continue

        # Only backfill paper_references if currently NULL
        new_refs = urls if (not existing_refs and urls) else None

        log.info(
            "  %s: readme_text=%d chars, %d URL(s)%s",
            accession_id,
            len(readme_text),
            len(urls),
            f", backfilling {len(new_refs)} paper_references" if new_refs else "",
        )

        if not DRY_RUN:
            values: dict = {"readme_text": readme_text}
            if new_refs is not None:
                values["paper_references"] = new_refs

            async with async_session_maker() as session:
                await session.execute(
                    update(BIDSDataset)
                    .where(BIDSDataset.id == dataset_id)
                    .values(**values)
                )
                await session.commit()

        updated += 1

    log.info(
        "Done. updated=%d  skipped=%d  mirror_hits=%d  s3_fetches=%d%s",
        updated,
        skipped,
        mirror_hit,
        s3_fetched,
        "  (DRY RUN — no writes)" if DRY_RUN else "",
    )


if __name__ == "__main__":
    asyncio.run(main())
