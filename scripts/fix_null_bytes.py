"""
scripts/fix_null_bytes.py
--------------------------
One-off script: remove null bytes from bids_objects.other_entities JSONB.

PostgreSQL text type cannot contain null bytes, and asyncpg rejects them at the
protocol level.  This script uses psycopg2 (synchronous driver) which reads the raw
JSON bytes, lets Python strip the null bytes, and writes back clean JSON.

Strategy:
  1. Server-side LIKE filter finds only the affected rows.
  2. psycopg2 decodes JSONB to Python dict; _has_nulls() checks recursively.
  3. _strip_nulls() cleans the dict; we re-serialise and UPDATE.

Usage:
    cd /path/to/BIDS-Eye
    python scripts/fix_null_bytes.py

Options (env vars):
    FIX_DRY_RUN=1    print what would be updated without writing
    FIX_BATCH=N      rows per SELECT page (default 1000)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DRY_RUN = os.environ.get("FIX_DRY_RUN", "").strip() == "1"
BATCH = int(os.environ.get("FIX_BATCH", "1000") or 1000)

# ── DB connection ──────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "BIDS-SQL"))
from settings import settings_obj  # noqa: E402

# Convert async URL to sync
_dsn = (
    settings_obj.DATABASE_URL
    .replace("postgresql+asyncpg://", "postgresql://")
    .replace("postgresql+psycopg2://", "postgresql://")
)

# The 6-char JSON escape sequence written as chr(92)+'u0000' to avoid the
# write tool interpreting the literal backslash-u-0-0-0-0 as a null byte.
_NULL_ESCAPE = chr(92) + "u0000"


def _has_nulls(obj) -> bool:
    """Return True if any string value in the JSON object contains a null byte."""
    if isinstance(obj, str):
        return "\x00" in obj
    if isinstance(obj, dict):
        return any(_has_nulls(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nulls(v) for v in obj)
    return False


def _strip_nulls(obj):
    """Recursively remove chr(0) from all string values in a JSON-decoded object."""
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


def main() -> None:
    if DRY_RUN:
        log.info("DRY RUN -- no DB writes will be made")

    conn = psycopg2.connect(_dsn)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    like_pat = "%" + _NULL_ESCAPE + "%"
    cur.execute(
        "SELECT COUNT(*) FROM bids_objects "
        "WHERE other_entities IS NOT NULL "
        "AND other_entities::text LIKE %s",
        (like_pat,),
    )
    total = cur.fetchone()["count"]
    log.info("Found %d rows with null bytes in other_entities -- fixing ...", total)

    updated = skipped = 0

    # Always fetch from OFFSET 0: as rows are cleaned they leave the LIKE result
    # set, so the next batch is always the next chunk of still-affected rows.
    while True:
        cur.execute(
            "SELECT id, other_entities FROM bids_objects "
            "WHERE other_entities IS NOT NULL "
            "AND other_entities::text LIKE %s "
            "ORDER BY id "
            "LIMIT %s",
            (like_pat, BATCH),
        )
        rows = cur.fetchall()
        if not rows:
            break

        for row in rows:
            obj_id = row["id"]
            raw_json: dict = row["other_entities"]  # psycopg2 auto-decodes JSONB -> dict

            if not _has_nulls(raw_json):
                skipped += 1
                continue

            cleaned = _strip_nulls(raw_json)
            cleaned_json = json.dumps(cleaned)

            log.debug("  %s: stripped null bytes", obj_id)
            if not DRY_RUN:
                cur.execute(
                    "UPDATE bids_objects SET other_entities = %s::jsonb WHERE id = %s",
                    (cleaned_json, obj_id),
                )
            updated += 1

        if not DRY_RUN:
            conn.commit()

        log.info("  updated=%d  skipped=%d  (remaining in batch: %d)", updated, skipped, len(rows))

    if DRY_RUN:
        log.info("DRY RUN complete -- would update %d row(s)", updated)
    else:
        conn.commit()
        log.info("Done. updated=%d  skipped=%d", updated, skipped)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
