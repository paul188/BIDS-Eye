#!/usr/bin/env python3
"""
scan_crawl_errors.py — Scan crawl job logs for non-AccessDenied failures.

Reports two categories:
  1. Dataset-level failures: [FAIL] lines and [crawler] Error lines
  2. File-level failures: [mirror] Failed to download lines (non-access-denied)

Excludes anything access-denied / restricted / embargoed.

Usage:
    python crawlers/scan_crawl_errors.py
    python crawlers/scan_crawl_errors.py --since 25540000
    python crawlers/scan_crawl_errors.py --file-failures
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path("/lustre/scratch/data/s24pjoha_hpc-llm_sql_data/logs")

_ACCESS_DENIED = [
    "accessdenied", "access denied", "access-denied",
    "restricted", "embargoed", "graphql fallback",
    "falling back to graphql", "is access-denied",
]

_FAIL_RE    = re.compile(r"\[FAIL\]\s+(ds\w+)\s+\|\s+(.+?)\s+\|\s+indexed=")
_CRAWLER_RE = re.compile(r"\[crawler\]\s+Error indexing\s+(ds\w+):\s+(.+)")
_MIRROR_RE  = re.compile(r"\[mirror\]\s+Failed to download\s+(\S+):\s+(.+)")


def is_access_denied(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _ACCESS_DENIED)


def job_id(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def scan_log(path: Path) -> tuple[list[tuple], list[tuple]]:
    """
    Returns:
      dataset_errors: [(job_id, accession_id, error_msg), ...]
      file_errors:    [(job_id, s3_key, reason), ...]
    """
    jid = str(job_id(path))
    dataset_errors, file_errors = [], []

    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception as e:
        print(f"  [warn] Could not read {path.name}: {e}", file=sys.stderr)
        return [], []

    seen_ds: set[str] = set()

    for line in lines:
        # Dataset-level [FAIL]
        m = _FAIL_RE.search(line)
        if m:
            ds, msg = m.group(1), m.group(2)
            if not is_access_denied(msg) and ds not in seen_ds:
                dataset_errors.append((jid, ds, msg))
                seen_ds.add(ds)
            continue

        # Dataset-level [crawler] Error
        m = _CRAWLER_RE.search(line)
        if m:
            ds, msg = m.group(1), m.group(2)
            if not is_access_denied(msg) and ds not in seen_ds:
                dataset_errors.append((jid, ds, msg))
                seen_ds.add(ds)
            continue

        # File-level mirror failure
        m = _MIRROR_RE.search(line)
        if m:
            key, reason = m.group(1), m.group(2)
            if not is_access_denied(reason):
                file_errors.append((jid, key, reason))

    return dataset_errors, file_errors


def categorise(msg: str) -> str:
    low = msg.lower()
    if "pybids" in low or "dictionary update" in low:
        return "pybids / BIDS parsing"
    if "int32" in low or "overflow" in low or "out of range" in low:
        return "integer overflow"
    if "json" in low or "decode" in low:
        return "JSON decode"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "connection" in low or "network" in low or "socket" in low:
        return "network / connection"
    if "empty" in low or "no objects" in low or "not found" in low:
        return "empty / missing"
    if "unicode" in low or "encoding" in low:
        return "encoding"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--since", type=int, default=0, metavar="JOB_ID",
                        help="Only scan logs with job ID >= this value")
    parser.add_argument("--file-failures", action="store_true",
                        help="Show individual file-level download failures")
    args = parser.parse_args()

    logs = sorted(args.log_dir.glob("crawl_*.log"), key=job_id)
    if not logs:
        sys.exit(f"No crawl_*.log files in {args.log_dir}")
    if args.since:
        logs = [p for p in logs if job_id(p) >= args.since]
    if not logs:
        sys.exit(f"No logs with job ID >= {args.since}")

    print(f"Scanning {len(logs)} log file(s)...\n")

    all_ds_errors: list[tuple] = []
    all_file_errors: list[tuple] = []

    for path in logs:
        ds_errs, f_errs = scan_log(path)
        all_ds_errors.extend(ds_errs)
        all_file_errors.extend(f_errs)

    # ── Dataset-level failures ────────────────────────────────────────────────
    if all_ds_errors:
        by_cat: dict[str, list] = defaultdict(list)
        for jid, ds, msg in all_ds_errors:
            by_cat[categorise(msg)].append((jid, ds, msg))

        print(f"{'='*60}")
        print(f"  DATASET FAILURES ({len(all_ds_errors)} total, access-denied excluded)")
        print(f"{'='*60}")
        for cat, entries in sorted(by_cat.items()):
            print(f"\n── {cat} ({len(entries)})")
            for jid, ds, msg in sorted(entries, key=lambda x: x[1]):
                print(f"  [{jid}] {ds}")
                print(f"    {msg[:120]}")
    else:
        print("No non-access-denied dataset failures found.")

    # ── File-level failures ───────────────────────────────────────────────────
    if args.file_failures:
        print(f"\n{'='*60}")
        if all_file_errors:
            print(f"  FILE DOWNLOAD FAILURES ({len(all_file_errors)} total, access-denied excluded)")
            print(f"{'='*60}")
            # Group by dataset
            by_ds: dict[str, list] = defaultdict(list)
            for jid, key, reason in all_file_errors:
                ds_m = re.match(r"(ds\w+)/", key)
                ds = ds_m.group(1) if ds_m else "__unknown__"
                by_ds[ds].append((jid, key, reason))
            for ds in sorted(by_ds):
                entries = by_ds[ds]
                print(f"\n  {ds} ({len(entries)} file(s))")
                for jid, key, reason in entries[:5]:
                    print(f"    [{jid}] {key.split('/', 1)[-1][:60]}")
                    print(f"          {reason[:80]}")
                if len(entries) > 5:
                    print(f"    ... and {len(entries) - 5} more")
        else:
            print("  No non-access-denied file download failures found.")
    elif all_file_errors:
        print(f"\n  ({len(all_file_errors)} non-access-denied file download failure(s) — run with --file-failures to see them)")

    print()


if __name__ == "__main__":
    main()
