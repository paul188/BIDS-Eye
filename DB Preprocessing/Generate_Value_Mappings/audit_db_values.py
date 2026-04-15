#!/usr/bin/env python3
"""
audit_db_values.py — Interactively map unknown DB metadata values.

For every distinct value in the DB (diagnosis, sex, handedness, task,
suffix, datatype) that is NOT yet in value_mappings.py, you are prompted
to provide a natural-language label — with the dataset accession IDs that
contain that value shown so you have context.

Answers are written directly into value_mappings.py.

Usage:
    python audit_db_values.py --db-url "postgresql://user:password@localhost:5432/bids_sql"

    # Only audit specific fields:
    python audit_db_values.py --db-url "..." --fields diagnosis sex

    # Just print a report without prompting:
    python audit_db_values.py --db-url "..." --report-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import value_mappings as _vm

MAPPINGS_FILE = Path(__file__).resolve().parent / "value_mappings.py"

# ── Field config ───────────────────────────────────────────────────────────────

FIELD_CONFIG = {
    "diagnosis": {
        "clean_fn":  _vm.clean_diagnosis,
        "map_var":   "DIAGNOSIS_MAP",
        "map_dict":  _vm.DIAGNOSIS_MAP,
        # returns (raw_value, count, accession_ids[])
        "query": """
            SELECT p.diagnosis AS value,
                   COUNT(*) AS n,
                   ARRAY_AGG(DISTINCT d.accession_id ORDER BY d.accession_id) FILTER (WHERE d.accession_id IS NOT NULL) AS sources
            FROM bids_participants p
            JOIN bids_datasets d ON d.id = p.dataset_id
            WHERE p.diagnosis IS NOT NULL AND p.diagnosis != ''
            GROUP BY p.diagnosis
            ORDER BY n DESC
        """,
    },
    "sex": {
        "clean_fn":  _vm.clean_sex,
        "map_var":   "SEX_LABEL",
        "map_dict":  _vm.SEX_LABEL,
        "query": """
            SELECT p.sex AS value,
                   COUNT(*) AS n,
                   ARRAY_AGG(DISTINCT d.accession_id ORDER BY d.accession_id) FILTER (WHERE d.accession_id IS NOT NULL) AS sources
            FROM bids_participants p
            JOIN bids_datasets d ON d.id = p.dataset_id
            WHERE p.sex IS NOT NULL AND p.sex != ''
            GROUP BY p.sex
            ORDER BY n DESC
        """,
    },
    "handedness": {
        "clean_fn":  _vm.clean_handedness,
        "map_var":   "HANDEDNESS_LABEL",
        "map_dict":  _vm.HANDEDNESS_LABEL,
        "query": """
            SELECT p.handedness AS value,
                   COUNT(*) AS n,
                   ARRAY_AGG(DISTINCT d.accession_id ORDER BY d.accession_id) FILTER (WHERE d.accession_id IS NOT NULL) AS sources
            FROM bids_participants p
            JOIN bids_datasets d ON d.id = p.dataset_id
            WHERE p.handedness IS NOT NULL AND p.handedness != ''
            GROUP BY p.handedness
            ORDER BY n DESC
        """,
    },
    "task": {
        "clean_fn":  _vm.clean_task,
        "map_var":   "TASK_LABEL",
        "map_dict":  _vm.TASK_LABEL,
        "query": """
            SELECT o.task AS value,
                   COUNT(*) AS n,
                   ARRAY_AGG(DISTINCT d.accession_id ORDER BY d.accession_id) FILTER (WHERE d.accession_id IS NOT NULL) AS sources
            FROM bids_objects o
            JOIN bids_datasets d ON d.id = o.dataset_id
            WHERE o.task IS NOT NULL AND o.task != ''
            GROUP BY o.task
            ORDER BY n DESC
        """,
    },
    "suffix": {
        "clean_fn":  _vm.clean_suffix,
        "map_var":   "SUFFIX_LABEL",
        "map_dict":  _vm.SUFFIX_LABEL,
        "query": """
            SELECT o.suffix AS value,
                   COUNT(*) AS n,
                   ARRAY_AGG(DISTINCT d.accession_id ORDER BY d.accession_id) FILTER (WHERE d.accession_id IS NOT NULL) AS sources
            FROM bids_objects o
            JOIN bids_datasets d ON d.id = o.dataset_id
            WHERE o.suffix IS NOT NULL AND o.suffix != ''
            GROUP BY o.suffix
            ORDER BY n DESC
        """,
    },
    "datatype": {
        "clean_fn":  _vm.clean_datatype,
        "map_var":   "DATATYPE_LABEL",
        "map_dict":  _vm.DATATYPE_LABEL,
        "query": """
            SELECT o.datatype AS value,
                   COUNT(*) AS n,
                   ARRAY_AGG(DISTINCT d.accession_id ORDER BY d.accession_id) FILTER (WHERE d.accession_id IS NOT NULL) AS sources
            FROM bids_objects o
            JOIN bids_datasets d ON d.id = o.dataset_id
            WHERE o.datatype IS NOT NULL AND o.datatype != ''
            GROUP BY o.datatype
            ORDER BY n DESC
        """,
    },
}

ALL_FIELDS = list(FIELD_CONFIG.keys())


# ── DB fetch ───────────────────────────────────────────────────────────────────

def fetch_field(engine, field: str) -> list[dict]:
    cfg = FIELD_CONFIG[field]
    with engine.connect() as conn:
        rows = conn.execute(text(cfg["query"])).fetchall()
    return [
        {
            "raw":     r[0],
            "count":   r[1],
            "sources": list(r[2] or [])[:8],   # cap at 8 dataset IDs for display
        }
        for r in rows
    ]


# ── Value_mappings.py updater ──────────────────────────────────────────────────

def _append_to_mappings(map_var: str, raw: str, label: str) -> None:
    """Append a single new entry to the correct dict in value_mappings.py."""
    source = MAPPINGS_FILE.read_text(encoding="utf-8")

    # Find the closing } of the target dict and insert before it
    # We look for the line that contains only "}" after the dict definition.
    pattern = re.compile(
        rf'({re.escape(map_var)}\s*:\s*dict\[.*?\]\s*=\s*\{{.*?)(^\}})',
        re.DOTALL | re.MULTILINE,
    )
    key_line = f'    {raw.strip().lower()!r}: {label!r},\n'

    match = pattern.search(source)
    if match:
        insert_pos = match.start(2)
        new_source = source[:insert_pos] + key_line + source[insert_pos:]
    else:
        # Fallback: just append at end of file
        new_source = source.rstrip() + f"\n\n# Added by audit_db_values.py\n{map_var}[{raw.strip().lower()!r}] = {label!r}\n"

    MAPPINGS_FILE.write_text(new_source, encoding="utf-8")


# ── Interactive audit ─────────────────────────────────────────────────────────

def audit_field(engine, field: str, report_only: bool) -> tuple[int, int, int]:
    """
    Audit one field.  Returns (total, already_mapped, newly_mapped).
    """
    cfg      = FIELD_CONFIG[field]
    clean_fn = cfg["clean_fn"]
    map_dict = cfg["map_dict"]
    map_var  = cfg["map_var"]

    rows = fetch_field(engine, field)
    total = len(rows)

    already_mapped = 0
    newly_mapped   = 0
    filtered_count = 0

    unknown = []
    for row in rows:
        raw = row["raw"]
        key = raw.strip().lower()
        if key in map_dict:
            already_mapped += 1
            continue
        label = clean_fn(raw)
        if label is None:
            filtered_count += 1
            if report_only:
                print(f"  [FILTERED]  {raw!r:40s}  ({row['count']}x)  from: {', '.join(row['sources'][:3])}")
        else:
            # Pass-through (raw value used as-is) — worth reviewing
            unknown.append({**row, "current_label": label})

    if unknown:
        print(f"\n{'─'*60}")
        print(f"  {field.upper()} — {len(unknown)} values not in map  "
              f"({already_mapped} already mapped, {filtered_count} filtered)")
        print(f"{'─'*60}")

    for row in unknown:
        raw    = row["raw"]
        count  = row["count"]
        label  = row["current_label"]
        sources = row["sources"]

        print(f"\n  value   : {raw!r}")
        print(f"  count   : {count} rows")
        print(f"  datasets: {', '.join(sources)}")
        print(f"  current : {label!r}  (pass-through — used as-is)")

        if report_only:
            continue

        print("  Options:")
        print(f"    Enter  → keep as-is ({label!r})")
        print(f"    text   → use this human label instead")
        print(f"    f      → filter out (never appear in prompts)")
        print(f"    q      → stop auditing this field")

        try:
            ans = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if ans == "q":
            print("  Skipping remaining values for this field.")
            break
        elif ans == "f":
            # Add a sentinel so clean_* returns None: we patch the allowlist approach.
            # For now we just note it — actual filtering is via the clean_* function guards.
            print(f"  Marked as filtered (add guard in value_mappings.py if needed).")
        elif ans:
            _append_to_mappings(map_var, raw, ans)
            # Reload the module so subsequent values see the new entry
            import importlib
            importlib.reload(_vm)
            cfg["map_dict"] = getattr(_vm, map_var)
            map_dict = cfg["map_dict"]
            print(f"  ✓ Added {raw.strip().lower()!r} → {ans!r} to {map_var}")
            newly_mapped += 1
        else:
            print(f"  Kept as pass-through: {label!r}")

    return total, already_mapped, newly_mapped


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--fields", nargs="+", choices=ALL_FIELDS, default=ALL_FIELDS,
                        metavar="FIELD",
                        help=f"Which fields to audit (default: all). Choices: {ALL_FIELDS}")
    parser.add_argument("--report-only", action="store_true",
                        help="Print report without prompting for input")
    args = parser.parse_args()

    engine = create_engine(args.db_url)

    print(f"Auditing {len(args.fields)} field(s): {', '.join(args.fields)}")
    if not args.report_only:
        print(f"value_mappings.py: {MAPPINGS_FILE}")
        print()

    total_new = 0
    for field in args.fields:
        _, _, newly = audit_field(engine, field, report_only=args.report_only)
        total_new += newly

    print(f"\n{'='*60}")
    if args.report_only:
        print("Report complete. Re-run without --report-only to update mappings.")
    else:
        print(f"Done. {total_new} new mapping(s) written to value_mappings.py.")
        if total_new:
            print("Re-run the data generation job to use the updated mappings.")


if __name__ == "__main__":
    main()
