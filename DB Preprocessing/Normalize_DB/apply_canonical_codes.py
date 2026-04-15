#!/usr/bin/env python3
"""
apply_canonical_codes.py
-------------------------
Reads 'canonical_code' entries from value_mappings.yaml and normalises the
database so every synonymous code is replaced with its single canonical form.

STEP 1 — Edit value_mappings.yaml
  Add a `standard_code` key to any leaf node whose codes you want to unify:

    healthy_control:
      label: healthy control
      standard_code: healthy_control   # ← this is the one code everything maps to
      codes:
        - ctl
        - ctr
        - ctrl
        - hc          # standard_code must be listed here too (or will be added)
        - control
        - healthy
        ...

  Only codes that differ from standard_code are updated; the standard_code value
  itself is left unchanged.  standard_code is case-insensitive at match time
  but written to the DB exactly as you typed it.

STEP 2 — Dry-run to preview changes
  python training_data_generation/apply_canonical_codes.py --dry-run

STEP 3 — Apply
  python training_data_generation/apply_canonical_codes.py

Sections updated (yaml section → db table.column):
  diagnosis  → bids_participants.diagnosis
  sex        → bids_participants.sex
  handedness → bids_participants.handedness
  task       → bids_objects.task
  suffix     → bids_objects.suffix
  datatype   → bids_objects.datatype

Usage:
  python training_data_generation/apply_canonical_codes.py \\
      --db-url postgresql://user:password@localhost:5429/bids_sql \\
      [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from sqlalchemy import create_engine, text

YAML_PATH = Path(__file__).with_name("value_mappings.yaml")

FIELD_MAP = {
    "diagnosis":  ("bids_participants", "diagnosis"),
    "sex":        ("bids_participants", "sex"),
    "handedness": ("bids_participants", "handedness"),
    "task":       ("bids_objects",      "task"),
    "suffix":     ("bids_objects",      "suffix"),
    "datatype":   ("bids_objects",      "datatype"),
}

_LEAF_KEYS = {"label", "description", "codes", "synonyms", "standard_code",
              "extra_codes", "dataset_codes"}


def collect_rewrites(d: dict) -> list[tuple[str, str, str, list[str] | None]]:
    """
    Walk the YAML and yield (path, from_code, canonical_code, datasets_or_None).

    datasets_or_None:
      None  — global rewrite (UPDATE ... WHERE col = from_code)
      list  — dataset-scoped rewrite (UPDATE ... WHERE col = from_code
                AND dataset_id IN (SELECT id FROM bids_datasets WHERE accession_id = ANY(:ids)))

    Sources:
      'codes'        — global rewrites  (datasets_or_None = None)
      'dataset_codes'— scoped rewrites  (datasets_or_None = list of accession_ids)
    """
    results: list[tuple[str, str, str, list[str] | None]] = []

    def walk(node: dict, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "standard_code" in node:
            canonical = str(node["standard_code"])

            # Global codes
            for c in node.get("codes", []):
                code_str = str(c)
                if code_str.lower() != canonical.lower():
                    results.append((path, code_str, canonical, None))

            # Dataset-scoped codes
            for dc in node.get("dataset_codes", []):
                raw = str(dc.get("raw", ""))
                datasets: list[str] = dc.get("datasets", [])
                if raw and datasets and raw.lower() != canonical.lower():
                    results.append((path, raw, canonical, datasets))

        for key, val in node.items():
            if key not in _LEAF_KEYS and isinstance(val, dict):
                child_path = f"{path}.{key}" if path else key
                walk(val, child_path)

    walk(d, "")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-url",
                        default="postgresql://user:password@localhost:5429/bids_sql",
                        help="SQLAlchemy DB URL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without modifying the database")
    args = parser.parse_args()

    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    engine = create_engine(args.db_url)
    grand_total = 0

    # Use a transaction so all changes commit atomically (or roll back on error)
    with engine.begin() as conn:
        for section, (table, col) in FIELD_MAP.items():
            if section not in data:
                continue

            rewrites = collect_rewrites(data[section])
            if not rewrites:
                continue

            # Split into global rewrites (datasets=None) and scoped rewrites (datasets=list)
            global_rewrites = [(p, fc, can) for p, fc, can, ds in rewrites if ds is None]
            scoped_rewrites  = [(p, fc, can, ds) for p, fc, can, ds in rewrites if ds is not None]

            bar = "─" * 60
            printed_header = False

            def _print_header() -> None:
                nonlocal printed_header
                if not printed_header:
                    print(f"\n{bar}")
                    print(f"  {section.upper()}  ({table}.{col})")
                    print(bar)
                    printed_header = True

            # ── Global rewrites ───────────────────────────────────────────────
            if global_rewrites:
                # Deduplicate: same from_code under multiple YAML paths — keep first.
                seen_from: dict[str, str] = {}
                deduped: list[tuple[str, str, str]] = []
                for path, from_code, canonical in global_rewrites:
                    key = from_code.lower()
                    if key not in seen_from:
                        seen_from[key] = canonical
                        deduped.append((path, from_code, canonical))

                from_lowers = [fc.lower() for _, fc, _ in deduped]
                count_rows = conn.execute(
                    text(
                        f"SELECT LOWER({col}) AS raw, COUNT(*) AS n "
                        f"FROM {table} "
                        f"WHERE LOWER({col}) = ANY(:codes) "
                        f"GROUP BY LOWER({col})"
                    ),
                    {"codes": from_lowers},
                ).fetchall()
                counts_by_raw = {row[0]: row[1] for row in count_rows}

                section_global_total = sum(counts_by_raw.values())
                if section_global_total > 0:
                    _print_header()
                    for path, from_code, canonical in sorted(deduped, key=lambda x: (x[2], x[1])):
                        count = counts_by_raw.get(from_code.lower(), 0)
                        if count == 0:
                            continue
                        marker = "[DRY]" if args.dry_run else "[UPD]"
                        print(f"  {marker} {from_code!r:30s} → {canonical!r:20s}  "
                              f"({count} row{'s' if count != 1 else ''})  [{path}]")

                    verb = "would be updated" if args.dry_run else "updated"
                    print(f"\n  → {section} (global): {section_global_total} row(s) {verb}")
                    grand_total += section_global_total

                    if not args.dry_run:
                        value_clauses = ", ".join(
                            f"(:v{i}_from, :v{i}_to)" for i in range(len(deduped))
                        )
                        params: dict = {}
                        for i, (_, fc, canon) in enumerate(deduped):
                            params[f"v{i}_from"] = fc.lower()
                            params[f"v{i}_to"]   = canon
                        conn.execute(
                            text(
                                f"UPDATE {table} "
                                f"SET {col} = m.canonical "
                                f"FROM (VALUES {value_clauses}) AS m(raw, canonical) "
                                f"WHERE LOWER({col}) = m.raw"
                            ),
                            params,
                        )

            # ── Dataset-scoped rewrites ───────────────────────────────────────
            for path, from_code, canonical, datasets in scoped_rewrites:
                count_row = conn.execute(
                    text(
                        f"SELECT COUNT(*) FROM {table} t "
                        f"JOIN bids_datasets d ON d.id = t.dataset_id "
                        f"WHERE LOWER(t.{col}) = LOWER(:fc) "
                        f"  AND d.accession_id = ANY(:ids)"
                    ),
                    {"fc": from_code, "ids": datasets},
                ).scalar()
                if not count_row:
                    continue

                _print_header()
                marker = "[DRY]" if args.dry_run else "[UPD]"
                ds_str = ", ".join(datasets[:5]) + ("…" if len(datasets) > 5 else "")
                print(f"  {marker} {from_code!r:30s} → {canonical!r:20s}  "
                      f"({count_row} row{'s' if count_row != 1 else ''})  "
                      f"[{path}]  datasets=[{ds_str}]")
                grand_total += count_row

                if not args.dry_run:
                    conn.execute(
                        text(
                            f"UPDATE {table} t "
                            f"SET {col} = :canonical "
                            f"FROM bids_datasets d "
                            f"WHERE t.dataset_id = d.id "
                            f"  AND LOWER(t.{col}) = LOWER(:fc) "
                            f"  AND d.accession_id = ANY(:ids)"
                        ),
                        {"canonical": canonical, "fc": from_code, "ids": datasets},
                    )

            if printed_header:
                verb = "would be updated" if args.dry_run else "updated"
                print(f"  → {section}: rows {verb} (global + scoped above)")

    if grand_total == 0:
        print("No rows to update — either no canonical_code entries are set in the YAML "
              "or all DB values already match their canonical codes.")
    elif args.dry_run:
        print(f"\nDry run complete.  {grand_total} row(s) would be updated.")
        print("Re-run without --dry-run to apply changes.")
    else:
        print(f"\nDone.  {grand_total} row(s) updated.")


if __name__ == "__main__":
    main()
