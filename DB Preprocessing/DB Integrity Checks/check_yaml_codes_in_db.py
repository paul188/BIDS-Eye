#!/usr/bin/env python3
"""
check_yaml_codes_in_db.py
--------------------------
Checks which code values listed in value_mappings.yaml are actually
present in the database.  Reports per-field which codes exist in the DB
and which are absent (stale / never-used codes you may want to remove).

Sections checked (yaml section → db table.column):
  diagnosis  → bids_participants.diagnosis
  sex        → bids_participants.sex
  handedness → bids_participants.handedness
  task       → bids_objects.task
  suffix     → bids_objects.suffix
  datatype   → bids_objects.datatype

Usage:
  python training_data_generation/check_yaml_codes_in_db.py \\
      --db-url postgresql://user:password@localhost:5429/bids_sql
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from sqlalchemy import create_engine, text

YAML_PATH = Path(__file__).with_name("value_mappings.yaml")

# yaml section → (db table, db column)
FIELD_MAP = {
    "diagnosis":  ("bids_participants", "diagnosis"),
    "sex":        ("bids_participants", "sex"),
    "handedness": ("bids_participants", "handedness"),
    "task":       ("bids_objects",      "task"),
    "suffix":     ("bids_objects",      "suffix"),
    "datatype":   ("bids_objects",      "datatype"),
}

# Keys that are not dict children to recurse into
_LEAF_KEYS = {"label", "description", "codes", "synonyms", "canonical_code"}


def collect_codes(d: dict) -> list[tuple[str, str]]:
    """
    Walk the YAML dict and yield (dotted_path, code_string) for every
    code found under any 'codes' list.  Recurses into all dict children
    that are not recognised leaf-metadata keys (so subcategories are also
    visited even when their parent already has 'codes').
    """
    results: list[tuple[str, str]] = []

    def walk(node: dict, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "codes" in node:
            for c in node["codes"]:
                results.append((path, str(c)))
        # Keep descending into any non-leaf sub-dicts (e.g. 'subcategories')
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
    args = parser.parse_args()

    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    engine = create_engine(args.db_url)

    with engine.connect() as conn:
        for section, (table, col) in FIELD_MAP.items():
            if section not in data:
                print(f"\n[{section}] — section not in YAML, skipping")
                continue

            # Collect all lowercase DB values for this column
            rows = conn.execute(text(
                f"SELECT DISTINCT LOWER({col}) FROM {table} "
                f"WHERE {col} IS NOT NULL AND {col} != ''"
            )).fetchall()
            db_values: set[str] = {r[0] for r in rows}

            entries = collect_codes(data[section])
            # Deduplicate (same code can appear under multiple paths)
            seen: dict[str, str] = {}
            for path, code in entries:
                seen.setdefault(code.lower(), path)

            present = [(seen[c], c) for c in seen if c in db_values]
            absent  = [(seen[c], c) for c in seen if c not in db_values]

            bar = "=" * 62
            print(f"\n{bar}")
            print(f"  {section.upper()}  |  {len(seen)} unique YAML codes  |  "
                  f"{len(db_values)} distinct DB values")
            print(f"  Present in DB: {len(present)}   "
                  f"Absent from DB: {len(absent)}")
            print(bar)

            if absent:
                print(f"\n  ABSENT from DB (stale / never-used codes):")
                for path, code in sorted(absent, key=lambda x: x[1]):
                    print(f"    {code!r:35s}  (from {path})")

            if present:
                print(f"\n  PRESENT in DB:")
                for path, code in sorted(present, key=lambda x: x[1]):
                    print(f"    {code!r:35s}  (from {path})")

    print()


if __name__ == "__main__":
    main()
