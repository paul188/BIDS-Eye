#!/usr/bin/env python3
"""
find_unmapped_db_values.py
---------------------------
Scans every distinct value stored in the database for each mapped field
and identifies values not covered by any 'codes' entry in value_mappings.yaml.

For every unknown value it records:
  - the raw value as it appears in the DB
  - how many rows contain it
  - which datasets (accession_id) it appears in

Also writes a dataset_context.json file mapping accession_id → {name, description}
for every dataset that uses an unmapped value.  auto_map_unmapped.py uses this
to give Gemini context about what a tag means in each dataset.

Usage:
  python training_data_generation/find_unmapped_db_values.py \\
      --db-url postgresql://user:password@localhost:5429/bids_sql \\
      --out    training_data_generation/unmapped_db_values.yaml
      [--dataset-context training_data_generation/dataset_context.json]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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

_LEAF_KEYS = {"label", "description", "codes", "synonyms", "canonical_code"}


def all_codes(d: dict) -> set[str]:
    """Return a lowercase set of every code value found anywhere in a YAML section."""
    codes: set[str] = set()

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if "codes" in node:
            for c in node["codes"]:
                codes.add(str(c).lower())
        for key, val in node.items():
            if key not in _LEAF_KEYS and isinstance(val, dict):
                walk(val)

    walk(d)
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-url",
                        default="postgresql://user:password@localhost:5429/bids_sql",
                        help="SQLAlchemy DB URL")
    parser.add_argument("--out",
                        default=str(Path(__file__).with_name("unmapped_db_values.yaml")),
                        help="Output YAML file path")
    parser.add_argument("--dataset-context",
                        default=str(Path(__file__).with_name("dataset_context.json")),
                        help="Output JSON file: accession_id → {name, description} "
                             "for every dataset that uses an unmapped value")
    parser.add_argument("--max-datasets", type=int, default=20,
                        help="Max dataset accession IDs to list per value (default 20)")
    args = parser.parse_args()

    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    engine = create_engine(args.db_url)
    output: dict[str, dict] = {}

    with engine.connect() as conn:
        for section, (table, col) in FIELD_MAP.items():
            known = all_codes(data.get(section, {}))

            # All distinct values with occurrence counts, most common first
            rows = conn.execute(text(f"""
                SELECT {col}, COUNT(*) AS cnt
                FROM {table}
                WHERE {col} IS NOT NULL AND {col} != ''
                GROUP BY {col}
                ORDER BY cnt DESC
            """)).fetchall()

            section_out: dict = {}
            for value, count in rows:
                if str(value).lower() in known:
                    continue

                # Unmapped — look up which datasets contain this value
                ds_rows = conn.execute(text(f"""
                    SELECT DISTINCT d.accession_id
                    FROM {table} t
                    JOIN bids_datasets d ON d.id = t.dataset_id
                    WHERE LOWER(t.{col}) = LOWER(:val)
                      AND d.accession_id IS NOT NULL
                    ORDER BY d.accession_id
                    LIMIT :lim
                """), {"val": value, "lim": args.max_datasets}).fetchall()

                datasets = [r[0] for r in ds_rows]
                section_out[str(value)] = {
                    "count": int(count),
                    "datasets": datasets,
                }

            if section_out:
                output[section] = section_out
                print(f"  {section}: {len(section_out)} unmapped value(s)")
            else:
                print(f"  {section}: fully covered ✓")

    if not output:
        print("\nAll DB values are covered by value_mappings.yaml — nothing to report.")
        return

    # ── Fetch dataset context (name + description) for every referenced dataset ─
    all_accession_ids: list[str] = []
    for section_data in output.values():
        for info in section_data.values():
            all_accession_ids.extend(info.get("datasets", []))
    unique_ids = list(set(all_accession_ids))

    dataset_context: dict[str, dict] = {}
    if unique_ids:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT accession_id, name, description_text "
                    "FROM bids_datasets "
                    "WHERE accession_id = ANY(:ids)"
                ),
                {"ids": unique_ids},
            ).fetchall()
        for accession_id, name, description_text in rows:
            dataset_context[accession_id] = {
                "name": name or "",
                "description": description_text or "",
            }

    context_path = Path(args.dataset_context)
    context_path.write_text(
        json.dumps(dataset_context, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote dataset context for {len(dataset_context)} dataset(s) to:\n  {context_path}")

    out_path = Path(args.out)
    header_lines = [
        "# Auto-generated by find_unmapped_db_values.py",
        f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "#",
        "# These DB values are not covered by any 'codes' entry in value_mappings.yaml.",
        "# Review each value and either:",
        "#   a) Add it to the appropriate 'codes' list in value_mappings.yaml",
        "#   b) Add a new leaf entry for it if it represents a new concept",
        "#   c) Leave it as-is if it should be ignored / filtered",
        "#",
        "# After editing value_mappings.yaml, re-run this script to verify full coverage.",
        "# Then add 'canonical_code' to relevant entries and run apply_canonical_codes.py.",
        "#",
        "# Format:",
        "#   <section>:",
        "#     <raw_db_value>:",
        "#       count: <number of rows in DB with this value>",
        "#       datasets: [<accession_id>, ...]   # up to --max-datasets examples",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=True)

    total = sum(len(v) for v in output.values())
    print(f"\nWrote {total} unmapped values across {len(output)} field(s) to:\n  {out_path}")


if __name__ == "__main__":
    main()
