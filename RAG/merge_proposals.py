"""
RAG/merge_proposals.py
----------------------
Atomically apply synonym proposals from expand_synonyms.py into the flat
SKOS value_mappings.yaml.

Flat structure means each concept is a top-level key within its category,
so navigation is O(1): schema[category][standard_code].

Modes:
  --mode replace (default): overwrite each concept's synonyms with proposed_synonyms
  --mode augment           : append proposed_additions, skip duplicates

Atomic write: temp file → os.replace(), so a crash cannot corrupt the YAML.

Usage:
    python RAG/merge_proposals.py \\
        --proposals RAG/proposals_batch1.yaml \\
        --yaml      RAG/value_mappings.yaml \\
        --mode      replace

    # Dry-run — show what would change without writing
    python RAG/merge_proposals.py --proposals RAG/proposals_batch1.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_CATEGORIES = [
    "diagnosis", "task", "suffix", "handedness", "sex",
    "datatype", "sidecar_fields", "participant_extra_fields",
]


def _find_concept(
    schema: dict,
    std_code: str,
) -> Optional[dict]:
    """Return the concept dict for std_code, or None if not found.

    Searches all categories. In the flat SKOS structure each concept is a
    top-level key within its category equal to its standard_code, so lookup
    is O(categories) ≈ O(1).
    """
    for cat in _CATEGORIES:
        cat_data = schema.get(cat, {})
        if std_code in cat_data and isinstance(cat_data[std_code], dict):
            return cat_data[std_code]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply synonym proposals into flat SKOS value_mappings.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--proposals", required=True,
                        help="Proposals YAML from expand_synonyms.py")
    parser.add_argument("--yaml",      default="RAG/value_mappings.yaml",
                        help="Path to value_mappings.yaml")
    parser.add_argument("--mode",      default="replace", choices=["replace", "augment"],
                        help="replace: overwrite synonyms; augment: append new terms")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show changes without writing to disk")
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    prop_path = Path(args.proposals)

    if not yaml_path.exists():
        sys.exit(f"Error: {yaml_path} not found")
    if not prop_path.exists():
        sys.exit(f"Error: {prop_path} not found")

    with open(yaml_path, encoding="utf-8") as fh:
        schema = yaml.safe_load(fh)

    with open(prop_path, encoding="utf-8") as fh:
        proposals = yaml.safe_load(fh)

    if not proposals:
        print("No proposals to apply.")
        return

    applied = skipped = 0

    for proposal in proposals:
        std_code = proposal.get("standard_code")
        if not std_code:
            print(f"  [warn] proposal missing standard_code: {proposal.get('path')}")
            continue

        node = _find_concept(schema, std_code)
        if node is None:
            print(f"  [warn] standard_code not found in YAML: {std_code}")
            skipped += 1
            continue

        # Resolve synonym list from proposal (supports both output key names)
        new_terms_raw: List[Any] = (
            proposal.get("proposed_synonyms") or
            proposal.get("proposed_additions") or
            []
        )
        if not new_terms_raw:
            skipped += 1
            continue

        # Normalise to [{term, weight}] dicts
        new_terms: List[dict] = []
        for entry in new_terms_raw:
            if isinstance(entry, str):
                new_terms.append({"term": entry, "weight": 1.0})
            elif isinstance(entry, dict) and "term" in entry:
                new_terms.append({"term": entry["term"], "weight": float(entry.get("weight", 1.0))})

        if args.mode == "replace":
            final = new_terms
        else:
            # augment: keep existing, append only non-duplicate terms
            existing = node.get("synonyms") or []
            existing_lower: set = set()
            for s in existing:
                if isinstance(s, str):
                    existing_lower.add(s.lower())
                elif isinstance(s, dict) and "term" in s:
                    existing_lower.add(s["term"].lower())
            additions = [t for t in new_terms if t["term"].lower() not in existing_lower]
            final = list(existing) + additions

        # Format: plain string when weight ≥ 1.0, weighted dict otherwise
        formatted: List[Any] = []
        for t in final:
            if isinstance(t, dict):
                if t.get("weight", 1.0) >= 1.0:
                    formatted.append(t["term"])
                else:
                    formatted.append({"term": t["term"], "weight": round(t["weight"], 2)})
            else:
                formatted.append(str(t))

        if args.dry_run:
            old_syns = node.get("synonyms", [])
            old_str  = str(old_syns[:3]) + ("..." if len(old_syns) > 3 else "")
            new_str  = str(formatted[:3]) + ("..." if len(formatted) > 3 else "")
            print(f"  {std_code}: {old_str} → {new_str}")
        else:
            node["synonyms"] = formatted

        applied += 1

    print(f"\nApplied: {applied}  Skipped: {skipped}")

    if args.dry_run:
        print("(dry-run — no file written)")
        return

    # Atomic write
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=yaml_path.parent, prefix=".merge_tmp_", suffix=".yaml"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(
                "# value_mappings.yaml — SKOS flat format\n"
                "# Migrated from nested-tree format by RAG/migrate_to_skos.py\n"
                "#\n"
                "# Structure:\n"
                "#   Each category (task, diagnosis, ...) contains a flat dict of concepts.\n"
                "#   Leaf concepts have 'standard_code' (maps to DB column values).\n"
                "#   Group concepts have 'is_group: true' (used for retrieval grouping only).\n"
                "#   'broader' lists immediate parents; add multiple entries for polyhierarchy.\n"
                "#\n"
                "# To add polyhierarchy: append additional keys to a concept's 'broader' list.\n\n"
            )
            yaml.dump(
                schema, fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        os.replace(tmp_path, yaml_path)
        print(f"Written: {yaml_path}")
    except Exception as exc:
        os.unlink(tmp_path)
        sys.exit(f"Error writing file: {exc}")


if __name__ == "__main__":
    main()
