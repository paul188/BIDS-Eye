"""
RAG/migrate_to_skos.py
----------------------
One-time migration: converts the nested-tree value_mappings.yaml into the
flat SKOS-inspired structure where every concept is a top-level key within
its category and polyhierarchy is expressed via explicit 'broader' lists.

Flat key assignment:
  - Leaf concepts (have standard_code): flat key = standard_code
  - Pure group nodes (no standard_code): flat key = YAML key
  - Dual nodes (have standard_code AND children): flat key = standard_code
    Their children reference them by standard_code in 'broader'.

'broader' assignment:
  - Single entry derived from immediate tree parent.
  - After migration, multiple entries can be added manually or by other tools.

Usage:
    python RAG/migrate_to_skos.py
    # Outputs RAG/value_mappings_flat.yaml

    # Inspect, then replace:
    mv RAG/value_mappings.yaml RAG/value_mappings_tree_backup.yaml
    mv RAG/value_mappings_flat.yaml RAG/value_mappings.yaml
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_HERE = Path(__file__).parent
_IN_YAML  = _HERE / "value_mappings.yaml"
_OUT_YAML = _HERE / "value_mappings_flat.yaml"

_METADATA_KEYS = frozenset({
    "label", "standard_code", "description",
    "synonyms", "codes", "extra_codes", "dataset_codes", "count",
})

_CATEGORIES = [
    "diagnosis", "task", "suffix", "handedness", "sex",
    "datatype", "sidecar_fields", "participant_extra_fields",
]

# Desired field order in output YAML for readability
_FIELD_ORDER = [
    "label", "standard_code", "is_group", "broader",
    "description", "synonyms", "codes", "dataset_codes", "count",
]


def _collect_all_standard_codes(data: Any, codes: set) -> None:
    """Pre-scan the nested tree to collect every standard_code value."""
    if not isinstance(data, dict):
        return
    for k, v in data.items():
        if k in _METADATA_KEYS or not isinstance(v, dict):
            continue
        if "standard_code" in v:
            codes.add(v["standard_code"])
        _collect_all_standard_codes(v, codes)


def _flat_key(yaml_key: str, value: dict, all_std_codes: set) -> str:
    """Return the flat concept identifier for a node.

    Leaf/dual nodes → standard_code (always unique after dedup fixes).
    Pure group nodes → YAML key, UNLESS that key equals some leaf's standard_code,
    in which case we append '_group' to avoid collision.
    """
    std = value.get("standard_code")
    if std:
        return std
    # Group node: use YAML key, but avoid collision with any standard_code
    if yaml_key in all_std_codes:
        return yaml_key + "_group"
    return yaml_key


def _ordered_entry(data: dict) -> dict:
    """Return a copy of data with fields in the preferred display order."""
    out: dict = {}
    for field in _FIELD_ORDER:
        if field in data:
            out[field] = data[field]
    # Any remaining fields not in the preferred list
    for k, v in data.items():
        if k not in out:
            out[k] = v
    return out


def _walk(
    data: dict,
    parent_flat_key: Optional[str],
    flat_concepts: Dict[str, dict],
    collision_log: List[str],
    all_std_codes: set,
) -> None:
    """Recursively walk the nested tree and populate flat_concepts."""
    for yaml_key, value in data.items():
        if yaml_key in _METADATA_KEYS or not isinstance(value, dict):
            continue

        this_flat_key = _flat_key(yaml_key, value, all_std_codes)
        std_code = value.get("standard_code")

        # Separate metadata from child nodes
        node_meta: dict = {}
        children: dict = {}
        for k, v in value.items():
            if k in _METADATA_KEYS:
                node_meta[k] = v
            elif isinstance(v, dict):
                children[k] = v
            # non-dict, non-metadata scalars (shouldn't normally exist) are dropped

        # Build the flat entry
        entry: dict = {}
        if node_meta.get("label"):
            entry["label"] = node_meta["label"]
        if std_code:
            entry["standard_code"] = std_code
        else:
            entry["is_group"] = True

        entry["broader"] = [parent_flat_key] if parent_flat_key else []

        if node_meta.get("description"):
            entry["description"] = node_meta["description"]
        if node_meta.get("synonyms"):
            entry["synonyms"] = node_meta["synonyms"]
        if node_meta.get("codes"):
            entry["codes"] = node_meta["codes"]
        if node_meta.get("dataset_codes"):
            entry["dataset_codes"] = node_meta["dataset_codes"]
        if "count" in node_meta:
            entry["count"] = node_meta["count"]
        if node_meta.get("extra_codes"):
            entry["extra_codes"] = node_meta["extra_codes"]

        # Collision handling: keep first occurrence, log duplicates
        if this_flat_key in flat_concepts:
            collision_log.append(
                f"  COLLISION: flat key '{this_flat_key}' already seen "
                f"(yaml_key='{yaml_key}', parent='{parent_flat_key}'). Skipping."
            )
        else:
            flat_concepts[this_flat_key] = _ordered_entry(entry)

        # Recurse into children, with this node as their parent
        if children:
            _walk(children, this_flat_key, flat_concepts, collision_log, all_std_codes)


def migrate_category(cat_data: dict) -> Tuple[dict, List[str]]:
    """Convert one category's nested tree to a flat SKOS dict."""
    flat: dict = {}
    collisions: List[str] = []
    # Pre-scan to collect all standard_codes in this category (for collision-safe group keys)
    all_std_codes: set = set()
    _collect_all_standard_codes(cat_data, all_std_codes)
    _walk(cat_data, parent_flat_key=None, flat_concepts=flat,
          collision_log=collisions, all_std_codes=all_std_codes)
    return flat, collisions


def main() -> None:
    print(f"Reading  : {_IN_YAML}")
    with open(_IN_YAML, encoding="utf-8") as fh:
        schema = yaml.safe_load(fh)

    out_schema: dict = {}
    total_nodes = 0
    total_collisions = 0

    for cat in _CATEGORIES:
        if cat not in schema:
            continue

        flat, collisions = migrate_category(schema[cat])
        out_schema[cat] = flat

        n = len(flat)
        total_nodes += n
        total_collisions += len(collisions)

        print(f"  {cat:30s}: {n:4d} concepts", end="")
        if collisions:
            print(f"  [{len(collisions)} collision(s)]")
            for msg in collisions:
                print(msg)
        else:
            print()

    print(f"\nTotal    : {total_nodes} concepts, {total_collisions} collision(s)")

    # Verify: check that all 'broader' references point to existing keys
    broken_refs: List[str] = []
    for cat, concepts in out_schema.items():
        for key, value in concepts.items():
            for parent_ref in value.get("broader", []):
                if parent_ref not in concepts:
                    broken_refs.append(
                        f"  {cat} > {key}: broader ref '{parent_ref}' not found in flat dict"
                    )

    if broken_refs:
        print(f"\nBroken broader references ({len(broken_refs)}):")
        for msg in broken_refs[:20]:
            print(msg)
        if len(broken_refs) > 20:
            print(f"  ... and {len(broken_refs) - 20} more")
    else:
        print("Broader references: all valid")

    # Write output
    header = (
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

    with open(_OUT_YAML, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.dump(
            out_schema, fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    print(f"\nWrote    : {_OUT_YAML}")
    print("\nNext steps:")
    print("  1. Inspect value_mappings_flat.yaml and spot-check ~20 nodes")
    print("  2. mv RAG/value_mappings.yaml RAG/value_mappings_tree_backup.yaml")
    print("  3. mv RAG/value_mappings_flat.yaml RAG/value_mappings.yaml")
    print("  4. Run: python RAG/yaml_to_llamaindex.py  (verify no errors)")
    print("  5. Run: python RAG/audit_yaml.py           (verify node count)")


if __name__ == "__main__":
    main()
