"""audit_yaml.py

Find every concept in the flat SKOS value_mappings.yaml that has neither a
'label' nor a 'synonyms' entry.  These concepts are invisible to the
fuzzy-matching layer of the RAG and should be enriched.

Usage:
    python audit_yaml.py [path/to/value_mappings.yaml]

Output:
    A table listing each missing concept, its category, key, and type
    (leaf = has standard_code; group = is_group: true).
"""

import sys
import yaml
from typing import Dict, List, Tuple

_CATEGORIES = [
    "diagnosis", "task", "suffix", "handedness", "sex",
    "datatype", "sidecar_fields", "participant_extra_fields",
]


def audit(yaml_path: str) -> List[Tuple[str, str, str, str]]:
    """Return list of (category, key, node_type, missing_fields) tuples."""
    with open(yaml_path, "r") as fh:
        schema = yaml.safe_load(fh)

    results: List[Tuple[str, str, str, str]] = []

    for cat in _CATEGORIES:
        cat_data = schema.get(cat)
        if not isinstance(cat_data, dict):
            continue

        for key, value in cat_data.items():
            if not isinstance(value, dict):
                continue

            has_label    = bool(value.get("label", ""))
            has_synonyms = bool(value.get("synonyms"))

            if not has_label and not has_synonyms:
                node_type = "group" if value.get("is_group") else "leaf"
                missing: List[str] = []
                if not has_label:
                    missing.append("label")
                if not has_synonyms:
                    missing.append("synonyms")
                results.append((cat, key, node_type, ", ".join(missing)))

    return results


def get_stats(yaml_path: str) -> Dict[str, int]:
    """Return concept count statistics per category."""
    with open(yaml_path, "r") as fh:
        schema = yaml.safe_load(fh)

    stats: Dict[str, int] = {}
    total = 0
    for cat in _CATEGORIES:
        cat_data = schema.get(cat)
        if isinstance(cat_data, dict):
            n = sum(1 for v in cat_data.values() if isinstance(v, dict))
            stats[cat] = n
            total += n
    stats["total"] = total
    return stats


def _print_report(results: List[Tuple[str, str, str, str]]) -> None:
    if not results:
        print("All concepts have at least a label or synonyms.")
        return

    cat_w  = max(len(r[0]) for r in results)
    key_w  = max(len(r[1]) for r in results)
    type_w = max(len(r[2]) for r in results)

    header = f"{'CATEGORY':<{cat_w}}  {'KEY':<{key_w}}  {'TYPE':<{type_w}}  MISSING"
    print(header)
    print("-" * len(header))

    current_cat = None
    for cat, key, node_type, missing in sorted(results):
        if cat != current_cat:
            print()
            current_cat = cat
        print(f"{cat:<{cat_w}}  {key:<{key_w}}  {node_type:<{type_w}}  {missing}")

    print(f"\nTotal: {len(results)} concept(s) missing label and/or synonyms.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "value_mappings.yaml"
    stats = get_stats(path)
    print(f"Concept counts: {stats}\n")
    results = audit(path)
    _print_report(results)
