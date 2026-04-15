"""audit_yaml.py

Find every node in value_mappings.yaml — whether a group or a leaf — that has
neither a 'label' nor a 'synonyms' entry.  These nodes are invisible to the
fuzzy-matching layer of the RAG and should be enriched.

Usage:
    python audit_yaml.py [path/to/value_mappings.yaml]

Output:
    A table listing each missing node, its YAML path, and whether it is a
    leaf (has standard_code) or a group node.
"""

import sys
import yaml
from typing import Any, List, Tuple

# Keys that are metadata fields *within* a node, not child node identifiers.
_METADATA_KEYS = {
    "label", "standard_code", "description",
    "synonyms", "codes", "extra_codes", "dataset_codes",
}

_CATEGORIES = [
    "diagnosis", "task", "suffix", "handedness", "sex",
    "datatype", "sidecar_fields", "participant_extra_fields",
]


def _audit(
    data: Any,
    path: List[str],
    results: List[Tuple[str, str, str]],  # (dot_path, node_type, reason)
) -> None:
    """Recursively walk the YAML, collecting nodes that lack label and synonyms."""
    if not isinstance(data, dict):
        return

    for key, value in data.items():
        if key in _METADATA_KEYS:
            continue
        if not isinstance(value, dict):
            continue

        current_path = path + [key]
        dot_path = " > ".join(current_path)

        has_label = bool(value.get("label", ""))
        has_synonyms = bool(value.get("synonyms"))

        if not has_label and not has_synonyms:
            node_type = "leaf" if "standard_code" in value else "group"
            missing = []
            if not has_label:
                missing.append("label")
            if not has_synonyms:
                missing.append("synonyms")
            results.append((dot_path, node_type, ", ".join(missing)))

        # Recurse regardless — children of a passing node may still be missing
        _audit(value, current_path, results)


def audit(yaml_path: str) -> List[Tuple[str, str, str]]:
    with open(yaml_path, "r") as fh:
        schema = yaml.safe_load(fh)

    results: List[Tuple[str, str, str]] = []
    for cat in _CATEGORIES:
        if cat in schema:
            _audit(schema[cat], [cat], results)

    return results


def _print_report(results: List[Tuple[str, str, str]]) -> None:
    if not results:
        print("All nodes have at least a label or synonyms.")
        return

    # Column widths
    path_w = max(len(r[0]) for r in results)
    type_w = max(len(r[1]) for r in results)

    header = f"{'PATH':<{path_w}}  {'TYPE':<{type_w}}  MISSING"
    print(header)
    print("-" * len(header))

    # Group by category for readability
    current_cat = None
    for dot_path, node_type, missing in sorted(results):
        cat = dot_path.split(" > ")[0]
        if cat != current_cat:
            print()
            current_cat = cat
        print(f"{dot_path:<{path_w}}  {node_type:<{type_w}}  {missing}")

    print(f"\nTotal: {len(results)} node(s) missing label and/or synonyms.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "value_mappings.yaml"
    results = audit(path)
    _print_report(results)
