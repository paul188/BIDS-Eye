from pathlib import Path

import yaml
from typing import Any, Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_leaf_codes(data: Any) -> List[str]:
    """Recursively collect all standard_codes from leaf nodes under a dict.

    A node is a *leaf* if it contains a 'standard_code' key.  Group nodes
    (no 'standard_code') are traversed to collect their children.
    """
    codes: List[str] = []
    if not isinstance(data, dict):
        return codes

    # If this specific level has a standard_code, collect it
    if "standard_code" in data:
        codes.append(data["standard_code"])
        # DO NOT return here! We must keep checking for children below this node.

    # Reserved keys that are metadata, not children
    reserved_keys = {"label", "description", "standard_code", "synonyms", "codes", "dataset_codes"}

    for key, value in data.items():
        if key not in reserved_keys and isinstance(value, dict):
            codes.extend(_collect_leaf_codes(value))

    # Deduplicate before returning
    return list(dict.fromkeys(codes))


def _parse_node(
    category: str,
    data: dict,
    path: List[str],
    leaf_db: Dict[str, Dict[str, str]],
    display_db: Dict[str, Dict[str, str]],
    group_db: Dict[str, Dict[str, List[str]]],
    group_display_db: Dict[str, Dict[str, List[str]]],
    hierarchy_db: Dict[str, Dict],
) -> None:
    """Walk one level of the YAML tree, registering leaves and groups.

    Two parallel lookup surfaces per layer:

    leaf_db / group_db
        All surface forms — used for *exact* matching.
        Includes codes, standard_code, and underscore-style keys so that
        programmatic or copy-pasted values are found precisely.

    display_db / group_display_db
        Human-readable forms only (label + synonyms) — used for *fuzzy* matching.
        Keeps the fuzzy search space clean; no underscore identifiers pollute it.
        Nodes with neither label nor synonyms are absent from these dicts
        (the audit script flags them so they can be filled in).
    """
    reserved_keys = {"label", "description", "standard_code", "synonyms", "codes", "dataset_codes"}

    for key, value in data.items():
        if not isinstance(value, dict):
            continue

        current_path = path + [key]
        
        # Safely extract labels and synonyms
        label: str = value.get("label", "")
        raw_syns = value.get("synonyms", [])
        if isinstance(raw_syns, str):
            raw_syns = [raw_syns]  # Fix common YAML typo where users forget the bullet dash
        synonyms: List[str] = [str(s).strip() for s in raw_syns if s]

        # Identify if this node has sub-dictionaries (children)
        children = {k: v for k, v in value.items() if k not in reserved_keys and isinstance(v, dict)}

        # ── 1. Leaf node behavior ──────────────────────────────────────────────────
        # Process leaf behavior if standard_code exists (regardless of whether it has children)
        if "standard_code" in value:
            std_code = value["standard_code"]

            hierarchy_db[std_code] = {
                "path": current_path,
                "label": label or key,
                "description": value.get("description", ""),
            }

            # leaf_db: all surface forms for exact lookup
            all_terms: set = set()
            if label:
                all_terms.add(label.lower())
            all_terms.add(std_code.lower())
            for syn in synonyms:
                all_terms.add(syn.lower())
            for code in value.get("codes", []):
                all_terms.add(str(code).lower())
            for dc in value.get("dataset_codes", []):
                if isinstance(dc, dict) and "raw" in dc:
                    all_terms.add(str(dc["raw"]).lower())
            
            for term in all_terms:
                if term:
                    leaf_db[category][term] = std_code

            # display_db: label + synonyms only, for fuzzy lookup
            for term in ([label] if label else []) + synonyms:
                t = term.lower()
                if t:
                    display_db[category][t] = std_code

        # ── 2. Group node behavior ─────────────────────────────────────────────────
        # Process group behavior if it has children (regardless of whether it has a standard_code)
        if children:
            leaf_codes = []
            if "standard_code" in value:
                leaf_codes.append(value["standard_code"])
            leaf_codes.extend(_collect_leaf_codes(children))
            
            # Deduplicate
            leaf_codes = list(dict.fromkeys(leaf_codes))

            if leaf_codes:
                # group_db: key name + label + synonyms, for exact lookup
                for term in [key] + ([label] if label else []) + synonyms:
                    t = str(term).lower()
                    if t:
                        existing = group_db[category].get(t, [])
                        group_db[category][t] = list(dict.fromkeys(existing + leaf_codes))

                # group_display_db: label + synonyms only, for fuzzy lookup
                # (Added label here! Previously only synonyms were searchable in groups)
                for term in ([label] if label else []) + synonyms:
                    t = str(term).lower()
                    if t:
                        existing = group_display_db[category].get(t, [])
                        group_display_db[category][t] = list(dict.fromkeys(existing + leaf_codes))

            # Recurse into children
            _parse_node(
                category, children, current_path,
                leaf_db, display_db, group_db, group_display_db, hierarchy_db,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "diagnosis", "task", "suffix", "handedness", "sex",
    "datatype", "sidecar_fields", "participant_extra_fields",
]


def build_knowledge_base(yaml_path: str) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict],
]:
    """Parse the value-mappings YAML into five complementary lookup structures.

    Returns
    -------
    leaf_db : {category: {surface_form_lower: standard_code}}
        Full exact-match lookup for leaf nodes.
        Covers label, standard_code, synonyms, codes, and dataset_codes.

    display_db : {category: {human_readable_lower: standard_code}}
        Fuzzy-match lookup for leaf nodes.
        Contains *only* label + synonyms — no underscore identifiers.
        Nodes with neither label nor synonyms are absent (see audit_yaml.py).

    group_db : {category: {term_lower: [standard_code, ...]}}
        Full exact-match lookup for group nodes.
        Keys are the YAML group key name + all group synonyms.

    group_display_db : {category: {synonym_lower: [standard_code, ...]}}
        Fuzzy-match lookup for group nodes.
        Contains *only* group synonyms — no underscore key names.

    hierarchy_db : {standard_code: {path, label, description}}
        Ancestry path and metadata for every leaf node.
    """
    with open(yaml_path, "r") as fh:
        schema = yaml.safe_load(fh)

    leaf_db: Dict[str, Dict[str, str]] = {cat: {} for cat in _CATEGORIES}
    display_db: Dict[str, Dict[str, str]] = {cat: {} for cat in _CATEGORIES}
    group_db: Dict[str, Dict[str, List[str]]] = {cat: {} for cat in _CATEGORIES}
    group_display_db: Dict[str, Dict[str, List[str]]] = {cat: {} for cat in _CATEGORIES}
    hierarchy_db: Dict[str, Dict] = {}

    for cat in _CATEGORIES:
        if cat in schema:
            _parse_node(
                cat, schema[cat], [],
                leaf_db, display_db, group_db, group_display_db, hierarchy_db,
            )

    return leaf_db, display_db, group_db, group_display_db, hierarchy_db


def get_group_summary(group_db: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Return a flat map of {category: [group_synonyms]} for use in LLM prompts.

    Provides the LLM with awareness of which broad/group-level terms exist so
    it can extract them when the user's query is at a category level rather
    than a specific diagnosis/task.
    """
    summary: Dict[str, List[str]] = {}
    for cat, groups in group_db.items():
        if groups:
            summary[cat] = sorted(groups.keys())
    return summary


# ---------------------------------------------------------------------------
# Module-level initialisation (used by downstream modules)
# ---------------------------------------------------------------------------

# Use an absolute path so this module is importable from any working directory.
_YAML_PATH = str(Path(__file__).parent / "value_mappings.yaml")
leaf_db, display_db, group_db, group_display_db, hierarchy_db = build_knowledge_base(_YAML_PATH)