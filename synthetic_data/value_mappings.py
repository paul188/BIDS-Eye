from __future__ import annotations
import yaml
from pathlib import Path
from typing import Optional

# ── Load mappings from YAML ────────────────────────────────────────────────────

_YAML_PATH = Path(__file__).with_name("value_mappings.yaml")

def _load() -> dict:
    if not _YAML_PATH.exists():
        return {}
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _build_reverse_map(nested_dict: dict) -> dict[str, Optional[str]]:
    """
    Inverts the YAML structure.
    Takes a nested dict where leaves have 'codes' (list) and 'label' (str).
    Returns a flat dict: { "raw_code": "Human Label" }
    If a category is mapped to null in YAML, the codes will map to None (filtered).
    """
    flat_map = {}

    def walk(d):
        for key, value in d.items():
            if isinstance(value, dict):
                # Check if this is a leaf node with 'codes'
                if "codes" in value:
                    label = value.get("label")
                    # Map every code in the list to the human-friendly label
                    for code in value["codes"]:
                        flat_map[str(code).lower()] = label
                else:
                    # Keep walking down (for 'healthy_subjects', 'psychiatric', etc.)
                    walk(value)
            elif value is None:
                # Handle explicit filters (e.g., some_task: null)
                flat_map[key.lower()] = None

    walk(nested_dict)
    return flat_map

def _build_concept_map(nested_dict: dict) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Walk the YAML hierarchy and return two structures:

    concept_expansion: {intermediate_node_key → [standard_code, ...]}
        Maps a broad concept (YAML intermediate node key) to the flat list of all
        leaf standard_codes beneath it, recursively.
        e.g. "epilepsy_spectrum" → ["epilepsy", "watanabe_syndrome", "hhe_syndrome", ...]
        e.g. "psychiatric"      → ["schizophrenia", "major_depressive_disorder", ...]

    standard_code_label: {standard_code → human label}   (leaf nodes only)
        e.g. "epilepsy" → "epilepsy"

    Leaf nodes are identified by having both 'codes' and 'standard_code'.
    Intermediate nodes (no 'codes') accumulate all standard_codes from descendants.
    """
    concept_expansion: dict[str, list[str]] = {}
    standard_code_label: dict[str, str] = {}
    _SKIP = {"label", "description", "codes", "synonyms", "standard_code", "extra_codes"}

    def walk(d: dict) -> list[str]:
        subtree: list[str] = []
        for key, val in d.items():
            if not isinstance(val, dict) or key in _SKIP:
                continue
            if "codes" in val and "standard_code" in val:
                # Leaf node
                sc = str(val["standard_code"])
                standard_code_label[sc] = val.get("label", sc)
                subtree.append(sc)
            elif "codes" not in val:
                # Intermediate node — recurse
                children = walk(val)
                if children:
                    concept_expansion[key] = children
                    subtree.extend(children)
            # Nodes with 'codes' but no 'standard_code': raw-code-only, not part of hierarchy
        return subtree

    walk(nested_dict)
    return concept_expansion, standard_code_label


_data = _load()

# Pre-flattened maps for O(1) lookup during pipeline execution
DIAGNOSIS_MAP:    dict[str, Optional[str]] = _build_reverse_map(_data.get("diagnosis", {}))
SEX_LABEL:        dict[str, Optional[str]] = _build_reverse_map(_data.get("sex", {}))
HANDEDNESS_LABEL: dict[str, Optional[str]] = _build_reverse_map(_data.get("handedness", {}))
TASK_LABEL:       dict[str, Optional[str]] = _build_reverse_map(_data.get("task", {}))
SUFFIX_LABEL:     dict[str, Optional[str]] = _build_reverse_map(_data.get("suffix", {}))
DATATYPE_LABEL:   dict[str, Optional[str]] = _build_reverse_map(_data.get("datatype", {}))

# Per-section hierarchy maps derived from the YAML tree.
# CONCEPT_EXPANSION: intermediate node key → flat list of all leaf standard_codes beneath it
# STANDARD_CODE_LABEL: leaf standard_code → human-readable label
CONCEPT_EXPANSION:   dict[str, list[str]]
STANDARD_CODE_LABEL: dict[str, str]
CONCEPT_EXPANSION, STANDARD_CODE_LABEL = _build_concept_map(_data.get("diagnosis", {}))

_TASK_CONCEPT_EXPANSION, _TASK_SC_LABEL = _build_concept_map(_data.get("task", {}))
_SUFFIX_CONCEPT_EXPANSION, _SUFFIX_SC_LABEL = _build_concept_map(_data.get("suffix", {}))
_DATATYPE_CONCEPT_EXPANSION, _DATATYPE_SC_LABEL = _build_concept_map(_data.get("datatype", {}))

# Combined lookup keyed by DB column name — used by sql_expander.py
FIELD_CONCEPT_EXPANSION: dict[str, dict[str, list[str]]] = {
    "diagnosis": CONCEPT_EXPANSION,
    "task":      _TASK_CONCEPT_EXPANSION,
    "suffix":    _SUFFIX_CONCEPT_EXPANSION,
    "datatype":  _DATATYPE_CONCEPT_EXPANSION,
}
FIELD_STANDARD_CODE_LABEL: dict[str, dict[str, str]] = {
    "diagnosis": STANDARD_CODE_LABEL,
    "task":      _TASK_SC_LABEL,
    "suffix":    _SUFFIX_SC_LABEL,
    "datatype":  _DATATYPE_SC_LABEL,
}


def _build_synonyms_map(nested_dict: dict) -> dict[str, list[str]]:
    """
    Return {standard_code_or_concept_key: [synonyms]} for a YAML section.

    Collects synonyms from:
      - Leaf nodes (standard_code entries): indexed by standard_code
      - Group/concept nodes (intermediate nodes with no standard_code):
        indexed by the YAML key (which is the concept key used in SQL)
    """
    result: dict[str, list[str]] = {}
    _SKIP = {"label", "description", "codes", "synonyms", "standard_code",
             "extra_codes", "dataset_codes"}

    def walk(d: dict) -> None:
        for key, val in d.items():
            if not isinstance(val, dict) or key in _SKIP:
                continue
            syns = [str(s) for s in val.get("synonyms", []) if s]
            if "standard_code" in val:
                # Leaf node — index by standard_code
                sc = str(val["standard_code"])
                if syns:
                    result[sc] = syns
            elif "codes" not in val:
                # Group/concept node — index by YAML key (the concept key)
                if syns:
                    result[key] = syns
                walk(val)

    walk(nested_dict)
    return result


# Synonyms for both leaf standard_codes and group concept keys, per field.
# Used by sample_diverse_prompts.py to generate natural-language question phrasings.
FIELD_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "diagnosis": _build_synonyms_map(_data.get("diagnosis", {})),
    "task":      _build_synonyms_map(_data.get("task", {})),
    "suffix":    _build_synonyms_map(_data.get("suffix", {})),
    "datatype":  _build_synonyms_map(_data.get("datatype", {})),
}

# Reverse lookup: human-readable label (lowercase) → standard_code, per field.
# Used to recover standard_codes from the label strings returned by clean_*() functions.
FIELD_LABEL_TO_CODE: dict[str, dict[str, str]] = {
    field: {label.lower(): sc for sc, label in sc_labels.items()}
    for field, sc_labels in FIELD_STANDARD_CODE_LABEL.items()
}

# Structured field catalogs: {field_key: {"label": str, "synonyms": [...], "codes": [...]}}
# Used by sample_diverse_prompts.py to generate natural-language questions without
# quoting raw field names.
def _build_field_catalog(section: dict) -> dict[str, dict]:
    """Return {entry_key: {label, synonyms, codes, description}} for a flat or nested section."""
    catalog: dict[str, dict] = {}
    def walk(d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                if "codes" in value:
                    catalog[key] = {
                        "label": value.get("label", key),
                        "synonyms": value.get("synonyms", []),
                        "codes": value.get("codes", [key]),
                        "description": value.get("description", ""),
                    }
                else:
                    walk(value)
    walk(section)
    return catalog

SIDECAR_FIELDS:          dict[str, dict] = _build_field_catalog(_data.get("sidecar_fields", {}))
PARTICIPANT_EXTRA_FIELDS: dict[str, dict] = _build_field_catalog(_data.get("participant_extra_fields", {}))


def expand_concept(concept: str, field: str = "diagnosis") -> list[str]:
    """
    Given a concept key and a DB field name, return all standard_codes it encompasses.

    - Intermediate node key (e.g. "epilepsy_spectrum", "resting_state") → all leaf standard_codes
    - Leaf standard_code (e.g. "epilepsy", "resting_state_general")     → [concept] itself
    - Unknown key                                                         → []

    field: one of "diagnosis", "task", "suffix", "datatype"
    """
    exp = FIELD_CONCEPT_EXPANSION.get(field, {})
    lbl = FIELD_STANDARD_CODE_LABEL.get(field, {})
    if concept in exp:
        return exp[concept]
    if concept in lbl:
        return [concept]
    return []


# ── Lookup helpers ─────────────────────────────────────────────────────────────

def clean_diagnosis(raw: str) -> Optional[str]:
    """Return human-readable diagnosis label, or None if raw value is unusable/filtered."""
    if not raw or "," in raw:
        return None
    
    stripped = raw.strip().lower()
    
    # Validation: ignore purely numeric codes or single chars if not in map
    if stripped not in DIAGNOSIS_MAP:
        if stripped.lstrip("-").replace(".", "").isdigit() or len(stripped) < 2:
            return None
            
    return DIAGNOSIS_MAP.get(stripped, raw.strip()) # Fallback to raw if not in map


def clean_sex(raw: str) -> Optional[str]:
    if not raw: return None
    return SEX_LABEL.get(raw.strip().lower())


def clean_handedness(raw: str) -> Optional[str]:
    if not raw: return None
    return HANDEDNESS_LABEL.get(raw.strip().lower())


def clean_task(raw: str) -> Optional[str]:
    """Returns the mapped label. If value is explicitly null in YAML, returns None."""
    if not raw: return None
    stripped = raw.strip()
    # Try exact then lower
    result = TASK_LABEL.get(stripped, TASK_LABEL.get(stripped.lower()))
    
    # If it's not in our map at all, return the original stripped string
    # If it is in the map but the value is None, it returns None (filtered)
    if result is None and stripped.lower() not in TASK_LABEL:
        return stripped
    return result


def clean_suffix(raw: str) -> Optional[str]:
    if not raw: return None
    stripped = raw.strip()
    return SUFFIX_LABEL.get(stripped, SUFFIX_LABEL.get(stripped.lower()))


def clean_datatype(raw: str) -> Optional[str]:
    if not raw: return None
    stripped = raw.strip()
    return DATATYPE_LABEL.get(stripped, DATATYPE_LABEL.get(stripped.lower(), stripped))