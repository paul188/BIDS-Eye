from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_YAML_PATH = Path(__file__).resolve().parent / "RAG" / "value_mappings.yaml"


def _load() -> dict:
    if not _YAML_PATH.exists():
        return {}
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_reverse_map(nested_dict: dict) -> dict[str, Optional[str]]:
    flat_map: dict[str, Optional[str]] = {}

    def walk(d):
        for key, value in d.items():
            if isinstance(value, dict):
                if "codes" in value:
                    label = value.get("label")
                    for code in value["codes"]:
                        flat_map[str(code).lower()] = label
                else:
                    walk(value)
            elif value is None:
                flat_map[key.lower()] = None

    walk(nested_dict)
    return flat_map


def _build_concept_map(nested_dict: dict) -> tuple[dict[str, list[str]], dict[str, str]]:
    concept_expansion: dict[str, list[str]] = {}
    standard_code_label: dict[str, str] = {}
    _SKIP = {"label", "description", "codes", "synonyms", "standard_code", "extra_codes"}

    def walk(d: dict) -> list[str]:
        subtree: list[str] = []
        for key, val in d.items():
            if not isinstance(val, dict) or key in _SKIP:
                continue
            if "codes" in val and "standard_code" in val:
                sc = str(val["standard_code"])
                standard_code_label[sc] = val.get("label", sc)
                subtree.append(sc)
            elif "codes" not in val:
                children = walk(val)
                if children:
                    concept_expansion[key] = children
                    subtree.extend(children)
        return subtree

    walk(nested_dict)
    return concept_expansion, standard_code_label


_data = _load()

DIAGNOSIS_MAP: dict[str, Optional[str]] = _build_reverse_map(_data.get("diagnosis", {}))
SEX_LABEL: dict[str, Optional[str]] = _build_reverse_map(_data.get("sex", {}))
HANDEDNESS_LABEL: dict[str, Optional[str]] = _build_reverse_map(_data.get("handedness", {}))
TASK_LABEL: dict[str, Optional[str]] = _build_reverse_map(_data.get("task", {}))
SUFFIX_LABEL: dict[str, Optional[str]] = _build_reverse_map(_data.get("suffix", {}))
DATATYPE_LABEL: dict[str, Optional[str]] = _build_reverse_map(_data.get("datatype", {}))

CONCEPT_EXPANSION: dict[str, list[str]]
STANDARD_CODE_LABEL: dict[str, str]
CONCEPT_EXPANSION, STANDARD_CODE_LABEL = _build_concept_map(_data.get("diagnosis", {}))

_TASK_CONCEPT_EXPANSION, _TASK_SC_LABEL = _build_concept_map(_data.get("task", {}))
_SUFFIX_CONCEPT_EXPANSION, _SUFFIX_SC_LABEL = _build_concept_map(_data.get("suffix", {}))
_DATATYPE_CONCEPT_EXPANSION, _DATATYPE_SC_LABEL = _build_concept_map(_data.get("datatype", {}))

FIELD_CONCEPT_EXPANSION: dict[str, dict[str, list[str]]] = {
    "diagnosis": CONCEPT_EXPANSION,
    "task": _TASK_CONCEPT_EXPANSION,
    "suffix": _SUFFIX_CONCEPT_EXPANSION,
    "datatype": _DATATYPE_CONCEPT_EXPANSION,
}

FIELD_STANDARD_CODE_LABEL: dict[str, dict[str, str]] = {
    "diagnosis": STANDARD_CODE_LABEL,
    "task": _TASK_SC_LABEL,
    "suffix": _SUFFIX_SC_LABEL,
    "datatype": _DATATYPE_SC_LABEL,
}
