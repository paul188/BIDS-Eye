"""Which few-shot example families a plan asks for, read from the plan's fields.

The families are the keys of few_shot_examples.json. pick_examples (examples.py)
prefers examples of these families when the SQL model writes a query. They are
derived here instead of asked from the plan model: the plan fields already say
what the question is about, a question can belong to several families at once
(most real requests mix two to four), and the plan model has one field less to
get wrong. Nothing else reads them.
"""
from __future__ import annotations

from typing import Any, Iterable, List

from services.translate.participant_query_compiler import derivatives_condition

# A family that several of the others combine into.
COMBINED = "combined_filter"

# Sidecar keys of common acquisition settings and of the scanner.
_ACQUISITION_KEYS = frozenset({"repetitiontime", "echotime", "flipangle", "multibandaccelerationfactor",
                               "magneticfieldstrength", "slicetiming", "samplingfrequency"})
_SCANNER_KEYS = frozenset({"manufacturer", "manufacturersmodelname", "manufacturermodelname",
                           "stationname", "softwareversions"})
_EXISTENCE_OPERATORS = frozenset({"exists", "not_exists"})


def _get(obj: Any, name: str, default: Any = None) -> Any:
    value = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if value is None else value


def _names_pipeline(terms: Iterable[str]) -> bool:
    return any(derivatives_condition(term) for term in terms if term)


def plan_families(plan: Any) -> List[str]:
    """The example families of *plan*, in a fixed order; COMBINED when there are several."""
    families: List[str] = []

    def add(family: str) -> None:
        if family not in families:
            families.append(family)

    groups = _get(plan, "groups", [])
    diagnosed = [g for g in groups if _get(g, "diagnosis_terms", [])]
    if diagnosed:
        add("concept_query")
    if len(diagnosed) >= 2 or (diagnosed and len(groups) > len(diagnosed)):
        add("clinical_design_query")   # patients next to another group, such as controls
    if any(_get(g, "sex", []) or _get(g, "handedness", []) or _get(g, "min_subjects") is not None
           or _get(g, "extra_participant_fields", {}) for g in groups):
        add("participant_filter")
    if _get(plan, "age_groups", []) or any(
            _get(g, "age_min") is not None or _get(g, "age_max") is not None for g in groups):
        add("age_query")
    if _get(plan, "count_comparison"):
        add("comparison_query")

    scans = _get(plan, "scan_requirements", [])
    required = [s for s in scans if _get(s, "required", True)]
    scan_terms = [t for s in scans for t in _get(s, "scan_terms", [])]
    if _names_pipeline(scan_terms + [_get(plan, "description_search", "")]):
        add("derivative_query")
    if required:
        add("scan_filter")
    if len(required) >= 2:
        add("multimodal_query")
    if len(required) < len(scans) or _get(plan, "excluded_diagnosis_terms", []):
        add("absence_query")
    if any(_get(s, "same_subject", False) for s in scans):
        add("subject_multimodal_query")

    if _get(plan, "author_names", []):
        add("author_query")
    if _get(plan, "funding_terms", []):
        add("funding_query")
    if (_get(plan, "institution_terms", []) or _get(plan, "excluded_institution_terms", [])
            or _get(plan, "location_terms", [])):
        add("institution_query")
    if _get(plan, "license_search"):
        add("license_query")
    if _get(plan, "doi_required") is not None:
        add("doi_query")
    description = _get(plan, "description_search", "")
    if (description and not derivatives_condition(description)) or _get(plan, "name_search") \
            or _get(plan, "topics", []):
        # A description that only names a pipeline ("fMRIPrep") is the derivatives lookup.
        add("description_search")

    for metadata in _get(plan, "metadata_filters", []):
        key = str(_get(metadata, "json_field", "")).lower()
        if key in _SCANNER_KEYS:
            add("scanner_query")
        elif key in _ACQUISITION_KEYS:
            add("acquisition_parameter_query")
        elif _get(metadata, "operator") in _EXISTENCE_OPERATORS:
            add("metadata_query")
        else:
            add("json_numeric_query")

    if _get(plan, "min_sessions") is not None or _get(plan, "study_structure") == "longitudinal":
        add("session_query")
    if _get(_get(plan, "ordering"), "sort_key"):
        add("ranking_query")

    if len(families) >= 2:
        add(COMBINED)
    return families
