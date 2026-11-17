"""Deterministic SQL compilation for supported participant-cohort searches.

The language model extracts cohort intent into QueryPlan. RAG resolves each
cohort's diagnosis terms to canonical codes. This module turns that typed data
into SQL without asking a second model to reconstruct the cohort logic.

Supported: one or more diagnosis groups (AND or OR between groups, the
diagnoses within one group are alternatives; "at least N patients with X"
counts the participants who have the diagnosis), optionally with plain scan
requirements ("fMRI datasets with …"). The scan conditions are the ones the
preprocessor already wrote into the augmented question, taken as they are.

Anything else declines with a reason and continues through the existing
SQL-generation path. The check is a whitelist: a plan field this module does
not know must hold its default, so a field added later never compiles by
accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cohort_count import placeholder as patients_placeholder
from text_terms import MODE_SHORT, diagnosis_alternatives, format_text_match


_CODE_RE = re.compile(r"^[a-z0-9_]+$")

# Plan fields the compiler reads, or that do not change the SQL.
_KNOWN_PLAN_FIELDS = frozenset({
    "query_family", "natural_language_summary",
    "groups", "group_operator", "scan_requirements",
    "order_direction",   # only meaningful with order_by, which must be unset
    "age_groups",        # read from its filter line (_age_conditions)
})

# For plans that are not pydantic models (tests, dicts): the fields that must be empty.
_UNSUPPORTED_PLAN_FIELDS = (
    "topics",
    "author_names",
    "name_search",
    "description_search",
    "license_search",
    "doi_required",
    "funding_terms",
    "institution_terms",
    "excluded_institution_terms",
    "metadata_filters",
    "max_size_gb",
    "min_size_gb",
    "min_sessions",
    "study_structure",
    "species",
    "count_comparison",
    "result_limit",
    "order_by",
    "rag_requests",
)

_GROUP_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
    ("task_terms", []), ("sex", []), ("handedness", []), ("age_min", None),
    ("age_max", None), ("extra_participant_fields", {}),
)

_FILTER_HEADER = "[Resolved DB filters"
# A plain scan line, or a list of alternatives that preprocess.py already
# joined with OR into one parenthesised condition ("scan any of ...").
_SCAN_LINE_RE = re.compile(
    r'^  scan(?: task)? "[^"]*": (?P<condition>.+)$'
    r'|^  scan any of "[^"]*"(?:, "[^"]*")+ \(alternatives, one is enough\): (?P<any_of>\(.+\))$'
)
# The age-group line (RAG/criteria.py age_condition), copied as it is.
_AGE_LINE_RE = re.compile(
    r"^  age group [a-z ]+ \(complete, no other age condition\): (?P<condition>\(.+\))$")
# Lines the compiler builds from the typed plan itself: the diagnosis of a
# group and, with a minimum, the count of its patients.
_DIAGNOSIS_LINE_RE = re.compile(r"^  (?:diagnosis\b[^:]*|patients \([^)]*\)): ")


@dataclass(frozen=True)
class Compiled:
    """The compiled SQL, or why this plan goes to the SQL model instead."""
    sql: Optional[str]
    reason: str = ""


def _decline(reason: str) -> Compiled:
    return Compiled(sql=None, reason=reason)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _empty(value: Any) -> bool:
    return value is None or value is False or value == "" or value == [] or value == {}


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _unsupported_field(plan: Any) -> Optional[str]:
    """The first plan field that asks for something the compiler cannot build."""
    fields = getattr(type(plan), "model_fields", None)
    if fields:
        for name, info in fields.items():
            if name in _KNOWN_PLAN_FIELDS:
                continue
            value = getattr(plan, name, None)
            default = info.get_default(call_default_factory=True)
            if value != default and not (_empty(value) and _empty(default)):
                return name
        return None
    for name in _UNSUPPORTED_PLAN_FIELDS:
        if not _empty(_value(plan, name)):
            return name
    return None


def _resolved_diagnoses(resolved: Iterable[Any]) -> Dict[str, List[str]]:
    terms: Dict[str, List[str]] = {}
    for item in resolved or []:
        field = _value(item, "field")
        if getattr(field, "value", field) != "diagnosis":
            continue
        for term, codes in (_value(item, "term_to_codes", {}) or {}).items():
            key = str(term).strip().casefold()
            bucket = terms.setdefault(key, [])
            for code in codes or []:
                code = str(code).strip().lower()
                if code not in bucket:
                    bucket.append(code)
    return terms


def _group_condition(group: Any, resolved: Dict[str, List[str]]) -> Optional[str]:
    raw_terms = [str(term).strip() for term in (_value(group, "diagnosis_terms", []) or [])]
    if not raw_terms or any(not term for term in raw_terms):
        return None

    codes: List[str] = []
    term_map: Dict[str, List[str]] = {}
    for term in raw_terms:
        matched = resolved.get(term.casefold())
        if not matched:
            return None
        valid = [code for code in matched if _CODE_RE.fullmatch(code)]
        if len(valid) != len(matched):
            return None
        term_map[term] = valid
        for code in valid:
            if code not in codes:
                codes.append(code)

    code_sql = ", ".join(_sql_string(code) for code in codes)
    parts = [
        "EXISTS (SELECT 1 FROM bids_participants p_cohort "
        "WHERE p_cohort.dataset_id = d.id "
        f"AND p_cohort.diagnosis IN ({code_sql}))"
    ]
    text = format_text_match(diagnosis_alternatives(term_map), MODE_SHORT)
    if text:
        parts.append(text)
    condition = "(" + " OR ".join(parts) + ")"
    minimum = _value(group, "min_subjects")
    if minimum is None:
        return condition
    # "At least N patients with X": the dataset is about X, and N of its
    # participants are known to have it (RAG/cohort_count.py; a dataset that
    # does not record this goes to the uncertain tier).
    count = patients_placeholder(codes, minimum) if isinstance(minimum, int) else None
    if count is None:
        return None
    return f"({condition} AND {count})"


def _scan_conditions(augmented_question: Optional[str]) -> Tuple[Optional[List[str]], str]:
    """The scan conditions of the augmented question, or (None, reason).

    Every line of the resolved-filters block must be a diagnosis line (built
    here from the typed plan instead) or a scan line. Any other line, and any
    note outside the block (vocabulary miss, contradiction), is a condition
    the compiler does not know, so the plan is declined.
    """
    if not augmented_question:
        return [], ""
    for marker in ("[VOCABULARY MISS", "[Query note", "[DB MISS", "[Topics"):
        if marker in augmented_question:
            return None, f"augmented question has {marker.strip('[')}"
    lines = augmented_question.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(_FILTER_HEADER))
    except StopIteration:
        return [], ""
    conditions: List[str] = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        if _DIAGNOSIS_LINE_RE.match(line):
            continue
        if _AGE_LINE_RE.match(line):
            continue  # _age_conditions
        match = _SCAN_LINE_RE.match(line)
        if not match:
            return None, f"filter line not supported: {line.strip()[:40]}"
        conditions.append((match.group("condition") or match.group("any_of")).strip())
    return conditions, ""


def _age_conditions(augmented_question: Optional[str]) -> List[str]:
    """The age-group condition of the augmented question, if it has one."""
    return [match.group("condition")
            for match in map(_AGE_LINE_RE.match, (augmented_question or "").splitlines())
            if match]


def compile_plan(plan: Any, resolved_terms: Iterable[Any],
                 augmented_question: Optional[str] = None) -> Compiled:
    """Compile a cohort plan (with optional scan requirements), or decline.

    Each group is an independent requirement; ``group_operator`` controls AND
    versus OR between groups, while diagnoses within one group are
    alternatives. Scan conditions are ANDed to the cohorts.
    """
    groups = _value(plan, "groups", []) or []
    operator = _value(plan, "group_operator", "AND")
    if operator not in {"AND", "OR"}:
        return _decline("unknown group operator")
    if not any(_value(group, "diagnosis_terms") for group in groups):
        return _decline("no diagnosis group")
    field = _unsupported_field(plan)
    if field:
        return _decline(f"plan field {field}")

    for group in groups:
        if _empty(_value(group, "diagnosis_terms")):
            # "patients and healthy controls" with the controls left out of the
            # plan: compiling the rest would silently drop a cohort.
            return _decline("group without diagnosis terms")
        for name, default in _GROUP_DEFAULTS:
            value = _value(group, name, default)
            if value != default and not _empty(value):
                return _decline(f"group field {name}")

    scans = _value(plan, "scan_requirements", []) or []
    for scan in scans:
        if _value(scan, "required", True) is not True or _value(scan, "same_subject", False):
            return _decline("scan requirement is negated or per subject")
    scan_sql, reason = _scan_conditions(augmented_question)
    if scan_sql is None:
        return _decline(reason)
    if scans and not scan_sql:
        return _decline("scan requirement without a resolved condition")
    if scan_sql and not scans:
        return _decline("scan condition without a scan requirement")

    resolved = _resolved_diagnoses(resolved_terms)
    conditions = [_group_condition(group, resolved) for group in groups]
    if any(condition is None for condition in conditions):
        return _decline("diagnosis term not resolved")

    cohorts = f" {operator} ".join(conditions)  # operator validated above
    if len(conditions) > 1:
        cohorts = f"({cohorts})"
    where = " AND ".join([*scan_sql, *_age_conditions(augmented_question), cohorts])
    return Compiled(sql=(
        "SELECT {{COLS}}\n"
        "FROM bids_datasets d\n"
        "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        f"WHERE {where}\n"
        "GROUP BY d.id"
    ))


def compile_participant_groups(plan: Any, resolved_terms: Iterable[Any],
                               augmented_question: Optional[str] = None) -> Optional[str]:
    """compile_plan(...).sql: the SQL, or ``None`` when the plan is declined."""
    return compile_plan(plan, resolved_terms, augmented_question).sql
