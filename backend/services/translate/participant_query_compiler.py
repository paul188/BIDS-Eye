"""Deterministic SQL compilation: the model reads the question, code writes the SQL.

The language model extracts the intent into a QueryPlan, RAG resolves its
terms to codes, and preprocess/ writes most conditions as complete SQL
lines of the augmented question ("scan", "species", "participant age",
"dataset text", ...). Asking a second model to AND those lines together added
variation without adding information: the same request came back as 0, 23 or
177 datasets in three runs. This module builds the query itself:

  - diagnosis groups from the typed plan (AND or OR between groups, the
    diagnoses of a group are alternatives, "at least N patients with X");
  - every filter line that is a complete condition, ANDed (an excluded
    diagnosis is one: "not diagnosis", RAG/criteria.py);
  - fixed conditions for plan fields that have no line: a primary paper
    (doi_required), sessions per subject (min_sessions), a named
    preprocessing pipeline (fMRIPrep, MRIQC, ... from bids_derivatives).

Anything that needs interpretation declines with a reason and goes through
the SQL model: sidecar fields, ordering, comparisons, institutions, authors,
licences, negated or per-subject scan requirements, contradictions. The check
is a whitelist: a plan field this module does not know must hold its default,
so a field added later never compiles by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cohort_count import placeholder as patients_placeholder
from text_terms import MODE_ALL, MODE_SHORT, diagnosis_alternatives, diagnosis_qualifier, format_text_match

from services.schema.tables import (
    dataset_papers as _dataset_papers, datasets as _datasets, derivatives as _derivatives, objects as _objects,
    participants as _participants,
)


_CODE_RE = re.compile(r"^[a-z0-9_]+$")


def _is_healthy_term(term: str) -> bool:
    try:
        from preprocess import _is_healthy_term as check
    except ImportError:  # pragma: no cover - outside the backend
        return False
    return check(term)

# Plan fields the compiler reads, or that do not change the SQL.
_KNOWN_PLAN_FIELDS = frozenset({
    "query_family", "natural_language_summary",
    "groups", "group_operator", "scan_requirements",
    "order_direction",   # only meaningful with order_by, which must be unset
    # Written by preprocess/ as complete filter lines, read from there:
    "age_groups", "description_search", "description_alternatives", "request_wishes",
    "species", "study_structure", "max_size_gb", "min_size_gb", "topics", "name_search",
    "location_terms",
    "excluded_diagnosis_terms",   # the "not diagnosis" lines
    # Built here (_fixed_conditions):
    "doi_required", "min_sessions",
})

# For plans that are not pydantic models (tests, dicts): the fields that must be empty.
_UNSUPPORTED_PLAN_FIELDS = (
    "author_names",
    "license_search",
    "funding_terms",
    "institution_terms",
    "excluded_institution_terms",
    "metadata_filters",
    "count_comparison",
    "result_limit",
    "order_by",
    "rag_requests",
)

# Group fields with no filter line. sex, age_min, age_max and min_subjects
# are written by preprocess/ as "participant ..." / "subject count" lines.
_GROUP_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
    ("task_terms", []), ("handedness", []), ("extra_participant_fields", {}),
)

_FILTER_HEADER = "[Resolved DB filters"
# A plain scan line, or a list of alternatives that preprocess/ already
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
    # "Healthy controls" is not searched and not filtered (preprocess/,
    # _is_healthy_term): most datasets include healthy people, few record it.
    raw_terms = [term for term in raw_terms if not _is_healthy_term(term)]
    if not raw_terms:
        return ""

    codes: List[str] = []
    term_map: Dict[str, List[str]] = {}
    unknown: List[str] = []
    for term in raw_terms:
        matched = resolved.get(term.casefold())
        if not matched:
            # Not in the vocabulary ("bipolar syndrome", "Covid-19"): the
            # dataset's title and description name it, or nothing does.
            if not re.fullmatch(r"[\w' -]+", term):
                return None
            unknown.append(term)
            continue
        valid = [code for code in matched if _CODE_RE.fullmatch(code)]
        if len(valid) != len(matched):
            return None
        term_map[term] = valid
        for code in valid:
            if code not in codes:
                codes.append(code)

    parts: List[str] = []
    qualifiers = {term: diagnosis_qualifier(term, codes_) for term, codes_ in term_map.items()}
    if any(qualifiers.values()):
        # A term whose codes miss one of its disease words ("Lewy body
        # dementia" → the dementia codes) needs those words in the dataset
        # text: each such term is its own alternative (text_terms.diagnosis_qualifier).
        for term, words in qualifiers.items():
            code_sql = ", ".join(_sql_string(code) for code in term_map[term])
            part = (f"(EXISTS (SELECT 1 FROM {_participants.name} p_cohort "
                    "WHERE p_cohort.dataset_id = d.id "
                    f"AND p_cohort.diagnosis IN ({code_sql}))")
            text = format_text_match(diagnosis_alternatives({term: term_map[term]}), MODE_SHORT)
            part += f" OR {text})" if text else ")"
            if words:
                part = f"({part} AND {format_text_match([' '.join(words)], MODE_ALL)})"
            parts.append(part)
    elif codes:
        code_sql = ", ".join(_sql_string(code) for code in codes)
        parts.append(
            f"EXISTS (SELECT 1 FROM {_participants.name} p_cohort "
            "WHERE p_cohort.dataset_id = d.id "
            f"AND p_cohort.diagnosis IN ({code_sql}))")
        text = format_text_match(diagnosis_alternatives(term_map), MODE_SHORT)
        if text:
            parts.append(text)
    for term in unknown:
        unknown_text = format_text_match([term.lower()], MODE_SHORT)
        if unknown_text:  # None when nothing searchable is left
            parts.append(unknown_text)
    if not parts:
        return None
    condition = "(" + " OR ".join(parts) + ")"
    minimum = _value(group, "min_subjects")
    if minimum is None:
        return condition
    if not codes:
        return None  # a patient count needs coded diagnoses
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


# Filter lines that are complete conditions: ANDed as they are.
_COMPLETE_LINE_RE = re.compile(
    r"^  (?:scan(?: task)? \"[^\"]*\""
    r"|scan any of \"[^\"]*\"(?:, \"[^\"]*\")+ \(alternatives, one is enough\)"
    r"|participant sex|participant age|participant sex and age"
    r"|species \(complete, no other species condition\)"
    r"|not diagnosis \"[^\"]*\" \(complete, datasets that record no diagnosis stay\)"
    r"|study design|age group [a-z ]+ \(complete, no other age condition\)"
    r"|topic \"[^\"]*\"|location \"[^\"]*\"|dataset text|dataset name|dataset size): (?P<condition>.+)$")
_HAVING_LINE_RE = re.compile(r"^  subject count: HAVING (?P<condition>.+?)(?: \[.*\])?$")
_IGNORED_LINE_RE = re.compile(r"^  not searchable \(add NO condition for these\): ")
_VOCAB_HINT_RE = re.compile(r"For free-text search use: (?P<hint>.+?)\]\s*$", re.MULTILINE)
_VOCAB_BLOCK_RE = re.compile(
    r"\[VOCABULARY MISS — these terms are not in the known concept vocabulary: (?P<fields>.*?)\. "
    r"Do NOT filter.*?\]", re.DOTALL)
_VOCAB_FIELD_RE = re.compile(r"(\w+): ((?:'(?:[^']|'')*'(?:, )?)+)")
_TEXT_MATCH_RE = re.compile(r"text_match\('(?:[^']|'')*'(?:, '[a-z]+')?\)")

# Named preprocessing outputs (bids_derivatives.pipeline: OpenNeuro
# Derivatives and the dataset's own derivatives/ folder). A request for
# "fMRIPrep" was searched as the README word: 53 of 365 correct answers.
_PIPELINES = {
    "fmriprep": ("fmriprep",), "mriqc": ("mriqc",), "xcpd": ("xcpd",), "xcp-d": ("xcpd",),
    "freesurfer": ("freesurfer",), "fitlins": ("fitlins",), "eeglab": ("eeglab",),
}
_PREPROCESSED_WORDS = frozenset({"preprocessed", "pre-processed", "preprocessing", "preproc"})
_PREPROCESSED_PIPELINES = ("fmriprep", "xcpd", "preprocessed", "preprocessed-data",
                           "preprocessing", "preproc")
_PIPELINE_FILLER = frozenset({"data", "dataset", "datasets", "with", "by", "using", "already",
                              "outputs", "output", "derivatives", "processed", "and", "or",
                              "available", "fmri"})


def _derivatives_condition(description: Optional[str]) -> Optional[str]:
    """bids_derivatives for a description that only names preprocessing ("fMRIPrep", "preprocessed")."""
    words = [w for w in re.findall(r"[a-z][a-z0-9-]*", (description or "").lower())
             if w not in _PIPELINE_FILLER]
    if not words:
        return None
    pipelines: List[str] = []
    for word in words:
        if word in _PIPELINES:
            pipelines += _PIPELINES[word]
        elif word in _PREPROCESSED_WORDS:
            pipelines += _PREPROCESSED_PIPELINES
        else:
            return None
    values = ", ".join(_sql_string(p) for p in dict.fromkeys(pipelines))
    return (f"EXISTS (SELECT 1 FROM {_derivatives.name} dv "
            f"WHERE dv.dataset_id = d.id AND dv.pipeline IN ({values}))")


def _filter_conditions(augmented_question: Optional[str], plan: Any,
                       ) -> Tuple[Optional[List[str]], List[str], str]:
    """(WHERE conditions, HAVING conditions, reason) from the filter block; None when declined."""
    text = augmented_question or ""
    for marker in ("[DB MISS",):
        if marker in text:
            return None, [], f"augmented question has {marker.strip('[')}"
    if "[Query note:" in text and "HARD CONTRADICTION" in text:
        return None, [], "augmented question has a contradiction"
    where: List[str] = []
    having: List[str] = []
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(_FILTER_HEADER)), None)
    derivatives = _derivatives_condition(_value(plan, "description_search"))
    if start is not None:
        for line in lines[start + 1:]:
            if not line.strip():
                break
            if _DIAGNOSIS_LINE_RE.match(line) or _IGNORED_LINE_RE.match(line):
                continue  # diagnoses: built from the typed plan
            if line.startswith("[VOCABULARY MISS") or line.startswith("[Query note"):
                continue  # handled below / a soft warning
            match = _HAVING_LINE_RE.match(line)
            if match:
                having.append(match.group("condition").strip())
                continue
            match = _COMPLETE_LINE_RE.match(line)
            if not match:
                return None, [], f"filter line not supported: {line.strip()[:40]}"
            condition = match.group("condition").strip()
            if derivatives and line.startswith("  dataset text: "):
                condition = derivatives  # the pipeline is a recorded fact, not a README word
                derivatives = None
            where.append(condition)
    if derivatives:
        where.append(derivatives)
    # Words the vocabulary does not know: the text search the block names for
    # each. Diagnoses are part of their group's condition (_group_condition).
    if "[VOCABULARY MISS" in text and not _VOCAB_BLOCK_RE.search(text):
        return None, [], "vocabulary miss not understood"
    for block in _VOCAB_BLOCK_RE.finditer(text):
        terms = [(field, term) for field, listed in _VOCAB_FIELD_RE.findall(block.group("fields"))
                 for term in re.findall(r"'((?:[^']|'')*)'", listed)]
        hint = _VOCAB_HINT_RE.search(block.group(0))
        calls = _TEXT_MATCH_RE.findall(hint.group("hint")) if hint else []
        if len(calls) != len(terms):
            return None, [], "vocabulary miss without a text search per term"
        for (field, term), call in zip(terms, calls):
            if field == "diagnosis":
                continue
            pipeline = _derivatives_condition(term)
            if pipeline:
                where.append(pipeline)
            elif re.search(r"[_.]|\b(?:nii|gz|tsv|json|edf|bdf|set)\b", term):
                return None, [], "vocabulary miss names a file"
            else:
                where.append(call)
    return where, having, ""


def _fixed_conditions(plan: Any) -> List[str]:
    """Conditions for plan fields that have no filter line."""
    conditions: List[str] = []
    if _value(plan, "doi_required"):
        # The dataset's own paper (bids_dataset_papers, role 'primary'), not d.doi:
        # that is the DOI of the dataset itself.
        conditions.append(f"EXISTS (SELECT 1 FROM {_dataset_papers.name} dp "
                          "WHERE dp.dataset_id = d.id AND dp.role = 'primary')")
    sessions = _value(plan, "min_sessions")
    if isinstance(sessions, int) and sessions > 1:
        # Every subject has at least N sessions; one grouped subquery over the
        # table (a per-dataset GROUP BY ran into the 300 s limit).
        conditions.append(
            f"d.id NOT IN (SELECT s.dataset_id FROM {_objects.name} s WHERE s.subject IS NOT NULL "
            f"GROUP BY s.dataset_id, s.subject HAVING COUNT(DISTINCT s.session) < {sessions})")
        conditions.append(f"EXISTS (SELECT 1 FROM {_objects.name} s2 WHERE s2.dataset_id = d.id "
                          "AND s2.subject IS NOT NULL)")
    return conditions


def compile_plan(plan: Any, resolved_terms: Iterable[Any],
                 augmented_question: Optional[str] = None) -> Compiled:
    """Compile the plan into SQL, or decline with the reason.

    Diagnosis groups are independent requirements; ``group_operator`` sets AND
    versus OR between them, the diagnoses within one group are alternatives.
    Everything else is ANDed.
    """
    groups = _value(plan, "groups", []) or []
    operator = _value(plan, "group_operator", "AND")
    if operator not in {"AND", "OR"}:
        return _decline("unknown group operator")
    field = _unsupported_field(plan)
    if field:
        return _decline(f"plan field {field}")
    for group in groups:
        for name, default in _GROUP_DEFAULTS:
            value = _value(group, name, default)
            if value != default and not _empty(value):
                return _decline(f"group field {name}")

    scans = _value(plan, "scan_requirements", []) or []
    for scan in scans:
        if _value(scan, "required", True) is not True or _value(scan, "same_subject", False):
            # The scan lines do not say which requirement is negated or must
            # hold per subject.
            return _decline("scan requirement is negated or per subject")

    where, having, reason = _filter_conditions(augmented_question, plan)
    if where is None:
        return _decline(reason)
    vocabulary_scans = re.search(r"VOCABULARY MISS[^\]]*?\bscan: ", augmented_question or "")
    if scans and not vocabulary_scans and not any(f"{_objects.name} o2" in c for c in where):
        return _decline("scan requirement without a resolved condition")

    resolved = _resolved_diagnoses(resolved_terms)
    cohort_conditions: List[str] = []
    for group in groups:
        if _empty(_value(group, "diagnosis_terms")):
            continue  # "all participants": no condition
        condition = _group_condition(group, resolved)
        if condition is None:
            return _decline("diagnosis term not resolved")
        if condition:
            cohort_conditions.append(condition)
    if operator == "OR" and cohort_conditions and len(cohort_conditions) < len(groups):
        # "patients or anyone": an alternative without a condition admits every dataset.
        return _decline("an alternative group has no condition")
    if cohort_conditions:
        cohorts = f" {operator} ".join(cohort_conditions)  # operator validated above
        where.append(f"({cohorts})" if len(cohort_conditions) > 1 else cohorts)

    where.extend(_fixed_conditions(plan))
    where = list(dict.fromkeys(where))
    if not where and not having:
        return _decline("nothing to filter")
    sql = ("SELECT {{COLS}}\n"
           f"FROM {_datasets.name} d\n"
           f"LEFT JOIN {_objects.name} o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
           + (f"WHERE {' AND '.join(where)}\n" if where else "")
           + "GROUP BY d.id"
           + (f"\nHAVING {' AND '.join(having)}" if having else ""))
    return Compiled(sql=sql)


def compile_participant_groups(plan: Any, resolved_terms: Iterable[Any],
                               augmented_question: Optional[str] = None) -> Optional[str]:
    """compile_plan(...).sql: the SQL, or ``None`` when the plan is declined."""
    return compile_plan(plan, resolved_terms, augmented_question).sql
