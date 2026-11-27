"""What to look up in the vocabulary for a plan, and the scored filters that come back."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional
from participant_codes import sex_code
from services.schema.coded_fields import CONCEPT_FIELDS
from join_registry import (
    field_to_db_col as _field_to_db_col,
)

from . import catalogue as _catalogue
from .plan import ParticipantGroup, QueryPlan, RAGField, RAGRequest, ScoredFilter

log = logging.getLogger("preprocess")


# "Healthy" is not searched and not filtered: 1,331 of the 1,718 classified
# datasets include healthy participants, 175 record it as a diagnosis and
# about half say it in their text. A condition on it removed a third of the
# correct answers of the real requests that asked for healthy people.
_HEALTHY_WORDS = frozenset("""
    healthy normal control controls neurotypical typical typically developing unaffected
    nonclinical non-clinical hc td
""".split())


_PERSON_WORDS = frozenset("""
    patients patient participants participant subjects subject people person persons
    individuals individual volunteers volunteer adults adult children child humans human
    men women man woman male female males females group groups cohort sample samples and or
""".split())


def _is_healthy_term(term: str) -> bool:
    """True for a participant term that only says "healthy" ("healthy controls", "normal adults")."""
    words = re.findall(r"[a-z-]+", (term or "").lower())
    return (any(w in _HEALTHY_WORDS for w in words)
            and all(w in _HEALTHY_WORDS or w in _PERSON_WORDS for w in words))


_ADULT_AGE = 18


_ONLY_RE = re.compile(r"\b(?:only|exclusively|solely|all[- ](?:female|male|women|men))\b", re.IGNORECASE)


def _participant_conditions(plan: QueryPlan) -> List[str]:
    """Fixed SQL for the sex and age the plan asks for.

    Written here, not by the SQL model: the model wrote the same request as
    three different conditions in three runs (sex, age and diagnosis on one
    participants row; a text search for "healthy & adult"; an age test on the
    dataset), with 0, 23 and 177 results.

    "Adults" without a number is a property of the sample: it excludes only
    datasets whose recorded ages are all below 18 (most datasets record no
    ages, and 193 such datasets are correct answers of one real request).
    A stated age with a stated sex ("male, 40 years old") describes one
    person: both must hold for the same participant where the dataset lists
    ages per participant. As two separate conditions the request returned 65
    datasets instead of 18.
    """
    def number(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else str(x)

    def sex_values(group: ParticipantGroup) -> Optional[str]:
        # The code the column holds for the words the plan gives ("women",
        # "F"): the vocabulary, backend/vocabulary/participant_codes.py.
        wanted = {code for v in group.sex if (code := sex_code(v))}
        return f"'{next(iter(wanted))}'" if len(wanted) == 1 else None

    participants = _catalogue.tables().participants.name
    rows = f"EXISTS (SELECT 1 FROM {participants} p2 WHERE p2.dataset_id = d.id AND "
    all_sexes = {s for s in (sex_values(g) for g in plan.groups) if s}
    lines: List[str] = []

    # One person with a stated sex and age.
    for group in plan.groups:
        low, high = group.age_min, group.age_max
        values = sex_values(group)
        stated = (low is not None or high is not None) and not (low == _ADULT_AGE and high is None)
        if values and stated and len(all_sexes) == 1:
            per_person = " AND ".join(
                ([f"p2.age >= {number(low)}"] if low is not None else [])
                + ([f"p2.age <= {number(high)}"] if high is not None else []))
            dataset_level = " AND ".join(
                ([f"d.age_max >= {number(low)}"] if low is not None else [])
                + ([f"d.age_min <= {number(high)}"] if high is not None else []))
            return [
                f"  participant sex and age: ({rows}p2.sex = {values} AND {per_person}) "
                f"OR (NOT {rows}p2.age IS NOT NULL) "
                f"AND {rows}p2.sex = {values}) AND {dataset_level}))"
            ]

    if len(all_sexes) == 1:
        values = next(iter(all_sexes))
        condition = f"{rows}p2.sex = {values})"
        if _ONLY_RE.search(plan.natural_language_summary or ""):
            # "Only female participants": every participant row says so. A sex
            # recorded for part of the participants is no evidence.
            condition += (f" AND NOT EXISTS (SELECT 1 FROM {participants} p3 "
                          "WHERE p3.dataset_id = d.id AND (p3.sex IS NULL "
                          f"OR p3.sex <> {values}))")
        lines.append(f"  participant sex: {condition}")
    lows = [g.age_min for g in plan.groups if g.age_min is not None]
    highs = [g.age_max for g in plan.groups if g.age_max is not None]
    low = min(lows) if lows else None
    high = max(highs) if highs else None
    said = " ".join([plan.description_search or ""]
                    + [t for g in plan.groups for t in g.diagnosis_terms]).lower()
    adults = (low is None and high is None and re.search(r"\badults?\b", said)) or (
        low == _ADULT_AGE and high is None)
    if adults:
        lines.append(
            f"  participant age: ({rows}p2.age >= {_ADULT_AGE}) OR d.age_max >= {_ADULT_AGE} "
            f"OR (d.age_min IS NULL AND d.age_max IS NULL AND NOT {rows}p2.age IS NOT NULL)))")
    elif low is not None and high is not None:
        lines.append(f"  participant age: ({rows}p2.age >= {number(low)} AND p2.age <= {number(high)}) "
                     f"OR (d.age_min <= {number(high)} AND d.age_max >= {number(low)}))")
    elif low is not None:
        lines.append(f"  participant age: ({rows}p2.age >= {number(low)}) OR d.age_max >= {number(low)})")
    elif high is not None:
        lines.append(f"  participant age: ({rows}p2.age <= {number(high)}) OR d.age_min <= {number(high)})")
    return lines


def build_rag_requests(plan: QueryPlan) -> List[RAGRequest]:
    """
    Derive the full list of RAGRequests from a QueryPlan.

    Scan terms from all ScanRequirements are collected into a single
    RAGRequest with field=RAGField.SCAN.  The retriever handles multi-field
    lookup (datatype + suffix + task) for these terms.

    All other terms (diagnosis, participant task, author, funding) get
    per-field RAGRequests as before.
    """
    requests: Dict[RAGField, RAGRequest] = {}

    def _merge(field: RAGField, terms: List[str]) -> None:
        if not terms:
            return
        if field not in requests:
            requests[field] = RAGRequest(field=field, terms=list(terms))
        else:
            for t in terms:
                if t not in requests[field].terms:
                    requests[field].terms.append(t)

    # Scan requirements → single SCAN request (multi-field RAG)
    all_scan_terms: List[str] = []
    for sr in plan.scan_requirements:
        all_scan_terms.extend(sr.scan_terms)
    _merge(RAGField.SCAN, all_scan_terms)

    # Participant groups — strip generic demographic words before RAG resolution
    # so that "patients" / "participants" never resolves to unknown_diagnosis
    generic_words = {
        "patients", "patient", "participants", "participant",
        "subjects", "subject", "people", "person",
        "individuals", "individual", "volunteers", "volunteer",
        "adults", "adult", "children", "child",
        "healthy", "normal",
    }
    for grp in plan.groups:
        real_diagnoses = [t for t in grp.diagnosis_terms
                          if t.lower().strip() not in generic_words
                          and not _is_healthy_term(t)]
        _merge(RAGField.DIAGNOSIS, real_diagnoses)
        _merge(RAGField.TASK,      grp.task_terms)
    # Excluded diagnoses resolve like asked ones: the exclusion is the
    # negation of the same condition (backend/vocabulary/criteria.py without_diagnosis).
    _merge(RAGField.DIAGNOSIS, [t for t in plan.excluded_diagnosis_terms
                                if t and t.strip() and t.lower().strip() not in generic_words
                                and not _is_healthy_term(t)])

    # Dataset-level text fields
    _merge(RAGField.AUTHOR,  plan.author_names)
    _merge(RAGField.FUNDING, plan.funding_terms)
    _merge(RAGField.INSTITUTION, plan.institution_terms + plan.excluded_institution_terms)

    # Merge any extra requests the LLM explicitly set
    for req in plan.rag_requests:
        if req.field == RAGField.DIAGNOSIS:
            req = req.model_copy(update={"terms": [t for t in req.terms if not _is_healthy_term(t)]})
            if not req.terms:
                continue
        if req.field not in requests:
            requests[req.field] = req.model_copy()
        else:
            for t in req.terms:
                if t not in requests[req.field].terms:
                    requests[req.field].terms.append(t)

    return list(requests.values())



def _build_scored_filters(
    rag_results: Dict[str, Dict[str, List[str]]],
    scores: Dict[str, Dict[str, float]],
) -> List[ScoredFilter]:
    """Flatten RAG results + scores into per-(field, code) ScoredFilters.

    Codes are taken from rag_results (already DB-pruned, so DB-miss codes are
    excluded automatically). Only fields that map to a DB column are kept.
    Returns [] when no scores were captured (e.g. an older retriever without the
    *_scored methods) so downstream ranking degrades gracefully.
    """
    if not scores:
        return []
    out: List[ScoredFilter] = []
    seen: set[tuple[str, str]] = set()
    for field, term_map in rag_results.items():
        # Only codes with a queryable column are scored: author and funding are
        # arrays on bids_datasets (exists_alias 'none').
        if field not in CONCEPT_FIELDS or _field_to_db_col(field) is None:
            continue
        field_scores = scores.get(field, {})
        for codes in term_map.values():
            for code in codes:
                if (field, code) in seen:
                    continue
                seen.add((field, code))
                out.append(
                    ScoredFilter(field=field, code=code,
                                 score=field_scores.get(code, 1.0))
                )
    return out
