"""Words of the question that the plan must not lose or misread.

Applied to every plan the model returns (intent.create_query_plan). Each rule
reads the question itself, because the model drops these words or puts them in
the wrong field, and the prompt cannot fix that without moving other plans.
"""
from __future__ import annotations

import re

from .plan import ParticipantAttribute, ParticipantCriterion, ParticipantGroup, QueryPlan
from .rag import _ADULT_AGE

# "adults", but not "older adults" or "young adults": those name an age group
# or a range the model writes itself.
_ADULTS_RE = re.compile(r"(?<![\w-])(?P<before>[\w-]+\s+)?adults?\b", re.IGNORECASE)
_NOT_PLAIN_ADULTS = frozenset({"older", "young", "younger", "elderly", "middle-aged", "aging", "ageing", "non"})

# "localizer ok", "a T1w is fine": the user allows something, they do not ask for it.
_ALLOWED_RE = re.compile(
    r"(?P<what>[^,;:()\[\]\n.!?/]+?)\s+(?:is\s+|are\s+|would\s+be\s+)?(?:ok|okay|fine|allowed|acceptable)\b",
    re.IGNORECASE)


def _key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _says_plain_adults(query: str) -> bool:
    for match in _ADULTS_RE.finditer(query or ""):
        before = (match.group("before") or "").strip().lower()
        if before not in _NOT_PLAIN_ADULTS:
            return True
    return False


def keep_adults(plan: QueryPlan, query: str) -> None:
    """"adults" without a number becomes the adult age condition (rag._participant_conditions).

    The prompt keeps the word out of age_groups and diagnosis_terms, so a plan
    without any age lost it ("adults with no history of stroke").
    """
    cohorts = plan.cohorts
    has_age = plan.age_groups or any(g.age_min is not None or g.age_max is not None for g in cohorts)
    if has_age or not _says_plain_adults(query):
        return
    if not cohorts:
        cohorts = [ParticipantGroup(group_name="all_participants")]
        plan.groups.append(cohorts[0])
    cohorts[0].criteria.append(ParticipantCriterion(attribute=ParticipantAttribute.AGE, min=_ADULT_AGE))


def drop_allowed_scans(plan: QueryPlan, query: str) -> None:
    """A required scan term the question only allows ("localizer ok") is no condition."""
    allowed = [_key(m.group("what")) for m in _ALLOWED_RE.finditer(query or "")]
    if not allowed:
        return
    said = _key(query)

    def only_allowed(term: str) -> bool:
        key = _key(term)
        if not key:
            return False
        inside = sum(f" {phrase} ".count(f" {key} ") for phrase in allowed)
        return inside > 0 and f" {said} ".count(f" {key} ") == inside

    for requirement in plan.scan_requirements:
        if requirement.required:
            requirement.scan_terms = [t for t in requirement.scan_terms if not only_allowed(t)]
    plan.scan_requirements = [r for r in plan.scan_requirements if r.scan_terms]


def apply(plan: QueryPlan, query: str) -> QueryPlan:
    keep_adults(plan, query)
    drop_allowed_scans(plan, query)
    return plan
