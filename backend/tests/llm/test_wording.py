"""Words of the question the plan must not lose: plain "adults", and scans the user only allows."""
from __future__ import annotations

import pytest

from preprocess import ParticipantGroup, QueryPlan, ScanRequirement
from preprocess import wording


def _plan(*requirements, **fields) -> QueryPlan:
    return QueryPlan(natural_language_summary="Summary.",
                     scan_requirements=list(requirements), **fields)


def test_plain_adults_becomes_the_adult_age() -> None:
    plan = wording.apply(_plan(ScanRequirement(scan_terms=["structural MRI"])),
                         "adults with no history of stroke, head injury, seizures; structural MRI")
    assert [g.age_min for g in plan.groups] == [18] and plan.groups[0].age_max is None


def test_adults_go_to_the_existing_group() -> None:
    plan = wording.apply(_plan(groups=[ParticipantGroup(group_name="g", diagnosis_terms=["ADHD"])]),
                         "resting-state fMRI in adults with ADHD")
    assert len(plan.groups) == 1 and plan.groups[0].age_min == 18


@pytest.mark.parametrize("question, fields", [
    ("memory in older adults", {}),
    ("young adults playing games", {}),
    ("children and adults, EEG", {"age_groups": ["children"]}),
    ("adults aged 20 to 40", {"groups": [ParticipantGroup(group_name="g", age_min=20, age_max=40)]}),
    ("EEG datasets", {}),
])
def test_other_ages_are_left_alone(question, fields) -> None:
    before = _plan(**fields)
    assert wording.apply(_plan(**fields), question).groups == before.groups


def test_an_allowed_scan_is_no_condition() -> None:
    plan = wording.apply(
        _plan(ScanRequirement(scan_terms=["fMRI", "decoding task", "localizer"]),
              ScanRequirement(scan_terms=["RS"], required=False)),
        "I need fMRI with some decoding task: block design, 4 or more subjects (more is fine), "
        "several sessions each, localizer ok; no RS")
    assert [(r.scan_terms, r.required) for r in plan.scan_requirements] == [
        (["fMRI", "decoding task"], True), (["RS"], False)]


def test_a_requirement_with_only_an_allowed_term_goes() -> None:
    plan = wording.apply(_plan(ScanRequirement(scan_terms=["MEG"]), ScanRequirement(scan_terms=["T1w"])),
                         "MEG of speech; a T1w is fine too")
    assert [r.scan_terms for r in plan.scan_requirements] == [["MEG"]]


@pytest.mark.parametrize("question", [
    "EEG with an oddball task, raw or preprocessed is fine",     # nothing of the plan is allowed only
    "fMRI and a localizer; any localizer is fine",              # also asked for outside the phrase
])
def test_a_term_also_asked_for_stays(question) -> None:
    plan = wording.apply(_plan(ScanRequirement(scan_terms=["fMRI", "localizer"])), question)
    assert plan.scan_requirements[0].scan_terms == ["fMRI", "localizer"]


def test_an_excluded_scan_is_not_touched() -> None:
    plan = wording.apply(_plan(ScanRequirement(scan_terms=["EEG"], required=False)), "no EEG, MEG is ok")
    assert plan.scan_requirements[0].scan_terms == ["EEG"]
