"""The scorer on hand-written plans: right plans score equal, each mistake is named."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_plans as sp  # noqa: E402
from build import load_own  # noqa: E402

ITEMS = {i["id"]: i for i in load_own()}


def scan(*terms, required=True, any_of=False):
    return {"scan_terms": list(terms), "required": required, "any_of": any_of}


def group(*diagnoses, **fields):
    return {"diagnosis_terms": list(diagnoses), **fields}


def result(item: str, **plan) -> dict:
    return sp.compare(ITEMS[item], plan)


RIGHT = {
    "own-01": {"scan_requirements": [scan("EEG", "motor imagery"), scan("resting-state", required=False)]},
    "own-02": {"scan_requirements": [scan("iEEG"), scan("scalp EEG", required=False), scan("MEG", required=False)]},
    "own-03": {"groups": [group("depression")], "excluded_diagnosis_terms": ["bipolar disorder"],
               "scan_requirements": [scan("resting state", "fMRI")]},
    "own-07": {"groups": [group("depression")], "description_search": "unmedicated",
               "scan_requirements": [scan("resting state fMRI")]},
    "own-09": {"groups": [group("epilepsy")], "description_search": "MRI-negative",
               "scan_requirements": [scan("stereo-EEG")]},
    "own-15": {"groups": [group(age_min=18)], "excluded_diagnosis_terms": ["stroke", "head injury", "seizures"],
               "scan_requirements": [scan("structural MRI")]},
    "own-20": {"scan_requirements": [scan("fMRI"), scan("go/no-go", "stop-signal", any_of=True)], "max_size_gb": 30},
    "own-24": {"scan_requirements": [scan("T1w"), scan("diffusion"), scan("resting state", "fMRI")]},
    "own-25": {"groups": [group("anorexia nervosa", "bulimia nervosa", "binge-eating disorder")],
               "scan_requirements": [scan("MRI")]},
    "own-36": {"scan_requirements": [scan("MPRAGE")], "groups": [group(age_min=5, age_max=21)],
               "metadata_filters": [{"json_field": "MagneticFieldStrength", "operator": "=", "value": 3.0}],
               "description_search": 'NOT ("prospective motion correction" OR vNavs)'},
    "own-41": {"scan_requirements": [scan("resting state", "EEG"), scan("sleep", required=False),
                                     scan("TMS", required=False)], "excluded_diagnosis_terms": ["epilepsy"]},
    "own-44": {"groups": [group("stroke", "aphasia", min_subjects=10)], "scan_requirements": [scan("diffusion MRI")]},
}


@pytest.mark.parametrize("item", RIGHT)
def test_a_right_plan_is_equal(item) -> None:
    r = result(item, **RIGHT[item])
    assert r["equal"], r
    assert not (r["dropped"] or r["invented"] or r["polarity"] or r["grouping"]), r


def test_a_lost_exclusion_is_dropped() -> None:
    r = result("own-01", scan_requirements=[scan("EEG", "motor imagery")])
    assert not r["equal"] and r["dropped"] == ["c3"] and r["negative_dropped"] == 1


def test_an_exclusion_turned_into_a_requirement_is_polarity() -> None:
    r = result("own-01", scan_requirements=[scan("EEG", "motor imagery", "resting state")])
    assert not r["equal"] and r["polarity"] == ["c3"] and r["negative_wrong"] == 1


def test_or_read_as_and_is_grouping() -> None:
    r = result("own-20", scan_requirements=[scan("fMRI"), scan("go/no-go", "stop-signal")], max_size_gb=30)
    assert not r["equal"] and r["grouping"] == ["c2+c3"] and not r["dropped"]


def test_and_read_as_or_is_grouping() -> None:
    r = result("own-24", scan_requirements=[scan("T1w", "diffusion", any_of=True), scan("resting state", "fMRI")])
    assert r["grouping"] == ["c1+c2"]


def test_an_added_condition_is_invented() -> None:
    r = result("own-24", **{**RIGHT["own-24"], "species": ["human"]})
    assert not r["equal"] and r["invented"] == ["species:human"]


def test_a_wish_used_as_a_condition_is_an_error_and_in_wishes_is_not() -> None:
    wanted = {"scan_requirements": [scan("language", "fMRI"), scan("DTI")]}
    assert result("own-34", **wanted, request_wishes=["something smaller"])["equal"]
    r = result("own-34", **wanted, description_search="smaller")
    assert not r["equal"] and r["invented"]


def test_any_accepted_reading_counts() -> None:
    both = result("own-23", scan_requirements=[scan("TOF"), scan("SWI")])
    either = result("own-23", scan_requirements=[scan("TOF", "SWI", any_of=True)])
    assert both["equal"] and either["equal"]


def test_a_narrower_diagnosis_is_not_the_wider_one() -> None:
    r = result("own-04", groups=[group("temporal lobe epilepsy")])
    assert not r["equal"] and "c1" in r["dropped"] and r["polarity"] == ["c2"]


def test_scalp_eeg_and_ieeg_are_different_clauses() -> None:
    r = result("own-02", scan_requirements=[scan("iEEG"), scan("EEG", required=False), scan("MEG", required=False)])
    assert r["equal"], r


def test_a_wrong_number_is_dropped_and_invented() -> None:
    r = result("own-44", groups=[group("stroke", "aphasia", min_subjects=50)], scan_requirements=[scan("diffusion")])
    assert r["dropped"] == ["c4"] and r["invented"] == ["count:50..None"]


def test_a_lower_bound_is_not_an_upper_bound() -> None:
    assert not result("own-20", scan_requirements=RIGHT["own-20"]["scan_requirements"], min_size_gb=30)["equal"]


def test_an_empty_plan_drops_everything() -> None:
    r = result("own-24")
    assert not r["equal"] and r["dropped"] == ["c1", "c2", "c3", "c4"]


def test_groups_joined_by_or_are_alternatives() -> None:
    plan = {"groups": [group("stroke"), group("aphasia")], "scan_requirements": [scan("diffusion")]}
    assert result("own-44", **plan, group_operator="AND")["grouping"] == ["c1+c2"]
    assert not result("own-44", **plan, group_operator="OR")["grouping"]


def test_an_equivalent_plan_is_accepted_and_the_item_stays_in_the_headline() -> None:
    parent = scan("non-invasive brain stimulation", "TMS", "tDCS", any_of=True)
    plain = scan("TMS", "tDCS", any_of=True)
    for need in (parent, plain):
        assert result("own-19", scan_requirements=[scan("EEG"), need])["equal"]
    assert not result("own-19", scan_requirements=[scan("EEG"), plain, scan("non-invasive brain stimulation")])["equal"]
    assert sp.kind(ITEMS["own-19"]) == sp.kind(ITEMS["own-36"]) == "headline"


def test_one_participant_may_be_a_minimum_count() -> None:
    plan = RIGHT["own-36"]
    counted = {**plan, "groups": [group(age_min=5, age_max=21, min_subjects=1)]}
    assert result("own-36", **counted)["equal"]
    more = {**plan, "groups": [group(age_min=5, age_max=21, min_subjects=20)]}
    assert result("own-36", **more)["invented"] == ["count:20..None"]


def test_every_item_is_in_exactly_one_group() -> None:
    kinds = [sp.kind(i) for i in ITEMS.values()]
    assert {k: kinds.count(k) for k in set(kinds)} == {"headline": 47, "ambiguous": 10, "not representable": 11}
