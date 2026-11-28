"""Which few-shot example families a plan asks for (services/translate/prompt/example_families.py)."""

from __future__ import annotations

import json

from preprocess import MetadataFilter, ParticipantGroup, QueryPlan, ScanRequirement
from services.translate.prompt import example_families
from services.translate.prompt.examples import _FEW_SHOT_PATH

COMBINED = example_families.COMBINED


def _families(**fields) -> list:
    return example_families.plan_families(QueryPlan(natural_language_summary="x", **fields))


def test_an_empty_plan_asks_for_no_family() -> None:
    assert _families() == []


def test_one_kind_of_condition_is_one_family() -> None:
    assert _families(author_names=["Poldrack"]) == ["author_query"]
    assert _families(license_search="CC0") == ["license_query"]
    assert _families(description_search="memory") == ["description_search"]


def test_a_mix_is_every_family_plus_the_combined_one() -> None:
    # qx12: human EEG under 10 GB with more than 30 subjects and a linked paper
    families = _families(
        groups=[ParticipantGroup(group_name="all", min_subjects=30, sex=["female"])],
        scan_requirements=[ScanRequirement(scan_terms=["EEG"])],
        doi_required=True,
    )
    assert families == ["participant_filter", "scan_filter", "doi_query", COMBINED]


def test_patients_with_controls_is_a_clinical_design() -> None:
    families = _families(groups=[
        ParticipantGroup(group_name="patients", diagnosis_terms=["schizophrenia"]),
        ParticipantGroup(group_name="controls"),
    ])
    assert families[:2] == ["concept_query", "clinical_design_query"]


def test_a_named_pipeline_is_a_derivative_request() -> None:
    assert "derivative_query" in _families(
        scan_requirements=[ScanRequirement(scan_terms=["MRIQC derivatives"])])
    assert _families(description_search="fMRIPrep") == ["derivative_query"]
    assert "derivative_query" not in _families(description_search="memory")


def test_scans_required_absent_or_per_subject() -> None:
    families = _families(scan_requirements=[
        ScanRequirement(scan_terms=["T1w"], same_subject=True),
        ScanRequirement(scan_terms=["bold"], same_subject=True),
        ScanRequirement(scan_terms=["EEG"], required=False),
    ])
    assert families[:4] == ["scan_filter", "multimodal_query", "absence_query", "subject_multimodal_query"]


def test_sidecar_filters_by_key_and_operator() -> None:
    def one(field, operator, value=None):
        return _families(metadata_filters=[MetadataFilter(json_field=field, operator=operator, value=value)])
    assert one("RepetitionTime", "<", 1.0) == ["acquisition_parameter_query"]
    assert one("Manufacturer", "=", "GE") == ["scanner_query"]
    assert one("AcquisitionTime", "exists") == ["metadata_query"]
    assert one("ImagingFrequency", ">", 100) == ["json_numeric_query"]


def test_every_family_it_names_is_in_the_pool() -> None:
    pool = json.loads(_FEW_SHOT_PATH.read_text())
    source = open(example_families.__file__).read()
    import re
    named = set(re.findall(r'add\("([a-z_]+)"\)', source)) | {COMBINED}
    assert named <= set(pool), named - set(pool)


def test_an_excluded_diagnosis_is_an_absence_not_a_cohort() -> None:
    # "depression, excluding bipolar disorder": one cohort, one exclusion.
    families = _families(groups=[
        ParticipantGroup(group_name="patients", diagnosis_terms=["depression"]),
        ParticipantGroup(group_name="excluded", quantifier="none", diagnosis_terms=["bipolar disorder"]),
    ])
    assert families == ["concept_query", "absence_query", COMBINED]
