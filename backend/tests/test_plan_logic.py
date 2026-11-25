"""Logic the plan states must reach the SQL: hand-written plans, no model.

Found 2026-11-25 (reports/Logical clause separation datasets.md): four places
where the code lost an "or" or a "not" of a correct plan. The negated scan is
fixed; the other three are expected failures, strict, so that a fix has to
remove the mark.
"""

from __future__ import annotations

import pytest

import preprocess as pp
from preprocess import QueryPlan, RAGField, ResolvedTerms, ScanRequirement, Topic
from preprocess.topics import _detach_topic_terms
from services.translate import participant_query_compiler as pqc

CODES = {
    RAGField.DATATYPE: {"EEG": ["electroencephalography"], "MEG": ["magnetoencephalography"],
                        "fMRI": ["functional_mri"], "MPRAGE": ["anatomical_mri"],
                        "MP2RAGE": ["anatomical_mri"], "DTI": ["diffusion_mri"],
                        "structural scan": ["anatomical_mri"], "T1w images": ["anatomical_mri"],
                        "T1 weighted images": ["anatomical_mri"]},
    RAGField.SUFFIX: {"EEG": ["eeg"], "MEG": ["meg"], "fMRI": ["fmri_bold"],
                      "MPRAGE": ["t1_weighted_mri"], "MP2RAGE": ["mp2rage"],
                      "task fMRI": ["fmri_bold", "bold_map"], "attention": ["att"],
                      "structural scan": ["t2_weighted_mri"], "T1w images": ["t1_weighted_mri"]},
    RAGField.TASK: {"grasping": ["grasping_task"], "reaching": ["reaching_task"],
                    "pointing": ["pointing_task"], "resting state": ["resting_state"], "attention": ["attention_task"]},
}


def _plan(*requirements: ScanRequirement, **fields) -> QueryPlan:
    return QueryPlan(query_family="scan_filter", natural_language_summary="Summary.",
                     scan_requirements=list(requirements), **fields)


def _need(*terms: str, **fields) -> ScanRequirement:
    return ScanRequirement(scan_terms=list(terms), **fields)


def _without(*terms: str, **fields) -> ScanRequirement:
    return ScanRequirement(scan_terms=list(terms), required=False, **fields)


def _resolved(plan: QueryPlan) -> list:
    asked = {t for r in plan.scan_requirements for t in r.scan_terms}
    return [ResolvedTerms(field=field, term_to_codes={t: c for t, c in codes.items() if t in asked})
            for field, codes in CODES.items()]


def _augmented(plan: QueryPlan, unresolved=()) -> str:
    return pp._build_augmented_question(
        plan, _resolved(plan), unresolved_terms={"scan": list(unresolved)} if unresolved else None)


def _scan_lines(augmented: str) -> dict:
    """label → condition of the scan lines."""
    lines = [line.strip() for line in augmented.splitlines()
             if line.startswith(("  scan", "  not scan"))]
    return dict(line.split(": ", 1) for line in lines)


def _sql(plan: QueryPlan, unresolved=()) -> pqc.Compiled:
    return pqc.compile_plan(plan, _resolved(plan), _augmented(plan, unresolved))


# ── 1. A negated scan ────────────────────────────────────────────────────────

def test_a_negated_scan_gets_a_negated_line_and_compiles() -> None:
    plan = _plan(_need("fMRI"), _without("EEG"))
    lines = _scan_lines(_augmented(plan))
    assert list(lines) == ['scan "fMRI"', 'not scan "EEG" (complete)']
    assert lines['not scan "EEG" (complete)'] == (
        "NOT (EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id AND o2.suffix = 'eeg'))")
    result = _sql(plan)
    assert result.sql is not None, result.reason
    assert "AND NOT (EXISTS (" in result.sql
    assert "electroencephalography" not in result.sql


def test_only_a_negated_scan_compiles() -> None:
    sql = _sql(_plan(_without("EEG"))).sql
    assert sql is not None and "WHERE NOT (EXISTS (" in sql


def test_a_sequence_is_excluded_without_the_rest_of_its_datatype() -> None:
    # qf27, "MPRAGE, not MP2RAGE": both resolve to the anatomical datatype too.
    plan = _plan(_need("MPRAGE"), _without("MP2RAGE"))
    excluded = _scan_lines(_augmented(plan))['not scan "MP2RAGE" (complete)']
    assert "o2.suffix = 'mp2rage'" in excluded and "anatomical_mri" not in excluded
    assert _sql(plan).sql is not None


def test_a_term_with_a_datatype_only_is_excluded_by_it() -> None:
    assert "o2.datatype = 'diffusion_mri'" in _scan_lines(
        _augmented(_plan(_without("DTI"))))['not scan "DTI" (complete)']


def test_a_negated_task_is_excluded_by_its_codes_and_its_words_in_the_short_fields() -> None:
    # As for an excluded diagnosis, a mention in a README alone does not exclude.
    condition = _scan_lines(_augmented(_plan(_need("fMRI"), _without("resting state"))))[
        'not scan "resting state" (complete)']
    assert condition.startswith("NOT (") and "o2.task = 'resting_state'" in condition
    assert "text_match('resting state', 'short')" in condition


@pytest.mark.parametrize("any_of", [True, False])
def test_several_negated_scans_of_one_kind_are_all_excluded(any_of) -> None:
    plan = _plan(_without("EEG", "MEG", any_of=any_of))
    assert list(_scan_lines(_augmented(plan))) == ['not scan "EEG" (complete)', 'not scan "MEG" (complete)']
    sql = _sql(plan).sql
    assert sql is not None and sql.count("NOT (EXISTS") == 2


def test_a_recording_and_its_task_are_excluded_only_together() -> None:
    # "EEG datasets without resting-state fMRI" keeps task fMRI and resting EEG.
    plan = _plan(_need("EEG"), _without("resting state", "fMRI"))
    lines = _scan_lines(_augmented(plan))
    condition = lines['not scan "resting state", "fMRI" (complete, only together)']
    assert condition.startswith("NOT (") and " AND EXISTS (" in condition
    assert _sql(plan).sql is not None


def test_a_term_that_is_also_required_is_not_excluded() -> None:
    # "fMRI, no resting-state fMRI": the plan repeats the shared modality.
    plan = _plan(_need("fMRI"), _without("resting state", "fMRI"))
    assert list(_scan_lines(_augmented(plan))) == ['scan "fMRI"', 'not scan "resting state" (complete)']
    assert _sql(plan).sql is not None


def test_an_exclusion_that_covers_a_required_scan_is_left_to_the_sql_model() -> None:
    # The vocabulary cannot tell task fMRI from fMRI: the exclusion would empty the search.
    plan = _plan(_need("fMRI"), _without("task fMRI"))
    assert list(_scan_lines(_augmented(plan))) == ['scan "fMRI"']
    assert _sql(plan) == pqc.Compiled(sql=None, reason="negated scan without a filter line")


@pytest.mark.parametrize("term", [
    "structural scan",      # l05: the datatype by name, T2w by similarity; "not T2w" kept 779 datasets
    "T1 weighted images",   # the anatomical datatype by similarity only
])
def test_a_term_the_codes_do_not_name_is_left_to_the_sql_model(term) -> None:
    plan = _plan(_need("fMRI"), _without(term))
    assert list(_scan_lines(_augmented(plan))) == ['scan "fMRI"']
    assert _sql(plan).reason == "negated scan without a filter line"


def test_a_filler_word_does_not_hide_a_listed_name() -> None:
    condition = _scan_lines(_augmented(_plan(_need("fMRI"), _without("T1w images"))))[
        'not scan "T1w images" (complete)']
    assert "o2.suffix = 't1_weighted_mri'" in condition and "anatomical_mri" not in condition


@pytest.mark.parametrize("field, term, listed", [
    ("suffix", "EEG", True), ("suffix", "T2w", True), ("suffix", "MP2-RAGE", True),
    ("suffix", "PETs", True), ("datatype", "structural scans", True), ("datatype", "MRI", True),
    ("suffix", "structural scan", False), ("suffix", "anatomical", False),
    ("datatype", "T1 weighted images", False),
])
def test_listed_names_of_the_vocabulary(field, term, listed) -> None:
    import retriever
    assert retriever.is_listed(field, term) is listed


def test_a_term_that_is_a_file_type_or_a_task_is_left_to_the_sql_model() -> None:
    plan = _plan(_need("fMRI"), _without("attention"))
    assert list(_scan_lines(_augmented(plan))) == ['scan "fMRI"']
    assert _sql(plan).reason == "negated scan without a filter line"


def test_a_negated_scan_outside_the_vocabulary_is_excluded_by_its_words() -> None:
    plan = _plan(_need("MPRAGE"), _without("vNavs"))
    augmented = _augmented(plan, unresolved=["vNavs"])
    assert "VOCABULARY MISS" not in augmented    # it would be searched for there
    assert _scan_lines(augmented)['not scan "vNavs" (complete)'].startswith("NOT (text_match('vnavs', 'short')")
    assert _sql(plan, ["vNavs"]).sql is not None


def test_not_both_in_one_subject_has_no_filter_line() -> None:
    plan = _plan(_without("EEG", "fMRI", same_subject=True))
    assert _scan_lines(_augmented(plan)) == {}
    assert _sql(plan).sql is None


def test_a_negated_scan_does_not_rank_or_contradict() -> None:
    plan = _plan(_need("fMRI"), _without("EEG"))
    results = {r.field.value: dict(r.term_to_codes) for r in _resolved(plan)}
    scores = {f: {c: 1.0 for codes in m.values() for c in codes} for f, m in results.items()}
    ranked = {f.code for f in pp.augment_with_rag(plan, results, scores=scores).scored_filters}
    assert ranked == {"functional_mri", "fmri_bold"}


# ── 2.–4. Still lost ─────────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="doi_required=False adds no condition")
def test_without_a_doi_is_a_condition_or_is_declined() -> None:
    sql = _sql(_plan(_need("fMRI"), doi_required=False)).sql
    assert sql is None or "NOT EXISTS (SELECT 1 FROM bids_dataset_papers" in sql


@pytest.mark.xfail(strict=True, reason="a topic removes a negated scan that repeats its head word")
def test_a_topic_keeps_the_negated_scan_that_repeats_it() -> None:
    plan = _plan(_need("fMRI"), _without("conflict"),
                 topics=[Topic(phrase="social conflict", head="conflict", qualifiers=["social"])])
    _detach_topic_terms(plan)
    assert [r.scan_terms for r in plan.scan_requirements if not r.required] == [["conflict"]]


@pytest.mark.xfail(strict=True, reason="a term of two any_of lists is kept in the last one only")
def test_a_term_of_two_lists_of_alternatives_stays_in_both() -> None:
    plan = _plan(_need("grasping", "reaching", any_of=True), _need("reaching", "pointing", any_of=True))
    assert list(_scan_lines(_augmented(plan))) == [
        'scan any of "grasping", "reaching" (alternatives, one is enough)',
        'scan any of "reaching", "pointing" (alternatives, one is enough)']
