"""The deterministic compiler for cohort plans (services/translate/participant_query_compiler.py)."""
from __future__ import annotations

import pytest

import preprocess
from preprocess import ParticipantGroup, QueryFamily, QueryPlan, RAGField, ResolvedTerms, ScanRequirement
from services.translate import participant_query_compiler as pqc

SCZ = ["schizophrenia", "schizophrenia_with_avh"]
HC = ["healthy_control", "healthy_volunteer"]
FMRI_LINE = ("EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id "
             "AND (o2.datatype = 'functional_mri' OR o2.suffix = 'fmri_bold'))")


def _plan(*groups, operator="AND", **fields) -> QueryPlan:
    return QueryPlan(
        query_family=QueryFamily.CONCEPT_QUERY if hasattr(QueryFamily, "CONCEPT_QUERY")
        else list(QueryFamily)[0],
        natural_language_summary="demo",
        groups=[ParticipantGroup(group_name=f"g{i}", diagnosis_terms=list(terms))
                for i, terms in enumerate(groups)],
        group_operator=operator,
        **fields,
    )


def _resolved(**term_to_codes) -> list:
    return [ResolvedTerms(field=RAGField.DIAGNOSIS,
                          term_to_codes={k.replace("_", " "): v for k, v in term_to_codes.items()})]


def _augmented(*lines) -> str:
    return ("[Join paths — table routing for EXISTS subqueries]\n  bids_objects (alias o2)\n\n"
            "[Resolved DB filters — copy each filter expression verbatim into the SQL WHERE clause]\n"
            + "\n".join(lines) + "\n\nFind datasets.")


RESOLVED = _resolved(schizophrenia=SCZ, healthy_controls=HC, adhd=["adhd"])


def test_two_groups_are_two_requirements() -> None:
    result = pqc.compile_plan(_plan(["schizophrenia"], ["ADHD"]), RESOLVED)
    assert result.sql is not None and result.reason == ""
    sql = result.sql
    assert sql.startswith("SELECT {{COLS}}\nFROM bids_datasets d\n")
    assert sql.endswith("GROUP BY d.id")
    assert sql.count("EXISTS (SELECT 1 FROM bids_participants p_cohort") == 2
    assert "p_cohort.diagnosis IN ('schizophrenia', 'schizophrenia_with_avh')" in sql
    assert "p_cohort.diagnosis IN ('adhd')" in sql
    assert ") AND (" in sql
    # Each cohort keeps its own short-field text fallback.
    assert sql.count("text_match(") == 2


def test_the_group_operator_is_kept() -> None:
    sql = pqc.compile_plan(_plan(["schizophrenia"], ["ADHD"], operator="OR"), RESOLVED).sql
    assert sql is not None and ") OR (" in sql and ") AND (" not in sql


def test_diagnoses_within_one_group_are_alternatives() -> None:
    sql = pqc.compile_plan(_plan(["schizophrenia", "ADHD"]), RESOLVED).sql
    assert sql is not None
    assert sql.count("bids_participants p_cohort") == 1
    assert "IN ('schizophrenia', 'schizophrenia_with_avh', 'adhd')" in sql


def test_a_single_diagnosis_compiles() -> None:
    sql = pqc.compile_plan(_plan(["schizophrenia"]), RESOLVED).sql
    assert sql is not None and sql.count("bids_participants p_cohort") == 1


def test_terms_match_whatever_their_case() -> None:
    assert pqc.compile_plan(_plan(["Schizophrenia "]), RESOLVED).sql is not None


def test_scan_conditions_come_from_the_augmented_question() -> None:
    plan = _plan(["schizophrenia"], ["ADHD"],
                 scan_requirements=[ScanRequirement(scan_terms=["fMRI"])])
    augmented = _augmented("  diagnosis: (EXISTS (...) OR text_match('x', 'short'))",
                           f'  scan "fMRI": {FMRI_LINE}')
    sql = pqc.compile_plan(plan, RESOLVED, augmented).sql
    assert sql is not None
    assert f"WHERE {FMRI_LINE} AND ((EXISTS" in sql
    # The cohorts stay two requirements next to the scan: no flattened IN list.
    assert sql.count("bids_participants p_cohort") == 2


@pytest.mark.parametrize("plan, augmented, reason", [
    (_plan(), None, "nothing to filter"),
    (_plan([]), None, "nothing to filter"),
    (_plan(["schizophrenia"], result_limit=10), None, "plan field result_limit"),
    (_plan(["schizophrenia"], order_by="size_gb"), None, "plan field order_by"),
    (_plan(["schizophrenia"], license_search="cc0"), None, "plan field license_search"),
    (_plan(["schizophrenia"], scan_requirements=[ScanRequirement(scan_terms=["EEG"], required=False)]),
     _augmented('  scan "EEG": EXISTS (...)'), "negated scan without a filter line"),
    (_plan(["schizophrenia"], scan_requirements=[ScanRequirement(scan_terms=["EEG"], same_subject=True)]),
     _augmented('  scan "EEG": EXISTS (...)'), "scan requirement is per subject"),
    (_plan(["schizophrenia"], scan_requirements=[ScanRequirement(scan_terms=["EEG"])]),
     _augmented("  diagnosis: (...)"), "scan requirement without a resolved condition"),
    (_plan(["schizophrenia"]), _augmented("  diagnosis: (...)", "  institution: d.x = 1"),
     "filter line not supported: institution: d.x = 1"),
    (_plan(["schizophrenia"]), _augmented("  diagnosis: (...)") + "\n[VOCABULARY MISS — x]",
     "vocabulary miss not understood"),
    (_plan(["schizophrenia"]), _augmented("  diagnosis: (...)", "  [DB MISS — x]"), "augmented question has DB MISS"),
])
def test_what_needs_interpretation_is_declined_with_a_reason(plan, augmented, reason) -> None:
    result = pqc.compile_plan(plan, RESOLVED, augmented)
    assert result.sql is None
    assert result.reason == reason


def test_group_fields_without_a_filter_line_are_declined() -> None:
    plan = _plan(["schizophrenia"])
    plan.groups[0].handedness = ["left"]
    assert pqc.compile_plan(plan, RESOLVED).reason == "group field handedness"


def test_complete_filter_lines_are_anded_without_a_diagnosis_group() -> None:
    plan = _plan(scan_requirements=[ScanRequirement(scan_terms=["fMRI"])], species=["primate"])
    augmented = _augmented(
        f'  scan "fMRI": {FMRI_LINE}',
        "  species (complete, no other species condition): d.species ~* 'macaque'",
        "  participant sex: EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id AND LOWER(p2.sex) IN ('male', 'm'))",
        "  dataset text: text_match('reward')",
        "  dataset size: d.size_bytes <= 10",
        "  not searchable (add NO condition for these): for machine learning",
        "  subject count: HAVING COUNT(DISTINCT o.subject) >= 20 [apply this HAVING clause on the outer query, not in a subquery]",
    )
    sql = pqc.compile_plan(plan, [], augmented).sql
    assert sql is not None
    assert FMRI_LINE in sql and "d.species ~* 'macaque'" in sql and "LOWER(p2.sex)" in sql
    assert "text_match('reward')" in sql and "d.size_bytes <= 10" in sql
    assert "machine learning" not in sql
    assert sql.endswith("GROUP BY d.id\nHAVING COUNT(DISTINCT o.subject) >= 20")


def test_healthy_groups_add_no_condition() -> None:
    sql = pqc.compile_plan(_plan(["schizophrenia"], ["healthy controls"]), RESOLVED,
                           _augmented("  diagnosis: (...)")).sql
    assert sql.count("p_cohort.diagnosis IN") == 1 and "healthy" not in sql


def test_a_diagnosis_the_vocabulary_does_not_know_is_searched_in_its_group() -> None:
    augmented = _augmented("  diagnosis: (...)") + (
        "\n[VOCABULARY MISS — these terms are not in the known concept vocabulary: diagnosis: "
        "'bipolar syndrome'. Do NOT filter bids_participants.diagnosis or bids_objects.task/datatype/suffix "
        "for these terms — those columns store canonical codes, not free text. For free-text search use: "
        "text_match('bipolar syndrome', 'loose')]")
    sql = pqc.compile_plan(_plan(["bipolar syndrome", "schizophrenia"], operator="OR"), RESOLVED, augmented).sql
    assert "OR text_match('bipolar syndrome', 'short'))" in sql
    assert "'loose'" not in sql   # not ANDed as its own condition


def test_an_unknown_term_with_nothing_searchable_is_left_out() -> None:
    # "-" passes the term check but leaves no word to search: it once put
    # None into the OR list and the join raised TypeError.
    sql = pqc.compile_plan(_plan(["-", "schizophrenia"]), RESOLVED).sql
    assert sql is not None and "p_cohort.diagnosis IN ('schizophrenia'" in sql
    # Nothing else in the group: declined, not an empty "()" condition.
    assert pqc.compile_plan(_plan(["-"]), RESOLVED).sql is None


def test_recorded_facts_replace_their_words() -> None:
    plan = _plan(description_search="fMRIPrep", doi_required=True, min_sessions=3)
    sql = pqc.compile_plan(plan, [], _augmented("  dataset text: text_match('fmriprep')")).sql
    assert "dv.pipeline IN ('fmriprep')" in sql and "text_match" not in sql
    assert "dp.role = 'primary'" in sql
    assert "HAVING COUNT(DISTINCT s.session) < 3" in sql
    vocabulary = _augmented() + (
        "\n[VOCABULARY MISS — these terms are not in the known concept vocabulary: scan: 'MRIQC derivatives'. "
        "Do NOT filter bids_participants.diagnosis or bids_objects.task/datatype/suffix for these terms — "
        "those columns store canonical codes, not free text. For free-text search use: "
        "text_match('mriqc derivatives', 'loose')]")
    plan = _plan(scan_requirements=[ScanRequirement(scan_terms=["MRIQC derivatives"])])
    assert "dv.pipeline IN ('mriqc')" in pqc.compile_plan(plan, [], vocabulary).sql


def test_a_code_that_is_not_a_plain_code_is_never_written_into_sql() -> None:
    resolved = _resolved(schizophrenia=["schizophrenia", "x') OR 1=1 --"])
    result = pqc.compile_plan(_plan(["schizophrenia"]), resolved)
    assert result.sql is None and result.reason == "diagnosis term not resolved"
    result = pqc.compile_plan(_plan(["x') OR 1=1 --"]), RESOLVED)
    assert result.sql is None and result.reason == "diagnosis term not resolved"


def test_every_plan_field_is_known_or_must_be_default() -> None:
    """A field added to QueryPlan later declines until the compiler learns it."""
    for name, info in QueryPlan.model_fields.items():
        if name in pqc._KNOWN_PLAN_FIELDS or name in ("rag_requests",):
            continue
        default = info.get_default(call_default_factory=True)
        annotation = str(info.annotation)
        if "bool" in annotation:
            value = not default
        elif "List" in annotation or "list" in annotation:
            continue   # needs a typed element; covered by the reasons above
        elif "int" in annotation or "float" in annotation:
            value = 3
        elif "Literal" in annotation:
            continue
        else:
            value = "x"
        try:
            plan = _plan(["schizophrenia"], **{name: value})
        except Exception:
            continue   # the field does not accept the probe value
        assert pqc.compile_plan(plan, RESOLVED).reason == f"plan field {name}", name


def test_the_wrapper_returns_only_the_sql() -> None:
    assert pqc.compile_participant_groups(_plan(["schizophrenia"]), RESOLVED).startswith("SELECT {{COLS}}")
    assert pqc.compile_participant_groups(_plan(), RESOLVED) is None


def test_compiled_sql_passes_the_rewriter_and_the_guard() -> None:
    from services import query_guard
    from services.translate.sql_rewriter import rewrite_generated_sql
    plan = _plan(["schizophrenia"], ["healthy controls"],
                 scan_requirements=[ScanRequirement(scan_terms=["fMRI"])])
    augmented = _augmented(f'  scan "fMRI": {FMRI_LINE}')
    rewritten = rewrite_generated_sql(pqc.compile_plan(plan, RESOLVED, augmented).sql, augmented)
    assert "{{COLS}}" not in rewritten.sql and "text_match(" not in rewritten.sql
    assert query_guard.check_static(rewritten.sql) is None


# ── The model path keeps the groups apart too (preprocess._diagnosis_lines) ────

TERMS = {"schizophrenia": SCZ, "healthy controls": HC}


def test_required_groups_get_one_diagnosis_line_each() -> None:
    lines = preprocess._diagnosis_lines(_plan(["Schizophrenia"], ["healthy controls"]), TERMS)
    assert [terms for _, terms in lines] == [{"schizophrenia": SCZ}, {"healthy controls": HC}]
    assert all(label.startswith("diagnosis (participant group") for label, _ in lines)
    # The compiler skips these lines like the single one.
    assert all(pqc._DIAGNOSIS_LINE_RE.match(f"  {label}: x") for label, _ in lines)


def test_one_line_when_the_groups_are_alternatives_or_one_group() -> None:
    for plan in (_plan(["schizophrenia"], ["healthy controls"], operator="OR"),
                 _plan(["schizophrenia", "healthy controls"]),
                 _plan(["schizophrenia"], []),          # a term no group owns
                 _plan()):
        assert preprocess._diagnosis_lines(plan, TERMS) == [("diagnosis", TERMS)]


def test_augmented_question_has_a_line_per_required_group() -> None:
    plan = _plan(["schizophrenia"], ["healthy controls"])
    resolved = [ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes=TERMS)]
    text = preprocess._build_augmented_question(plan, resolved)
    assert text.count("  diagnosis (participant group") == 2
    assert "IN ('schizophrenia', 'schizophrenia_with_avh'))" in text
    assert "IN ('healthy_control', 'healthy_volunteer'))" in text
    assert "'schizophrenia_with_avh', 'healthy_control'" not in text
