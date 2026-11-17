""""At least N patients with X" counts the participants who have the diagnosis.
Datasets that do not record it go to the uncertain tier; datasets whose
participant rows show fewer are left out."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("LLM_preprocessor", "RAG"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

import cohort_count  # noqa: E402
from preprocess import (  # noqa: E402
    ParticipantGroup, QueryPlan, RAGField, ResolvedTerms, ScanRequirement,
    _build_augmented_question,
)
from services import uncertain_tier  # noqa: E402
from services.participant_query_compiler import compile_plan  # noqa: E402
from services.sql_rewriter import rewrite_generated_sql  # noqa: E402

PD = ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes={
    "Parkinson's disease": ["parkinsons_disease", "de_novo_parkinsons"]})
EEG = ResolvedTerms(field=RAGField.DATATYPE, term_to_codes={"EEG": ["electroencephalography"]})


def _plan(minimum=40, diagnosis=("Parkinson's disease",), scans=()):
    return QueryPlan(
        query_family="participant_filter", natural_language_summary="Summary.",
        groups=[ParticipantGroup(group_name="patients", diagnosis_terms=list(diagnosis),
                                 min_subjects=minimum)],
        scan_requirements=[ScanRequirement(scan_terms=list(scans))] if scans else [],
    )


# ── the count itself ────────────────────────────────────────────────────────────

def test_the_placeholder_and_its_expansion():
    text = cohort_count.placeholder(["parkinsons_disease", "de_novo_parkinsons"], 40)
    assert text == "patients_with('parkinsons_disease, de_novo_parkinsons') >= 40"
    sql = cohort_count.expand(f"SELECT 1 FROM bids_datasets d WHERE {text}")
    assert "patients_with" not in sql
    assert "cohort_n.diagnosis IN ('parkinsons_disease', 'de_novo_parkinsons')" in sql
    assert sql.rstrip().endswith(">= 40")
    assert cohort_count.parse_expanded(sql.split("WHERE ", 1)[1]) == (
        ["parkinsons_disease", "de_novo_parkinsons"], 40)


def test_more_than_is_one_more():
    sql = cohort_count.expand("WHERE patients_with('adhd') > 39")
    assert sql.rstrip().endswith(">= 40")


def test_group_columns_count_under_the_names_of_the_vocabulary():
    labels = cohort_count.labels_for(["parkinsons_disease"])
    assert {"pd", "parkinson", "parkinson's disease"} <= set(labels)
    sql = cohort_count.confirmed_sql(["parkinsons_disease"], 40)
    assert "LOWER(cohort_g.key) IN ('group'" in sql and "'pd'" in sql
    # Rows without extra columns hold JSON null, which has no keys to read.
    assert "JSON_TYPEOF(cohort_n.extra) = 'object'" in sql


def test_unknown_means_rows_say_nothing_and_the_dataset_is_large_enough():
    sql = cohort_count.unknown_sql(["parkinsons_disease"], 40)
    assert sql.startswith("(NOT (EXISTS(") and sql.rstrip().endswith(">= 40)")
    # A control label is recognised, so a dataset of controls only is known.
    assert "'control'" in sql and "'ctl'" in sql and "'pd'" in sql
    assert "COUNT(DISTINCT cohort_s.subject)" in sql


@pytest.mark.parametrize("codes, n", [([], 40), (["bad code!"], 40), (["adhd"], 0)])
def test_no_placeholder_without_a_usable_code_or_number(codes, n):
    assert cohort_count.placeholder(codes, n) is None


# ── the filter line and the compiler ────────────────────────────────────────────

def test_the_filter_line_counts_patients_and_not_all_subjects():
    augmented = _build_augmented_question(_plan(), [PD])
    line = next(l for l in augmented.splitlines() if l.startswith("  patients ("))
    assert line.endswith("patients_with('parkinsons_disease, de_novo_parkinsons') >= 40")
    assert "add no other count" in line
    assert "COUNT(DISTINCT o.subject)" not in augmented


def test_a_minimum_without_a_diagnosis_is_still_the_size_of_the_dataset():
    plan = QueryPlan(query_family="participant_filter", natural_language_summary="Summary.",
                     groups=[ParticipantGroup(group_name="all", min_subjects=50)],
                     description_search="memory")
    augmented = _build_augmented_question(plan, [])
    assert "HAVING COUNT(DISTINCT o.subject) >= 50" in augmented
    assert "patients_with" not in augmented


def test_the_compiler_builds_the_count_without_the_sql_model():
    plan = _plan(scans=("EEG",))
    resolved = [PD, EEG]
    compiled = compile_plan(plan, resolved, _build_augmented_question(plan, resolved))
    assert compiled.sql, compiled.reason
    where = " ".join(compiled.sql.split("WHERE", 1)[1].split("GROUP BY")[0].split())
    assert "patients_with('parkinsons_disease, de_novo_parkinsons') >= 40" in where
    assert "electroencephalography" in where and "HAVING" not in compiled.sql


def test_two_groups_each_with_its_own_count():
    plan = QueryPlan(
        query_family="participant_filter", natural_language_summary="Summary.",
        groups=[ParticipantGroup(group_name="pd", diagnosis_terms=["Parkinson's disease"], min_subjects=40),
                ParticipantGroup(group_name="hc", diagnosis_terms=["healthy controls"], min_subjects=20)])
    resolved = [ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes={
        "Parkinson's disease": ["parkinsons_disease"], "healthy controls": ["healthy_control"]})]
    compiled = compile_plan(plan, resolved, _build_augmented_question(plan, resolved))
    assert compiled.sql, compiled.reason
    assert "patients_with('parkinsons_disease') >= 40" in compiled.sql
    assert "patients_with('healthy_control') >= 20" in compiled.sql


# ── the rewriter and the tier ───────────────────────────────────────────────────

def _final_sql(monkeypatch, tier: bool) -> str:
    monkeypatch.setenv("UNCERTAIN_TIER", "true" if tier else "false")
    plan = _plan()
    augmented = _build_augmented_question(plan, [PD])
    compiled = compile_plan(plan, [PD], augmented)
    assert compiled.sql, compiled.reason
    return " ".join(rewrite_generated_sql(compiled.sql, augmented).sql.split())


def test_without_the_tier_only_counted_patients_pass(monkeypatch):
    sql = _final_sql(monkeypatch, tier=False)
    assert "patients_with" not in sql
    assert re.search(r"cohort_n\.diagnosis IN \('parkinsons_disease', 'de_novo_parkinsons'\).*>= 40", sql)
    assert "HAVING" not in sql.upper()
    assert uncertain_tier.LEVEL_COLUMN not in sql


def test_with_the_tier_an_unrecorded_count_is_uncertain_and_a_smaller_one_is_out(monkeypatch):
    sql = _final_sql(monkeypatch, tier=True)
    assert uncertain_tier.LEVEL_COLUMN in sql
    # The count is one of the soft conditions: confirmed, or unknown with its reason.
    assert "cohort_k.diagnosis IS NULL" in sql
    assert cohort_count.REASON_UNKNOWN in sql
    # Its state has no "possible" branch: not confirmed and not unknown is excluded.
    state = re.search(r"CASE WHEN ev\.confirmed_(\d) THEN 0 WHEN ev\.probable_\1 THEN 1 ELSE 9 END", sql)
    assert state, "no confirmed / uncertain / excluded state for the count"


def test_the_tier_recognises_the_count_after_the_sql_was_printed_again():
    import sqlglot
    expanded = cohort_count.confirmed_sql(["parkinsons_disease"], 40)
    select = sqlglot.parse_one(f"SELECT 1 FROM bids_datasets AS d WHERE {expanded}", read="postgres")
    soft = uncertain_tier.classify(select.args["where"].this, "d")
    assert soft is not None and soft.kind == "patient_count"
    assert soft.known is None and "cohort_k" in soft.probable
    assert soft.keep_sql().count("cohort_n") >= 1 and "cohort_k" in soft.keep_sql()


def test_the_final_sql_passes_the_static_guard(monkeypatch):
    from services import query_guard
    sql = _final_sql(monkeypatch, tier=True)
    assert query_guard.check_static(sql) is None


def test_the_filter_chips_can_read_the_query(monkeypatch):
    from services import query_filters
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    plan = _plan()
    augmented = _build_augmented_question(plan, [PD])
    rewritten = rewrite_generated_sql(compile_plan(plan, [PD], augmented).sql, augmented)
    chips = query_filters.describe(rewritten.filter_sql or rewritten.sql)
    assert chips, "no chips for a diagnosis with a patient count"
