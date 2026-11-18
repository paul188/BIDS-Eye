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
from services.fulltext import uncertain_tier  # noqa: E402
from services.translate.sql_rewriter import rewrite_generated_sql  # noqa: E402

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


def test_a_stored_subject_count_replaces_the_file_table():
    sql = cohort_count.unknown_sql(["parkinsons_disease"], 40, subject_count="COALESCE(d.n_subjects, 0)")
    assert "bids_objects" not in sql and "GREATEST(COALESCE(d.n_subjects, 0)," in sql


def test_unknown_means_rows_say_nothing_and_the_dataset_is_large_enough():
    sql = cohort_count.unknown_sql(["parkinsons_disease"], 40)
    assert sql.startswith("(NOT (EXISTS(") and sql.rstrip().endswith(">= 40)")
    # A control label is recognised, so a dataset of controls only is known.
    assert "'control'" in sql and "'ctl'" in sql and "'pd'" in sql
    assert "COUNT(DISTINCT cohort_s.subject)" in sql


@pytest.mark.parametrize("codes, n", [([], 40), (["bad code!"], 40), (["adhd"], 0)])
def test_no_placeholder_without_a_usable_code_or_number(codes, n):
    assert cohort_count.placeholder(codes, n) is None


# ── the filter line ─────────────────────────────────────────────────────────────

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


def test_healthy_controls_are_not_counted_as_a_diagnosis():
    plan = QueryPlan(
        query_family="participant_filter", natural_language_summary="Summary.",
        groups=[ParticipantGroup(group_name="hc", diagnosis_terms=["healthy controls"], min_subjects=20)])
    resolved = [ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes={"healthy controls": ["healthy_control"]})]
    augmented = _build_augmented_question(plan, resolved)
    assert "patients_with" not in augmented
    assert "HAVING COUNT(DISTINCT o.subject) >= 20" in augmented


# ── the rewriter and the tier ───────────────────────────────────────────────────

# What the SQL model writes from the filter lines: the diagnosis condition and
# the copied placeholder.
MODEL_SQL = (
    "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
    "WHERE (EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id "
    "AND p.diagnosis IN ('parkinsons_disease', 'de_novo_parkinsons')) OR text_match('parkinson', 'short')) "
    "AND patients_with('parkinsons_disease, de_novo_parkinsons') >= 40 "
    "GROUP BY d.id"
)


def _final_sql(monkeypatch, tier: bool) -> str:
    monkeypatch.setenv("UNCERTAIN_TIER", "true" if tier else "false")
    augmented = _build_augmented_question(_plan(), [PD])
    return " ".join(rewrite_generated_sql(MODEL_SQL, augmented).sql.split())


def test_without_the_tier_only_counted_patients_pass(monkeypatch):
    sql = _final_sql(monkeypatch, tier=False)
    assert "patients_with" not in sql
    assert re.search(r"cohort_n\.diagnosis IN \('parkinsons_disease', 'de_novo_parkinsons'\).*>= 40", sql)
    assert "HAVING" not in sql.upper()
    assert uncertain_tier.LEVEL_COLUMN not in sql


def test_with_the_tier_an_unrecorded_count_is_uncertain_and_a_smaller_one_is_out(monkeypatch):
    sql = _final_sql(monkeypatch, tier=True)
    assert uncertain_tier.LEVEL_COLUMN in sql
    assert "cohort_k.diagnosis IS NULL" in sql
    assert cohort_count.REASON_UNKNOWN in sql
    # Not confirmed and not unknown is excluded: the count has no "possible" branch.
    state = re.search(r"CASE WHEN ev\.confirmed_(\d) THEN 0 WHEN ev\.probable_\1 THEN 1 ELSE 9 END", sql)
    assert state, "no confirmed / uncertain / excluded state for the count"


def test_the_tier_recognises_the_count_after_the_sql_was_printed_again():
    import sqlglot
    expanded = cohort_count.confirmed_sql(["parkinsons_disease"], 40)
    select = sqlglot.parse_one(f"SELECT 1 FROM bids_datasets AS d WHERE {expanded}", read="postgres")
    soft = uncertain_tier.classify(select.args["where"].this, "d")
    assert soft is not None and soft.kind == "patient_count"
    assert soft.known is None and "cohort_k" in soft.probable


def test_the_tier_reads_the_stored_subject_count_when_there_is_one(monkeypatch):
    from services.catalogue import schema_capabilities
    monkeypatch.setattr(schema_capabilities, "_has_subject_counts", True)
    soft = uncertain_tier._patient_count(cohort_count.confirmed_sql(["parkinsons_disease"], 40), "d")
    assert "d.n_subjects" in soft.probable and "bids_objects" not in soft.probable


def test_the_filter_chips_can_read_the_query(monkeypatch):
    from services.results import query_filters
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    augmented = _build_augmented_question(_plan(), [PD])
    rewritten = rewrite_generated_sql(MODEL_SQL, augmented)
    chips = query_filters.describe(rewritten.filter_sql or rewritten.sql)
    assert chips, "no chips for a diagnosis with a patient count"


# ── "at least N participants" without a diagnosis ───────────────────────────────

_JOINED = ("SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
           "%s JOIN bids_participants p ON p.dataset_id = d.id %s GROUP BY d.id "
           "HAVING COUNT(DISTINCT p.participant_id) >= 100")


@pytest.mark.parametrize("side", ["LEFT", ""])
def test_a_join_on_participants_only_to_count_them_becomes_the_subject_count(monkeypatch, side):
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    sql = rewrite_generated_sql(_JOINED % (side, ""), "datasets with at least 100 participants").sql
    assert "bids_participants" not in sql
    assert re.search(r"HAVING COUNT\(DISTINCT o\.subject\) >= 100", sql)


def test_a_count_of_filtered_participants_is_left_alone(monkeypatch):
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    sql = rewrite_generated_sql(_JOINED % ("", "WHERE p.sex = 'female'"), "at least 100 women").sql
    assert "bids_participants" in sql and "p.participant_id" in sql


# ── the count is put in by the rewriter, not left to the model ──────────────────

OWN_COUNT_SQL = (
    "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
    "WHERE (EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id "
    "AND p.diagnosis IN ('parkinsons_disease', 'de_novo_parkinsons')) OR text_match('parkinson', 'short')) "
    "AND (SELECT COUNT(DISTINCT p.participant_id) FROM bids_participants p WHERE p.dataset_id = d.id "
    "AND p.diagnosis IN ('parkinsons_disease', 'de_novo_parkinsons')) >= 40 "
    "GROUP BY d.id%s"
)
_OWN_HAVING = (" HAVING (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id "
               "AND p.diagnosis = 'parkinsons_disease') >= 40")


@pytest.mark.parametrize("having", ["", _OWN_HAVING])
def test_the_models_own_count_is_replaced(monkeypatch, having):
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    augmented = _build_augmented_question(_plan(), [PD])
    sql = " ".join(rewrite_generated_sql(OWN_COUNT_SQL % having, augmented).sql.split())
    assert sql.count("cohort_n.diagnosis IN") == 1
    assert "COUNT(DISTINCT p.participant_id)" not in sql and "HAVING" not in sql.upper()
    # The diagnosis condition itself stays.
    assert "p.diagnosis IN ('parkinsons_disease', 'de_novo_parkinsons')" in sql


def test_the_count_is_added_when_the_model_wrote_none(monkeypatch):
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    augmented = _build_augmented_question(_plan(), [PD])
    plain = MODEL_SQL.replace(" AND patients_with('parkinsons_disease, de_novo_parkinsons') >= 40", "")
    assert "patients_with" not in plain
    sql = rewrite_generated_sql(plain, augmented).sql
    assert sql.count("cohort_n.diagnosis IN") == 1


def test_a_copied_count_is_not_added_twice(monkeypatch):
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    augmented = _build_augmented_question(_plan(), [PD])
    assert rewrite_generated_sql(MODEL_SQL, augmented).sql.count("cohort_n.diagnosis IN") == 1


def test_no_filter_line_no_count(monkeypatch):
    monkeypatch.setenv("UNCERTAIN_TIER", "false")
    sql = rewrite_generated_sql(OWN_COUNT_SQL % "", "datasets with Parkinson's patients").sql
    assert "cohort_n" not in sql and "COUNT(DISTINCT p.participant_id)" in sql
