"""Diagnosis terms the vocabulary covers only in part (2026-11-17).

"frontotemporal dementia" resolved to the dementia group (Alzheimer's, MCI,
Parkinson's MCI), and its text fallback searched "alzheimer" and "parkinson
disease": 60 results, 2 about FTD. Runs against the real value_mappings.yaml,
without the embedding fallback (production has no sentence_transformers).
"""

from __future__ import annotations

import pytest

import retriever
from retriever import _resolve_yaml_term
from text_terms import diagnosis_qualifier, diagnosis_words


@pytest.fixture(autouse=True)
def _no_embedding_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever, "_embedding_fallback_scored", lambda field, term: [])


# ── Vocabulary ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("term", [
    "frontotemporal dementia", "FTD", "bvFTD", "frontotemporal lobar degeneration",
    "semantic dementia", "primary progressive aphasia", "Pick's disease",
])
def test_ftd_and_its_spectrum_resolve_to_ftd(term: str) -> None:
    assert _resolve_yaml_term("diagnosis", term) == ["frontotemporal_dementia"]


def test_dementia_includes_ftd() -> None:
    codes = _resolve_yaml_term("diagnosis", "dementia")
    assert "frontotemporal_dementia" in codes and "alzheimers_disease" in codes


# ── Aphasia stays aphasia ─────────────────────────────────────────────────────

@pytest.mark.parametrize("term", ["aphasia", "post-stroke aphasia", "aphasic patients"])
def test_aphasia_is_not_frontotemporal_dementia(term: str) -> None:
    # The FTD entry lists "primary progressive aphasia"; fuzzy containment
    # would otherwise take a plain "aphasia" there.
    assert _resolve_yaml_term("diagnosis", term) == ["aphasia"]


def test_course_and_filler_words_name_no_disease() -> None:
    assert diagnosis_words("Early-onset Alzheimer's disease patients") == {"alzheimer"}
    assert diagnosis_words("treatment-resistant depression") == {"depression"}


# ── A broader label: its codes plus the words they miss ───────────────────────

@pytest.mark.parametrize("term, words", [
    ("Lewy body dementia", ["lewy", "body"]),
    ("vascular dementia", ["vascular"]),
    ("Parkinson's disease with dementia", ["dementia"]),
    ("postpartum depression", ["postpartum"]),
    ("childhood absence epilepsy", ["absence"]),
])
def test_words_the_codes_do_not_cover_are_required(term: str, words: list) -> None:
    codes = _resolve_yaml_term("diagnosis", term)
    assert codes, term
    assert diagnosis_qualifier(term, codes) == words


@pytest.mark.parametrize("term", [
    "frontotemporal dementia", "schizophrenia", "major depressive disorder",
    "chronic stroke", "Alzheimer's disease", "type 2 diabetes",
])
def test_covered_terms_need_nothing_more(term: str) -> None:
    assert diagnosis_qualifier(term, _resolve_yaml_term("diagnosis", term)) == []


# ── The filter line ───────────────────────────────────────────────────────────

def _augmented(term: str, codes: list) -> str:
    import preprocess as p

    plan = p.QueryPlan(query_family="concept_query", natural_language_summary=term,
                       groups=[p.ParticipantGroup(group_name="g", diagnosis_terms=[term])])
    resolved = [p.ResolvedTerms(field=p.RAGField.DIAGNOSIS, term_to_codes={term: codes})]
    return p._build_augmented_question(plan, resolved)


def test_ftd_line_searches_no_alzheimer_or_parkinson_words() -> None:
    line = next(ln for ln in _augmented("frontotemporal dementia", ["frontotemporal_dementia"]).splitlines()
                if ln.startswith("  diagnosis:"))
    assert "p2.diagnosis IN ('frontotemporal_dementia')" in line
    assert "frontotemporal dementia" in line
    assert "alzheimer" not in line and "parkinson" not in line
    assert " AND text_match" not in line


def test_broader_codes_get_the_missing_words_as_a_required_text() -> None:
    codes = _resolve_yaml_term("diagnosis", "Lewy body dementia")
    line = next(ln for ln in _augmented("Lewy body dementia", codes).splitlines()
                if ln.startswith("  diagnosis:"))
    assert line.endswith("AND text_match('lewy body'))")
    assert "p2.diagnosis IN (" in line


def test_a_known_diagnosis_without_participant_rows_keeps_its_line(monkeypatch) -> None:
    """No participant row says frontotemporal_dementia yet; the term must stay, not
    become a DB MISS that tells the SQL model to use the parent code."""
    import preprocess as p

    class Retriever:
        def retrieve_for_field_multi_scored(self, field, terms):
            return {t: [("frontotemporal_dementia", 1.0)] for t in terms}

        def db_verify_codes(self, field, codes):
            return [], list(codes)

    plan = p.QueryPlan(query_family="concept_query", natural_language_summary="FTD",
                       groups=[p.ParticipantGroup(group_name="g",
                                                  diagnosis_terms=["frontotemporal dementia"])])
    monkeypatch.setattr(p.pipeline, "create_query_plan", lambda question: plan)
    augmented = p.run_pipeline("frontotemporal dementia", retriever=Retriever()).augmented_question
    assert "DB MISS" not in augmented
    assert "p2.diagnosis IN ('frontotemporal_dementia')" in augmented


def test_a_diagnosis_keeps_its_own_code_when_only_a_narrower_one_has_rows(monkeypatch) -> None:
    """Participant rows say hippocampal_sclerosis, none says temporal_lobe_epilepsy. "TLE" must stay
    temporal lobe epilepsy: its code first, its words searched in the dataset text."""
    import preprocess as p

    from services.database_stats.database_stats import ValueCounts
    from services.database_stats.database_stats import database_stats

    monkeypatch.setattr(database_stats, "values",
                        ValueCounts({"diagnosis": {"hippocampal_sclerosis": 1}}))
    codes = _resolve_yaml_term("diagnosis", "TLE")
    assert codes == ["temporal_lobe_epilepsy", "hippocampal_sclerosis"]

    class Retriever:
        def retrieve_for_field_multi_scored(self, field, terms):
            return {t: [(code, 1.0) for code in codes] for t in terms}

    plan = p.QueryPlan(query_family="concept_query", natural_language_summary="TLE",
                       groups=[p.ParticipantGroup(group_name="g", diagnosis_terms=["TLE"])])
    monkeypatch.setattr(p.pipeline, "create_query_plan", lambda question: plan)
    augmented = p.run_pipeline("20 patients with TLE", retriever=Retriever()).augmented_question
    line = next(ln for ln in augmented.splitlines() if ln.startswith("  diagnosis:"))
    assert "DB MISS" not in augmented
    assert "p2.diagnosis IN ('temporal_lobe_epilepsy', " in line
    assert "temporal lobe epilepsy" in line
