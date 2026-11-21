from __future__ import annotations

import re

import pytest

import institution_index as ii
from institution_index import InstitutionIndex, text_alternatives, tokens

# A slice of the real bids_datasets.institutions values (production, Nov 2026).
NAMES = [
    "MIT", "mit", "Brains_Unlimited", "Brains Unlimited",
    "Leibniz Institute for Neurobiology; University of Marburg; Goldsmiths, University of London",
    "McGovern_Institute_for_Brain_Research_(3T1)",
    "Washington_University", "Washington University", "washington_university",
    "Washington University School of Medicine", "University of Washington", "Wash_U",
    "BeiJing_Normal_University", "Beijing Normal University", "Beijing Normal Universtiy/2773C4/",
    "Jiangsu Normal University", "South China Normal University",
    "University of Oxford", "UGent", "UZ_Gent",
    "Stanford University", "Stanford_Lucas_Center", "Stanford CNI",
    "Technische Universitaet Berlin", "Technische Universitat Berlin",
    "Queen's University", "Queens_University",
    "CAMRI, Nortwestern University", "Northwestern_University",
    "University of California", "University of California San Diego", "UCSD",
    "Radboudumc", "Penn_State_SLEIC", "NIH_FMRIF", "NIH FMRIF", "NIMH",
]


@pytest.fixture
def index() -> InstitutionIndex:
    return InstitutionIndex(NAMES)


def _names(index: InstitutionIndex, term: str) -> set:
    return {name for name, _ in index.resolve(term)}


def test_tokens_fold_spelling_variants() -> None:
    assert tokens("Technische Universität") == tokens("Technische_Universitaet")
    assert tokens("Queen's University") == tokens("Queens_University") == tokens("Queen s University")
    assert tokens("Max-Planck-Institut") == ("max", "planck", "institut")


def test_acronym_matches_whole_tokens_only(index: InstitutionIndex) -> None:
    found = _names(index, "MIT")
    assert {"MIT", "mit", "McGovern_Institute_for_Brain_Research_(3T1)"} <= found
    assert not any("Brains" in name or "Goldsmiths" in name for name in found)


def test_word_order_keeps_different_institutions_apart(index: InstitutionIndex) -> None:
    washu = _names(index, "Washington University")
    assert "University of Washington" not in washu
    assert {"Washington University", "Washington_University", "Wash_U"} <= washu
    assert _names(index, "University of Washington") == {"University of Washington"}


def test_typos_in_stored_names_are_tolerated(index: InstitutionIndex) -> None:
    found = _names(index, "Beijing Normal University")
    assert "Beijing Normal Universtiy/2773C4/" in found
    assert "Jiangsu Normal University" not in found
    assert "South China Normal University" not in found
    assert "CAMRI, Nortwestern University" in _names(index, "Northwestern")


def test_short_tokens_are_not_fuzzy(index: InstitutionIndex) -> None:
    assert "NIMH" not in _names(index, "NIH")


def test_distinctive_words_used_when_no_phrase_matches(index: InstitutionIndex) -> None:
    assert _names(index, "Oxford University") >= {"University of Oxford"}
    assert _names(index, "Ghent") == {"UGent", "UZ_Gent"}


def test_scores_rank_exact_above_alias_and_fuzzy(index: InstitutionIndex) -> None:
    scores = dict(index.resolve("MIT"))
    assert scores["MIT"] == ii.SCORE_PHRASE
    assert scores["McGovern_Institute_for_Brain_Research_(3T1)"] < scores["MIT"]


def test_unknown_or_empty_terms_resolve_to_nothing(index: InstitutionIndex) -> None:
    assert index.resolve("Cambridge") == []
    assert index.resolve("") == []
    assert index.resolve("the") == []


def test_too_vague_terms_are_ignored() -> None:
    many = InstitutionIndex([f"University Hospital {i}" for i in range(ii.MAX_NAMES_PER_TERM + 1)])
    assert many.resolve("University Hospital") == []


def test_parent_institution_finds_its_hospitals_but_not_the_reverse() -> None:
    # Production spellings (Nov 2026): no scan header says "Harvard" except BCH's.
    index = InstitutionIndex([
        "BCH, Harvard Medical School", "MGH 7T", "MGH-MR1L6", "Martinos_Center_Bay_3",
        "Athinoula A. Martinos Center for Biomedical Imaging", "Mclean Hospital", "MGB_IMG_BELM",
        "Brigham_Young_University", "MIT", "McGovern_Institute_for_Brain_Research_(3T1)",
        "Policlinico G. Martino",
    ])
    harvard = _names(index, "Harvard")
    assert harvard == {
        "BCH, Harvard Medical School", "MGH 7T", "MGH-MR1L6", "Martinos_Center_Bay_3",
        "Athinoula A. Martinos Center for Biomedical Imaging", "Mclean Hospital", "MGB_IMG_BELM",
    }
    assert _names(index, "Harvard Medical School") == harvard
    assert "Mclean Hospital" not in _names(index, "MGH")
    assert "massachusetts general hospital" in text_alternatives("Harvard")


def test_text_alternatives_drop_short_acronyms_and_keep_spelling() -> None:
    assert text_alternatives("MIT") == ["massachusetts institute of technology", "mcgovern institute"]
    assert text_alternatives("Queen's University") == ["Queen's University"]
    assert "Universität Zürich" in text_alternatives("Universität Zürich")


# ── Pipeline integration ──────────────────────────────────────────────────────

def _augmented(term_to_names: dict, available=("institutions",), unresolved=()) -> str:
    from preprocess import QueryPlan, RAGField, ResolvedTerms, _build_augmented_question

    plan = QueryPlan(
        query_family="combined_filter",
        natural_language_summary="Find datasets from the given institution.",
        institution_terms=list(term_to_names) + list(unresolved),
    )
    resolved = [ResolvedTerms(field=RAGField.INSTITUTION, term_to_codes=term_to_names)] if term_to_names else []
    return _build_augmented_question(
        plan, resolved,
        unresolved_terms={"institution": list(unresolved)} if unresolved else {},
        available_columns=set(available),
    )


def test_resolved_institution_becomes_exact_array_filter_with_text_backup() -> None:
    augmented = _augmented({"Stanford": ["Stanford University", "Stanford_Lucas_Center"]})
    assert (
        "institution \"Stanford\": (d.institutions && ARRAY['Stanford University', "
        "'Stanford_Lucas_Center']::text[] OR text_match('stanford', 'short'))"
    ) in augmented


def test_quotes_in_names_are_escaped() -> None:
    augmented = _augmented({"Queen's University": ["Queen's University"]})
    assert "ARRAY['Queen''s University']::text[]" in augmented


def test_unknown_institution_falls_back_to_full_text_and_not_vocabulary_miss() -> None:
    augmented = _augmented({}, unresolved=["Donders"])
    assert "institution \"Donders\": text_match('donders | donders institute" in augmented
    assert "VOCABULARY MISS" not in augmented


def test_without_the_column_only_text_search_is_used() -> None:
    augmented = _augmented({"Stanford": ["Stanford University"]}, available=())
    assert "d.institutions" not in augmented
    assert "institution \"Stanford\": text_match('stanford')" in augmented


def test_array_filter_survives_sql_rewrite() -> None:
    from services.translate.sql_rewriter import rewrite_generated_sql

    sql = (
        "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
        "AND o.subject IS NOT NULL WHERE (d.institutions && ARRAY['Queen''s University']::text[] "
        "OR text_match('queens university', 'short')) GROUP BY d.id"
    )
    rewritten = rewrite_generated_sql(sql, "").sql
    # sqlglot may render the cast as ::text[] or CAST(... AS TEXT[]); only the
    # overlap on the exact, quote-escaped name matters.
    assert re.search(r"d\.institutions && (?:CAST\()?ARRAY\['Queen''s University'\]", rewritten)
    assert "text_match" not in rewritten
    assert "search_tsv @@" in rewritten


def test_retriever_resolves_institutions_from_shared_index(tmp_path) -> None:
    import json

    from retriever import MetadataRetriever

    path = tmp_path / "name_index.json"
    path.write_text(json.dumps({"authors": [], "funding_sources": [], "institutions": ["Stale Name"]}))
    previous = ii._index
    try:
        ii.set_names(NAMES)
        retriever = MetadataRetriever(path)
        scored = retriever.retrieve_for_field_multi_scored("institution", ["Ghent", "Cambridge"])
        assert {name for name, _ in scored["Ghent"]} == {"UGent", "UZ_Gent"}
        assert "Cambridge" not in scored
    finally:
        ii._index = previous


def test_excluded_institution_keeps_datasets_without_recorded_institution() -> None:
    from preprocess import QueryPlan, RAGField, ResolvedTerms, _build_augmented_question, build_rag_requests

    plan = QueryPlan(
        query_family="combined_filter",
        natural_language_summary="Find fMRI datasets not collected at Stanford.",
        excluded_institution_terms=["Stanford"],
    )
    assert [r.terms for r in build_rag_requests(plan) if r.field == RAGField.INSTITUTION] == [["Stanford"]]
    resolved = [ResolvedTerms(field=RAGField.INSTITUTION, term_to_codes={"Stanford": ["Stanford University"]})]
    augmented = _build_augmented_question(plan, resolved, available_columns={"institutions"})
    assert (
        "not institution \"Stanford\": NOT COALESCE(d.institutions && "
        "ARRAY['Stanford University']::text[], false)"
    ) in augmented
    assert "  institution \"Stanford\"" not in augmented

    unknown = _build_augmented_question(plan, [], unresolved_terms={"institution": ["Stanford"]},
                                        available_columns={"institutions"})
    assert "not institution \"Stanford\": NOT text_match('stanford', 'short')" in unknown


def test_city_after_in_or_at_narrows_to_that_site() -> None:
    index = InstitutionIndex([
        "MPI_Leipzig", "Max Planck Institute for Human Cognitive and Brain Sciences",
        "Max Planck Institute for Biological Cybernetics", "Univ of Texas Austin",
    ])
    found = _names(index, "Max Planck Institute in Leipzig")
    assert found == {"MPI_Leipzig", "Max Planck Institute for Human Cognitive and Brain Sciences"}
    assert _names(index, "Univ of Texas at Austin") == {"Univ of Texas Austin"}
