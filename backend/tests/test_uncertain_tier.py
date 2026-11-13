from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

from services import text_search, uncertain_tier as ut  # noqa: E402

BASE = (
    "SELECT d.id, d.name, d.accession_id, COUNT(DISTINCT o.subject) AS subject_count "
    "FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
    "WHERE {where} GROUP BY d.id"
)
MODALITY = ("EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
            "AND o2.datatype = 'functional_mri')")


@pytest.fixture(autouse=True)
def _counts(monkeypatch):
    # Rarity without a database: "children" is rarer than "young".
    counts = {"young": 300, "children": 120, "evaluated": 500, "resective": 30, "surgery": 90}
    monkeypatch.setattr(text_search, "_lookup_counts", lambda qs: [counts.get(q, 1000) for q in qs])
    monkeypatch.setenv("UNCERTAIN_POSSIBLE", "true")  # the tests cover both levels


def test_relax_keeps_the_rarer_half_and_drops_scope_labels() -> None:
    assert ut.relax_tsquery("young & children") == "children"
    assert ut.relax_tsquery("evaluated & resective & surgery") == "resective & surgery"
    assert ut.relax_tsquery("language:AB") == "language"
    assert ut.relax_tsquery("children") is None


def test_age_range_and_group_words() -> None:
    sql = ("(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
           "AND p2.age BETWEEN 10 AND 19) OR d.age_min >= 10)")
    low, high = ut.asked_age_range(sql)
    assert (low, high) == (10, 19)
    words = ut.age_group_words(low, high)
    assert "adolescents" in words and "children" in words and "adults" not in words
    assert "adolescent" not in ut.age_group_words(18, 120)
    assert ut.age_group_words(0, 120) == []


def test_modality_stays_hard_and_text_becomes_a_tier() -> None:
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'young & children')")
    tiered = ut.apply(sql)
    assert "uncertain_level" in tiered
    # The modality filter is kept as it is; the text filter gets its weaker form.
    assert "o2.datatype = 'functional_mri'" in tiered
    assert "TO_TSQUERY('english', 'children')" in tiered
    assert "ut.uncertain_level < 9" in tiered
    sqlglot.parse_one(tiered, read="postgres")


def test_age_condition_knows_missing_ages_and_contradictions() -> None:
    age = "(d.age_max <= 12)"
    sql = BASE.format(where=f"{MODALITY} AND {age} AND d.search_tsv @@ to_tsquery('english', 'adhd')")
    tiered = ut.apply(sql)
    assert "d.age_min IS NOT NULL OR d.age_max IS NOT NULL" in tiered   # recorded-ages check
    assert "ages not recorded" in tiered
    assert "children" in tiered                                          # README hint for the age group
    # Recorded ages that do not fit are a contradiction (9), never "possible".
    assert "ELSE 9 END" in tiered


def test_without_the_possible_tier_missing_fields_need_a_hint(monkeypatch) -> None:
    monkeypatch.setenv("UNCERTAIN_POSSIBLE", "false")
    sql = BASE.format(where="(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
                            "AND p2.diagnosis = 'adhd'))")
    tiered = ut.apply(sql)
    assert "THEN 2" not in tiered


def test_negated_and_unparseable_queries_are_left_alone() -> None:
    negated = BASE.format(where=f"{MODALITY} AND NOT d.search_tsv @@ to_tsquery('english', 'young & children')")
    assert ut.apply(negated) == negated
    assert ut.apply("SELECT 1") == "SELECT 1"
    assert ut.apply("not sql at all (") == "not sql at all ("


def test_page_query_puts_the_tier_first() -> None:
    from services.sql_rewriter import build_page_sql
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'young & children')")
    page_sql, _ = build_page_sql(ut.apply(sql), [], True, 20, 0, rank_tsquery="young | children")
    assert "ORDER BY page_result.uncertain_level, page_result.relevance_score + page_result.text_rank DESC" in page_sql


def test_keyword_query_drops_question_noise() -> None:
    query = text_search.keyword_query("Find EEG datasets of children with ADHD please")
    assert query == "eeg | children | adhd"
    assert text_search.keyword_query("show me all datasets") is None


def test_weaker_form_does_not_repeat_the_file_check() -> None:
    expensive = ("(EXISTS(SELECT 1 FROM bids_objects AS bo WHERE bo.dataset_id = d.id AND bo.task = 'x') "
                 "OR d.search_tsv @@ to_tsquery('english', 'evaluated & resective & surgery'))")
    tiered = ut.apply(BASE.format(where=expensive))
    weaker = tiered.split("THEN 0 WHEN (", 1)[1].split(") THEN 1", 1)[0]
    assert weaker == "d.search_tsv @@ TO_TSQUERY('english', 'resective & surgery')"


def test_paper_text_is_second_tier_evidence() -> None:
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'placebo')")
    assert ut.apply(sql, papers=False) == sql            # one word, nothing to relax
    tiered = ut.apply(sql, papers=True)
    assert "d.search_tsv_papers @@ TO_TSQUERY('english', 'placebo')" in tiered
    assert ut.PAPER_REASON in tiered
    # Confirmation still uses the dataset's own text.
    assert "CASE WHEN (d.search_tsv @@ TO_TSQUERY('english', 'placebo')) THEN 0" in tiered
