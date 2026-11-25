from __future__ import annotations

import re

import pytest

sqlglot = pytest.importorskip("sqlglot")

from services.fulltext import tsquery, word_frequencies  # noqa: E402
from services.uncertain import text as tier_text, tier as ut  # noqa: E402
from services.uncertain.fields import age as tier_age  # noqa: E402
from services.fulltext.ranking import RankQuery  # noqa: E402

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
    counts = {"young": 300, "children": 120, "evaluated": 500, "resective": 30, "surgery": 90,
              "resective & surgery": 7}
    monkeypatch.setattr(word_frequencies, "_lookup_counts", lambda qs: [counts.get(q, 1000) for q in qs])
    monkeypatch.setenv("UNCERTAIN_POSSIBLE", "true")  # the tests cover both levels


def test_relax_keeps_the_rarer_half_and_drops_scope_labels() -> None:
    assert tier_text.relax_tsquery("young & children") == "children"
    assert tier_text.relax_tsquery("evaluated & resective & surgery") == "resective & surgery"
    assert tier_text.relax_tsquery("language:AB") == "language"
    assert tier_text.relax_tsquery("children") is None


def test_age_range_and_group_words() -> None:
    sql = ("(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
           "AND p2.age BETWEEN 10 AND 19) OR d.age_min >= 10)")
    low, high = tier_age.asked_age_range(sql)
    assert (low, high) == (10, 19)
    words = tier_age.age_group_words(low, high)
    assert "adolescents" in words and "children" not in words and "adults" not in words
    assert "adolescent" not in tier_age.age_group_words(18, 120)
    older = tier_age.age_group_words(60, 120)
    assert "elderly" in older and "adults" not in older and "aging" not in older
    assert "elderly" not in tier_age.age_group_words(18, 65)
    assert tier_age.age_group_words(0, 120) == []


def test_modality_stays_hard_and_text_becomes_a_tier() -> None:
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'young & children')")
    tiered = ut.apply(sql)
    assert "uncertain_level" in tiered
    # The modality filter is kept as it is; the text filter gets its weaker form.
    assert "o2.datatype = 'functional_mri'" in tiered
    assert "TO_TSQUERY('english', 'children')" in tiered
    assert "evidence.u0 < 9" in tiered
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
    from services.translate.sql_rewriter import build_page_sql
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'young & children')")
    page_sql, _ = build_page_sql(ut.apply(sql), [], True, 20, 0, rank_query=RankQuery("young | children"))
    assert "ORDER BY page_result.uncertain_level, page_result.relevance_score + page_result.text_rank DESC" in page_sql


def test_keyword_query_drops_question_noise() -> None:
    query = tsquery.keyword_query("Find EEG datasets of children with ADHD please")
    assert query == "eeg | children | child | adhd"      # every word in its other forms too
    assert tsquery.keyword_query("show me all datasets") is None


def test_weaker_form_does_not_repeat_the_file_check() -> None:
    expensive = ("(EXISTS(SELECT 1 FROM bids_objects AS bo WHERE bo.dataset_id = d.id AND bo.task = 'x') "
                 "OR d.search_tsv @@ to_tsquery('english', 'evaluated & resective & surgery'))")
    tiered = ut.apply(BASE.format(where=expensive))
    weaker = tiered.split("AS confirmed_0, (", 1)[1].split(") AS probable_0", 1)[0]
    assert weaker == "d.search_tsv @@ TO_TSQUERY('english', 'resective & surgery')"


def test_paper_text_is_second_tier_evidence() -> None:
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'placebo')")
    assert ut.apply(sql, papers=False) == sql            # one word, nothing to relax
    tiered = ut.apply(sql, papers=True)
    assert ("d.search_tsv_papers @@ TO_TSQUERY('english', 'placebo') "
            "AND NOT d.search_tsv @@ TO_TSQUERY('english', 'placebo')") in tiered
    assert "found only in a paper the dataset names" in tiered
    # Confirmation still uses the dataset's own text.
    assert "(d.search_tsv @@ TO_TSQUERY('english', 'placebo')) AS confirmed_0" in tiered
    assert "CASE WHEN ev.confirmed_0 THEN 0 WHEN ev.probable_0 THEN 1 ELSE 9 END AS u0" in tiered


def test_the_dataset_alias_of_the_query_is_used() -> None:
    sql = ("SELECT ds.id, COUNT(DISTINCT o.subject) AS subject_count FROM bids_datasets ds "
           "LEFT JOIN bids_objects o ON o.dataset_id = ds.id "
           "WHERE ds.search_tsv @@ to_tsquery('english', 'young & children') GROUP BY ds.id")
    for hoist in (False, True):
        tiered = ut.apply(sql, papers=True, hoist_terms=hoist)
        assert "uncertain_level" in tiered
        assert "WHERE ds.id = tier_base.id" in tiered
        assert not re.search(r"\bd\.", tiered)      # no leftover default alias


def test_each_condition_is_evaluated_once_per_evidence_kind() -> None:
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'placebo')")
    tiered = ut.apply(sql, papers=True)
    # State, reason and filter read the named flags; none repeats the condition.
    for flag in ("confirmed_0", "probable_0"):
        assert tiered.count(f"AS {flag}") == 1
    # The paper is the only weaker evidence here: probable_0 already says it.
    assert "paper_0" not in tiered
    assert "WHEN evidence.u0 = 1 THEN 'found only in a paper the dataset names'" in tiered
    assert "ELSE 9 ELSE 9" not in tiered and "THEN 9 ELSE 9" not in tiered
    # With a second kind of weaker evidence a flag tells the paper apart.
    two = ut.apply(BASE.format(where=f"{MODALITY} AND d.search_tsv @@ "
                                     "to_tsquery('english', 'young & children')"), papers=True)
    assert two.count("AS paper_0") == 1 and two.count("ev.paper_0") == 1
    assert "WHEN evidence.u0 = 1 AND evidence.paper_0 THEN 'found only in a paper" in two
    assert "WHEN evidence.u0 = 1 THEN 'matches only part of the search; missing: young'" in two


def test_hoisted_terms_name_each_tsquery_once_and_pass_the_guard() -> None:
    from services import query_guard
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'placebo')")
    plain = ut.apply(sql, papers=True, hoist_terms=False)
    hoisted = ut.apply(sql, papers=True, hoist_terms=True)
    assert "_qf_tier_terms" not in plain
    assert hoisted.count("TO_TSQUERY('english', 'placebo')") == 1
    assert plain.count("TO_TSQUERY('english', 'placebo')") > 3
    for tiered in (plain, hoisted):
        sqlglot.parse_one(tiered, read="postgres")
        assert query_guard.check_static(tiered) is None


def test_modifiers_are_not_kept_alone() -> None:
    # Even when "young" is rarer than "children".
    original = word_frequencies._lookup_counts
    word_frequencies._lookup_counts = lambda qs: [{"young": 54, "children": 63}.get(q, 1000) for q in qs]
    try:
        assert tier_text.relax_tsquery("young & children") == "children"
    finally:
        word_frequencies._lookup_counts = original


def test_top_n_and_ordered_questions_keep_their_shape() -> None:
    limited = BASE.format(where="d.age_max <= 12 AND d.search_tsv @@ to_tsquery('english', 'autism')") \
        + " ORDER BY d.size_bytes DESC LIMIT 10"
    assert ut.apply(limited) == limited


def test_sidecar_conditions_stay_hard() -> None:
    sidecar = ("(EXISTS(SELECT 1 FROM bids_objects AS bo WHERE bo.dataset_id = d.id "
               "AND CAST((bo.other_entities ->> 'MagneticFieldStrength') AS DOUBLE PRECISION) >= 9.4) "
               "OR d.search_tsv @@ to_tsquery('english', 'mri & brain'))")
    sql = BASE.format(where=sidecar)
    assert ut.apply(sql) == sql


def test_institution_conditions_stay_hard() -> None:
    # Loosened to the README, 'Harvard' matched mice from a lab that only
    # thanked Harvard Medical School for the knockout line.
    institution = ("(d.institutions && CAST(ARRAY['MGH 7T'] AS TEXT[]) "
                   "OR d.search_tsv @@ to_tsquery('english', 'harvard:AB'))")
    sql = BASE.format(where=f"{MODALITY} AND {institution}")
    assert ut.apply(sql) == sql
    with_text = BASE.format(where=f"{institution} AND d.search_tsv @@ to_tsquery('english', 'young & children')")
    rewritten = ut.apply(with_text)
    assert "'harvard'" not in rewritten and "'children'" in rewritten

    species = ("(d.species ~* 'mouse' OR (d.species IS NULL "
               "AND d.search_tsv @@ to_tsquery('english', 'mouse:AB')))")
    sql = BASE.format(where=f"{MODALITY} AND {species}")
    assert ut.apply(sql) == sql


def test_reasons_name_what_is_missing() -> None:
    sql = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'young & children')")
    assert "missing: young" in ut.apply(sql)
    scoped = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', 'language:AB')")
    assert '"language" appears only further down in the README' in ut.apply(scoped)

    synonyms = BASE.format(where=f"{MODALITY} AND d.search_tsv @@ to_tsquery('english', "
                                 "'parkinson:AB | (paralysis:AB <-> agitans:AB)')")
    assert '"parkinson" / "paralysis" … appears only further down in the README' in ut.apply(synonyms)


def test_and_in_nested_parentheses_is_split_into_conditions() -> None:
    """"spatiotemporal epilepsy" (live, 2026-11-20): the AND sat in two pairs of
    parentheses and stayed one condition, whose weaker form was any one of
    its text searches, so a paper saying "spatiotemporal" was enough."""
    where = ("(((EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
             "AND p2.diagnosis IN ('epilepsy')) OR d.search_tsv @@ to_tsquery('english', 'epilepsy:AB')) "
             "AND d.search_tsv @@ to_tsquery('english', 'spatiotemporal')))")
    tree = sqlglot.parse_one(BASE.format(where=where), read="postgres")
    parts = ut._flatten_and(tree.args["where"].this)
    assert len(parts) == 2
    tiered = ut.apply(BASE.format(where=where), papers=True)
    # Each condition is judged alone: one may be unsure, not both.
    assert "evidence.u0" in tiered and "evidence.u1" in tiered


def test_guarded_pair_in_extra_parentheses_stays_whole() -> None:
    pair = ("((d.search_tsv @@ to_tsquery('english', 'age') "
            "AND d.search_tsv_exact @@ to_tsquery('simple', 'ageing | aging')))")
    tree = sqlglot.parse_one(BASE.format(where=pair), read="postgres")
    assert len(ut._flatten_and(tree.args["where"].this)) == 1


def test_rare_never_stands_for_rare_diseases() -> None:
    # "rare" alone found the RARE MRI sequence of mouse studies (live, 2026-11-20).
    assert tier_text.relax_tsquery("rare & diseases") is None
