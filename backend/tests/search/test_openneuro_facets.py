"""Filters that OpenNeuro's search page offers and BIDS-Eye missed
(benchmarks/openneuro_facets): study design and participant sex."""

from __future__ import annotations

from services.translate.sql_rewriter import SqlRewriteService

SERVICE = SqlRewriteService()


# ── study design ──────────────────────────────────────────────────────────────

def _augmented(structure, available=("study_longitudinal",), description_search=None) -> str:
    from preprocess import QueryPlan, _build_augmented_question

    plan = QueryPlan(
        natural_language_summary="Find datasets with the given study design.",
        study_structure=structure,
        description_search=description_search,
    )
    return _build_augmented_question(plan, [], available_columns=set(available))


def test_longitudinal_reads_the_uploader_label_first() -> None:
    augmented = _augmented("longitudinal")
    assert ("  study design: (NOT (COALESCE(d.study_longitudinal, '') ~* 'cross' AND "
            "COALESCE(d.study_longitudinal, '') !~* 'longitudinal') AND "
            "(d.study_longitudinal ~* 'longitudinal' OR text_match('longitudinal')))") in augmented


def test_cross_sectional_excludes_datasets_labelled_longitudinal() -> None:
    augmented = _augmented("cross-sectional")
    assert "COALESCE(d.study_longitudinal, '') ~* 'longitudinal' AND COALESCE(d.study_longitudinal, '') !~* 'cross'" \
        in augmented
    assert "(d.study_longitudinal ~* 'cross' OR text_match('cross-sectional'))" in augmented


def test_design_word_is_not_searched_twice() -> None:
    augmented = _augmented("longitudinal", description_search="longitudinal")
    assert "dataset text:" not in augmented


def test_other_description_words_stay() -> None:
    # Not an age group: "pediatric" is the age filter (backend/vocabulary/criteria.py AGE_GROUP).
    augmented = _augmented("longitudinal", description_search="bilingual")
    assert "dataset text: text_match('bilingual')" in augmented


def test_without_the_column_only_the_text_is_searched() -> None:
    augmented = _augmented("longitudinal", available=())
    assert "  study design: text_match('longitudinal')" in augmented
    assert "study_longitudinal" not in augmented


def test_no_design_asked_no_line() -> None:
    assert "  study design:" not in _augmented(None, description_search="pediatric")


# ── participant sex ───────────────────────────────────────────────────────────

# The column holds 'male', 'female' or 'other_diverse' (the crawler's
# remote/participant_codes.py); the rewriter writes whatever the SQL model
# typed as that code, from the vocabulary (backend/vocabulary/participant_codes.py).

def test_a_sex_literal_becomes_the_stored_code() -> None:
    for typed in ("female", "Female", "FEMALE", "F", "women"):
        sql = f"SELECT 1 FROM bids_participants p WHERE p.sex = '{typed}'"
        assert SERVICE._standard_sex_codes(sql) == (
            "SELECT 1 FROM bids_participants p WHERE p.sex = 'female'")


def test_sex_in_list_and_abbreviations() -> None:
    sql = "EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.sex IN ('M', 'male'))"
    assert SERVICE._standard_sex_codes(sql) == (
        "EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.sex = 'male')")
    both = "WHERE p.sex IN ('F', 'female', 'M')"
    assert SERVICE._standard_sex_codes(both) == "WHERE p.sex IN ('female', 'male')"


def test_sex_negation_kept() -> None:
    assert SERVICE._standard_sex_codes("WHERE p.sex <> 'Male'") == "WHERE p.sex <> 'male'"
    assert SERVICE._standard_sex_codes("WHERE p.sex NOT IN ('M', 'male')") == "WHERE p.sex <> 'male'"


def test_values_the_vocabulary_does_not_know_are_left() -> None:
    for sql in ("WHERE p.sex = 'other_diverse'", "WHERE p.sex = 'xyz'", "WHERE p.sex = '1'"):
        assert SERVICE._standard_sex_codes(sql) == sql


def test_wrapped_columns_and_null_checks_untouched() -> None:
    # LOWER(p.sex) IN ('female', 'f') is what stored interpretations from
    # before the codes were unified hold; it still finds 'female'.
    for sql in ("WHERE LOWER(p.sex) = 'female'", "WHERE LOWER(p2.sex) IN ('female', 'f')",
                "WHERE p.sex IS NOT NULL", "WHERE p.sex ILIKE 'f%'", "WHERE p.sex IN (SELECT x FROM t)"):
        assert SERVICE._standard_sex_codes(sql) == sql


def test_full_rewrite_keeps_counting_comparison() -> None:
    raw_sql = (
        "SELECT {{COLS}} FROM bids_datasets AS d "
        "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL "
        "WHERE EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'Female') "
        "GROUP BY d.id"
    )
    rewritten = SERVICE.rewrite_generated_sql(raw_sql, "datasets with female participants").sql
    assert "p.sex = 'female'" in rewritten


# ── number of participants ────────────────────────────────────────────────────

def test_participant_count_in_having_uses_both_sources() -> None:
    sql = ("SELECT {{COLS}} FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
           "AND o.subject IS NOT NULL GROUP BY d.id HAVING (SELECT COUNT(DISTINCT p.participant_id) "
           "FROM bids_participants AS p WHERE p.dataset_id = d.id) >= 100")
    rewritten = SERVICE._rewrite_participant_count_to_having(sql)
    assert rewritten.endswith(
        "HAVING GREATEST(COUNT(DISTINCT o.subject), (SELECT COUNT(DISTINCT p.participant_id) "
        "FROM bids_participants AS p WHERE p.dataset_id = d.id)) >= 100")
    assert SERVICE._rewrite_participant_count_to_having(rewritten) == rewritten


def test_conditional_participant_count_untouched() -> None:
    sql = ("SELECT 1 FROM bids_datasets AS d GROUP BY d.id HAVING (SELECT COUNT(DISTINCT p.participant_id) "
           "FROM bids_participants AS p WHERE p.dataset_id = d.id AND p.sex = 'female') >= 50")
    assert SERVICE._rewrite_participant_count_to_having(sql) == sql


def test_participant_count_in_where_still_moves_to_having() -> None:
    sql = ("SELECT 1 FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
           "WHERE d.id IS NOT NULL AND (SELECT COUNT(DISTINCT p.participant_id) FROM bids_participants p "
           "WHERE p.dataset_id = d.id) >= 20 GROUP BY d.id")
    rewritten = SERVICE._rewrite_participant_count_to_having(sql)
    assert "HAVING COUNT(DISTINCT o.subject) >= 20" in rewritten
    assert "GREATEST" not in rewritten


# ── species words are not a README search ─────────────────────────────────────

def test_species_only_description_search_is_dropped() -> None:
    for words in ("mouse", "mice", "rats", "non-human primates", "rats and mice", "monkeys", "contain monkeys",
                  "find all datasets that contain monkeys", "baboons", "marmosets", "rhesus macaques",
                  "animal studies", "animal research", "data from rats"):
        assert "dataset text:" not in _augmented(None, available=(), description_search=words), words


def test_species_with_other_words_stays() -> None:
    for words in ("mouse visual cortex", "monkey grid cells", "visual cortex in rats", "memory"):
        assert "dataset text:" in _augmented(None, available=(), description_search=words), words


# ── a participants join next to the files join ────────────────────────────────

_CHILDREN_100 = (
    "SELECT d.id, COUNT(DISTINCT o.subject) AS subject_count FROM bids_datasets AS d "
    "LEFT JOIN bids_objects AS o ON o.dataset_id = d.id AND NOT o.subject IS NULL "
    "LEFT JOIN bids_participants AS p ON p.dataset_id = d.id WHERE p.age <= 12 "
    "GROUP BY d.id HAVING COUNT(DISTINCT p.participant_id) >= 100"
)


def test_participants_join_becomes_subqueries() -> None:
    rewritten = SERVICE._participants_join_to_subqueries(_CHILDREN_100)
    assert "JOIN bids_participants" not in rewritten
    assert ("WHERE EXISTS(SELECT 1 FROM bids_participants AS p WHERE p.dataset_id = d.id AND p.age <= 12)"
            in rewritten)
    assert ("HAVING (SELECT COUNT(DISTINCT p.participant_id) FROM bids_participants AS p "
            "WHERE p.dataset_id = d.id AND p.age <= 12) >= 100") in rewritten
    assert "LEFT JOIN bids_objects AS o" in rewritten


def test_participants_join_keeps_other_conditions_and_on_conditions() -> None:
    sql = ("SELECT d.id FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
           "JOIN bids_participants AS p ON p.dataset_id = d.id AND p.sex = 'female' "
           "WHERE d.species IS NULL AND p.age > 65 GROUP BY d.id")
    rewritten = SERVICE._participants_join_to_subqueries(sql)
    assert "JOIN bids_participants" not in rewritten
    assert "d.species IS NULL AND EXISTS(SELECT 1 FROM bids_participants AS p WHERE p.dataset_id = d.id " \
           "AND p.sex = 'female' AND p.age > 65)" in rewritten


def test_mixed_condition_leaves_query_alone() -> None:
    sql = ("SELECT d.id FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
           "LEFT JOIN bids_participants AS p ON p.dataset_id = d.id WHERE p.age > d.age_min GROUP BY d.id")
    assert SERVICE._participants_join_to_subqueries(sql) == sql


def test_participants_join_without_files_join_left_alone() -> None:
    sql = ("SELECT d.id FROM bids_datasets AS d JOIN bids_participants AS p ON p.dataset_id = d.id "
           "WHERE p.age < 12 GROUP BY d.id")
    assert SERVICE._participants_join_to_subqueries(sql) == sql


def test_full_rewrite_of_children_question() -> None:
    raw = _CHILDREN_100.replace("SELECT d.id, COUNT(DISTINCT o.subject) AS subject_count", "SELECT {{COLS}}")
    rewritten = SERVICE.rewrite_generated_sql(raw, "datasets with at least 100 children").sql
    assert "JOIN bids_participants" not in rewritten


def test_label_guard_survives_the_uncertain_tier_paper_evidence() -> None:
    """The tier ORs paper evidence onto text conditions; a dataset labelled
    'Cross-Sectional' must still be excluded from a longitudinal question."""
    import re as _re
    from services.uncertain import tier as uncertain_tier
    augmented = _augmented("longitudinal")
    condition = _re.search(r"  study design: (.+)", augmented).group(1)
    base = ("SELECT d.id FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
            f"WHERE {condition} GROUP BY d.id")
    from services.fulltext.text_search import expand_text_search
    tiered = uncertain_tier.apply(expand_text_search(base, papers=False).sql, papers=True)
    guard = "NOT (COALESCE(d.study_longitudinal, '') ~* 'cross' AND NOT COALESCE(d.study_longitudinal, '') ~* 'longitudinal')"
    where = tiered.split("WHERE", 1)[1]
    assert guard in where
    # the guard is a conjunct of its own, never inside an OR with the paper evidence
    for part in where.split(" OR "):
        assert "search_tsv_papers" not in part or "'cross'" not in part
