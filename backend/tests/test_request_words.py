"""Fixes from the 70-request gold set (benchmarks/demand/results/2026-11-15_failure_causes_5f91230.md)."""
import preprocess
from services.uncertain import text as tier_text


def test_request_words_are_not_searched_as_text():
    assert preprocess._content_words("standard manner") == ""
    assert preprocess._content_words("very simple randomized") == "randomized"
    assert preprocess._content_words("pediatric") == "pediatric"


def test_a_generic_word_is_no_weak_evidence(monkeypatch):
    from services.fulltext import word_frequencies
    monkeypatch.setattr(word_frequencies, "_lookup_counts", lambda parts: [5 for _ in parts])
    assert tier_text.relax_tsquery("resting:AB & state:AB") is None
    assert tier_text.relax_tsquery("bipolar:AB & syndrome:AB") == "bipolar"
    assert tier_text.relax_tsquery("young & children") == "children"


def test_text_filters_the_sql_model_invented_are_dropped():
    from services.translate.sql_rewriter import SqlRewriteService
    svc = SqlRewriteService.__new__(SqlRewriteService)
    augmented = "[Resolved DB filters]\n  dataset text: text_match('pediatric')\n"
    sql = ("SELECT {{COLS}} FROM bids_datasets AS d WHERE d.license = 'CC0' AND text_match('pediatric') "
           "AND text_match('standard manner') AND (d.species ~* 'macaque' OR text_match('macaque | monkey', 'short')) "
           "GROUP BY d.id")
    out = svc._drop_invented_text_filters(sql, augmented)
    assert "standard manner" not in out
    assert "pediatric" in out and "macaque | monkey" in out and "CC0" in out
    only = "SELECT {{COLS}} FROM bids_datasets AS d WHERE text_match('dyslexia') GROUP BY d.id"
    assert svc._drop_invented_text_filters(only, "") == only


def test_wish_phrases_are_cut_from_the_text_search():
    assert preprocess._without_phrases("preprocessed in a Standard Manner", ["standard manner"]) == "preprocessed in a"


def test_free_text_no_dataset_contains_is_not_searched(monkeypatch):
    from services.fulltext import tsquery
    monkeypatch.setattr(tsquery, "count_matches", lambda alts, mode: [1 for _ in alts])
    assert preprocess._matches_too_few("adjacency matrices graphml")
    monkeypatch.setattr(tsquery, "count_matches", lambda alts, mode: [40 for _ in alts])
    assert not preprocess._matches_too_few("pediatric")
    monkeypatch.setattr(tsquery, "count_matches", lambda alts, mode: None)
    assert not preprocess._matches_too_few("pediatric")


def test_relaxed_query_drops_only_free_text_conditions():
    from services.translate.sql_rewriter import SqlRewriteService
    svc = SqlRewriteService.__new__(SqlRewriteService)
    sql = ("SELECT {{COLS}} FROM bids_datasets AS d WHERE d.license = 'CC0' AND text_match('isolated words') "
           "AND (d.species ~* 'macaque' OR text_match('macaque', 'short')) GROUP BY d.id")
    out = svc._without_text_only_filters(sql)
    assert out and "isolated words" not in out and "macaque" in out and "CC0" in out
    assert svc._without_text_only_filters(
        "SELECT {{COLS}} FROM bids_datasets AS d WHERE text_match('dyslexia') GROUP BY d.id") is None
    assert svc._without_text_only_filters(
        "SELECT {{COLS}} FROM bids_datasets AS d WHERE d.license = 'CC0' GROUP BY d.id") is None


# --- second round: the four regressions of the two-against-two comparison (2026-11-15) ---

def _svc():
    from services.translate.sql_rewriter import SqlRewriteService
    return SqlRewriteService.__new__(SqlRewriteService)


_FROM = "SELECT d.id FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id WHERE "


def test_healthy_is_never_a_filter():
    svc = _svc()
    one_row = svc._drop_healthy_filters(
        _FROM + "EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id AND p2.sex = 'male' "
        "AND p2.age >= 18 AND p2.diagnosis = 'healthy_volunteer') GROUP BY d.id")
    assert "healthy" not in one_row and "p2.sex = 'male'" in one_row and "p2.age >= 18" in one_row
    alone = svc._drop_healthy_filters(
        _FROM + "o.suffix = 'eeg' AND (EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
        "AND p2.diagnosis IN ('healthy_control', 'neurotypical')) OR text_match('healthy | control', 'short')) GROUP BY d.id")
    assert "bids_participants" not in alone and "healthy" not in alone and "o.suffix = 'eeg'" in alone


def test_a_disease_list_and_no_healthy_stay():
    svc = _svc()
    mixed = _FROM + ("EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
                     "AND p2.diagnosis IN ('alzheimers', 'healthy_control')) GROUP BY d.id")
    assert svc._drop_healthy_filters(mixed) == mixed
    negated = _FROM + ("NOT EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
                       "AND p2.diagnosis = 'healthy_control') GROUP BY d.id")
    assert svc._drop_healthy_filters(negated) == negated


def test_sessions_per_subject_is_one_grouped_subquery():
    svc = _svc()
    out = svc._decorrelate_grouped_exists(
        _FROM + "NOT EXISTS(SELECT 1 FROM bids_objects AS s WHERE s.dataset_id = d.id AND NOT s.subject IS NULL "
        "GROUP BY s.subject HAVING COUNT(DISTINCT s.session) < 3) GROUP BY d.id")
    assert "NOT d.id IN (SELECT s.dataset_id FROM bids_objects AS s WHERE NOT s.subject IS NULL " \
           "GROUP BY s.dataset_id, s.subject HAVING COUNT(DISTINCT s.session) < 3)" in out
    plain = _FROM + "EXISTS(SELECT 1 FROM bids_objects AS x WHERE x.dataset_id = d.id AND x.task = 'rest') GROUP BY d.id"
    assert svc._decorrelate_grouped_exists(plain) == plain


def test_healthy_terms_are_not_looked_up_as_a_diagnosis():
    assert preprocess._is_healthy_term("healthy controls")
    assert preprocess._is_healthy_term("normal adults")
    assert not preprocess._is_healthy_term("Alzheimer's disease")
    assert not preprocess._is_healthy_term("patients")
    plan = preprocess.QueryPlan.model_validate({
        "natural_language_summary": "x", "query_family": "participant_filter",
        "groups": [{"group_name": "a", "diagnosis_terms": ["healthy controls", "epilepsy"]}],
    })
    diagnosis = [r for r in preprocess.build_rag_requests(plan) if r.field == preprocess.RAGField.DIAGNOSIS]
    assert [r.terms for r in diagnosis] == [["epilepsy"]]


def test_sex_and_age_are_written_by_preprocessing():
    plan = preprocess.QueryPlan.model_validate({
        "natural_language_summary": "healthy adult male EEG during rest", "query_family": "participant_filter",
        "description_search": "healthy adult controls",
        "groups": [{"group_name": "a", "sex": ["male"]}],
    })
    lines = preprocess._participant_conditions(plan)
    assert lines == [
        "  participant sex: EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND LOWER(p2.sex) IN ('male', 'm'))",
        # adults: only datasets whose recorded ages are all below 18 are excluded
        "  participant age: (EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND p2.age >= 18) OR d.age_max >= 18 OR (d.age_min IS NULL AND d.age_max IS NULL AND NOT "
        "EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id AND p2.age IS NOT NULL)))",
    ]
    question = preprocess._build_augmented_question(plan, [])
    assert "text_match" not in question and "participant sex:" in question
    assert "not searchable (add NO condition for these): healthy" in question
    children = preprocess.QueryPlan.model_validate({
        "natural_language_summary": "x", "query_family": "comparison_query",
        "groups": [{"group_name": "a", "sex": ["male"], "age_min": 2, "age_max": 12}, {"group_name": "b", "sex": ["female"]}],
    })
    assert preprocess._participant_conditions(children) == [
        "  participant age: (EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND p2.age >= 2 AND p2.age <= 12) OR (d.age_min <= 12 AND d.age_max >= 2))"]


def test_a_stated_age_and_sex_describe_one_participant():
    plan = preprocess.QueryPlan.model_validate({
        "natural_language_summary": "male, 40 years old", "query_family": "participant_filter",
        "groups": [{"group_name": "a", "sex": ["male"], "age_min": 40, "age_max": 40}],
    })
    assert preprocess._participant_conditions(plan) == [
        "  participant sex and age: (EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND LOWER(p2.sex) IN ('male', 'm') AND p2.age >= 40 AND p2.age <= 40) "
        "OR (NOT EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id AND p2.age IS NOT NULL) "
        "AND EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id AND LOWER(p2.sex) IN ('male', 'm')) "
        "AND d.age_max >= 40 AND d.age_min <= 40))"]


def test_more_request_words():
    assert preprocess._content_words("analyse emotional state") == "emotional state"
    assert preprocess._content_words("BIDS-valid language") == "language"


def test_only_one_sex_means_every_participant():
    plan = preprocess.QueryPlan.model_validate({
        "natural_language_summary": "Find datasets containing only female participants.",
        "query_family": "participant_filter", "groups": [{"group_name": "a", "sex": ["female"]}],
    })
    (line,) = preprocess._participant_conditions(plan)
    assert "LOWER(p2.sex) IN ('female', 'f'))" in line
    assert "NOT EXISTS (SELECT 1 FROM bids_participants p3 WHERE p3.dataset_id = d.id AND (p3.sex IS NULL" in line
