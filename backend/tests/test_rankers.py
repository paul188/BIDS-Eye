"""The order of the results is a ranker's (services/results/rankers.py); the
page query only writes its sort key."""

from __future__ import annotations

from services.fulltext.ranking import RankQuery
from services.results import rankers
from services.results.rankers import ResultColumns, SortTerm
from services.translate.sql_rewriter import build_page_sql
from services.uncertain import tier

BASE = ("SELECT d.id, d.name, COUNT(DISTINCT o.subject) AS subject_count FROM bids_datasets AS d "
        "LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
        "WHERE d.search_tsv @@ to_tsquery('english', 'young & children') GROUP BY d.id")
EVERYTHING = ResultColumns("r", has_evidence=True, has_judgment=True, has_other_words=True,
                           has_scores=True, has_subjects=True)


def _page_order(page_sql: str) -> str:
    tail = page_sql[page_sql.rfind(") AS page_result") + len(") AS page_result"):]
    return " ".join(tail.split("LIMIT")[0].split())


def test_the_default_order_is_written_in_one_place() -> None:
    assert rankers.order_by(rankers.DEFAULT_RANKER, EVERYTHING) == (
        "ORDER BY r.uncertain_level, "
        "CASE WHEN r.id = ANY(CAST(:_rr_top AS uuid[])) THEN 0 "
        "WHEN r.id = ANY(CAST(:_rr_low AS uuid[])) THEN 2 ELSE 1 END, "
        "array_position(CAST(:_rr_top AS uuid[]), r.id), "
        "r.other_words_only, r.relevance_score + r.text_rank DESC, "
        "r.subject_count DESC NULLS LAST, r.id")


def test_a_ranker_uses_only_what_the_rows_carry() -> None:
    assert rankers.order_by(rankers.DEFAULT_RANKER, ResultColumns("r")) == "ORDER BY r.id"
    assert rankers.order_by(rankers.DEFAULT_RANKER, ResultColumns("r", has_scores=True)) == (
        "ORDER BY r.relevance_score + r.text_rank DESC, r.id")


def test_every_key_ends_with_the_dataset_id() -> None:
    class Largest:
        def sort_key(self, columns):
            return [columns.subjects]

    class ById:
        def sort_key(self, columns):
            return [columns.dataset_id]

    assert rankers.order_by(Largest(), EVERYTHING) == "ORDER BY r.subject_count DESC NULLS LAST, r.id"
    assert rankers.order_by(ById(), EVERYTHING) == "ORDER BY r.id"


def test_another_ranker_changes_the_page_order_and_nothing_else() -> None:
    class ScoreBeforeEvidence:
        """A probable result with a high score before a confirmed one with a low score."""

        def sort_key(self, columns):
            key = [columns.score] if columns.has_scores else []
            if columns.has_evidence:
                key.append(columns.strength)
            return key

    tiered = tier.apply(BASE, papers=True)
    rank = RankQuery("young & children")
    default, params = build_page_sql(tiered, [], True, 20, 0, rank_query=rank)
    swapped, swapped_params = build_page_sql(tiered, [], True, 20, 0, rank_query=rank,
                                             ranker=ScoreBeforeEvidence())
    assert _page_order(default) == (
        "ORDER BY page_result.uncertain_level, page_result.relevance_score + page_result.text_rank DESC, "
        "page_result.subject_count DESC NULLS LAST, page_result.id")
    assert _page_order(swapped) == (
        "ORDER BY page_result.relevance_score + page_result.text_rank DESC, "
        "page_result.uncertain_level, page_result.id")
    assert params == swapped_params
    # Same rows, same columns: only the two ORDER BY clauses differ.
    cut = lambda sql: sql[:sql.rfind(") AS ranked")]  # noqa: E731
    assert cut(default) == cut(swapped)


def test_a_ranker_can_tell_apart_which_condition_is_unsure() -> None:
    """Unsure about the ages before unsure about the text, whatever the score."""
    class TextEvidenceFirst:
        def sort_key(self, columns):
            key = [columns.strength]
            text = columns.strength_for("text")
            if text is not None:
                key.append(text)
            assert columns.strength_for("sex") is None   # no such condition here
            return key + [columns.score]

    tiered = tier.apply(BASE.replace("WHERE ", f"WHERE {AGES} AND "), papers=True)
    assert "GREATEST(evidence.u0, evidence.u1) AS uncertain_level" in tiered
    assert ", evidence.u0 AS uncertain_level_age, evidence.u1 AS uncertain_level_text" in tiered
    assert tier.kind_level_columns(tiered) == {"age": "uncertain_level_age",
                                               "text": "uncertain_level_text"}
    page, _ = build_page_sql(tiered, [], True, 20, 0, rank_query=RankQuery("young & children"),
                             ranker=TextEvidenceFirst())
    assert _page_order(page) == (
        "ORDER BY page_result.uncertain_level, page_result.uncertain_level_text, "
        "page_result.relevance_score + page_result.text_rank DESC, page_result.id")


def test_conditions_of_one_kind_share_a_column() -> None:
    two_texts = tier.apply(BASE.replace("GROUP BY", "AND d.search_tsv @@ to_tsquery('english', "
                                        "'resting & state') GROUP BY"), papers=True)
    assert "GREATEST(evidence.u0, evidence.u1) AS uncertain_level_text" in two_texts
    assert tier.kind_level_columns(BASE) == {}


AGES = ("(d.age_max < 3 OR EXISTS(SELECT 1 FROM bids_participants AS p2 "
        "WHERE p2.dataset_id = d.id AND p2.age < 3))")


def test_a_piece_of_evidence_has_a_score_measured_or_by_its_strength() -> None:
    from services.evidence.base import DEFAULT_SCORES, Strength
    from services.evidence.possible import NotRecorded
    from services.evidence.probable import PaperOnly, TextSearch

    assert DEFAULT_SCORES[Strength.CONFIRMED] == 1 and DEFAULT_SCORES[Strength.CONTRADICTED] == 0
    assert PaperOnly((TextSearch("x"),)).score() == DEFAULT_SCORES[Strength.PROBABLE]
    assert NotRecorded("participant sex").score() == DEFAULT_SCORES[Strength.POSSIBLE]
    assert NotRecorded("ages", 0.33).score() == 0.33


def test_the_query_score_multiplies_the_scores_of_its_clauses() -> None:
    """A clause has the score of the piece of evidence that decides it; the
    query, being their AND, the product."""
    tiered = tier.apply(BASE.replace("WHERE ", f"WHERE {AGES} AND "), papers=True)
    ages = ("(CASE WHEN evidence.u0 = 0 THEN 1 WHEN evidence.u0 = 1 THEN 0.53 "
            "WHEN evidence.u0 = 2 THEN 0.33 ELSE 0 END)")
    # A paper and the dataset's own text are two sources: the stronger in
    # full, the other at SHARED_EVIDENCE_DISCOUNT, on the scale -ln(1 - score).
    paper = "CASE WHEN evidence.paper_1 THEN 0.5108 ELSE 0 END"
    own_text = "CASE WHEN evidence.text_1 THEN 0.4620 ELSE 0 END"
    text = (f"(CASE WHEN evidence.u1 = 0 THEN 1 WHEN evidence.u1 = 1 THEN 1 - EXP(-("
            f"GREATEST({paper}, {own_text}) + 0.5 * ({paper} + {own_text} - GREATEST({paper}, {own_text}))"
            f")) ELSE 0 END)")
    assert f"{ages} * {text} AS evidence_score" in tiered


def test_the_weakest_clause_can_count_for_more_than_the_others(monkeypatch) -> None:
    from services.uncertain import conditions

    monkeypatch.setattr(conditions, "CLAUSE_INDEPENDENCE", 0.0)
    tiered = tier.apply(BASE.replace("WHERE ", f"WHERE {AGES} AND "), papers=False)
    assert "COALESCE(LEAST((CASE WHEN evidence.u0 = 0 THEN 1" in tiered
    assert ", 0), 0) AS evidence_score" in tiered


def test_a_ranker_can_order_by_the_evidence_score() -> None:
    class ByEvidenceScore:
        def sort_key(self, columns):
            return [columns.evidence_score, columns.score]

    page, _ = build_page_sql(tier.apply(BASE, papers=True), [], True, 20, 0,
                             rank_query=RankQuery("young & children"), ranker=ByEvidenceScore())
    assert _page_order(page) == (
        "ORDER BY page_result.evidence_score DESC, "
        "page_result.relevance_score + page_result.text_rank DESC, page_result.id")


def test_a_term_that_is_not_a_plain_column_can_be_ranked_by() -> None:
    class RecordedFirst:
        def sort_key(self, columns):
            return [SortTerm(f"({columns.row}.uncertain_level = 0)", descending=True), columns.score]

    page, _ = build_page_sql(tier.apply(BASE, papers=True), [], True, 20, 0,
                             rank_query=RankQuery("young & children"), ranker=RecordedFirst())
    assert _page_order(page).startswith("ORDER BY (page_result.uncertain_level = 0) DESC, ")


def test_an_order_the_user_asked_for_is_not_the_rankers() -> None:
    class Never:
        def sort_key(self, columns):
            raise AssertionError("not asked")

    page, _ = build_page_sql(BASE + " ORDER BY d.size_bytes DESC LIMIT 10", [], False, 20, 0,
                             ranker=Never())
    assert _page_order(page) == ""
