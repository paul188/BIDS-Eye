"""Other words for a description search: "dead people" also finds "post-mortem".

The language model lists the words dataset authors use for the user's
everyday word (QueryPlan.description_alternatives); preprocess/ checks them
against the catalogue and adds them to the text filter.
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

import preprocess as pp  # noqa: E402
from services.results import query_filters  # noqa: E402
from services.translate import sql_rewriter  # noqa: E402
from services.fulltext import ranking, text_search, word_frequencies  # noqa: E402
from services import sql_tree  # noqa: E402
from services.uncertain import text as tier_text  # noqa: E402


def _plan(**fields) -> pp.QueryPlan:
    return pp.QueryPlan(query_family=pp.QueryFamily.DESCRIPTION_SEARCH,
                        natural_language_summary="datasets from deceased people", **fields)


def _counts(table, default=5):
    return lambda alternatives: [table.get(a, default) for a in alternatives]


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.setattr(word_frequencies, "_lookup_counts", lambda queries: None)


def test_alternatives_are_checked_against_the_catalogue() -> None:
    plan = _plan(description_search="dead people",
                 description_alternatives=["post-mortem", "ex vivo", "People, dead", "cadaver",
                                           "brain", "ex-vivo", "postmortem"])
    dropped = pp.check_description_alternatives(
        plan, _counts({"cadaver": 0, "brain": 900}))
    # "People, dead" repeats the user's words; "ex-vivo" repeats "ex vivo".
    assert plan.description_alternatives == ["post-mortem", "ex vivo", "postmortem"]
    assert dropped == {"cadaver": "no dataset uses it", "brain": "too common (900 datasets)"}


def test_an_alternative_the_search_cuts_to_one_word_is_dropped() -> None:
    # "epilepsy ONLY postictal": "after seizure" is searched as "seizure" and
    # gave 25 epilepsy datasets (live, 2026-11-23).
    plan = _plan(description_search="postictal",
                 description_alternatives=["after seizure", "post-event", "seizure recovery"])
    dropped = pp.check_description_alternatives(plan, _counts({}))
    assert plan.description_alternatives == ["post-event", "seizure recovery"]
    assert dropped == {"after seizure": "one word is left once the search drops its stop words"}


def test_without_a_description_search_nothing_is_added() -> None:
    plan = _plan(description_alternatives=["post-mortem"])
    pp.check_description_alternatives(plan, _counts({}))
    assert plan.description_alternatives == []
    assert "post-mortem" not in pp.augment_with_rag(plan, {}).augmented_question


def test_without_counts_alternatives_are_kept_unchecked() -> None:
    plan = _plan(description_search="dead people", description_alternatives=["post-mortem"] * 3)
    pp.check_description_alternatives(plan, lambda alternatives: None)
    assert plan.description_alternatives == ["post-mortem"]


# 'also': searched like 'near', but a dataset that has only the other words
# ranks after those with the user's own (services/fulltext/ranking.py).
CONDITION = ("(text_match('dead people') OR "
             "text_match('post-mortem | ex vivo | deceased', 'also'))")


def test_filter_line_adds_the_alternatives_to_the_users_words() -> None:
    plan = _plan(description_search="dead people",
                 description_alternatives=["post-mortem", "ex vivo", "deceased"])
    question = pp.augment_with_rag(plan, {}).augmented_question
    assert f"dataset text: {CONDITION}" in question


def test_words_no_dataset_says_stay_a_filter_when_they_have_alternatives(monkeypatch) -> None:
    monkeypatch.setattr(pp.augment, "_matches_too_few", lambda text: True)
    plan = _plan(description_search="dead people",
                 description_alternatives=["post-mortem", "ex vivo", "deceased"])
    assert f"dataset text: {CONDITION}" in pp.augment_with_rag(plan, {}).augmented_question
    # Without alternatives such words are not searched next to another filter, as before.
    alone = pp.augment_with_rag(_plan(description_search="dead people", name_search="brain"),
                                {}).augmented_question
    assert "dataset text:" not in alone and "not searchable" in alone


def test_words_no_dataset_says_stay_when_they_are_the_whole_question(monkeypatch) -> None:
    # "bonn" (2026-11-18) answered with all 1,721 datasets: its only condition
    # was dropped as matching too few, which left no condition at all.
    monkeypatch.setattr(pp.augment, "_matches_too_few", lambda text: True)
    question = pp.augment_with_rag(_plan(description_search="Bonn"), {}).augmented_question
    assert "dataset text: text_match('bonn')" in question and "not searchable" not in question


def test_the_search_the_chip_and_the_tier_read_the_condition() -> None:
    sql = text_search.expand_text_search(
        f"SELECT d.id FROM bids_datasets AS d WHERE {CONDITION} GROUP BY d.id",
        papers=False).sql
    assert "'dead & people'" in sql
    assert "(post <-> mortem) | (ex <1> vivo" in sql and "| deceased'" in sql

    chips = query_filters.describe(sql)
    assert len(chips) == 1 and chips[0].label == "dead people"
    assert "post mortem" in chips[0].detail and "ex vivo" in chips[0].detail

    # Only the user's own words get a weaker form, not the alternatives; and
    # "people" alone is none (without counts the longer word would be kept).
    weaker = tier_text.relax_text_condition(sql_tree.parse(sql), "d")
    relaxed = weaker.sql() if weaker else None
    assert relaxed is None or ("mortem" not in relaxed and "vivo" not in relaxed)
    assert relaxed is None or "'people'" not in relaxed


def test_rewriter_restores_alternatives_the_model_left_out() -> None:
    service = sql_rewriter.sql_rewrite_service
    condition = "(text_match('dead people') OR text_match('post-mortem | ex vivo', 'near'))"
    question = f"[Resolved DB filters]\n  dataset text: {condition}\n\nsummary"
    left_out = ("SELECT d.id FROM bids_datasets AS d WHERE text_match('dead people') "
                "AND d.license = 'CC0' GROUP BY d.id")
    restored = service._restore_description_alternatives(left_out, question)
    assert f"WHERE {condition} AND d.license" in restored
    copied = f"SELECT d.id FROM bids_datasets AS d WHERE {condition} GROUP BY d.id"
    assert service._restore_description_alternatives(copied, question) == copied
    # Left out altogether: left alone (the model used a table instead).
    absent = ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_derivatives AS dv "
              "WHERE dv.dataset_id = d.id AND dv.pipeline = 'fmriprep') GROUP BY d.id")
    assert service._restore_description_alternatives(absent, question) == absent
    plain = "[Resolved DB filters]\n  dataset text: text_match('complex')\n\nsummary"
    assert service._restore_description_alternatives(left_out, plain) == left_out


# ── Words no dataset says ─────────────────────────────────────────────────────

def _question(monkeypatch, **fields) -> str:
    monkeypatch.setattr(pp.augment, "_matches_too_few", lambda text: True)
    return pp.augment_with_rag(_plan(**fields), {}).augmented_question


def test_words_that_are_the_whole_question_are_searched_even_when_rare(monkeypatch) -> None:
    """Not searching them returned the whole catalogue ("datasets with astronauts")."""
    question = _question(monkeypatch, description_search="astronauts")
    assert "dataset text: text_match('astronauts')" in question
    assert "not searchable" not in question


def test_an_or_stays_an_or_when_it_is_the_whole_question(monkeypatch) -> None:
    question = _question(monkeypatch, description_search="drawings or art")
    line = next(ln for ln in question.splitlines() if "dataset text:" in ln)
    assert "drawings" in line and "art" in line and "drawings art" not in line
    assert "not searchable" not in question


def test_a_not_alone_does_not_count_as_a_filter(monkeypatch) -> None:
    question = _question(monkeypatch, description_search="astronauts but not sleep")
    line = next(ln for ln in question.splitlines() if "dataset text:" in ln)
    assert "astronauts" in line and "NOT" in line and "sleep" in line


def test_rare_words_next_to_another_filter_are_still_not_searched(monkeypatch) -> None:
    question = _question(monkeypatch, description_search="adjacency matrices graphml",
                         name_search="connectome")
    assert "dataset text:" not in question
    assert "not searchable (add NO condition for these): adjacency matrices graphml" in question


# ── Ranking: the user's own words first ───────────────────────────────────────

def _rank(condition: str) -> ranking.RankQuery:
    return text_search.expand_text_search(
        f"SELECT d.id FROM bids_datasets AS d WHERE {condition} GROUP BY d.id",
        papers=False).rank_query


def test_a_match_on_the_alternatives_only_ranks_below_the_users_words() -> None:
    rank = _rank("(text_match('test-retest reliability') OR "
                 "text_match('reproducibility | stability', 'also'))")
    assert "reproducibility" in rank.stemmed and "reliability" in rank.stemmed and rank.exact is None
    assert rank.alternative == "(test <-> retest) & reliability"
    # Its own column, not part of the rank: the ranker says where such a dataset goes.
    assert ranking.other_words_only_expression("ft", rank) == (
        "(ft.search_tsv @@ to_tsquery('english', '(test <-> retest) & reliability')) IS NOT TRUE")
    assert "reliability')) IS" not in ranking.rank_expression("ft", rank)
    # Without the alternative part every match ranks alike, as a 'near' search does.
    flat = rank.without_alternative()
    assert flat == ranking.RankQuery(rank.stemmed)
    assert ranking.other_words_only_expression("ft", flat) is None
    assert ranking.rank_expression("ft", flat) == ranking.rank_expression("ft", rank)


def test_other_searches_rank_as_before() -> None:
    rank = _rank("text_match('ketamine') AND text_match('social conflict', 'near')")
    assert rank.alternative is None
    assert ranking.other_words_only_expression("ft", rank) is None
    # 'also' without a search before it has nothing to rank below.
    alone = _rank("text_match('post-mortem | ex vivo', 'also')")
    assert alone is not None and alone.alternative is None


def test_the_page_puts_the_users_words_before_the_other_words() -> None:
    from services.translate import sql_rewriter

    expanded = text_search.expand_text_search(
        "SELECT d.id, d.name, COUNT(DISTINCT o.subject) AS subject_count FROM bids_datasets AS d "
        "LEFT JOIN bids_objects AS o ON o.dataset_id = d.id WHERE "
        "(text_match('dead people') OR text_match('ex vivo', 'also')) GROUP BY d.id",
        papers=False)
    page, _ = sql_rewriter.build_page_sql(
        base_sql=expanded.sql, scored_filters=[], apply_relevance=True, limit=20, offset=0,
        rank_query=expanded.rank_query)
    assert "AS text_rank" in page and "AS other_words_only" in page
    assert ("ORDER BY page_result.other_words_only, "
            "page_result.relevance_score + page_result.text_rank DESC") in page
