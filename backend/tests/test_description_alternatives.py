"""Other words for a description search: "dead people" also finds "post-mortem".

The language model lists the words dataset authors use for the user's
everyday word (QueryPlan.description_alternatives); preprocess.py checks them
against the catalogue and adds them to the text filter.
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

import preprocess as pp  # noqa: E402
from services import query_filters, sql_rewriter, text_search, uncertain_tier  # noqa: E402


def _plan(**fields) -> pp.QueryPlan:
    return pp.QueryPlan(query_family=pp.QueryFamily.DESCRIPTION_SEARCH,
                        natural_language_summary="datasets from deceased people", **fields)


def _counts(table, default=5):
    return lambda alternatives: [table.get(a, default) for a in alternatives]


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.setattr(text_search, "_lookup_counts", lambda queries: None)


def test_alternatives_are_checked_against_the_catalogue() -> None:
    plan = _plan(description_search="dead people",
                 description_alternatives=["post-mortem", "ex vivo", "People, dead", "cadaver",
                                           "brain", "ex-vivo", "postmortem"])
    dropped = pp.check_description_alternatives(
        plan, _counts({"cadaver": 0, "brain": 900}))
    # "People, dead" repeats the user's words; "ex-vivo" repeats "ex vivo".
    assert plan.description_alternatives == ["post-mortem", "ex vivo", "postmortem"]
    assert dropped == {"cadaver": "no dataset uses it", "brain": "too common (900 datasets)"}


def test_without_a_description_search_nothing_is_added() -> None:
    plan = _plan(description_alternatives=["post-mortem"])
    pp.check_description_alternatives(plan, _counts({}))
    assert plan.description_alternatives == []
    assert "post-mortem" not in pp.augment_with_rag(plan, {}).augmented_question


def test_without_counts_alternatives_are_kept_unchecked() -> None:
    plan = _plan(description_search="dead people", description_alternatives=["post-mortem"] * 3)
    pp.check_description_alternatives(plan, lambda alternatives: None)
    assert plan.description_alternatives == ["post-mortem"]


CONDITION = ("(text_match('dead people') OR "
             "text_match('post-mortem | ex vivo | deceased', 'near'))")


def test_filter_line_adds_the_alternatives_to_the_users_words() -> None:
    plan = _plan(description_search="dead people",
                 description_alternatives=["post-mortem", "ex vivo", "deceased"])
    question = pp.augment_with_rag(plan, {}).augmented_question
    assert f"dataset text: {CONDITION}" in question


def test_words_no_dataset_says_stay_a_filter_when_they_have_alternatives(monkeypatch) -> None:
    monkeypatch.setattr(pp, "_matches_too_few", lambda text: True)
    plan = _plan(description_search="dead people",
                 description_alternatives=["post-mortem", "ex vivo", "deceased"])
    assert f"dataset text: {CONDITION}" in pp.augment_with_rag(plan, {}).augmented_question
    # Without alternatives such words are not searched, as before.
    alone = pp.augment_with_rag(_plan(description_search="dead people"), {}).augmented_question
    assert "dataset text:" not in alone and "not searchable" in alone


def test_the_search_the_chip_and_the_tier_read_the_condition() -> None:
    sql = text_search.expand_text_search(
        f"SELECT d.id FROM bids_datasets AS d WHERE {CONDITION} GROUP BY d.id",
        fts=True, papers=False).sql
    assert "'dead & people'" in sql
    assert "(post <-> mortem) | (ex <1> vivo" in sql and "| deceased'" in sql

    chips = query_filters.describe(sql)
    assert len(chips) == 1 and chips[0].label == "dead people"
    assert "post mortem" in chips[0].detail and "ex vivo" in chips[0].detail

    # Only the user's own words get a weaker form, not the alternatives.
    relaxed, _ = uncertain_tier.relax_text_condition(sql)
    assert "mortem" not in relaxed and "vivo" not in relaxed


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
    # Left out altogether: the condition is added.
    absent = "SELECT d.id FROM bids_datasets AS d WHERE d.license = 'CC0' GROUP BY d.id"
    added = service._restore_description_alternatives(absent, question).lower()
    assert "text_match('post-mortem | ex vivo', 'near')" in added and "cc0" in added
    plain = "[Resolved DB filters]\n  dataset text: text_match('complex')\n\nsummary"
    assert service._restore_description_alternatives(left_out, plain) == left_out
