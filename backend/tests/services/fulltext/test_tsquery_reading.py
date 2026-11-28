"""Reading generated tsquery strings back (services/fulltext/tsquery.py)."""

from __future__ import annotations

import pytest

from services.fulltext import tsquery

NEAR = "(social:AB <1> conflict:AB | social:AB <2> conflict:AB | conflict:AB <1> social:AB) | aggression:AB"


def test_labels_and_prefix_marks_go():
    assert tsquery.without_labels("parkinson:AB & disease:A | child:*AB") == "parkinson & disease | child"


def test_split_keeps_parentheses_whole():
    assert tsquery.split_top_level("(a | b) & c & (d & e)", "&") == ["(a | b)", "c", "(d & e)"]
    assert tsquery.split_top_level("(a | b) & c", "|") == ["(a | b) & c"]


def test_near_search_is_read_in_its_first_word_order():
    assert tsquery.is_near(NEAR) and not tsquery.is_near("follow <-> up")
    assert tsquery.near_parts(tsquery.split_top_level(NEAR, "|")[0]) == ["social:AB", "conflict:AB"]
    assert tsquery.query_phrases(NEAR) == ["social conflict", "aggression"]


@pytest.mark.parametrize("query, words", [
    (NEAR, ["social", "conflict", "aggression"]),
    ("(behaviour | behavior) & follow <-> up", ["behaviour", "behavior", "follow", "up"]),
    ("(ageing:AB | aging:AB) & weight", ["ageing", "aging", "weight"]),
])
def test_query_words(query, words):
    assert tsquery.query_words(query) == words


def test_phrases_of_alternatives():
    assert tsquery.query_phrases("(alzheimer:AB & disease:AB) | ad:AB") == ["alzheimer disease", "ad"]
