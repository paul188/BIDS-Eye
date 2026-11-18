"""Qualified topics: a subject whose meaning a qualifier fixes ("social conflict").

Covers the sense inventory (RAG/senses.py), topic resolution and its filter
line (LLM_preprocessor/preprocess/), the 'near' text search
(services/fulltext/text_search.py), its weaker forms in the uncertain tier and the
topic chips. See reports/Qualified topics plan.md.
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

import preprocess as pp  # noqa: E402
import senses  # noqa: E402
from services.results import query_filters  # noqa: E402
from services.fulltext import text_search, uncertain_tier  # noqa: E402
from text_terms import parse_alternatives  # noqa: E402

SOCIAL = pp.Topic(
    phrase="social conflict", typed="social conflicts", head="conflict", qualifiers=["social"],
    gloss="disagreement or hostility between people",
    alternatives=["conflict", "aggression", "conflict resolution", "Stroop interference",
                  "social interaction"],
    not_meant=["cognitive conflict", "Stroop"],
)


class _Retriever:
    """Vocabulary stand-in: knows one phrase."""

    def __init__(self, known=None):
        self.known = known or {}
        self.asked = []

    def retrieve_for_field_multi_scored(self, field, terms):
        self.asked.extend(terms)
        return {t: self.known[t] for t in terms if t in self.known}


def _counts(table, default=5):
    return lambda alternatives: [table.get(a, default) for a in alternatives]


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.setattr(text_search, "_lookup_counts", lambda queries: None)


# ── Sense inventory ───────────────────────────────────────────────────────────

def test_qualifier_picks_a_sense_and_excludes_the_others() -> None:
    pick = senses.pick_sense("conflicts", ["social"])
    assert pick.sense.name == "social"
    assert "aggression" in pick.alternatives
    assert {"stroop", "simon", "flanker"} <= set(pick.not_meant)
    assert "simon_task" in pick.excluded_tasks


def test_no_qualifier_or_unknown_head_picks_nothing() -> None:
    assert senses.pick_sense("conflict", []).sense is None
    assert senses.pick_sense("conflict", ["social", "cognitive"]).sense is None  # tie
    assert senses.pick_sense("memory", ["working"]) is None


def test_a_cue_in_the_phrase_outweighs_another_qualifier() -> None:
    pick = senses.pick_sense("stress", ["psychosocial", "induction"], phrase="psychosocial stress")
    assert pick.sense.name == "social"


# ── Resolution ────────────────────────────────────────────────────────────────

def test_alternatives_are_checked_before_they_are_searched() -> None:
    resolved = pp.resolve_topic(SOCIAL, None, _counts({"social interaction": 300,
                                                       "negotiation": 0}))
    assert resolved.sense == "social"
    assert resolved.dropped["conflict"] == "only the head word or a qualifier"
    assert resolved.dropped["Stroop interference"] == "names a meaning the search excludes"
    assert resolved.dropped["social interaction"].startswith("too common")
    assert resolved.dropped["negotiation"] == "no dataset uses it"
    assert "aggression" in resolved.alternatives
    assert "conflict resolution" in resolved.alternatives      # the model's own phrase
    assert "conflict" not in resolved.alternatives


def test_curated_alternatives_are_not_too_common() -> None:
    # "aggression" is in senses.yaml; a broad count does not drop it.
    resolved = pp.resolve_topic(SOCIAL, None, _counts({"aggression": 500}))
    assert "aggression" in resolved.alternatives


def test_without_counts_alternatives_are_kept_unchecked() -> None:
    resolved = pp.resolve_topic(SOCIAL, None, lambda alternatives: None)
    assert "conflict resolution" in resolved.alternatives


def test_codes_come_from_the_phrase_and_unwanted_codes_are_removed() -> None:
    retriever = _Retriever({"social conflict": [("simon_task", 0.9), ("competition_task", 0.9)]})
    resolved = pp.resolve_topic(SOCIAL, retriever, _counts({}))
    # "conflict" is asked only to subtract what the head word alone would give.
    assert retriever.asked == ["social conflict", "conflict"]
    assert resolved.codes == ["competition_task"]


def test_condition_guards_the_alternatives_not_the_phrase() -> None:
    resolved = pp.resolve_topic(SOCIAL, None, _counts({}))
    condition = resolved.condition
    assert condition.startswith("(text_match('social conflict', 'near') OR ((text_match('")
    # Several words together anywhere; one word only in the short fields.
    assert "interpersonal conflict" in condition and "'near') OR text_match('aggression" in condition
    assert "', 'short')) AND NOT text_match('cognitive conflict | stroop" in condition


def test_abbreviation_searches_typed_and_written_out_forms() -> None:
    emo = pp.Topic(phrase="emotion", typed="emo", head="emotion")
    retriever = _Retriever({"emotion": [("emotion_general", 1.0)]})
    resolved = pp.resolve_topic(emo, retriever, _counts({}))
    assert resolved.condition == (
        "(EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id "
        "AND o2.task = 'emotion_general') OR text_match('emotion | emo', 'near'))"
    )


def test_scan_terms_inside_a_topic_are_removed() -> None:
    plan = pp.QueryPlan(
        query_family="combined_filter", natural_language_summary="x", topics=[SOCIAL],
        scan_requirements=[pp.ScanRequirement(scan_terms=["conflict"]),
                           pp.ScanRequirement(scan_terms=["fMRI"])],
        description_search="social conflicts",
    )
    pp._detach_topic_terms(plan)
    assert [r.scan_terms for r in plan.scan_requirements] == [["fMRI"]]
    assert plan.description_search is None


def test_topic_becomes_a_filter_line_and_ranks_its_codes() -> None:
    plan = pp.QueryPlan(query_family="description_search", natural_language_summary="x",
                        topics=[SOCIAL])
    retriever = _Retriever({"social conflict": [("competition_task", 0.8)]})
    topics = pp.resolve_topics(plan, retriever, _counts({}))
    augmented = pp.augment_with_rag(plan, {}, topics=topics)
    assert '  topic "social conflict": (EXISTS' in augmented.augmented_question
    assert [(f.field, f.code, f.score) for f in augmented.scored_filters] == [
        ("task", "competition_task", 0.8)]


# ── Near search ───────────────────────────────────────────────────────────────

def test_near_spells_out_both_orders_and_small_gaps() -> None:
    query = text_search.build_filter_tsquery(parse_alternatives("social conflict | aggression"), "near")
    assert query == ("(social <1> conflict | social <2> conflict | conflict <1> social | "
                     "conflict <2> social) | aggression")
    assert text_search.build_rank_tsquery(parse_alternatives("social conflict"), "near") == \
        "social & conflict"


def test_near_keeps_the_order_of_three_words() -> None:
    query = text_search.build_filter_tsquery(parse_alternatives("approach avoidance conflict"), "near")
    assert query.count("|") == 3
    assert query.startswith("(approach <1> avoidance <1> conflict")


def test_count_matches_maps_counts_back_to_terms(monkeypatch) -> None:
    monkeypatch.setattr(text_search, "_lookup_counts", lambda queries: [7] * len(queries))
    assert text_search.count_matches(["aggression", "???", "family conflict"]) == [7, 0, 7]


# ── Uncertain tier ────────────────────────────────────────────────────────────

NEAR = "(social <1> conflict | social <2> conflict | conflict <1> social | conflict <2> social)"


def test_near_relaxes_to_all_words_never_to_the_head_alone() -> None:
    assert uncertain_tier.relax_tsquery(f"{NEAR} | aggression") == "(social & conflict) | aggression"
    assert uncertain_tier.query_words(NEAR) == ["social", "conflict"]


def test_excluded_meaning_demotes_instead_of_removing() -> None:
    sql = ("(d.search_tsv @@ TO_TSQUERY('english', '" + NEAR + "') OR "
           "(d.search_tsv @@ TO_TSQUERY('english', 'aggression') AND NOT "
           "d.search_tsv @@ TO_TSQUERY('english', 'stroop')))")
    relaxed, words = uncertain_tier.relax_text_condition(sql)
    assert "'(social & conflict)'" in relaxed
    assert "'aggression'" in relaxed            # kept, without the NOT
    reason = uncertain_tier.partial_reason(words)
    assert reason == ("the words social, conflict occur, but not together, or a related term "
                      "occurs next to a meaning the search excludes (stroop)")


def test_only_the_phrase_of_a_topic_is_loosened() -> None:
    sql = ("(d.search_tsv @@ TO_TSQUERY('english', '" + NEAR + "') OR "
           "d.search_tsv @@ TO_TSQUERY('english', '(interpersonal <1> conflict | "
           "conflict <1> interpersonal)'))")
    relaxed, _ = uncertain_tier.relax_text_condition(sql)
    assert relaxed == "d.search_tsv @@ TO_TSQUERY('english', '(social & conflict)')"


# ── Chips ─────────────────────────────────────────────────────────────────────

def _expanded(condition: str) -> str:
    sql = f"SELECT d.id FROM bids_datasets AS d WHERE {condition} AND d.license = 'CC0' GROUP BY d.id"
    return text_search.expand_text_search(sql, fts=True, papers=False).sql


def test_topic_gives_a_topic_chip_and_a_removable_exclusion() -> None:
    sql = _expanded(pp.resolve_topic(SOCIAL, None, _counts({})).condition)
    chips = {c.id: c for c in query_filters.describe(sql)}
    assert chips["w0"].kind == "topic" and chips["w0"].label == "social conflict"
    assert "stroop" not in chips["w0"].text_words
    assert chips["w0n"].kind == "sense" and chips["w0n"].text_words == []
    assert chips["w0n"].label.startswith("cognitive conflict, stroop")

    without = query_filters.remove(sql, ["w0n"])
    assert "NOT" not in without and "aggression" in without and "CC0" in without
    assert [c.id for c in query_filters.describe(without)] == ["w0", "w1"]


# ── SQL rewriting ─────────────────────────────────────────────────────────────

def test_rewriter_adds_a_topic_the_model_left_out() -> None:
    from services.translate import sql_rewriter

    condition = pp.resolve_topic(SOCIAL, None, _counts({})).condition
    question = f'[Resolved DB filters]\n  topic "social conflict": {condition}\n\nsummary'
    service = sql_rewriter.sql_rewrite_service
    left_out = "SELECT d.id FROM bids_datasets AS d WHERE d.license = 'CC0' GROUP BY d.id"
    added = service._inject_topics(left_out, question)
    # sqlglot writes the placeholder in capitals; its expansion ignores case.
    assert "text_match('social conflict', 'near')" in added.lower() and "CC0" in added
    copied = f"SELECT d.id FROM bids_datasets AS d WHERE {condition} GROUP BY d.id"
    assert service._inject_topics(copied, question) == copied


def test_codes_the_head_word_alone_gives_do_not_count() -> None:
    inhibition = pp.Topic(phrase="cortical inhibition", head="inhibition", qualifiers=["cortical"])
    retriever = _Retriever({
        "cortical inhibition": [("stop_signal_task", 0.9), ("gaba_spectroscopy", 0.9)],
        "inhibition": [("stop_signal_task", 1.0)],
    })
    resolved = pp.resolve_topic(inhibition, retriever, _counts({}))
    assert resolved.codes == ["gaba_spectroscopy"]


def test_a_narrower_phrase_keeps_its_codes() -> None:
    spatial = pp.Topic(phrase="spatial memory", head="memory", qualifiers=["spatial"])
    retriever = _Retriever({
        "spatial memory": [("spatial_span_task", 1.0)],
        "memory": [("spatial_span_task", 1.0), ("n_back", 1.0)],
    })
    assert pp.resolve_topic(spatial, retriever, _counts({})).codes == ["spatial_span_task"]


def test_only_ambiguous_or_abbreviated_topics_stay_topics() -> None:
    plan = pp.QueryPlan(
        query_family="combined_filter", natural_language_summary="x",
        topics=[SOCIAL,
                pp.Topic(phrase="emotion", typed="emo", head="emotion"),
                pp.Topic(phrase="sign language processing", typed="sign language processing",
                         head="language", qualifiers=["sign"], not_meant=["spoken language"])],
    )
    pp._keep_ambiguity_topics(plan)
    assert [t.phrase for t in plan.topics] == ["social conflict", "emotion"]
    assert [r.scan_terms for r in plan.scan_requirements] == [["sign language processing"]]
