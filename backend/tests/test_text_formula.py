"""The free-text part of a search as a formula: "drawings or art" is an OR.

"datasets related to drawings or art" required both words (2026-11-17): "or"
was dropped as a stopword and every word of a text_match() term must match.
RAG/text_formula.py reads description_search as phrases joined by AND, OR and
NOT; preprocess.py writes the 'dataset text:' line from it.
"""

from __future__ import annotations

from typing import Optional

import pytest

sqlglot = pytest.importorskip("sqlglot")

import preprocess as pp  # noqa: E402
import text_formula as tf  # noqa: E402
from services.fulltext import text_search, tier_text, tsquery, vectors, word_frequencies  # noqa: E402
from services.translate import sql_rewriter  # noqa: E402


@pytest.mark.parametrize("text, sql", [
    # The two questions found in the query log.
    ("drawings or art", "text_match('drawings | art')"),
    ("alcohol consumption or alcohol dependence",
     "text_match('alcohol consumption | alcohol dependence')"),
    # Without OR or NOT nothing changes: one term, every word required.
    ("drawings art", "text_match('drawings art')"),
    ("structural and functional connectivity", "text_match('structural functional connectivity')"),
    ("resting-state", "text_match('resting-state')"),
    # Lists, and/or, symbols.
    ("drawings, paintings or sculptures", "text_match('drawings | paintings | sculptures')"),
    ("autism and/or ADHD", "text_match('autism | adhd')"),
    ("eeg | meg", "text_match('eeg | meg')"),
    ("AUDIT, ADS", "text_match('audit ads')"),
    # Negation: a separate condition, not parenthesised with the rest.
    ("memory but not sleep", "text_match('memory') AND NOT text_match('sleep')"),
    ("memory AND NOT sleep", "text_match('memory') AND NOT text_match('sleep')"),
    ("(EEG or MEG) and not sleep", "text_match('eeg | meg') AND NOT text_match('sleep')"),
    ("not (rats or mice)", "NOT text_match('rats | mice')"),
    ("children and not (autism or adhd) and language",
     "text_match('children language') AND NOT text_match('autism | adhd')"),
    # Precedence: NOT, then AND, then OR.
    ("(memory and sleep) or dreams", "text_match('memory sleep | dreams')"),
    ("eeg or meg and sleep", "text_match('eeg | meg sleep')"),
    ("dreams or (memory and not sleep)",
     "(text_match('dreams') OR (text_match('memory') AND NOT text_match('sleep')))"),
    ("not not sleep", "text_match('sleep')"),
    # Broken input still gives a search, or none.
    ("( unbalanced", "text_match('unbalanced')"),
    ("or", None),
    ("not", None),
    ("the or a", None),
    ("", None),
])
def test_formula_to_sql(text, sql) -> None:
    assert tf.compile_formula(tf.parse_formula(text)) == sql


def test_operators_are_only_or_and_not() -> None:
    assert not tf.has_operators(tf.parse_formula("structural and functional"))
    assert tf.has_operators(tf.parse_formula("drawings or art"))
    assert tf.has_operators(tf.parse_formula("memory but not sleep"))


def test_phrases_are_cleaned_one_by_one() -> None:
    node = tf.parse_formula("healthy adults or children and not autism")
    cleaned = tf.map_phrases(node, lambda p: p.replace("healthy", "").replace("adults", ""))
    # The emptied phrase drops out of its OR; the rest stays as written.
    assert tf.compile_formula(cleaned) == "text_match('children') AND NOT text_match('autism')"
    assert tf.phrases(node, positive=False) == ["autism"]


# ── The 'dataset text:' line ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.setattr(word_frequencies, "_lookup_counts", lambda queries: None)


def _question(**fields) -> str:
    plan = pp.QueryPlan(query_family=pp.QueryFamily.DESCRIPTION_SEARCH,
                        natural_language_summary="test", **fields)
    return pp.augment_with_rag(plan, {}).augmented_question


def test_or_reaches_the_filter_line() -> None:
    assert "dataset text: text_match('drawings | art')" in _question(description_search="drawings or art")


def test_not_reaches_the_filter_line() -> None:
    question = _question(description_search="memory AND NOT sleep")
    assert "dataset text: text_match('memory') AND NOT text_match('sleep')" in question


def test_alternatives_join_an_or_but_not_a_negation() -> None:
    question = _question(description_search="dead people or corpses",
                         description_alternatives=["post-mortem"])
    assert ("dataset text: (text_match('dead people | corpses') OR "
            "text_match('post-mortem', 'near'))") in question
    negated = _question(description_search="dead people and not mice",
                        description_alternatives=["post-mortem"])
    assert "post-mortem" not in negated
    assert "text_match('dead people') AND NOT text_match('mice')" in negated


def test_rare_parts_are_not_searched_but_a_negation_stays(monkeypatch) -> None:
    monkeypatch.setattr(pp.augment, "_matches_too_few", lambda text: text == "zyx")
    # Next to another filter; alone the parts stay searched
    # (test_description_alternatives.py, words no dataset says).
    question = _question(description_search="zyx and not sleep", name_search="connectome")
    assert "dataset text: NOT text_match('sleep')" in question
    assert "not searchable (add NO condition for these): zyx" in question


def test_an_or_counts_its_phrases_together(monkeypatch) -> None:
    monkeypatch.setattr(tsquery, "count_matches",
                        lambda alternatives, mode="near": [2 if a == "drawings" else 1 for a in alternatives])
    # 2 + 1 datasets: still searched (three is enough).
    assert "text_match('drawings | art')" in _question(description_search="drawings or art")
    monkeypatch.setattr(tsquery, "count_matches", lambda alternatives, mode="near": [1, 1])
    assert "dataset text:" not in _question(description_search="drawings or art", name_search="connectome")


def test_healthy_inside_a_formula_is_not_searched() -> None:
    question = _question(description_search="healthy or depressed")
    assert "dataset text: text_match('depressed')" in question
    assert "healthy (most datasets" in question


# ── Search, tier and rewriter ────────────────────────────────────────────────

def test_an_or_has_no_weaker_form_in_the_tier() -> None:
    sql = text_search.expand_text_search(
        "SELECT d.id FROM bids_datasets AS d WHERE text_match('drawings | art') GROUP BY d.id",
        fts=True, papers=False).sql
    assert "to_tsquery('english', 'drawings | art')" in sql
    # Its weaker form is the same search: no single word ("art") on its own.
    relaxed, _ = tier_text.relax_text_condition(sql)
    assert relaxed is None or "'(drawings | art)'" in relaxed


def test_rewriter_adds_a_left_out_negation() -> None:
    service = sql_rewriter.sql_rewrite_service
    question = ("[Resolved DB filters]\n"
                "  dataset text: text_match('memory') AND NOT text_match('sleep')\n\nsummary")
    left_out = "SELECT d.id FROM bids_datasets AS d WHERE text_match('memory') GROUP BY d.id"
    restored = service._restore_description_formula(left_out, question).lower()
    assert "not text_match('sleep')" in restored and "text_match('memory')" in restored
    copied = ("SELECT d.id FROM bids_datasets AS d WHERE text_match('memory') "
              "AND NOT text_match('sleep') GROUP BY d.id")
    assert service._restore_description_formula(copied, question) == copied
    # A positive copy of the negated part is not the negation.
    wrong = ("SELECT d.id FROM bids_datasets AS d WHERE text_match('memory') "
             "AND text_match('sleep') GROUP BY d.id")
    assert "not text_match('sleep')" in service._restore_description_formula(wrong, question).lower()
    # A positive part is never put back: the model may have used a table for it
    # (fMRIPrep → bids_derivatives), and the text would empty that result.
    table = ("SELECT d.id FROM bids_datasets AS d WHERE NOT text_match('sleep') AND EXISTS "
             "(SELECT 1 FROM bids_derivatives x WHERE x.dataset_id = d.id) GROUP BY d.id")
    assert service._restore_description_formula(table, question) == table
    single = "[Resolved DB filters]\n  dataset text: text_match('drawings | art')\n\nsummary"
    assert service._restore_description_formula(left_out, single) == left_out
    # A line of exclusions only: each missing one comes back.
    neither = ("[Resolved DB filters]\n  dataset text: NOT text_match('sleep') AND "
               "NOT text_match('dreams')\n\nsummary")
    restored = service._restore_description_formula(
        "SELECT d.id FROM bids_datasets AS d WHERE NOT text_match('sleep') GROUP BY d.id", neither).lower()
    assert "not text_match('dreams')" in restored and restored.count("text_match('sleep')") == 1


def test_top_level_and_parts() -> None:
    assert sql_rewriter._top_level_and_parts(
        "text_match('a') AND NOT text_match('b | c') AND (text_match('d') AND text_match('e'))"
    ) == ["text_match('a')", "NOT text_match('b | c')", "(text_match('d') AND text_match('e'))"]


# ── More phrasings ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, sql", [
    ("either drawings or art", "text_match('drawings | art')"),
    ("both memory and sleep", "text_match('memory sleep')"),
    ("neither sleep nor dreams", "NOT text_match('sleep') AND NOT text_match('dreams')"),
    ("memory except sleep", "text_match('memory') AND NOT text_match('sleep')"),
    ("memory excluding sleep", "text_match('memory') AND NOT text_match('sleep')"),
    ("((drawings or art) and (children or adolescents))",
     "text_match('drawings | art') AND text_match('children | adolescents')"),
    ("drawings OR art OR drawings", "text_match('drawings | art')"),
    ("Drawings Or ART", "text_match('drawings | art')"),
    ("drawings || art && children", "text_match('drawings | art children')"),
    ("music; dance or theatre", "text_match('music | dance | theatre')"),
    ("visual (art or music)", "text_match('visual') AND text_match('art | music')"),
    ("memory and not (sleep or dreams) and not anesthesia",
     "text_match('memory') AND NOT text_match('sleep | dreams') AND NOT text_match('anesthesia')"),
    ("not memory or not sleep", "(NOT text_match('memory') OR NOT text_match('sleep'))"),
    # "without" is no operator: "with and without MCI" means both.
    ("patients with and without MCI", "text_match('patients without mci')"),
    # Stray operators and parentheses never break the search.
    ("or drawings", "text_match('drawings')"),
    ("drawings or", "text_match('drawings')"),
    ("drawings and", "text_match('drawings')"),
    ("drawings ) or ( art", "text_match('drawings | art')"),
    ("()", None),
    ("not not", None),
])
def test_more_phrasings(text, sql) -> None:
    assert tf.compile_formula(tf.parse_formula(text)) == sql


@pytest.mark.parametrize("text", [
    "resting state", "pediatric", "structural and functional connectivity",
    "alcohol consumption questionnaires AUDIT ADS", "post-mortem brains", "Parkinson's disease",
    "7T MP2RAGE", "n-back working memory", "the effects of music on the brain",
])
def test_without_or_and_not_the_search_is_what_it_always_was(text) -> None:
    """The old line was format_text_match([text]) with "and" dropped as a stopword."""
    assert not tf.has_operators(tf.parse_formula(text))
    assert tf.compile_formula(tf.parse_formula(text)) == pp.augment._format_text_match([text])


@pytest.mark.parametrize("text", [
    "x'); DROP TABLE bids_datasets; --", "a' OR '1'='1", "art\" or \"music", "ä ö ü or ß",
    "drawings % or _ art", "\\'; select 1",
])
def test_nothing_but_words_reaches_the_placeholder(text) -> None:
    """Whatever the model writes, the SQL holds only text_match() calls over [a-z0-9 |-]."""
    import re
    sql = tf.compile_formula(tf.parse_formula(text)) or ""
    calls = re.findall(r"text_match\('([^']*)'\)", sql)
    for argument in calls:
        assert set(argument) <= set("abcdefghijklmnopqrstuvwxyz0123456789 |-")
    # Outside the calls only the operators remain.
    rest = re.sub(r"text_match\('[^']*'\)", "", sql)
    assert re.fullmatch(r"[\sANDORT()]*", rest), rest


def test_render_shows_the_grouping() -> None:
    node = tf.parse_formula("dreams or (memory and not sleep)")
    assert tf.render(node) == "dreams OR (memory AND NOT sleep)"
    assert tf.render(tf.parse_formula("not (rats or mice)")) == "NOT (rats OR mice)"


def test_formula_counts_without_a_database_keep_the_part(monkeypatch) -> None:
    monkeypatch.setattr(tsquery, "count_matches", lambda alternatives, mode="near": None)
    assert not pp._formula_matches_too_few(tf.parse_formula("drawings or art"))
    monkeypatch.setattr(tsquery, "count_matches", lambda alternatives, mode="near": [None, 4])
    assert not pp._formula_matches_too_few(tf.parse_formula("drawings or art"))
    # A part with a NOT inside is no plain text search and is never too few.
    assert not pp._formula_matches_too_few(tf.parse_formula("dreams or not sleep"))


def test_request_wishes_are_cut_out_of_a_formula() -> None:
    question = _question(description_search="preprocessed or denoised for machine learning",
                         request_wishes=["for machine learning"])
    assert "dataset text: text_match('preprocessed | denoised')" in question


def test_adults_inside_a_formula_go_to_the_participant_lines() -> None:
    plan = pp.QueryPlan(query_family=pp.QueryFamily.DESCRIPTION_SEARCH, natural_language_summary="t",
                        description_search="adults and (drawings or art)")
    question = pp.augment_with_rag(plan, {}).augmented_question
    assert "participant age" in question
    assert "dataset text: text_match('drawings | art')" in question


def test_a_formula_line_is_not_taken_for_alternatives() -> None:
    """_restore_description_alternatives reads '(own OR other words)'; a formula is not that."""
    service = sql_rewriter.sql_rewrite_service
    question = ("[Resolved DB filters]\n  dataset text: (text_match('dreams') OR "
                "(text_match('memory') AND NOT text_match('sleep')))\n\nsummary")
    copied = ("SELECT d.id FROM bids_datasets AS d WHERE (text_match('dreams') OR "
              "(text_match('memory') AND NOT text_match('sleep'))) GROUP BY d.id")
    assert service._restore_description_alternatives(copied, question) == copied
    assert service._restore_description_formula(copied, question) == copied


# ── The whole rewrite: model SQL → executed SQL, tier, fallback, chips ───────

@pytest.fixture
def full_text(monkeypatch):
    from services.catalogue import schema_capabilities
    monkeypatch.setattr(schema_capabilities, "has_fts", lambda: True)
    monkeypatch.setattr(vectors, "papers_enabled", lambda: False)


def _rewrite(monkeypatch, description: str, model_where: Optional[str] = None, tier: bool = True):
    from services.results import query_filters
    monkeypatch.setenv("UNCERTAIN_TIER", "true" if tier else "false")
    question = _question(description_search=description)
    line = next(ln for ln in question.splitlines() if "dataset text:" in ln)
    where = model_where if model_where is not None else line.split("dataset text: ", 1)[1]
    model_sql = ("SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
                 f"WHERE {where} GROUP BY d.id")
    result = sql_rewriter.rewrite_generated_sql(model_sql, question)
    chips = query_filters.describe(result.filter_sql or result.sql)
    return " ".join(result.sql.split()), result, chips


def _negated_only(sql: str, word: str) -> bool:
    """Every full-text search for *word* in *sql* stands directly under a NOT."""
    import re
    search = rf"d\.search_tsv @@ to_tsquery\('english', '{word}'\)"
    every = len(re.findall(search, sql, re.IGNORECASE))
    negated = len(re.findall(rf"\bNOT\s+\(?\s*{search}", sql, re.IGNORECASE))
    return every > 0 and every == negated


@pytest.mark.parametrize("tier", [True, False])
def test_a_negation_stays_hard_through_the_rewrite(monkeypatch, full_text, tier) -> None:
    sql, result, chips = _rewrite(monkeypatch, "memory but not sleep", tier=tier)
    # Also inside the tier's CASE: a sleep dataset is never "probably relevant".
    assert _negated_only(sql, "sleep")
    assert "to_tsquery('english', 'memory')" in sql.lower()
    labels = {(c.label, c.negated) for c in chips}
    assert ("“memory”", False) in labels and ("“sleep”", True) in labels


def test_a_left_out_negation_is_restored_in_the_whole_rewrite(monkeypatch, full_text) -> None:
    sql, _, _ = _rewrite(monkeypatch, "memory but not sleep", model_where="text_match('memory')")
    assert _negated_only(sql, "sleep")


def test_the_fallback_for_an_empty_result_keeps_the_negation(monkeypatch, full_text) -> None:
    where = ("text_match('memory') AND NOT text_match('sleep') AND EXISTS (SELECT 1 FROM bids_objects o2 "
             "WHERE o2.dataset_id = d.id AND o2.datatype = 'eeg')")
    _, result, _ = _rewrite(monkeypatch, "memory but not sleep", model_where=where, tier=False)
    relaxed = " ".join((result.relaxed_sql or "").split()).lower()
    assert "'memory'" not in relaxed            # the free-text part is what the fallback drops
    assert "not d.search_tsv @@ to_tsquery('english', 'sleep')" in relaxed


def test_an_or_is_one_search_with_one_chip(monkeypatch, full_text) -> None:
    sql, result, chips = _rewrite(monkeypatch, "drawings or art")
    assert "to_tsquery('english', 'drawings | art')" in sql.lower()
    assert "& art" not in sql and "drawings &" not in sql
    # No "matches only part of the search" tier: neither word alone is weaker.
    assert "missing:" not in sql
    assert result.rank_tsquery and "drawings | art" in result.rank_tsquery
    assert [(c.field, c.label, c.negated) for c in chips] == [("Text", "“drawings” or “art”", False)]


def test_an_and_of_two_ors_is_two_conditions(monkeypatch, full_text) -> None:
    sql, _, chips = _rewrite(monkeypatch, "(drawings or art) and (music or dance)", tier=False)
    assert ("d.search_tsv @@ to_tsquery('english', 'drawings | art') AND "
            "d.search_tsv @@ to_tsquery('english', 'music | dance')") in sql
    assert sorted(c.label for c in chips) == ["“drawings” or “art”", "“music” or “dance”"]


def test_the_old_and_search_is_unchanged_through_the_rewrite(monkeypatch, full_text) -> None:
    sql, _, chips = _rewrite(monkeypatch, "structural functional connectivity", tier=False)
    assert "'structural & functional & connectivity'" in sql
    assert chips[0].label == "“structural functional connectivity”"
