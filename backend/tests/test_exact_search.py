"""Exact word forms end to end: the SQL a search writes for a word that must
match as asked, and everything that reads that SQL afterwards (chips, the
uncertain tier, ranking, README excerpts) plus the intent prompt.

Switched on with TEXT_SEARCH_EXACT; services/fulltext/text_search.py has the design."""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

import pytest
import sqlglot
import yaml

import preprocess as pp
import word_forms
from services.results import query_filters as qf
from services.catalogue.catalogue import ColumnCoverage, catalogue
from services.fulltext import ranking, text_search as ts, tsquery, vectors, word_frequencies
from services.uncertain import text as tier_text, tier as ut
from services.fulltext import word_form_counts
from services.translate.sql_rewriter import SqlRewriteService, build_page_sql
from services.fulltext.ranking import RankQuery
from word_forms import WordEvidence

SERVICE = SqlRewriteService()
HEAD = ("SELECT d.id, d.name, COUNT(DISTINCT o.subject) AS subject_count "
        "FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
        "AND NOT o.subject IS NULL ")
MODALITY = ("EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
            "AND o2.datatype = 'functional_mri')")
FORMS = {"ageing": ("ageing", "aging"), "weight": ("weight", "weights")}
AGEING = WordEvidence("ageing", "age", 291, (
    ("age", 220), ("aged", 44), ("aging", 34), ("ages", 30), ("ageing", 4)))


def _where(condition: str) -> str:
    return f"{HEAD}WHERE {condition} GROUP BY d.id"


def _condition(expanded: ts.ExpandedSearch) -> str:
    return expanded.sql.split("WHERE ", 1)[1].rsplit(" GROUP BY", 1)[0]


@pytest.fixture(autouse=True)
def _database_with_full_text(tmp_path, monkeypatch):
    """Exact search off, no counts, and an empty reviewed list."""
    path = tmp_path / "word_forms.yaml"
    path.write_text("{}")
    monkeypatch.setattr(word_forms, "_PATH", path)
    word_forms._reviewed.cache_clear()
    monkeypatch.delenv(vectors.EXACT_ENV, raising=False)
    catalogue.coverage = ColumnCoverage.full(())
    word_frequencies.set_frequency_lookup(None)
    word_form_counts.set_lookup(None)
    yield path
    word_frequencies.set_frequency_lookup(None)
    word_form_counts.reset_lookup()
    word_forms._reviewed.cache_clear()


@pytest.fixture
def exact(monkeypatch):
    monkeypatch.setenv(vectors.EXACT_ENV, "true")


# ── Switch ────────────────────────────────────────────────────────────────────

def test_without_the_switch_nothing_changes() -> None:
    sql = _where("text_match('quantitative mri ageing') AND NOT text_match('weight', 'short')")
    plain = ts.expand_text_search(sql)
    assert ts.expand_text_search(sql, exact_forms=FORMS) == plain
    assert "search_tsv_exact" not in plain.sql and plain.rank_query.exact is None


# ── What the search writes ────────────────────────────────────────────────────

def test_an_exact_word_gets_a_guard_next_to_its_search(exact) -> None:
    expanded = ts.expand_text_search(_where("text_match('quantitative mri ageing')"),
                                     exact_forms=FORMS)
    assert _condition(expanded) == (
        "(d.search_tsv @@ to_tsquery('english', 'quantitative & mri & ageing') "
        "AND d.search_tsv_exact @@ to_tsquery('simple', '(ageing | aging)'))")
    # Other words are searched as before.
    assert _condition(ts.expand_text_search(_where("text_match('memory')"), exact_forms=FORMS)) \
        == "d.search_tsv @@ to_tsquery('english', 'memory')"


def test_the_guard_keeps_scope_distances_and_negation(exact) -> None:
    short = ts.expand_text_search(_where("text_match('weight', 'short')"), exact_forms=FORMS)
    assert "to_tsquery('english', 'weight:AB')" in short.sql
    assert "to_tsquery('simple', '(weight:AB | weights:AB)')" in short.sql
    near = _condition(ts.expand_text_search(_where("text_match('healthy ageing', 'near')"),
                                            exact_forms=FORMS))
    assert "healthy <1> ageing" in near and near.endswith("'(ageing | aging)'))")
    negated = _condition(ts.expand_text_search(_where("NOT text_match('weight')"),
                                               exact_forms=FORMS))
    assert negated == ("NOT (d.search_tsv @@ to_tsquery('english', 'weight') "
                       "AND d.search_tsv_exact @@ to_tsquery('simple', '(weight | weights)'))")
    # A compound stays together by adjacency; its parts get no guard.
    compound = ts.expand_text_search(_where("text_match('t1-weight')"), exact_forms=FORMS)
    assert "search_tsv_exact" not in compound.sql


def test_alternatives_split_into_pairs_and_the_rest(exact) -> None:
    expanded = ts.expand_text_search(
        _where("text_match('obesity | body weight | adiposity', 'short')"), exact_forms=FORMS)
    assert _condition(expanded) == (
        "(d.search_tsv @@ to_tsquery('english', 'obesity:AB | adiposity:AB') OR "
        "(d.search_tsv @@ to_tsquery('english', 'body:AB & weight:AB') "
        "AND d.search_tsv_exact @@ to_tsquery('simple', '(weight:AB | weights:AB)')))")


def test_paper_text_uses_the_paper_twin_of_the_guard(exact) -> None:
    sql = _where("text_match('ageing') AND NOT text_match('weight')")
    with_papers = _condition(ts.expand_text_search(sql, papers=True, exact_forms=FORMS))
    assert with_papers.startswith(
        "(d.search_tsv_papers @@ to_tsquery('english', 'ageing') "
        "AND d.search_tsv_exact_papers @@ to_tsquery('simple', '(ageing | aging)'))")
    # NOT terms stay on the dataset's own text, guard included.
    assert "NOT (d.search_tsv @@ to_tsquery('english', 'weight') AND d.search_tsv_exact @@" \
        in with_papers
    without = ts.expand_text_search(sql, papers=False, exact_forms=FORMS).sql
    assert ts.with_paper_text(without) == ts.expand_text_search(
        sql, papers=True, exact_forms=FORMS).sql


def test_the_reviewed_list_needs_no_decision(exact, _database_with_full_text) -> None:
    _database_with_full_text.write_text(yaml.safe_dump(
        {"exact": [{"forms": ["severe", "severely", "severity"], "excludes": ["several"]}]}))
    word_forms._reviewed.cache_clear()
    expanded = ts.expand_text_search(_where("text_match('severe tbi', 'short')"))
    assert "to_tsquery('simple', '(severe:AB | severely:AB | severity:AB)')" in expanded.sql


def test_the_pair_survives_sqlglot_and_is_one_condition(exact) -> None:
    expanded = ts.expand_text_search(
        _where(f"{MODALITY} AND text_match('ageing') AND d.size_bytes > 5"), exact_forms=FORMS)
    tree = sqlglot.parse_one(expanded.sql, read="postgres")
    assert sqlglot.parse_one(tree.sql(dialect="postgres"), read="postgres") == tree
    conditions = qf._flatten_and(tree.args["where"].this)
    assert len(conditions) == 3 and ts.is_guarded_pair(conditions[1])
    assert len(ut._flatten_and(tree.args["where"].this)) == 3
    assert not ts.is_guarded_pair(sqlglot.parse_one("(a = 1 AND b = 2)", read="postgres"))


# ── Counting ──────────────────────────────────────────────────────────────────

def test_counts_ask_with_the_guard_only_when_there_is_one(exact) -> None:
    asked = []

    def lookup(queries, guards=None):
        asked.append((queries, guards))
        return [5] * len(queries)

    word_frequencies.set_frequency_lookup(lookup)
    assert tsquery.count_matches(["memory", "healthy ageing"], exact_forms=FORMS) == [5, 5]
    assert asked[-1][1] == [None, "(ageing | aging)"]
    tsquery.count_matches(["memory"], exact_forms=FORMS)
    assert asked[-1] == (["memory"], None)
    sql, params = word_frequencies._count_statement(["a", "ageing"], [None, "(ageing | aging)"])
    assert "search_tsv_exact @@ to_tsquery('simple', q.guard)" in sql
    assert params == {"queries": ["a", "ageing"], "guards": [None, "(ageing | aging)"]}
    assert word_frequencies._count_statement(["a"], [None]) == (word_frequencies._COUNT_SQL, {"queries": ["a"]})


def test_a_loose_term_ranks_rarity_by_the_forms_that_count(exact) -> None:
    # By stem "ageing" is in 291 datasets and "connectome" in 40; in its
    # forms "ageing" is the rare word.
    def lookup(queries, guards=None):
        guards = guards or [None] * len(queries)
        counts = {("ageing", None): 291, ("ageing", "(ageing | aging)"): 30,
                  ("connectome", None): 40,
                  ("ageing & connectome", "(ageing | aging)"): 2}
        return [counts.get((q, g), 0) for q, g in zip(queries, guards)]

    word_frequencies.set_frequency_lookup(lookup)
    expanded = ts.expand_text_search(_where("text_match('ageing connectome', 'loose')"),
                                     exact_forms=FORMS)
    assert _condition(expanded) == (
        "(d.search_tsv @@ to_tsquery('english', 'ageing') "
        "AND d.search_tsv_exact @@ to_tsquery('simple', '(ageing | aging)'))")


# ── Ranking ───────────────────────────────────────────────────────────────────

def test_exact_words_rank_on_the_unstemmed_vector(exact, monkeypatch) -> None:
    alone = ts.expand_text_search(_where("text_match('ageing')"), exact_forms=FORMS)
    assert alone.rank_query == RankQuery(exact="(ageing | aging)")
    mixed = ts.expand_text_search(_where("text_match('quantitative mri ageing')"),
                                  exact_forms=FORMS)
    # Next to other words the term still ranks where all of them meet.
    assert mixed.rank_query == RankQuery("quantitative & mri & ageing", "(ageing | aging)")
    # Stored with a translation as JSON; a part that is not a safe tsquery is dropped.
    assert ranking.rank_from_json(ranking.rank_to_json(mixed.rank_query)) == mixed.rank_query
    assert ranking.rank_from_json({"stemmed": "memory", "exact": "x'; DROP"}) == RankQuery("memory")
    assert ranking.rank_from_json("memory#(ageing | aging)") is None

    assert ranking.rank_expression("ft", RankQuery("memory")) == ranking.ts_rank_cd_expression(
        "ft.search_tsv", "memory")
    assert ranking.rank_expression("ft", alone.rank_query) == ranking.ts_rank_cd_expression(
        "ft.search_tsv_exact", "(ageing | aging)", "simple")
    both = ranking.rank_expression("ft", mixed.rank_query)
    assert "ft.search_tsv, to_tsquery('english', 'quantitative & mri & ageing'), 0)" in both
    assert "ft.search_tsv_exact, to_tsquery('simple', '(ageing | aging)'), 0)" in both
    assert ranking.rank_expression("ft", None) is None
    # A cached rank from before the switch went off: the exact part is ignored.
    monkeypatch.delenv(vectors.EXACT_ENV)
    assert ranking.rank_expression("ft", mixed.rank_query) == ranking.ts_rank_cd_expression(
        "ft.search_tsv", "quantitative & mri & ageing")
    assert ranking.rank_expression("ft", alone.rank_query) is None


def test_the_page_query_orders_by_both_ranks(exact) -> None:
    rewritten = SERVICE.rewrite_generated_sql(
        "SELECT d.id FROM bids_datasets d WHERE text_match('healthy ageing') GROUP BY d.id",
        "healthy ageing", FORMS)
    assert rewritten.rank_query == RankQuery("healthy & ageing", "(ageing | aging)")
    assert "d.search_tsv_exact @@ to_tsquery('simple', '(ageing | aging)')" in rewritten.sql
    page, _ = build_page_sql(rewritten.sql, [], True, 20, 0, rank_query=rewritten.rank_query)
    assert "ft.search_tsv_exact" in page and "AS text_rank" in page
    sqlglot.parse_one(page, read="postgres")


# ── Chips ─────────────────────────────────────────────────────────────────────

def _expanded(condition: str, **kwargs) -> str:
    return ts.expand_text_search(_where(condition), exact_forms=FORMS, **kwargs).sql


def test_a_guarded_search_is_one_chip_and_goes_as_one(exact) -> None:
    sql = _expanded(f"{MODALITY} AND text_match('healthy ageing')")
    chips = qf.describe(sql)
    assert [(c.id, c.kind) for c in chips] == [("w0", "modality"), ("w1", "text")]
    text = chips[1]
    assert text.label == "“healthy ageing”"
    assert text.text_words == ["healthy ageing", "ageing", "aging"]
    assert "Only these word forms count: ageing, aging." in text.detail
    removed = qf.remove(sql, ["w1"])
    assert "search_tsv" not in removed and "functional_mri" in removed


def test_a_negated_guard_is_not_a_search_word(exact) -> None:
    sql = _expanded("text_match('memory') AND NOT text_match('weight')")
    chips = {c.id: c for c in qf.describe(sql)}
    assert chips["w1"].negated and chips["w1"].text_words == ["weight", "weights"]
    assert qf.highlight_exact_words(sql) == frozenset()
    assert qf.highlight_terms(sql) == [("memory",)]


# ── Uncertain tier ────────────────────────────────────────────────────────────

def test_guard_for_follows_the_words_that_remain() -> None:
    guard = "(ageing:AB | aging:AB) & weight:AB"
    assert tier_text.guard_for("ageing & brain", guard) == "(ageing | aging)"
    assert tier_text.guard_for("brain", guard) is None
    assert tier_text.guard_for("weight & ageing", guard) == "(ageing | aging) & weight"
    assert tier_text.guard_for("ageing", None) is None


def test_the_weaker_form_keeps_the_word_forms(exact, monkeypatch) -> None:
    monkeypatch.setattr(word_frequencies, "_lookup_counts", lambda queries, guards=None: [
        {"healthy": 300, "ageing": 30, "cohort": 100}.get(q, 1000) for q in queries])
    sql = _expanded(f"{MODALITY} AND text_match('healthy ageing cohort')")
    tiered = ut.apply(sql, papers=False)
    weaker = tiered.split("AS confirmed_0, (", 1)[1].split(") AS probable_0", 1)[0]
    # The rarer half, still in the forms asked for: not every README with an age.
    assert weaker == ("(d.search_tsv @@ TO_TSQUERY('english', 'ageing & cohort') "
                      "AND d.search_tsv_exact @@ TO_TSQUERY('simple', '(ageing | aging)'))")
    # Confirmation is the pair as written.
    assert ("(d.search_tsv @@ TO_TSQUERY('english', 'healthy & ageing & cohort') "
            "AND d.search_tsv_exact @@ TO_TSQUERY('simple', '(ageing | aging)')) AS confirmed_0") in tiered


def test_scope_widening_keeps_the_guard_without_labels(exact) -> None:
    relaxed, words = tier_text.relax_text_condition(
        "d.search_tsv @@ TO_TSQUERY('english', 'weight:AB') "
        "AND d.search_tsv_exact @@ TO_TSQUERY('simple', '(weight:AB | weights:AB)')")
    assert relaxed == ("(d.search_tsv @@ TO_TSQUERY('english', 'weight') "
                       "AND d.search_tsv_exact @@ TO_TSQUERY('simple', '(weight | weights)'))")
    assert words == ["\x00readme weight"]


def test_paper_evidence_needs_the_word_forms_in_the_paper(exact) -> None:
    sql = _expanded(f"{MODALITY} AND text_match('ageing')", papers=False)
    tiered = ut.apply(sql, papers=True)
    assert (
        "(d.search_tsv_papers @@ TO_TSQUERY('english', 'ageing') "
        "AND d.search_tsv_exact_papers @@ TO_TSQUERY('simple', '(ageing | aging)')) "
        "AND NOT (d.search_tsv @@ TO_TSQUERY('english', 'ageing') "
        "AND d.search_tsv_exact @@ TO_TSQUERY('simple', '(ageing | aging)'))") in tiered
    sqlglot.parse_one(tiered, read="postgres")


# ── README excerpts ───────────────────────────────────────────────────────────

def test_marks_of_exact_words_must_be_the_word_as_written() -> None:
    from services.results.readme_snippets import keep_justified_marks
    headline = "⟦Age⟧ range 60-80; healthy ⟦ageing⟧ and ⟦Aging⟧'s effects, ⟦aged⟧ adults"
    marks = ["age", "age", "age", "age"]
    terms = [("ageing",), ("aging",)]
    lexemes = {"ageing": {"age"}, "aging": {"age"}}
    assert keep_justified_marks(headline, marks, terms, lexemes, exact={"ageing", "aging"}) == (
        "Age range 60-80; healthy ⟦ageing⟧ and ⟦Aging⟧'s effects, aged adults")
    # Without the exact set every stem match stays, as before.
    assert keep_justified_marks(headline, marks, terms, lexemes) == headline


class _Session:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def begin_nested(self):
        class _Nested:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False
        return _Nested()

    async def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        return self.rows


def test_the_excerpt_is_the_passage_with_the_exact_word(exact) -> None:
    from services.results import readme_snippets
    from schemas import DatasetSchema

    datasets = [DatasetSchema(id=UUID(int=n), name=f"ds{n}", dataset_type="raw",
                              source_type="openneuro", participants=[]) for n in (1, 2)]
    session = _Session([
        # The stemmed passage is about ages; the unstemmed one has the word.
        (UUID(int=1), "⟦Age⟧ range 20-30. Mean ⟦age⟧ 25.", ["age", "age"], ["age", "age"],
         "a study of healthy ⟦ageing⟧ in adults"),
        # No exact form in this README: the stemmed passage, marks filtered.
        (UUID(int=2), "Mean ⟦age⟧ 25 years.", ["age"], ["age", "age"], "Mean age 25 years."),
    ])
    asyncio.run(readme_snippets.attach_snippets(
        session, datasets, [("ageing",), ("aging",)], {"ageing", "aging"}))
    sql, params = session.calls[0]
    assert "ts_headline('simple'" in sql and params["exact_q"] == "ageing | aging"
    assert datasets[0].snippet == "… a study of healthy ⟦ageing⟧ in adults …"
    assert datasets[1].snippet == "Mean age 25 years."


def test_exact_words_come_from_the_filter_sql(exact) -> None:
    sql = _expanded(f"{MODALITY} AND text_match('healthy ageing')")
    assert qf.highlight_exact_words(sql) == {"ageing", "aging"}
    assert ("healthy", "ageing") in qf.highlight_terms(sql)
    assert ("aging",) in qf.highlight_terms(sql)
    assert qf.highlight_exact_words(_where(MODALITY)) == frozenset()
    assert qf.highlight_exact_words(None) == frozenset()


# ── Intent prompt and pipeline ────────────────────────────────────────────────

PLAN = {"query_family": "description_search", "natural_language_summary": "x",
        "description_search": "ageing"}


def _model(monkeypatch, answer: dict, schemas: list | None = None) -> list:
    prompts = []

    def call(prompt, api_key, *, json_schema=None):
        prompts.append(prompt)
        if schemas is not None:
            schemas.append(json_schema)
        return json.dumps(answer)

    monkeypatch.setattr(pp.intent, "_call_gemini", call)
    return prompts


def test_the_prompt_only_changes_for_a_colliding_word(monkeypatch) -> None:
    prompts = _model(monkeypatch, PLAN)
    pp.preprocess_query("MRI on ageing", api_key="k")
    pp.preprocess_query("MRI on ageing", api_key="k", evidence={})
    scans = WordEvidence("scans", "scan", 502, (("scans", 280), ("scan", 252)))
    pp.preprocess_query("MRI on ageing", api_key="k", evidence={"scans": scans})
    assert prompts[0] == prompts[1] == prompts[2]
    assert "word_forms" not in prompts[0]

    pp.preprocess_query("MRI on ageing", api_key="k", evidence={"ageing": AGEING})
    assert '"user_query": "MRI on ageing"' in prompts[3]
    assert '"catalogue_word_form_evidence"' in prompts[3]
    assert "ageing: age (220), aged (44), aging (34), ages (30), ageing (4)" in prompts[3]
    assert '"word_forms"' in prompts[3]


def test_the_models_choice_is_read_outside_the_plan(monkeypatch) -> None:
    _model(monkeypatch, {**PLAN, "word_forms": [
        {"word": "ageing", "keep": ["ageing", "aging"]}, "nonsense", {"word": 3}]})
    choices = []
    plan = pp.preprocess_query("MRI on ageing", api_key="k", evidence={"ageing": AGEING},
                               word_choices=choices)
    assert plan.description_search == "ageing"
    assert choices == [("ageing", ["ageing", "aging"])]
    # No answer for the word: nothing decided, the plan is still read.
    _model(monkeypatch, PLAN)
    choices = []
    pp.preprocess_query("MRI on ageing", api_key="k", evidence={"ageing": AGEING},
                        word_choices=choices)
    assert choices == []


def test_structured_output_schema_includes_conditional_word_forms(monkeypatch) -> None:
    schemas = []
    prompts = _model(monkeypatch, PLAN, schemas)
    pp.preprocess_query("Find datasets on memory", api_key="k")
    base_schema = schemas[-1]
    assert "word_forms" not in base_schema["properties"]
    assert "QueryPlan JSON schema" not in prompts[-1]
    assert "default" not in json.dumps(base_schema)

    _model(monkeypatch, {**PLAN, "word_forms": [
        {"word": "ageing", "keep": ["ageing", "aging"]}]}, schemas)
    pp.preprocess_query("MRI on ageing", api_key="k", evidence={"ageing": AGEING})
    form_schema = schemas[-1]
    assert "word_forms" in form_schema["required"]
    assert form_schema["properties"]["word_forms"]["items"]["required"] == ["word", "keep"]


def test_the_pipeline_hands_the_decided_forms_to_the_search(exact, monkeypatch) -> None:
    word_form_counts.set_lookup(lambda words: [
        ("ageing", "age", 291, ["age:220", "aged:44", "aging:34", "ages:30", "ageing:4"])])
    _model(monkeypatch, {**PLAN, "word_forms": [{"word": "ageing", "keep": ["aging", "ageing"]}]})
    augmented = pp.run_pipeline("Quantitative MRI datasets on ageing", api_key="k")
    assert augmented.word_forms == {"ageing": ["ageing", "aging"]}
    assert "text_match('ageing')" in augmented.augmented_question
    rewritten = SERVICE.rewrite_generated_sql(
        "SELECT d.id FROM bids_datasets d WHERE text_match('ageing') GROUP BY d.id",
        augmented.augmented_question, augmented.word_forms)
    assert "search_tsv_exact @@ to_tsquery('simple', '(ageing | aging)')" in rewritten.sql.replace(
        "TO_TSQUERY", "to_tsquery")


def test_the_pipeline_is_untouched_without_the_switch(monkeypatch) -> None:
    word_form_counts.set_lookup(lambda words: pytest.fail("looked up with the switch off"))
    prompts = _model(monkeypatch, PLAN)
    augmented = pp.run_pipeline("Quantitative MRI datasets on ageing", api_key="k")
    assert augmented.word_forms == {} and "word_forms" not in prompts[0]
