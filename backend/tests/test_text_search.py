from __future__ import annotations

import asyncio
import contextlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "crawlers" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

import text_terms  # noqa: E402
from services.catalogue import schema_capabilities as caps  # noqa: E402
from services.translate import sql_rewriter  # noqa: E402
from services.fulltext import text_search  # noqa: E402
from services.translate.sql_rewriter import SqlRewriteService, build_count_sql, build_page_sql  # noqa: E402

SERVICE = SqlRewriteService()
BASE = (
    "SELECT {{COLS}} FROM bids_datasets d "
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL "
)


@pytest.fixture(autouse=True)
def _full_text_available():
    caps.set_fts(True)
    caps.set_available_columns(())
    text_search.set_frequency_lookup(None)
    yield
    caps.set_fts(False)
    caps.set_available_columns(())
    text_search.set_frequency_lookup(None)


# ── Terms and placeholders ────────────────────────────────────────────────────

def test_split_words_keeps_compounds_and_drops_noise() -> None:
    assert text_terms.split_words("Resting-state fMRI of the brain") == [
        ["resting", "state"], ["fmri"], ["brain"],
    ]
    assert text_terms.split_words("Parkinson's") == [["parkinson"]]
    assert text_terms.split_words("Alzheimer’s disease") == [["alzheimer"], ["disease"]]
    assert text_terms.split_words("naïve  café") == [["naive"], ["cafe"]]
    assert text_terms.split_words("'; DROP TABLE x --") == [["drop"], ["table"], ["x"]]
    assert text_terms.split_words("???") == []


def test_placeholder_round_trip_uses_a_restricted_alphabet() -> None:
    macro = text_terms.format_text_match(["N-Back", "Working Memory", "n-back"], "short")
    assert macro == "text_match('n-back | working memory', 'short')"
    assert text_terms.format_text_match(["alcohol"]) == "text_match('alcohol')"
    assert text_terms.format_text_match(["", "??"]) is None
    assert re.fullmatch(r"text_match\('[a-z0-9 |-]+'(, '[a-z]+')?\)", macro)
    assert text_terms.parse_alternatives("n-back | working memory") == [
        [["n", "back"]], [["working"], ["memory"]],
    ]
    with pytest.raises(ValueError):
        text_terms.format_text_match(["x"], "everything")


def test_task_text_fallback_drops_query_scaffolding_but_keeps_the_construct() -> None:
    assert text_terms.format_task_text_match(["assessments related to memory"]) == \
        "text_match('memory', 'short')"
    assert text_terms.format_task_text_match(["decision making tasks"]) == \
        "text_match('decision making', 'short')"
    assert text_terms.format_task_text_match(["music perception paradigm"]) == \
        "text_match('music perception', 'short')"
    assert text_terms.format_task_text_match(["task assessments"]) is None


def test_task_text_fallback_requires_every_word_of_the_phrase() -> None:
    # In 'loose' mode, without search_tsv, only the longest word was searched:
    # "working memory" → %working%, "face perception" → %perception%.
    for phrase, words in (("working memory", ("working", "memory")),
                          ("face perception", ("face", "perception")),
                          ("cognitive control", ("cognitive", "control"))):
        macro = text_terms.format_task_text_match([phrase])
        sql = text_search.expand_text_search(f"SELECT 1 FROM bids_datasets d WHERE {macro}", fts=False).sql
        for word in words:
            assert f"ILIKE '%{word}%'" in sql, (phrase, sql)
        assert sql.count(" AND ") >= 1, (phrase, sql)
        assert "readme_text" not in sql


def test_task_text_fallback_skips_short_lone_words() -> None:
    # Substring search: "rest" is in "interest", "pain" in "Spain".
    assert text_terms.format_task_text_match(["rest"]) is None
    assert text_terms.format_task_text_match(["pain tasks"]) is None
    assert text_terms.format_task_text_match(["n-back"]) == "text_match('n-back', 'short')"
    assert text_terms.format_task_text_match(["stroop task"]) == "text_match('stroop', 'short')"


_VOCABULARY = {
    "epilepsy": [("epilepsy", 1.0), ("seizures", 1.0), ("epileptic", 0.9),
                 ("epileptic patients", 0.9), ("TLE", 0.3), ("refractory epilepsy", 0.3)],
    "focal_cortical_dysplasia": [("focal cortical dysplasia", 1.0), ("FCD", 1.0)],
    "parkinsons_disease": [("Parkinson's disease", 1.0), ("PD", 1.0), ("PD patients", 1.0),
                           ("Parkinson", 0.97), ("paralysis agitans", 0.1)],
    "alzheimers_disease": [("Alzheimer's disease", 1.0), ("ad", 1.0), ("load", 0.1),
                           ("alzheimers dementia", 0.1)],
    "healthy_control": [("healthy control (standard)", 1.0), ("normal subjects", 0.9)],
}


def _lookup(code):
    return _VOCABULARY.get(code, [])


def test_diagnosis_alternatives_add_synonyms_the_stemmer_misses() -> None:
    found = text_terms.diagnosis_alternatives(
        {"epilepsy": ["epilepsy", "focal_cortical_dysplasia"]}, lookup=_lookup
    )
    # "refractory epilepsy" is covered by "epilepsy"; "epileptic patients" by
    # "epileptic"; "TLE" is too short to search documents for.
    assert found == ["epilepsy", "seizures", "epileptic", "focal-cortical-dysplasia"] or \
        found == ["epilepsy", "seizures", "epileptic", "focal cortical dysplasia"]


def test_diagnosis_alternatives_drop_generic_words_and_risky_abbreviations() -> None:
    found = text_terms.diagnosis_alternatives(
        {"Parkinson's disease": ["parkinsons_disease"], "AD": ["alzheimers_disease"]},
        lookup=_lookup,
    )
    # "disease" would hide 'Basal Ganglia in Parkinson'; "PD" is also proton
    # density; "load" would match 'cognitive load'.
    assert found == ["parkinson", "paralysis agitans", "alzheimer", "alzheimers dementia"]

    assert text_terms.diagnosis_alternatives(
        {"healthy controls": ["healthy_control"]}, lookup=_lookup
    ) == ["healthy controls", "healthy control"]


def test_synonyms_keep_their_generic_words() -> None:
    vocabulary = {"dyslexia": [("dyslexia", 1.0), ("reading disorder", 0.9),
                               ("dyslexic disorder", 0.5)],
                  "ocd": [("obsessive-compulsive disorder", 1.0), ("OC disorder", 0.5),
                          ("compulsive disorder", 0.5)]}
    found = text_terms.diagnosis_alternatives({"dyslexia": ["dyslexia"]}, lookup=vocabulary.get)
    # "reading" alone would match every reading study.
    assert found == ["dyslexia", "reading disorder", "dyslexic disorder"]
    found = text_terms.diagnosis_alternatives({"OCD": ["ocd"]}, lookup=vocabulary.get)
    # "disorder" does not make "oc" long enough to search for.
    assert found == ["ocd", "obsessive-compulsive", "compulsive disorder"]


def test_diagnosis_alternatives_work_without_a_vocabulary() -> None:
    assert text_terms.diagnosis_alternatives(
        {"long covid": ["unknown_code"]}, lookup=lambda code: []
    ) == ["long covid"]


# ── tsquery building ──────────────────────────────────────────────────────────

def test_filter_query_scopes_by_weight_label() -> None:
    terms = text_terms.parse_alternatives("epilepsy | seizure disorder")
    assert text_search.build_filter_tsquery(terms, "short") == "epilepsy:AB | (seizure:AB & disorder:AB)"
    assert text_search.build_filter_tsquery(terms, "all") == "epilepsy | (seizure & disorder)"
    # Ranking always looks at every field.
    assert text_search.build_rank_tsquery(terms, "short") == "epilepsy | (seizure & disorder)"


def test_hyphenated_words_must_be_adjacent() -> None:
    terms = text_terms.parse_alternatives("resting-state fmri")
    assert text_search.build_filter_tsquery(terms, "all") == "(resting <-> state) & fmri"


def test_loose_term_keeps_its_rarest_words_while_enough_datasets_match() -> None:
    terms = text_terms.parse_alternatives("virtual reality navigation")
    # No frequencies: the longest word, as the ILIKE search did.
    assert text_search.build_filter_tsquery(terms, "loose") == "navigation"

    word_counts = {"virtual": 40, "reality": 13, "navigation": 60}
    combined_counts = {"reality & virtual & navigation": 2, "reality & virtual": 9}
    asked = []

    def lookup(queries):
        asked.append(queries)
        return [word_counts.get(q, combined_counts.get(q, 0)) for q in queries]

    text_search.set_frequency_lookup(lookup)
    assert text_search.build_filter_tsquery(terms, "loose") == "reality & virtual"
    assert asked == [["virtual", "reality", "navigation"],
                     ["reality & virtual & navigation", "reality & virtual"]]
    # Every word ranks.
    assert text_search.build_rank_tsquery(terms, "loose") == "virtual | reality | navigation"

    # Too rare together: the rarest word alone.
    combined_counts.update({"reality & virtual": 1})
    assert text_search.build_filter_tsquery(terms, "loose") == "reality"


def test_loose_term_survives_a_failing_or_empty_lookup() -> None:
    terms = text_terms.parse_alternatives("complex motor task")

    def broken(queries):
        raise RuntimeError("database gone")

    text_search.set_frequency_lookup(broken)
    assert text_search.build_filter_tsquery(terms, "loose") == "complex"
    text_search.set_frequency_lookup(lambda queries: None)
    assert text_search.build_filter_tsquery(terms, "loose") == "complex"
    # A word no dataset contains: every word is required, so nothing matches
    # ("sickle cell disease" must not become a search for "cell").
    text_search.set_frequency_lookup(
        lambda queries: [0, 40, 900] if len(queries) == 3 else [2]
    )
    assert text_search.build_filter_tsquery(terms, "loose") == "complex & motor & task"


def test_without_an_installed_lookup_counts_come_from_an_own_connection(monkeypatch) -> None:
    """Entry points that skip main.py's startup (benchmark, one-off runs) still get frequencies."""
    asked = []

    def own(queries):
        asked.append(queries)
        return [900, 40] if len(queries) == 2 else [12]

    monkeypatch.setattr(text_search, "_own_connection_lookup", own)
    monkeypatch.setattr(text_search, "_frequency_lookup", text_search._OWN_CONNECTION)
    terms = text_terms.parse_alternatives("eye tracking")
    assert text_search.build_filter_tsquery(terms, "loose") == "tracking & eye"
    assert asked == [["eye", "tracking"], ["tracking & eye"]]


def test_own_connection_lookup_stays_off_without_the_column_or_on_the_loop() -> None:
    caps.set_fts(False)
    assert text_search._own_connection_lookup(["eye"]) is None
    caps.set_fts(True)

    async def on_loop():
        return text_search._own_connection_lookup(["eye"])

    assert asyncio.run(on_loop()) is None


def test_frequency_lookup_refuses_to_block_the_event_loop() -> None:
    async def scenario():
        lookup = text_search.make_frequency_lookup(engine=object(), loop=asyncio.get_running_loop())
        return lookup(["alcohol"])

    assert asyncio.run(scenario()) is None


# ── Expansion ─────────────────────────────────────────────────────────────────

def test_expand_replaces_placeholders_and_collects_the_rank_query() -> None:
    sql = (
        BASE + "WHERE (EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND p2.diagnosis IN ('epilepsy')) OR TEXT_MATCH('epilepsy | epileptic', 'short')) "
        "AND text_match('alcohol') GROUP BY d.id"
    )
    expanded = text_search.expand_text_search(sql)
    assert "text_match" not in expanded.sql.lower()
    assert "d.search_tsv @@ to_tsquery('english', 'epilepsy:AB | epileptic:AB')" in expanded.sql
    assert "d.search_tsv @@ to_tsquery('english', 'alcohol')" in expanded.sql
    assert expanded.rank_tsquery == "(epilepsy | epileptic) | alcohol"
    assert text_search.is_safe_tsquery(expanded.rank_tsquery)


def test_negated_and_unsearchable_placeholders() -> None:
    expanded = text_search.expand_text_search(
        BASE + "WHERE NOT text_match('control') AND text_match('??') GROUP BY d.id"
    )
    assert "NOT d.search_tsv @@ to_tsquery('english', 'control')" in expanded.sql
    assert "AND FALSE" in expanded.sql
    # What the user excluded must not pull datasets up.
    assert expanded.rank_tsquery is None


def test_expand_uses_the_dataset_alias_of_the_query() -> None:
    expanded = text_search.expand_text_search(
        "SELECT ds.id FROM bids_datasets AS ds WHERE text_match('pain')"
    )
    assert "ds.search_tsv @@" in expanded.sql


def test_title_search_stays_ilike_but_ranks_on_the_name() -> None:
    expanded = text_search.expand_text_search(BASE + "WHERE d.name ILIKE '%Stroop%' GROUP BY d.id")
    assert "d.name ILIKE '%Stroop%'" in expanded.sql
    assert expanded.rank_tsquery == "stroop:A"


def test_stray_ilike_on_long_text_becomes_full_text_search() -> None:
    expanded = text_search.expand_text_search(
        BASE + "WHERE d.description_text ILIKE '%bipolar%' OR d.readme_text ILIKE '%7 Tesla%' "
        "OR d.readme_text NOT ILIKE '%pilot%' OR d.readme_text ILIKE '%doi.org%' "
        "OR d.license ILIKE '%CC0%' GROUP BY d.id"
    )
    assert "to_tsquery('english', 'bipolar:AB')" in expanded.sql
    assert "to_tsquery('english', '7 & tesla')" in expanded.sql
    assert "NOT d.search_tsv @@ to_tsquery('english', 'pilot')" in expanded.sql
    # URL fragments and short fields keep substring search.
    assert "d.readme_text ILIKE '%doi.org%'" in expanded.sql
    assert "d.license ILIKE '%CC0%'" in expanded.sql


def test_without_the_column_placeholders_fall_back_to_ilike() -> None:
    caps.set_fts(False)
    caps.set_available_columns({"dx_status", "study_domain"})
    sql = BASE + (
        "WHERE text_match('epilepsy | seizure disorder', 'short') "
        "AND text_match('drinking alcohol', 'loose') "
        "AND d.description_text ILIKE '%pain%' GROUP BY d.id"
    )
    expanded = text_search.expand_text_search(sql)
    assert "search_tsv" not in expanded.sql
    assert expanded.rank_tsquery is None
    assert "d.dx_status ILIKE '%epilepsy%'" in expanded.sql
    assert "d.name ILIKE '%seizure%'" in expanded.sql and "d.name ILIKE '%disorder%'" in expanded.sql
    assert "ILIKE '%drinking%'" in expanded.sql and "%alcohol%" not in expanded.sql
    assert "readme_text" not in expanded.sql
    negated = text_search.expand_text_search(BASE + "WHERE NOT text_match('eye tracking') GROUP BY d.id")
    assert "NOT ((d.name ILIKE '%eye%'" in negated.sql
    # Stray ILIKE is left alone: there is nothing better to turn it into.
    assert "d.description_text ILIKE '%pain%'" in expanded.sql


# ── Rewriter integration ──────────────────────────────────────────────────────

def test_rewriter_expands_and_returns_the_rank_query() -> None:
    rewritten = SERVICE.rewrite_generated_sql(
        BASE + "WHERE text_match('ketamine') GROUP BY d.id", "datasets mentioning ketamine"
    )
    assert "search_tsv @@" in rewritten.sql
    assert rewritten.params == {}
    assert rewritten.rank_tsquery == "ketamine"


def test_vocabulary_miss_filter_is_added_once_and_keeps_or_groups_intact() -> None:
    augmented = (
        "[VOCABULARY MISS — these terms are not in the known concept vocabulary: scan: "
        "'eye tracking'. For free-text search use: text_match('eye tracking', 'loose')]"
    )
    raw = BASE + "WHERE d.license ILIKE '%CC0%' OR d.doi IS NOT NULL GROUP BY d.id"
    rewritten = SERVICE.rewrite_generated_sql(raw, augmented)
    filter_sql = "d.search_tsv @@ to_tsquery('english', 'tracking')"
    assert filter_sql.lower() in rewritten.sql.lower()
    if sql_rewriter.parse_one is not None:
        # The old filter is parenthesised, so its OR cannot bypass the new one.
        assert re.search(r"@@ TO_TSQUERY\('english', 'tracking'\) AND \(", rewritten.sql, re.I)
    assert rewritten.rank_tsquery == "eye | tracking"

    copied = BASE + "WHERE TEXT_MATCH('eye tracking', 'loose') GROUP BY d.id"
    again = SERVICE.rewrite_generated_sql(copied, augmented)
    assert again.sql.lower().count("search_tsv @@") == 1


def test_page_sql_orders_by_structured_score_plus_text_rank() -> None:
    base = SERVICE.rewrite_generated_sql(
        BASE + "WHERE text_match('epilepsy', 'short') GROUP BY d.id", "epilepsy"
    )
    page_sql, params = build_page_sql(
        base.sql,
        scored_filters=[{"field": "diagnosis", "code": "epilepsy", "score": 0.9}],
        apply_relevance=True, limit=20, offset=0, rank_tsquery=base.rank_tsquery,
    )
    assert "matched_codes" in page_sql and "TS_RANK_CD" in page_sql.upper()
    assert re.search(r"relevance_score \+ ranked\.text_rank DESC", page_sql)
    assert params == {"_limit": 20, "_offset": 0}

    # Text terms alone are enough to rank.
    text_only, _ = build_page_sql(base.sql, [], True, 20, 0, rank_tsquery="epilepsy")
    assert "matched_codes" not in text_only and "text_rank" in text_only

    # The user asked for an order: nothing is layered on top.
    plain, _ = build_page_sql(base.sql, [], False, 20, 0, rank_tsquery="epilepsy")
    assert "text_rank" not in plain

    # A query this module did not build is never inlined.
    unsafe, _ = build_page_sql(base.sql, [], True, 20, 0, rank_tsquery="x'); DROP TABLE users; --")
    assert "DROP TABLE" not in unsafe and "text_rank" not in unsafe


def test_generated_sql_survives_the_sqlglot_round_trip() -> None:
    """Run this when the sqlglot pin in constraints.txt changes.

    sqlglot 30.x renders a `:name` bind as `%(name)s`, which SQLAlchemy does not
    bind. Text queries are therefore inlined; only LIMIT/OFFSET are bound, and
    they are added after the last sqlglot pass.
    """
    base = SERVICE.rewrite_generated_sql(
        BASE + "WHERE text_match('resting-state | n-back', 'short') GROUP BY d.id", "q"
    )
    page_sql, _ = build_page_sql(base.sql, [], True, 20, 0, rank_tsquery=base.rank_tsquery)
    for sql in (build_count_sql(base.sql), page_sql):
        assert "%(" not in sql
        assert "(resting:AB <-> state:AB) | (n:AB <-> back:AB)" in sql
    assert ":_limit" in page_sql and ":_offset" in page_sql


# ── Augmented question (what the SQL model is given to copy) ──────────────────

def test_augmented_question_hands_the_model_text_match_placeholders(monkeypatch) -> None:
    import preprocess

    monkeypatch.setattr(text_terms, "_lookup_strings", _lookup)
    plan = preprocess.QueryPlan(
        query_family="combined_filter",
        natural_language_summary="Bilingual epilepsy datasets with eye tracking",
        description_search="bilingual",
    )
    resolved = [preprocess.ResolvedTerms(
        field=preprocess.RAGField.DIAGNOSIS, term_to_codes={"epilepsy": ["epilepsy"]}
    )]
    question = preprocess._build_augmented_question(
        plan, resolved, unresolved_terms={"scan": ["eye tracking"]}
    )
    assert (
        "diagnosis: (EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND p2.diagnosis IN ('epilepsy')) OR text_match('epilepsy | seizures | epileptic', 'short'))"
    ) in question
    assert "dataset text: text_match('bilingual')" in question
    assert "For free-text search use: text_match('eye tracking', 'loose')]" in question
    assert "ILIKE '%" not in question


def test_unresolved_term_alone_still_gets_its_text_filter() -> None:
    import preprocess

    plan = preprocess.QueryPlan(query_family="scan_filter",
                                natural_language_summary="Datasets with eye tracking")
    question = preprocess._build_augmented_question(plan, [], unresolved_terms={"scan": ["eye tracking"]})
    assert "text_match('eye tracking', 'loose')" in question
    rewritten = SERVICE.rewrite_generated_sql(BASE + "GROUP BY d.id", question)
    assert "search_tsv @@" in rewritten.sql


def test_resolved_task_filter_keeps_a_readme_text_fallback(monkeypatch) -> None:
    import preprocess

    plan = preprocess.QueryPlan(
        query_family="scan_filter",
        natural_language_summary="Assessments related to memory",
    )
    resolved = [preprocess.ResolvedTerms(
        field=preprocess.RAGField.TASK,
        term_to_codes={"assessments related to memory": ["memory_task"]},
    )]
    question = preprocess._build_augmented_question(plan, resolved)
    assert (
        'scan "assessments related to memory": '
        "(EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id "
        "AND o2.task = 'memory_task') OR text_match('memory', 'short'))"
    ) in question
    assert "bids_objects o2" in question


# ── Capability detection and DDL ──────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def mappings(self):
        return self

    def one(self):
        return self._rows[0]

    def scalar(self):
        return None  # no summary table


class _FakeConnection:
    def __init__(self, columns):
        self._columns = columns

    async def execute(self, statement, params=None):
        if "information_schema.columns" in str(statement):
            return _FakeResult([(name,) for name in self._columns])
        return _FakeResult([{"total": 10}])

    def begin_nested(self):
        return contextlib.nullcontext()


def test_full_text_capability_follows_the_column() -> None:
    caps.set_fts(False)
    asyncio.run(caps.detect_available_columns(_FakeConnection(["id", "name", "search_tsv"])))
    assert caps.has_fts()
    asyncio.run(caps.detect_available_columns(_FakeConnection(["id", "name"])))
    assert not caps.has_fts()


def test_migration_and_crawler_define_the_same_search_vector() -> None:
    from remote import openneuro_metadata as meta

    def squash(sql: str) -> str:
        return re.sub(r"\s+", "", sql)

    migration = (ROOT / "crawlers" / "migrate_add_search_tsv.sql").read_text()
    assert squash(meta.SEARCH_VECTOR_EXPRESSION) in squash(migration)
    statements = meta.search_vector_sql()
    # The profile columns the expression reads come first.
    profile_statements = statements[: len(meta.PROFILE_COLUMNS)]
    assert all("ADD COLUMN IF NOT EXISTS llm_" in st for st in profile_statements)
    assert "ADD COLUMN IF NOT EXISTS search_tsv" in statements[-2]
    assert "CREATE INDEX IF NOT EXISTS" in statements[-1]
    # The weights the ranking assumes: name A, short fields B, LLM profile C, README D.
    assert re.findall(r"'([ABCD])'\)", meta.SEARCH_VECTOR_EXPRESSION) == ["A", "B", "C", "D"]
    assert meta.SEARCH_VECTOR_EXPRESSION.count("translate(") == 4
    assert "llm_conditions" in meta.SEARCH_VECTOR_EXPRESSION.split("'B')")[0]

    profiles = (ROOT / "crawlers" / "migrate_add_readme_profiles.sql").read_text()
    added = dict(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)\s+([A-Z]+)", profiles))
    assert added == meta.PROFILE_COLUMNS
