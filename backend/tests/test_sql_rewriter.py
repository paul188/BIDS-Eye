from __future__ import annotations

from services.sql_rewriter import FALLBACK_SQL, SqlRewriteService, build_count_sql, build_page_sql


SERVICE = SqlRewriteService()


def test_extract_sql_pulls_sql_from_fenced_response() -> None:
    raw = "Here you go:\n```sql\nSELECT 1\n```"
    assert SERVICE.extract_sql(raw) == "SELECT 1"


def test_rewrite_generated_sql_replaces_projection_in_plain_select() -> None:
    raw_sql = (
        "SELECT {{COLS}}\n"
        "FROM bids_datasets d\n"
        "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        "GROUP BY d.id"
    )

    rewritten = SERVICE.rewrite_generated_sql(raw_sql, "Show me datasets")

    assert "{{COLS}}" not in rewritten.sql
    assert "COUNT(DISTINCT o.subject) AS subject_count" in rewritten.sql
    assert rewritten.params == {}


def test_rewrite_generated_sql_handles_cte_and_nested_selects() -> None:
    raw_sql = (
        "WITH base AS (SELECT 1 AS n)\n"
        "SELECT {{COLS}}\n"
        "FROM (\n"
        "  SELECT * FROM base\n"
        ") inner_query\n"
        "LEFT JOIN bids_objects o ON o.dataset_id = inner_query.n AND o.subject IS NOT NULL\n"
        "GROUP BY inner_query.n"
    )

    rewritten = SERVICE.rewrite_generated_sql(raw_sql, "Show me datasets")

    assert rewritten.sql.startswith("WITH base AS")
    assert "COUNT(DISTINCT o.subject) AS subject_count" in rewritten.sql
    assert "inner_query" in rewritten.sql


def test_rewrite_generated_sql_strips_bad_outer_filters_and_adds_vocab_miss_clause() -> None:
    raw_sql = (
        "SELECT {{COLS}}\n"
        "FROM bids_datasets d\n"
        "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        "WHERE o.task = 'complex_motor_task' AND d.dataset_type = 'raw'\n"
        "GROUP BY d.id"
    )
    augmented_question = (
        "[VOCABULARY MISS] task term not resolved. For free-text search use: "
        "text_match('alcohol', 'loose')]"
    )

    # A database without the full-text column: the placeholder becomes ILIKE.
    rewritten = SERVICE.rewrite_generated_sql(raw_sql, augmented_question)

    assert "o.task = 'complex_motor_task'" not in rewritten.sql
    assert "text_match" not in rewritten.sql.lower()
    assert "d.name ILIKE '%alcohol%'" in rewritten.sql
    assert "d.description_text ILIKE '%alcohol%'" in rewritten.sql
    assert "d.dataset_type = 'raw'" in rewritten.sql
    assert rewritten.params == {}
    assert rewritten.rank_tsquery is None


def test_build_count_and_page_sql_wrap_base_query() -> None:
    base_sql = SERVICE.rewrite_generated_sql(
        "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL GROUP BY d.id",
        "Show me datasets",
    ).sql

    assert build_count_sql(base_sql).startswith("SELECT COUNT(*) FROM (")

    page_sql, params = build_page_sql(
        base_sql,
        scored_filters=[{"field": "task", "code": "resting_state", "score": 0.87}],
        apply_relevance=True,
        limit=20,
        offset=40,
    )

    assert "relevance_score" in page_sql
    assert "matched_codes" in page_sql
    assert params == {"_limit": 20, "_offset": 40}


def test_fallback_sql_remains_available() -> None:
    assert "LIMIT 200" in FALLBACK_SQL



def test_extract_sql_drops_a_trailing_closing_tag() -> None:
    """The prompt ends with "[SQL]"; Gemini then often answers SQL + "[/SQL]"."""
    sql = "SELECT {{COLS}} FROM bids_datasets d\n\nWHERE d.license = 'CC0'\nGROUP BY d.id"
    for raw in (
        sql + "[/SQL]",
        sql + "\n[/SQL]\nThis query finds CC0 datasets.",
        "[SQL]\n" + sql + "\n```",
        "```sql\n" + sql + "\n[/SQL]\n```",
    ):
        assert SERVICE.extract_sql(raw) == sql, raw
    # No marker: prose after a blank line is still dropped.
    assert SERVICE.extract_sql("SELECT 1\n\nThis returns one.") == "SELECT 1"
