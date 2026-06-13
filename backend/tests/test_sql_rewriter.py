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
        "d.name ILIKE '%alcohol%' OR d.description_text ILIKE '%alcohol%' ]"
    )

    rewritten = SERVICE.rewrite_generated_sql(raw_sql, augmented_question)

    assert "o.task = 'complex_motor_task'" not in rewritten.sql
    assert "ILIKE" in rewritten.sql
    assert rewritten.params == {"vocab_miss_0": "%alcohol%"}
    assert ":vocab_miss_0" in rewritten.sql


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

