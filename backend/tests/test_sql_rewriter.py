from __future__ import annotations

import pytest

from services.translate.sql_rewriter import FALLBACK_SQL, SqlRewriteService, build_count_sql, build_page_sql


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


def test_number_casts_of_json_values_cannot_crash_the_query() -> None:
    """'7 Tesla': one sidecar value like 'n/a' used to fail the whole query."""
    from services.translate import sql_rewriter

    if sql_rewriter.parse_one is None:
        return
    raw_sql = (
        "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
        "AND o.subject IS NOT NULL WHERE EXISTS (SELECT 1 FROM bids_objects bo WHERE "
        "bo.dataset_id = d.id AND CAST((bo.other_entities ->> 'MagneticFieldStrength') AS DOUBLE PRECISION) "
        "BETWEEN 6.5 AND 7.5) AND EXISTS (SELECT 1 FROM bids_participants p WHERE "
        "p.dataset_id = d.id AND (p.extra->>'bmi')::numeric > 30) GROUP BY d.id"
    )
    sql = SERVICE.rewrite_generated_sql(raw_sql, "7 Tesla datasets").sql
    assert sql.count("CASE WHEN") == 2
    assert "~ '^[[:space:]]*[-+]?" in sql
    assert "THEN CAST((bo.other_entities ->> 'MagneticFieldStrength') AS DOUBLE PRECISION) END" in sql
    assert "THEN CAST((p.extra ->> 'bmi') AS DECIMAL) END" in sql


def test_already_guarded_and_non_number_casts_are_left_alone() -> None:
    from services.translate import sql_rewriter

    if sql_rewriter.parse_one is None:
        return
    guarded = (
        "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
        "AND o.subject IS NOT NULL WHERE EXISTS (SELECT 1 FROM bids_objects bo WHERE bo.dataset_id = d.id "
        "AND CASE WHEN (bo.other_entities->>'EchoTime') ~ '^[0-9.]+$' "
        "THEN (bo.other_entities->>'EchoTime')::float > 0.03 ELSE FALSE END "
        "AND (bo.other_entities->>'Manufacturer')::text = 'Siemens') GROUP BY d.id"
    )
    sql = SERVICE.rewrite_generated_sql(guarded, "q").sql
    assert sql.count("CASE WHEN") == 1


def _outer_order(page_sql: str) -> str:
    """The ORDER BY of the page query itself (after the last subquery)."""
    tail = page_sql[page_sql.rfind(") AS page_result") + len(") AS page_result"):]
    return tail.split("LIMIT")[0]


def test_every_page_ends_its_order_with_the_dataset_id() -> None:
    """LIMIT/OFFSET over a tied or unordered result can repeat or skip rows."""
    base_sql = SERVICE.rewrite_generated_sql(
        "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
        "AND o.subject IS NOT NULL WHERE d.license = 'CC0' GROUP BY d.id",
        "CC0 datasets",
    ).sql
    ranked, _ = build_page_sql(
        base_sql,
        scored_filters=[{"field": "task", "code": "resting_state", "score": 1.0}],
        apply_relevance=True, limit=20, offset=20,
    )
    assert _outer_order(ranked).strip().endswith("page_result.id")
    assert "page_result.relevance_score + page_result.text_rank DESC" in _outer_order(ranked)

    unranked, _ = build_page_sql(base_sql, [], True, 20, 20)
    assert _outer_order(unranked).strip() == (
        "ORDER BY page_result.subject_count DESC NULLS LAST, page_result.id"
    )


def test_a_requested_order_is_kept_on_the_page() -> None:
    base_sql = SERVICE.rewrite_generated_sql(
        "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
        "AND o.subject IS NOT NULL GROUP BY d.id ORDER BY d.size_bytes DESC LIMIT 10",
        "the ten largest datasets",
    ).sql
    page_sql, _ = build_page_sql(base_sql, [], False, 20, 0)
    assert "ORDER BY" not in _outer_order(page_sql)


_UNIFIED = ("COALESCE(bo.other_entities ->> 'ManufacturersModelName', "
            "bo.other_entities ->> 'ManufacturerModelName')")


@pytest.mark.parametrize("key", ["ManufacturersModelName", "ManufacturerModelName"])
def test_the_scanner_model_is_read_under_both_sidecar_keys(key: str) -> None:
    """BIDS names it ManufacturersModelName, a few datasets write the DICOM
    ManufacturerModelName: whichever the model writes, both are searched."""
    raw_sql = ("SELECT d.id FROM bids_datasets d WHERE EXISTS (SELECT 1 FROM bids_objects bo "
               f"WHERE bo.dataset_id = d.id AND bo.other_entities->>'{key}' ILIKE '%Prisma%')")
    sql = SERVICE.rewrite_generated_sql(raw_sql, "Prisma scanner").sql
    assert f"{_UNIFIED} ILIKE '%Prisma%'" in sql
    assert sql.count("ModelName") == 2


def test_the_unified_scanner_model_and_other_keys_are_left_alone() -> None:
    raw_sql = ("SELECT d.id FROM bids_datasets d WHERE EXISTS (SELECT 1 FROM bids_objects bo "
               f"WHERE bo.dataset_id = d.id AND {_UNIFIED} ILIKE '%Skyra%' "
               "AND bo.other_entities ->> 'Manufacturer' ILIKE '%Siemens%')")
    once = SERVICE._unify_scanner_model_key(raw_sql)
    assert SERVICE._unify_scanner_model_key(once) == once
    assert once.count("ModelName") == 2
    assert "bo.other_entities ->> 'Manufacturer' ILIKE" in once


def test_a_condition_is_anded_onto_every_branch_of_a_union() -> None:
    service = SqlRewriteService()
    sql = ("SELECT d.id FROM bids_datasets d WHERE d.name = 'a' OR d.name = 'b' "
           "UNION SELECT d.id FROM bids_datasets d")
    out = service._and_into_where(sql, "d.n_subjects >= 10")
    assert out.count("d.n_subjects >= 10") == 2
    # The old filter keeps its OR inside brackets.
    assert "(d.name = 'a' OR d.name = 'b')" in out


def test_a_condition_is_not_added_to_sql_that_cannot_be_parsed() -> None:
    # The guard refuses such SQL; the old string edit put the condition in
    # front of the filter without brackets.
    sql = "SELECT d.id FROM bids_datasets d WHERE EXISTS (TABLE users)"
    assert SqlRewriteService()._and_into_where(sql, "d.n_subjects >= 10") == sql
