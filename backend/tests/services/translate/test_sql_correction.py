"""Self-correction repairs the query that ran; it does not rewrite it again.

The query sent for repair already went through every rewrite, the uncertain
tier included. Rewriting the model's repair a second time put the dataset
columns (d.id, ..., COUNT(DISTINCT o.subject)) over the tier's wrapper, whose
outer query has no table d: every repaired tiered query failed and the search
showed the 200-dataset sample (qf12, qf27 of the gold requests)."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlglot")

from sqlglot import exp, parse_one  # noqa: E402

from config.settings import settings_obj
from services.uncertain import tier as uncertain_tier  # noqa: E402
from services.translate import text_to_sql  # noqa: E402
from services.translate.sql_rewriter import (  # noqa: E402
    FALLBACK_SQL, finish_corrected_sql, rewrite_generated_sql,
)

MODEL_SQL = (
    "SELECT {{COLS}} FROM bids_datasets d "
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL "
    "WHERE EXISTS (SELECT 1 FROM bids_objects bo WHERE bo.dataset_id = d.id "
    "AND bo.datatype = 'functional_mri') "
    "AND (EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
    "AND p2.diagnosis IN ('adhd')) OR text_match('adhd | attention deficit', 'short')) "
    "GROUP BY d.id"
)
QUESTION = "resting-state fMRI in adults with ADHD"


@pytest.fixture
def tiered_sql(monkeypatch) -> str:
    monkeypatch.setattr(settings_obj, "UNCERTAIN_TIER", True)
    sql = rewrite_generated_sql(MODEL_SQL, QUESTION).sql
    assert uncertain_tier.LEVEL_COLUMN in sql
    return sql


def _outer_columns(sql: str) -> list:
    return [e.sql(dialect="postgres") for e in parse_one(sql, read="postgres").expressions]


def _repair(sql: str) -> str:
    """What a model repair looks like: one condition changed."""
    assert "'functional_mri'" in sql
    return sql.replace("'functional_mri'", "'anatomical_mri'")


def test_a_repaired_tiered_query_keeps_the_tier(tiered_sql):
    repaired = finish_corrected_sql(_repair(tiered_sql), tiered_sql)
    assert "'anatomical_mri'" in repaired
    assert _outer_columns(repaired) == _outer_columns(tiered_sql)
    assert uncertain_tier.LEVEL_COLUMN in _outer_columns(repaired)[1]


def test_a_repaired_tiered_query_has_no_dataset_columns_outside_the_wrapper(tiered_sql):
    """The old second rewrite made the outer query read d.id with no d in its FROM."""
    repaired = parse_one(finish_corrected_sql(_repair(tiered_sql), tiered_sql), read="postgres")
    outer_tables = {c.table for e in repaired.expressions for c in e.find_all(exp.Column)}
    assert outer_tables <= {"tier_base", "evidence"}


def test_a_changed_column_list_is_put_back(tiered_sql):
    model_answer = tiered_sql.replace("tier_base.*", "tier_base.id", 1)
    repaired = finish_corrected_sql(_repair(model_answer), tiered_sql)
    assert _outer_columns(repaired) == _outer_columns(tiered_sql)


def test_an_unchanged_answer_returns_the_failed_query(tiered_sql):
    """search_runner then goes to the fallback sample without running it twice."""
    reformatted = " ".join(tiered_sql.split())
    assert finish_corrected_sql(reformatted, tiered_sql) == tiered_sql


def test_an_answer_without_sql_is_the_fallback(tiered_sql):
    assert finish_corrected_sql("I cannot fix this query.", tiered_sql) == FALLBACK_SQL


def test_an_untiered_query_is_repaired_the_same_way(monkeypatch):
    monkeypatch.setattr(settings_obj, "UNCERTAIN_TIER", False)
    sql = rewrite_generated_sql(MODEL_SQL, QUESTION).sql
    repaired = finish_corrected_sql(f"```sql\n{_repair(sql)}\n```", sql)
    assert "'anatomical_mri'" in repaired
    assert _outer_columns(repaired) == _outer_columns(sql)


def test_the_correction_call_returns_a_runnable_tiered_query(tiered_sql, monkeypatch):
    prompts = []

    def fake_generate(prompt, **kwargs):
        prompts.append(prompt)
        return _repair(tiered_sql)

    monkeypatch.setattr(text_to_sql, "llm_generate", fake_generate)
    corrected = text_to_sql._correct_sql(
        tiered_sql, 'invalid input value for enum: "functional_mri"', QUESTION)
    assert prompts and tiered_sql in prompts[0]
    assert "'anatomical_mri'" in corrected
    assert _outer_columns(corrected) == _outer_columns(tiered_sql)
