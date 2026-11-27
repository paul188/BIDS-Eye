"""Vague task words in generated SQL (sql_rewriter._rewrite_task_ilike_concepts)."""

from __future__ import annotations

import pytest

from services import sql_tree
from services.translate.sql_rewriter import SqlRewriteService

_FILES = "(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id AND {})"
_COGNITIVE = "o2.task ILIKE '%cognitive%'"


def _rewritten(condition: str) -> str:
    sql = SqlRewriteService()._rewrite_task_ilike_concepts(
        f"SELECT d.id FROM bids_datasets AS d WHERE {condition}")
    tree = sql_tree.parse_select(sql)
    where = tree.args.get("where")
    return where.this.sql(dialect="postgres") if where is not None else ""


@pytest.mark.parametrize("condition, rewritten", [
    ("EXISTS" + _FILES.format(_COGNITIVE), "EXISTS" + _FILES.format("TRUE")),
    # Denied: the condition goes, what stands next to it stays.
    ("EXISTS" + _FILES.format("NOT " + _COGNITIVE),
     "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id)"),
    ("EXISTS" + _FILES.format("o2.task NOT ILIKE '%cognitive%'"),
     "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id)"),
    ("NOT EXISTS" + _FILES.format(_COGNITIVE), ""),
    ("d.age_min > 3 AND NOT EXISTS" + _FILES.format(_COGNITIVE), "d.age_min > 3"),
    ("d.age_min > 3 AND NOT (EXISTS" + _FILES.format(_COGNITIVE) + ")", "d.age_min > 3"),
    ("NOT EXISTS" + _FILES.format("(o2.task = 'rest' OR " + _COGNITIVE + ")"),
     "NOT EXISTS" + _FILES.format("(o2.task = 'rest')")),
    ("NOT (d.species = 'Mouse' OR EXISTS" + _FILES.format(_COGNITIVE) + ")", "NOT (d.species = 'Mouse')"),
    ("d.age_min > 3 AND NOT d.id IN (SELECT dataset_id FROM bids_objects AS o2 WHERE " + _COGNITIVE + ")",
     "d.age_min > 3"),
    ("NOT EXISTS" + _FILES.format(_COGNITIVE) + " AND NOT EXISTS" + _FILES.format("o2.task ILIKE '%social%'"), ""),
    ("o.task ILIKE '%flanker%'", "o.task ILIKE '%flanker%'"),
    ("t.task ILIKE '%cognitive%'", "t.task ILIKE '%cognitive%'"),
])
def test_a_vague_word(condition, rewritten):
    assert _rewritten(condition) == rewritten


@pytest.mark.parametrize("condition", ["NOT o2.task ILIKE '%naturalistic%'", "o2.task NOT ILIKE '%naturalistic%'"])
def test_a_category_is_expanded_under_either_not(condition):
    sql = _rewritten("EXISTS" + _FILES.format(condition))
    assert "NOT o2.task IN (" in sql and "'movie_viewing'" in sql


def test_sql_without_such_a_word_is_not_touched():
    sql = "SELECT d.id FROM bids_datasets d WHERE d.name  ILIKE '%cognitive%'"
    assert SqlRewriteService()._rewrite_task_ilike_concepts(sql) == sql
