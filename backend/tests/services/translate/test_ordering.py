"""The order and number of results the user asks for (QueryPlan.ordering).

The compiler writes them as the base query's own ORDER BY and LIMIT; the
rewriter and the page builder keep both.
"""
from __future__ import annotations

import pytest

from preprocess import Ordering, ParticipantGroup, QueryPlan, RAGField, ResolvedTerms, SortKey
from services.translate import participant_query_compiler as pqc
from services.translate import sql_rewriter

RESOLVED = [ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes={"schizophrenia": ["schizophrenia"]})]


def _plan(*diagnoses: str, **ordering) -> QueryPlan:
    groups = [ParticipantGroup(group_name="g0", diagnosis_terms=list(diagnoses))] if diagnoses else []
    return QueryPlan(natural_language_summary="demo", groups=groups,
                     ordering=Ordering(**ordering) if ordering else None)


def test_every_sort_key_has_a_column() -> None:
    assert set(pqc.ORDER_COLUMNS) == {key.value for key in SortKey}


def test_the_largest_n_compile_with_order_and_limit() -> None:
    sql = pqc.compile_plan(_plan("schizophrenia", sort_key="subjects", limit=10), RESOLVED).sql
    assert sql is not None
    assert sql.endswith("GROUP BY d.id\nORDER BY subject_count DESC NULLS LAST, d.id\nLIMIT 10")


def test_ascending_without_a_limit() -> None:
    sql = pqc.compile_plan(_plan("schizophrenia", sort_key="name", direction="ASC"), RESOLVED).sql
    assert sql is not None
    assert sql.endswith("ORDER BY d.name ASC NULLS LAST, d.id")
    assert "LIMIT" not in sql


def test_an_order_alone_may_list_every_dataset() -> None:
    sql = pqc.compile_plan(_plan(sort_key="published", limit=5), []).sql
    assert sql is not None
    assert "WHERE" not in sql
    assert sql.endswith("ORDER BY d.publish_date DESC NULLS LAST, d.id\nLIMIT 5")


def test_a_limit_without_a_sort_key_keeps_the_relevance_order() -> None:
    plan = _plan("schizophrenia", limit=5)
    assert not plan.user_ordered
    sql = pqc.compile_plan(plan, RESOLVED).sql
    assert sql is not None and "ORDER BY" not in sql and "LIMIT" not in sql
    # Nothing else to filter: no search to run.
    assert pqc.compile_plan(_plan(limit=5), []).reason == "nothing to filter"


@pytest.mark.parametrize("ordering, expected", [
    ({"sort_key": "subjects"}, True),
    ({"sort_key": "subjects", "limit": 0}, True),
    ({"limit": 3}, False),
    ({"sort_key": "impact_factor"}, None),   # unknown key and no limit: no ordering at all
    ({}, None),
])
def test_the_plan_keeps_only_an_ordering_it_can_use(ordering, expected) -> None:
    plan = QueryPlan.model_validate({"natural_language_summary": "demo", "ordering": ordering})
    if expected is None:
        assert plan.ordering is None and not plan.user_ordered
    else:
        assert plan.user_ordered is expected
    if ordering.get("limit") == 0:
        assert plan.ordering.limit is None


def test_old_plan_keys_are_ignored() -> None:
    # Stored plans from before the ordering field load without one.
    plan = QueryPlan.model_validate({"natural_language_summary": "demo", "result_limit": 10,
                                     "order_by": "subject_count", "order_direction": "ASC"})
    assert plan.ordering is None


def test_the_rewriter_and_the_page_keep_the_order_and_the_limit() -> None:
    sql = pqc.compile_plan(_plan("schizophrenia", sort_key="size", limit=3), RESOLVED).sql
    rewritten = sql_rewriter.rewrite_generated_sql(sql, "the 3 largest schizophrenia datasets").sql
    assert "ORDER BY d.size_bytes DESC NULLS LAST" in rewritten
    assert "LIMIT 3" in rewritten
    page, _ = sql_rewriter.build_page_sql(rewritten, [], apply_relevance=False, limit=20, offset=0)
    # No second ORDER BY over the page: it would undo the user's order.
    assert page.count("ORDER BY") == 1
