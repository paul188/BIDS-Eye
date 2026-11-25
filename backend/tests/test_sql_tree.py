"""Reading the tree of a generated query (services/sql_tree.py)."""

from __future__ import annotations

import pytest
from sqlglot import exp

from services import sql_tree


def _where(condition: str):
    tree = sql_tree.parse_select(f"SELECT d.id FROM bids_datasets AS d WHERE {condition}")
    assert tree is not None
    return tree.args["where"].this


@pytest.mark.parametrize("sql", [None, "", "not sql", "SELECT FROM WHERE ("])
def test_unreadable_sql_gives_none(sql):
    assert sql_tree.parse_select(sql) is None


def test_only_a_select_is_a_select():
    union = "SELECT 1 UNION SELECT 2"
    assert sql_tree.parse(union) is not None and sql_tree.parse_select(union) is None


@pytest.mark.parametrize("written, read", [
    ("d.name NOT ILIKE 'a'", "NOT d.name ILIKE 'a'"),
    ("d.name NOT LIKE 'a'", "NOT d.name LIKE 'a'"),
    ("NOT d.name NOT ILIKE 'a'", "NOT NOT d.name ILIKE 'a'"),
    ("d.name NOT ILIKE 'a' ESCAPE '!'", "NOT d.name ILIKE 'a' ESCAPE '!'"),
    ("d.name NOT ILIKE 'a' AND d.name NOT ILIKE 'b'", "NOT d.name ILIKE 'a' AND NOT d.name ILIKE 'b'"),
    ("d.name NOT ILIKE 'a' OR d.id = 1", "NOT d.name ILIKE 'a' OR d.id = 1"),
    ("(d.name NOT ILIKE 'a') IS TRUE", "(NOT d.name ILIKE 'a') IS TRUE"),
    ("EXISTS(SELECT 1 FROM t WHERE t.x NOT LIKE 'a')", "EXISTS(SELECT 1 FROM t WHERE NOT t.x LIKE 'a')"),
    # Not the same as NOT (… ANY …): left as it is.
    ("d.name NOT ILIKE ANY(ARRAY['a'])", "d.name NOT ILIKE ANY(ARRAY['a'])"),
    ("d.name ILIKE 'a'", "d.name ILIKE 'a'"),
])
def test_not_like_is_read_as_a_not(written, read):
    node = _where(written)
    assert node.sql(dialect="postgres") == read
    # Written back and read again it is the same.
    assert _where(read) == node


@pytest.mark.parametrize("condition, negated", [
    ("d.name ILIKE 'a'", False),
    ("NOT d.name ILIKE 'a'", True),
    ("NOT (d.id = 1 OR d.name ILIKE 'a')", True),
    ("NOT (NOT d.name ILIKE 'a')", True),
    ("d.name NOT ILIKE 'a'", True),
    ("d.name NOT ILIKE ANY(ARRAY['a'])", True),
    ("(d.name ILIKE 'a') IS FALSE", True),
    ("(d.name ILIKE 'a') = FALSE", True),
    ("FALSE = (d.name ILIKE 'a')", True),
    ("(d.name ILIKE 'a') IS NOT TRUE", True),
    ("(d.name ILIKE 'a') IS TRUE", False),
    ("d.name ILIKE 'a' AND NOT d.species = 'b'", False),
])
def test_under_negation(condition, negated):
    node = _where(condition)
    column = next(c for c in node.find_all(exp.Column) if c.name == "name")
    assert sql_tree.under_negation(column, node) is negated


def test_ancestors_include_the_root_and_never_the_node():
    node = _where("NOT d.species = 'a'")
    column = node.find(exp.Column)
    assert [type(a) for a in sql_tree.ancestors(column, node)] == [exp.EQ, exp.Not]
    assert list(sql_tree.ancestors(node, node)) == []


def test_in_subquery():
    node = _where("d.id IN (SELECT p.x FROM p)")
    inside = {c.name: sql_tree.in_subquery(c, node) for c in node.find_all(exp.Column)}
    assert inside == {"id": False, "x": True}


def test_or_branches_go_through_parentheses_and_not_through_not():
    node = _where("((d.a = 1 OR (d.b = 2 OR d.c = 3)) OR NOT (d.e = 4 OR d.f = 5))")
    assert [b.sql() for b in sql_tree.or_branches(node)] == [
        "d.a = 1", "d.b = 2", "d.c = 3", "NOT (d.e = 4 OR d.f = 5)"]


def test_without_literals_holds_no_value():
    node = _where("d.name ILIKE '%parkinson%' AND d.age_min < 3")
    assert sql_tree.without_literals(node) == "d.name ILIKE ? AND d.age_min < ?"
    assert "parkinson" in node.sql()   # the node itself is unchanged
