"""Participant handedness compared with the stored codes.

Four real requests for left-handed participants found 0 datasets
(2026-11-18): the SQL model wrote p2.handedness = 'left' and
p.handedness ~* '\\mleft\\M', the database stores 'left_handed'.
"""

from __future__ import annotations

import pytest

from constants import build_schema_ddl
from services.translate.sql_rewriter import SqlRewriteService

SERVICE = SqlRewriteService()


@pytest.mark.parametrize("value, code", [
    ("left", "left_handed"), ("L", "left_handed"), ("Left-handed", "left_handed"),
    ("left handers", "left_handed"), ("LH", "left_handed"), ("left_handed", "left_handed"),
    ("right", "right_handed"), ("R", "right_handed"), ("right-handed", "right_handed"),
    ("Ambidextrous", "ambidextrous"), ("mixed", "ambidextrous"),
    ("unknown", "unknown_handedness"),
])
def test_equality_becomes_the_stored_code(value: str, code: str) -> None:
    sql = f"WHERE p.handedness = '{value}'"
    assert SERVICE._handedness_codes(sql) == f"WHERE LOWER(p.handedness) = '{code}'"


def test_negation_kept() -> None:
    assert SERVICE._handedness_codes("WHERE p.handedness <> 'right'") == (
        "WHERE LOWER(p.handedness) <> 'right_handed'")


def test_in_list() -> None:
    sql = "WHERE p2.handedness IN ('L', 'left', 'ambidextrous')"
    assert SERVICE._handedness_codes(sql) == (
        "WHERE LOWER(p2.handedness) IN ('left_handed', 'ambidextrous')")


@pytest.mark.parametrize("condition", [
    r"p.handedness ~* '\mleft\M'", "p.handedness ~* 'left'", "p.handedness ILIKE '%left%'",
    "p.handedness LIKE 'left%'", "p.handedness ~* '^(l|left)$'",
])
def test_patterns_become_the_stored_code(condition: str) -> None:
    assert SERVICE._handedness_codes(f"WHERE {condition}") == (
        "WHERE LOWER(p.handedness) = 'left_handed'")


def test_negated_pattern() -> None:
    assert SERVICE._handedness_codes("WHERE p.handedness NOT ILIKE '%right%'") == (
        "WHERE LOWER(p.handedness) <> 'right_handed'")


@pytest.mark.parametrize("sql", [
    "WHERE LOWER(p.handedness) = 'left_handed'",
    "WHERE p.handedness IS NOT NULL",
    "WHERE p.handedness = '3.34'",                  # an Edinburgh score, not a code
    "WHERE p.handedness IN ('left', '3.34')",
    "WHERE p.handedness IN (SELECT x FROM t)",
    "WHERE p.handedness ~* '(left|right)'",          # two codes: not ours to pick
    "WHERE p.handedness ILIKE '%'",
])
def test_left_alone(sql: str) -> None:
    assert SERVICE._handedness_codes(sql) == sql


def test_logged_request_finds_left_handed_participants() -> None:
    """"Dataset with at least 20 patients all left handed" (query_log, 2026-11-18)."""
    raw_sql = (
        "SELECT {{COLS}} FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
        "AND NOT o.subject IS NULL GROUP BY d.id HAVING (SELECT COUNT(DISTINCT p2.participant_id) "
        "FROM bids_participants AS p2 WHERE p2.dataset_id = d.id AND p2.handedness = 'left') >= 20"
    )
    rewritten = SERVICE.rewrite_generated_sql(raw_sql, "Dataset with at least 20 patients all left handed").sql
    assert "LOWER(p2.handedness) = 'left_handed'" in rewritten
    assert "handedness = 'left'" not in rewritten


def test_schema_names_the_stored_codes() -> None:
    line = next(line for line in build_schema_ddl().splitlines() if line.strip().startswith("handedness"))
    assert "'left_handed'" in line and "'L'" not in line
