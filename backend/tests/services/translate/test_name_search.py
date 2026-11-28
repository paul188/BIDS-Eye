"""A dataset name the user asks for is data, whatever characters it holds.

name_search was written into the ILIKE pattern unescaped: "Alzheimer's" broke
the query and "%" matched every name. The Postgres part needs a scratch
database (BIDSEYE_TEST_PG_DSN, tests/postgres.py).
"""

from __future__ import annotations

import asyncio

import pytest

sqlglot = pytest.importorskip("sqlglot")
from sqlglot import exp  # noqa: E402

import preprocess as pp  # noqa: E402
from services import query_guard  # noqa: E402
from services.results import query_filters  # noqa: E402
from services.translate import sql_rewriter  # noqa: E402
from services.translate.participant_query_compiler import compile_plan  # noqa: E402
from sql_text import contains_pattern, pattern_words  # noqa: E402

from tests.postgres import fresh_schema, needs_postgres  # noqa: E402

NAMES = [
    "abide", "Alzheimer's", "100% sleep", "HCP_YA", "back\\slash", "ends\\", "study :rest", "study:rest", "a::b",
    "x\\:y", "text_match('a')", "it''s", "$$x$$", "a -- b", "a /* b",
    "x%' OR 1=1 --", "x'; DROP TABLE bids_datasets; --",
    "x' UNION SELECT usename FROM pg_user --", "\\' OR TRUE --",
]


def _sql(name: str) -> str:
    plan = pp.QueryPlan(natural_language_summary="test", name_search=name)
    augmented = pp.augment_with_rag(plan, {})
    compiled = compile_plan(plan, augmented.resolved, augmented.augmented_question)
    assert compiled.sql, compiled.reason
    return sql_rewriter.rewrite_generated_sql(compiled.sql, augmented.augmented_question).sql


@pytest.mark.parametrize("name", NAMES)
def test_the_name_stays_one_literal(name):
    sql = _sql(name)
    assert query_guard.check_static(sql) is None
    where = sqlglot.parse_one(sql, read="postgres").args["where"].this
    assert isinstance(where, exp.ILike)
    assert pattern_words(where.expression.this) == name


@pytest.mark.parametrize("name", NAMES)
def test_the_chip_shows_the_name_as_asked(name):
    chip, = query_filters.describe(_sql(name))
    assert (chip.field, chip.label) == ("Name", name) or name in chip.label or name in chip.values


def test_a_line_break_cannot_start_another_filter_line():
    plan = pp.QueryPlan(natural_language_summary="test",
                        name_search="abide\n  dataset size: TRUE")
    lines = pp.augment_with_rag(plan, {}).augmented_question.splitlines()
    assert not any(line.startswith("  dataset size:") for line in lines)


@pytest.mark.parametrize("name", ["", "  ", "%", "%%", None])
def test_no_name_no_condition(name):
    assert contains_pattern(name) is None


def test_the_model_s_own_wildcards_are_dropped():
    assert contains_pattern("%abide%") == "'%abide%'"


def test_a_name_with_an_apostrophe_ranks():
    from services.fulltext import text_search
    expanded = text_search.expand_text_search(f"SELECT d.id FROM bids_datasets d WHERE d.name ILIKE "
                                              f"{contains_pattern('Alzheimer' + chr(39) + 's_study')}")
    assert "alzheimer" in expanded.rank_query.stemmed.lower()


@needs_postgres
def test_only_the_named_dataset_is_found():
    from sqlalchemy import text

    async def run():
        async with fresh_schema("name_search") as engine, engine.begin() as connection:
            await connection.execute(text("CREATE TABLE bids_datasets (name TEXT)"))
            for name in NAMES + ["unrelated"]:
                await connection.execute(text("INSERT INTO bids_datasets VALUES (:name)"), {"name": name})
            for name in NAMES:
                found = (await connection.execute(text(
                    f"SELECT d.name FROM bids_datasets d WHERE d.name ILIKE {contains_pattern(name)}"))).scalars().all()
                expected = [n for n in NAMES if name.lower() in n.lower()]
                assert sorted(found) == sorted(expected), name

    asyncio.run(run())
