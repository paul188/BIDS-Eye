"""The evidence score of the uncertain tier (services/evidence/SCORING.md), on
a real Postgres: which pieces of evidence hold for a dataset, and what they
add up to.

Skipped unless BIDSEYE_TEST_PG_DSN names a scratch database
(tests/postgres.py); works in a schema of its own.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from services.uncertain import tier as ut

from tests.postgres import fresh_schema, needs_postgres, run_statements

pytestmark = needs_postgres

_SCHEMA_SQL = """
CREATE TABLE bids_datasets (id UUID PRIMARY KEY, name TEXT, description_text TEXT, readme_text TEXT,
                            age_min DOUBLE PRECISION, age_max DOUBLE PRECISION,
                            search_tsv TSVECTOR, search_tsv_papers TSVECTOR);
CREATE TABLE bids_participants (dataset_id UUID, participant_id TEXT, age DOUBLE PRECISION)
"""
# key → (README, text of a paper the dataset names, recorded ages)
_DATASETS = {
    "every_word": ("Young children watched cartoons.", "", [5.0]),
    "one_word": ("Children watched cartoons.", "", [5.0]),
    "paper_only": ("A cartoon study.", "Young children watched cartoons.", [5.0]),
    "one_word_and_paper": ("Children watched cartoons.", "The young children were five.", [5.0]),
    "neither": ("A cartoon study.", "Adults watched cartoons.", [5.0]),
    "infants_recorded": ("Young children slept.", "", [1.0]),
    "infants_named": ("Young children: sleep of infants.", "", []),
    "ages_unknown": ("Young children slept.", "", []),
}
_HEAD = "SELECT d.id FROM bids_datasets AS d WHERE "
TEXT = "d.search_tsv @@ to_tsquery('english', 'young & children')"
INFANTS = ("(d.age_max < 3 OR EXISTS(SELECT 1 FROM bids_participants AS p2 "
           "WHERE p2.dataset_id = d.id AND p2.age < 3))")


async def _scores(condition: str, hoist: bool, possible: bool = False) -> dict:
    """{dataset key: (level, score, reasons)} of the tier query for *condition*."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    ids = {key: uuid.uuid4() for key in _DATASETS}
    async with fresh_schema("score") as engine:
        async with engine.begin() as conn:
            await run_statements(conn, _SCHEMA_SQL)
            for key, (readme, paper, ages) in _DATASETS.items():
                await conn.execute(text(
                    "INSERT INTO bids_datasets VALUES (:id, :name, '', :readme, NULL, NULL, "
                    "to_tsvector('english', :readme), to_tsvector('english', :readme || ' ' || :paper))"),
                    {"id": ids[key], "name": key, "readme": readme, "paper": paper})
                for n, age in enumerate(ages):
                    await conn.execute(text("INSERT INTO bids_participants VALUES (:id, :p, :age)"),
                                       {"id": ids[key], "p": f"sub-{n}", "age": age})
        os.environ["UNCERTAIN_POSSIBLE"] = "true" if possible else "false"
        try:
            sql = ut.apply(_HEAD + condition, papers=True, hoist_terms=hoist)
        finally:
            del os.environ["UNCERTAIN_POSSIBLE"]
        async with AsyncSession(engine) as session:
            rows = (await session.execute(text(sql))).mappings().all()
        by_id = {value: key for key, value in ids.items()}
        return {by_id[r["id"]]: (r[ut.LEVEL_COLUMN], float(r[ut.SCORE_COLUMN]), r[ut.REASONS_COLUMN])
                for r in rows}


@pytest.mark.parametrize("hoist", [False, True])
def test_two_sources_for_one_clause_count_for_more_than_either(hoist) -> None:
    rows = asyncio.run(_scores(TEXT, hoist))
    assert set(rows) == {"every_word", "one_word", "paper_only", "one_word_and_paper",
                         "infants_recorded", "infants_named", "ages_unknown"}
    assert rows["every_word"][:2] == (0, 1.0)
    # One piece of evidence: its own score.
    assert rows["one_word"][0] == 1 and rows["one_word"][1] == pytest.approx(0.37, abs=1e-3)
    assert rows["paper_only"][0] == 1 and rows["paper_only"][1] == pytest.approx(0.48, abs=1e-3)
    # The dataset's own text and a paper: the stronger in full, the other at half weight.
    both = 1 - (1 - 0.48) * (1 - 0.37) ** 0.5
    assert rows["one_word_and_paper"][0] == 1
    assert rows["one_word_and_paper"][1] == pytest.approx(both, abs=1e-3)
    assert 0.48 < both < 1 - (1 - 0.48) * (1 - 0.37)
    # The card names the first reason in the list, as before.
    assert rows["one_word_and_paper"][2] == "found only in a paper the dataset names"
    assert rows["one_word"][2] == "matches only part of the search; missing: young"


def test_a_field_that_is_not_recorded_is_counted_once() -> None:
    rows = asyncio.run(_scores(f"{INFANTS} AND {TEXT}", hoist=False))
    assert rows["infants_recorded"][:2] == (0, 1.0)
    # The hint's score already contains what "ages not recorded" is worth.
    assert rows["infants_named"][0] == 1 and rows["infants_named"][1] == pytest.approx(0.53, abs=1e-3)
    assert "ages_unknown" not in rows and "every_word" not in rows   # recorded ages of 5: contradicted
    with_possible = asyncio.run(_scores(f"{INFANTS} AND {TEXT}", hoist=False, possible=True))
    assert with_possible["ages_unknown"][0] == 2
    assert with_possible["ages_unknown"][1] == pytest.approx(0.33, abs=1e-3)
