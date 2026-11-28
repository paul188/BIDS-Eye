from __future__ import annotations

import asyncio
import uuid

from services.results.display import dataset_cards, result_page

from tests.postgres import fresh_schema, needs_postgres, run_statements

_SCHEMA_SQL = """
CREATE TABLE bids_participants (
    dataset_id UUID, participant_id TEXT, age DOUBLE PRECISION, diagnosis TEXT, sex TEXT
);
"""


async def _summaries(people: dict[str, list[tuple]]) -> dict[str, object]:
    """people: dataset → [(diagnosis, sex, age)]; the summary of each."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    ids = {key: uuid.uuid4() for key in people}
    async with fresh_schema("participant_summary") as engine:
        async with engine.begin() as conn:
            await run_statements(conn, _SCHEMA_SQL)
            for key, rows in people.items():
                for i, (diagnosis, sex, age) in enumerate(rows):
                    await conn.execute(text(
                        "INSERT INTO bids_participants VALUES (:id, :p, :age, :dx, :sex)"),
                        {"id": ids[key], "p": f"sub-{i:03d}", "age": age, "dx": diagnosis, "sex": sex})
        async with AsyncSession(engine) as session:
            found = await dataset_cards.participant_summaries(session, [*ids.values(), uuid.uuid4()])
    return {key: found.get(ids[key]) for key in people}


@needs_postgres
def test_summary_counts_every_participant() -> None:
    # 69 mice: more than the 50 participants the card used to be sent.
    mice = ([("wild_type_animal_model", "female", 0.3)] * 30
            + [("genetic_knockout_animal_model", "male", None)] * 39)
    summary = asyncio.run(_summaries({"mice": mice}))["mice"]
    assert summary.total == 69
    assert summary.diagnoses == {"wild_type_animal_model": 30, "genetic_knockout_animal_model": 39}
    assert (summary.female, summary.male) == (30, 39)
    assert summary.age_min == summary.age_max == 0.3


@needs_postgres
def test_summary_skips_blank_values_and_datasets_without_participants() -> None:
    found = asyncio.run(_summaries({
        "blank": [("  ", None, 0.0), ("\tcontrol ", "other_diverse", -1.0), ("control", None, 70.0)],
        "none": [],
    }))
    blank = found["blank"]
    assert blank.total == 3
    assert blank.diagnoses == {"control": 2}
    assert blank.female == blank.male == 0
    assert blank.age_min == blank.age_max == 70.0
    assert found["none"] is None


def test_no_datasets_no_query() -> None:
    assert asyncio.run(dataset_cards.participant_summaries(None, [])) == {}


def test_found_message_uses_the_right_plural() -> None:
    assert result_page.found_message(1) == "Found 1 dataset matching your query."
    assert result_page.found_message(2) == "Found 2 datasets matching your query."
