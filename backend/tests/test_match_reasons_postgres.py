"""Match reasons and README marks against a real Postgres.

The SQL of services/results/match_reasons.py is built from the generated
query's own conditions; only a database shows that it runs and that each
branch holds where it should. Skipped unless BIDSEYE_TEST_PG_DSN names a
scratch database (tests/postgres.py says how to start one); works in a schema
of its own.
"""

from __future__ import annotations

import asyncio
import uuid

from criteria import age_condition, species_condition
from schemas import DatasetSchema, ParticipantSummarySchema
from services.fulltext import text_search
from services.results import match_reasons, query_filters, readme_snippets

from tests.postgres import fresh_schema, needs_postgres, run_statements

pytestmark = needs_postgres

_SCHEMA_SQL = """
CREATE TABLE bids_datasets (
    id UUID PRIMARY KEY,
    name TEXT,
    description_text TEXT,
    readme_text TEXT,
    species TEXT,
    age_min DOUBLE PRECISION,
    age_max DOUBLE PRECISION,
    -- The weights of production (crawlers/migrate_add_search_tsv.sql): a
    -- word labelled A or B matches the name and description, not the README.
    search_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description_text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(readme_text, '')), 'D')) STORED
);
CREATE TABLE bids_participants (
    dataset_id UUID, participant_id TEXT, age DOUBLE PRECISION, diagnosis TEXT, sex TEXT
);
CREATE TABLE bids_objects (
    dataset_id UUID, subject TEXT, datatype TEXT, suffix TEXT, task TEXT
);
"""

# name: (description, readme, species, age_min, age_max, [(age, diagnosis)])
_DATASETS = {
    # Ages recorded; the README never says "infant". Parents are listed too.
    "ages": ("EEG during sleep", "We recorded 6-month-old participants and their parents.",
             "Human", None, None,
             [(0.33, None), (1.0, None), (1.92, None), (30, None), (34, None)]),
    # No participant ages, only the dataset's age range.
    "range": ("Resting MRI", "Scans of very young participants.", "Human", 0.5, 2, []),
    # Nothing recorded: the description names newborns. The README names
    # infants, which the excerpt marks.
    "words": ("MRI of newborns", "Structural images. Every infant was scanned asleep.",
              None, None, None, [(None, None)]),
    # Adults: no venue holds.
    "adults": ("Adult fMRI", "Adults performed a task.", "Human", 18, 40,
               [(25, None), (40, None)]),
    # Parkinson's disease coded for two of three participants, named in the README.
    "parkinson": ("Gait study", "Patients with Parkinson's disease walked.", "Human", None, None,
                  [(None, "parkinsons_disease"), (None, "parkinsons_disease"), (None, "control")]),
    # Parkinson's in the description, nothing coded.
    "pd_text": ("Parkinson's disease and gait", "Walking data.", "Human", None, None,
                [(None, None)]),
    # Parkinson's in the README only: the :AB search does not reach it.
    "pd_readme": ("Gait", "Some patients had Parkinson's disease.", "Human", None, None,
                  [(None, None)]),
    # A mouse dataset without a species label.
    "mice": ("Two-photon imaging in mice", "Mice were imaged.", "Other", None, None, []),
}
DIAGNOSIS = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.diagnosis IN ('parkinsons_disease')) OR d.search_tsv @@ "
    "to_tsquery('english', 'parkinson:AB | parkinsons:AB'))"
)


def _query(condition: str) -> str:
    sql = ("SELECT d.id, d.name FROM bids_datasets AS d LEFT JOIN bids_objects AS o "
           f"ON o.dataset_id = d.id WHERE {condition} GROUP BY d.id")
    return text_search.expand_text_search(sql, papers=False).sql


async def _run(check):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    ids = {key: uuid.uuid4() for key in _DATASETS}
    async with fresh_schema("match_reasons") as engine:
        async with engine.begin() as conn:
            await run_statements(conn, _SCHEMA_SQL)
            for key, (description, readme, species, low, high, people) in _DATASETS.items():
                await conn.execute(text(
                    "INSERT INTO bids_datasets (id, name, description_text, readme_text, species, "
                    "age_min, age_max) VALUES (:id, :name, :d, :r, :s, :lo, :hi)"),
                    {"id": ids[key], "name": f"Dataset {len(key)}", "d": description, "r": readme, "s": species,
                     "lo": low, "hi": high})
                for i, (age, diagnosis) in enumerate(people):
                    await conn.execute(text(
                        "INSERT INTO bids_participants VALUES (:id, :p, :age, :dx, NULL)"),
                        {"id": ids[key], "p": f"sub-{i}", "age": age, "dx": diagnosis})
        async with AsyncSession(engine) as session:
            await check(session, ids)


async def _page(session, ids, condition):
    """The datasets the generated query returns, with their match reasons."""
    from sqlalchemy import text

    sql = _query(condition)
    rows = (await session.execute(text(sql))).mappings().all()
    by_id = {value: key for key, value in ids.items()}
    datasets = []
    for row in rows:
        key = by_id[row["id"]]
        total = len(_DATASETS[key][5])
        datasets.append(DatasetSchema(
            id=row["id"], name=key, dataset_type="raw", source_type="openneuro",
            participant_summary=ParticipantSummarySchema(total=total) if total else None))
    await match_reasons.attach_match_reasons(session, datasets, sql)
    return sql, {d.name: d for d in datasets}


def test_infant_search_explains_each_match():
    async def check(session, ids):
        _, page = await _page(session, ids, age_condition(["infants"]))
        assert set(page) == {"ages", "range", "words"}
        assert page["ages"].match_reasons == ["Age: 3 of 5 participants aged 4–23 months"]
        assert page["range"].match_reasons == ["Age: dataset age range 0.5–2 years"]
        assert page["words"].match_reasons == ["Age: named in the title or description"]
    asyncio.run(_run(check))


def test_diagnosis_and_species_venues():
    async def check(session, ids):
        _, page = await _page(session, ids, DIAGNOSIS)
        assert set(page) == {"parkinson", "pd_text"}
        assert page["parkinson"].match_reasons == ["Diagnosis: coded for 2 of 3 participants"]
        assert page["pd_text"].match_reasons == ["Diagnosis: named in the title or description"]
        # Without weight labels the search reads the README too.
        anywhere = DIAGNOSIS.replace(":AB", "")
        _, page = await _page(session, ids, anywhere)
        assert set(page) == {"parkinson", "pd_text", "pd_readme"}
        assert page["parkinson"].match_reasons == [
            "Diagnosis: coded for 2 of 3 participants; named in the README or description"]
        assert page["pd_readme"].match_reasons == ["Diagnosis: named in the README or description"]
        _, page = await _page(session, ids, species_condition(["mouse"]))
        assert set(page) == {"mice"}
        assert page["mice"].match_reasons == ["Species: named in the title or description"]
    asyncio.run(_run(check))


def test_readme_excerpt_marks_the_age_group_words():
    async def check(session, ids):
        sql, page = await _page(session, ids, age_condition(["infants"]))
        datasets = list(page.values())
        await readme_snippets.attach_snippets(
            session, datasets, query_filters.highlight_terms(sql),
            query_filters.highlight_exact_words(sql), query_filters.highlight_field_strengths(sql))
        assert "⟦infant⟧" in page["words"].snippet
        # No age word in this README: its start, unmarked; the reason line explains it.
        assert "⟦" not in page["ages"].snippet
    asyncio.run(_run(check))


def test_uncertain_tier_query_still_runs_with_reasons():
    # The tier wraps the query; reasons come from the unwrapped filter SQL.
    from sqlalchemy import text
    from services.uncertain import tier as uncertain_tier

    async def check(session, ids):
        sql = _query(age_condition(["infants"]))
        tiered = uncertain_tier.apply(sql, papers=False, hoist_terms=False)
        rows = (await session.execute(text(tiered))).mappings().all()
        assert {r["id"] for r in rows} >= {ids["ages"], ids["range"], ids["words"]}
        datasets = [DatasetSchema(id=r["id"], name="x", dataset_type="raw",
                                  source_type="openneuro") for r in rows]
        await match_reasons.attach_match_reasons(session, datasets, sql)
        reasons = {d.id: d.match_reasons for d in datasets}
        assert reasons[ids["ages"]] == ["Age: 3 participants aged 4–23 months"]
    asyncio.run(_run(check))


def test_a_paper_is_named_only_where_the_own_text_does_not_match():
    """search_tsv_papers holds the dataset's own text too."""
    from sqlalchemy import text

    async def check(session, ids):
        await session.execute(text("ALTER TABLE bids_datasets ADD COLUMN search_tsv_papers TSVECTOR"))
        await session.execute(text(
            "UPDATE bids_datasets SET search_tsv_papers = search_tsv || setweight(to_tsvector("
            "'english', CASE WHEN id = :id THEN 'Parkinson patients' ELSE '' END), 'D')"), {"id": ids["adults"]})
        condition = DIAGNOSIS.replace(":AB", "").replace("search_tsv", "search_tsv_papers")
        _, page = await _page(session, ids, condition)
        assert set(page) == {"parkinson", "pd_text", "pd_readme", "adults"}
        assert page["pd_readme"].match_reasons == ["Diagnosis: named in the README or description"]
        assert page["adults"].match_reasons == ["Diagnosis: named in a linked paper"]
    asyncio.run(_run(check))


_SHAPES = [
    "d.search_tsv @@ to_tsquery('english', 'parkinson')",
    "d.search_tsv @@ to_tsquery('english', 'parkinson:AB')",
    "NOT d.search_tsv @@ to_tsquery('english', 'parkinson')",
    "(d.search_tsv @@ to_tsquery('english', 'parkinson')) IS FALSE",
    "d.description_text ILIKE '%parkinson%'",
    "d.description_text NOT ILIKE '%parkinson%'",
    "LOWER(d.description_text) LIKE '%parkinson%'",
    "d.description_text ~* 'parkinson'",
    "d.description_text !~* 'parkinson'",
    "d.description_text NOT ILIKE ANY(ARRAY['%parkinson%', '%gait%'])",
    "d.readme_text NOT LIKE '%Adults%' OR d.species = 'Other'",
    "EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id AND p2.age < 3)",
    "NOT EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id AND p2.age < 3)",
    "d.id IN (SELECT dataset_id FROM bids_participants WHERE diagnosis = 'parkinsons_disease')",
    "d.id NOT IN (SELECT dataset_id FROM bids_participants WHERE diagnosis = 'parkinsons_disease')",
    "d.species IS NULL OR d.species = 'Other'",
    "d.species IS NOT NULL",
    "d.age_min < 3 OR d.description_text NOT ILIKE '%a%'",
    "d.species IS NULL AND NOT d.search_tsv @@ to_tsquery('english', 'mouse')",
    DIAGNOSIS,
]


def test_evidence_holds_only_where_its_condition_does_and_covers_it():
    """Each piece of evidence of a condition holds for datasets of the
    condition only, and for an evaluable condition they cover all of it."""
    from sqlalchemy import text
    from services import sql_tree
    from services.evidence.branches import confirming_evidence

    async def check(session, ids):
        names = {value: key for key, value in ids.items()}

        async def holds(condition):
            sql = f"SELECT d.id FROM bids_datasets AS d WHERE {condition}"
            rows = await session.execute(text(sql.replace(":AB", r"\:AB")))
            return {names[row.id] for row in rows}

        for shape in _SHAPES:
            tree = sql_tree.parse_select(f"SELECT d.id FROM bids_datasets AS d WHERE {shape}")
            node = tree.args["where"].this
            wanted = await holds(shape)
            # Reading the query and writing it back changes no result.
            assert await holds(node.sql(dialect="postgres")) == wanted, shape
            ways = confirming_evidence(text_search.Clause.read(node), "d")
            held = [await holds(way.sql()) for way in ways]
            assert all(found <= wanted for found in held), shape
            if len(ways) == len(sql_tree.or_branches(node)):
                assert set().union(*held) == wanted, shape
    asyncio.run(_run(check))
