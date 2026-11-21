"""Sex and scanner text venues of the uncertain tier, on a real Postgres.

Skipped unless BIDSEYE_TEST_PG_DSN names a scratch database (see
tests/test_match_reasons_postgres.py); works in a schema of its own.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid

import pytest

from services.fulltext import tier_scanner, uncertain_tier as ut

from tests.test_scanner_sex_venues import (
    FEMALE, GE_OR_PHILIPS, ONLY_FEMALE, PRISMA, STRENGTH_7T, STRENGTH_7T_FMRI, _HEAD)

DSN = os.getenv("BIDSEYE_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="BIDSEYE_TEST_PG_DSN not set")

_SCHEMA_SQL = """
CREATE TABLE bids_datasets (
    id UUID PRIMARY KEY, name TEXT, description_text TEXT, readme_text TEXT,
    search_tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description_text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(readme_text, '')), 'D')) STORED
);
CREATE TABLE bids_participants (dataset_id UUID, participant_id TEXT, age DOUBLE PRECISION,
                                diagnosis TEXT, sex TEXT);
CREATE TABLE bids_objects (dataset_id UUID, subject TEXT, datatype TEXT, suffix TEXT, task TEXT,
                           other_entities JSON);
CREATE TABLE bids_dataset_facets (dataset_id UUID, datatype TEXT, suffix TEXT, task TEXT)
"""

FUNC = ("functional_mri", "fmri_bold")
ANAT = ("anatomical_mri", "t1w")
# key: (readme, [(datatype, suffix, sidecar)], [sex per participant])
_DATASETS = {
    "sidecar_7t": ("Visual cortex study.", [FUNC + ({"MagneticFieldStrength": 7},)], []),
    "readme_7t_func": ("Data were acquired at 7 Tesla.", [FUNC + ({},)], []),
    "readme_7t_anat": ("Anatomy at 7T.", [ANAT + ({},)], []),
    "sidecar_3t": ("Unlike our earlier 7T study, this used 3T.",
                   [FUNC + ({"MagneticFieldStrength": 3},)], []),
    "readme_3t": ("Scanned on a 3T scanner.", [FUNC + ({},)], []),
    "sidecar_siemens": ("Resting state.", [FUNC + ({"Manufacturer": "Siemens"},)], []),
    "readme_prisma": ("Siemens Prisma fit, 64-channel coil.", [FUNC + ({},)], []),
    "readme_ge": ("Acquired on a GE Healthcare Discovery MR750.", [FUNC + ({},)], []),
    "sidecar_philips_readme_ge": ("Earlier pilots used GE Healthcare scanners.",
                                  [FUNC + ({"Manufacturer": "Siemens"},)], []),
    "sex_recorded_f": ("Memory task.", [], ["F", "M"]),
    "sex_recorded_m": ("Memory task; women were excluded.", [], ["M", "M"]),
    "sex_text_women": ("22 women took part.", [], [None, None]),
    "sex_text_only": ("An all-female sample of 30 adults.", [], [None]),
    "sex_text_mixed": ("Both women and men took part.", [], []),
}


async def _run(check):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    schema = f"venues_{uuid.uuid4().hex[:8]}"
    admin = create_async_engine(DSN)
    async with admin.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_async_engine(DSN, connect_args={"server_settings": {"search_path": schema}})
    ids = {key: uuid.uuid4() for key in _DATASETS}
    try:
        async with engine.begin() as conn:
            for statement in _SCHEMA_SQL.split(";"):
                if statement.strip():
                    await conn.execute(text(statement))
            for i, (key, (readme, files, sexes)) in enumerate(_DATASETS.items()):
                await conn.execute(text(
                    "INSERT INTO bids_datasets (id, name, description_text, readme_text) "
                    "VALUES (:id, :name, NULL, :readme)"),
                    {"id": ids[key], "name": f"Dataset {i}", "readme": readme})
                for datatype, suffix, sidecar in files:
                    await conn.execute(text(
                        "INSERT INTO bids_objects VALUES (:id, 'sub-01', :dt, :sx, NULL, "
                        "CAST(:se AS JSON))"),
                        {"id": ids[key], "dt": datatype, "sx": suffix, "se": json.dumps(sidecar)})
                    await conn.execute(text(
                        "INSERT INTO bids_dataset_facets VALUES (:id, :dt, :sx, NULL)"),
                        {"id": ids[key], "dt": datatype, "sx": suffix})
                for n, sex in enumerate(sexes):
                    await conn.execute(text(
                        "INSERT INTO bids_participants VALUES (:id, :p, NULL, NULL, :sex)"),
                        {"id": ids[key], "p": f"sub-{n}", "sex": sex})
        async with AsyncSession(engine) as session:
            await check(session, ids)
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        await admin.dispose()


async def _tiers(session, ids, condition, hoist=True) -> dict:
    """{dataset key: (level, reasons)} of the tier query for *condition*."""
    from sqlalchemy import text

    sql = ut.apply(_HEAD + condition, papers=False, hoist_terms=hoist)
    assert ut.LEVEL_COLUMN in sql, "the condition was not softened"
    rows = (await session.execute(text(sql))).mappings().all()
    by_id = {value: key for key, value in ids.items()}
    return {by_id[r["id"]]: (r[ut.LEVEL_COLUMN], r[ut.REASONS_COLUMN]) for r in rows}


def test_field_strength_from_sidecars_then_text():
    async def check(session, ids):
        for hoist in (True, False):
            found = await _tiers(session, ids, STRENGTH_7T, hoist)
            assert found == {
                "sidecar_7t": (0, None),
                "readme_7t_func": (1, "field strength not in the file metadata; "
                                      "the description names 7 T"),
                "readme_7t_anat": (1, "field strength not in the file metadata; "
                                      "the description names 7 T"),
            }   # a recorded 3T is not overruled by "our earlier 7T study"
        fmri = await _tiers(session, ids, STRENGTH_7T_FMRI)
        assert set(fmri) == {"sidecar_7t", "readme_7t_func"}   # the anatomy-only one has no fMRI
        # The same with the file part read from the facets summary.
        from services.catalogue import schema_capabilities
        facets = schema_capabilities.has_dataset_facets
        schema_capabilities.has_dataset_facets = lambda: True
        try:
            assert await _tiers(session, ids, STRENGTH_7T_FMRI) == fmri
        finally:
            schema_capabilities.has_dataset_facets = facets
        # With a 'possible' level the full recorded check gives the same states.
        os.environ["UNCERTAIN_POSSIBLE"] = "true"
        try:
            possible = await _tiers(session, ids, STRENGTH_7T)
        finally:
            del os.environ["UNCERTAIN_POSSIBLE"]
        assert {k: v for k, v in possible.items() if v[0] < 2} == found
    asyncio.run(_run(check))


def test_manufacturer_and_model_from_sidecars_then_text():
    async def check(session, ids):
        siemens = GE_OR_PHILIPS.replace("'%GE%'", "'%Siemens%'").replace("'%Philips%'", "'%Siemens%'")
        found = await _tiers(session, ids, siemens)
        assert found["sidecar_siemens"] == (0, None)
        assert found["readme_prisma"][0] == 1
        assert "sidecar_philips_readme_ge" in found and found["sidecar_philips_readme_ge"][0] == 0
        ge = await _tiers(session, ids, GE_OR_PHILIPS)
        # Text counts only where nothing is recorded: a recorded Siemens stays out.
        assert set(ge) == {"readme_ge"} and ge["readme_ge"][1].endswith("names GE or Philips")
        model = await _tiers(session, ids, PRISMA)
        assert set(model) == {"readme_prisma"}
    asyncio.run(_run(check))


def test_participant_sex_from_records_then_text():
    async def check(session, ids):
        female = await _tiers(session, ids, FEMALE)
        assert female["sex_recorded_f"] == (0, None)
        assert "sex_recorded_m" not in female   # recorded, and no woman; its README text does not count
        for key in ("sex_text_women", "sex_text_only", "sex_text_mixed"):
            assert female[key] == (1, "participant sex not recorded; the description names "
                                      "female participants"), key
        only = await _tiers(session, ids, ONLY_FEMALE)
        assert set(only) == {"sex_text_only"}
        assert only["sex_text_only"][1].endswith("all participants are female")
    asyncio.run(_run(check))


@pytest.mark.parametrize("strengths", [[7.0], [1.5], [3.0, 7.0], [9.4, 10.5, 11.7, 14.0]])
def test_field_strength_pattern_is_the_same_in_python_and_postgres(strengths):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    texts = [
        "7T", "7 T", "7-Tesla", "7.0T", "7 tesla", "at 7 T.", "(7T)", "7T/3T", "17T", "0.7T",
        "ses-7T", "7 T1w", "200 T/m/s", "1.5T", "1,5 T", "1.5 Tesla", "11.5T", "3T", "3.0 T",
        "a 3T scanner", "9.4T", "10.5 T", "14T", "4.7T", "TESLA", "7TESLA", "x7T", "7Tx", "_7T",
    ]
    python = re.compile(tier_scanner.field_strength_regex(strengths))
    postgres = tier_scanner.field_strength_regex(strengths, postgres=True)
    prefilter = tier_scanner._strength_tsquery(strengths)

    async def compare():
        engine = create_async_engine(DSN)
        try:
            async with engine.connect() as conn:
                for sample in texts:
                    hit = (await conn.execute(text("SELECT CAST(:t AS TEXT) ~ CAST(:p AS TEXT)"),
                                              {"t": sample, "p": postgres})).scalar()
                    assert hit is bool(python.search(sample)), (sample, strengths)
                    if hit:   # the indexed prefilter never loses a match
                        assert (await conn.execute(text(
                            "SELECT to_tsvector('english', CAST(:t AS TEXT)) @@ "
                            "to_tsquery('english', CAST(:q AS TEXT))"),
                            {"t": f"Data at {sample} here", "q": prefilter})).scalar(), (sample, prefilter)
        finally:
            await engine.dispose()
    asyncio.run(compare())
