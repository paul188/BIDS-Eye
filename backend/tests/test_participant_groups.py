"""Participant diagnoses from a group column and participants.json (remote/participant_groups.py).

Uses the real vocabulary (backend/vocabulary/value_mappings.yaml). The cases are taken from
the dry run on the live database (2026-11-17).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "ingest" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

from remote import participant_groups as pg  # noqa: E402

FTD_SIDECAR = {"Group": {"Description": "Participant group label", "Levels": {
    "A": "Alzheimer Disease Group", "F": "Frontotemporal Dementia Group", "C": "Healthy Group"}}}


def _rows(column, values):
    return [(f"sub-{i:03d}", {column: v, "Age": str(50 + i)}) for i, v in enumerate(values)]


# ── Labels ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, code", [
    ("Frontotemporal Dementia Group", "frontotemporal_dementia"),
    ("Alzheimer Disease Group", "alzheimers_disease"),
    ("Healthy Group", "healthy_control"),
    ("PD", "parkinsons_disease"),
    ("CTL", "healthy_control"),
    ("Mild AD", "alzheimers_disease"),
    ("First Episode Psychosis", "schizophrenia"),
    ("participants with ADHD", "adhd"),
    ("Healthy control participants", "healthy_control"),
])
def test_labels_that_name_a_diagnosis(text, code) -> None:
    assert pg.label_code(text) == code


@pytest.mark.parametrize("text", [
    "1", "2", "yes", "No", "Present", "No ADHD diagnosis", "participant did not have bipolar disorder",
    "Placebo", "Experimental", "Musician", "Schizophrenia/Schizoaffective",
])
def test_labels_that_do_not(text) -> None:
    # Fuzzy matching had turned "1" into neurofibromatosis type 1 and
    # "placebo" into a control code; only exact vocabulary matches count.
    assert pg.label_code(text) is None


def test_age_groups_and_lifestyle_codes_are_no_diagnosis() -> None:
    assert pg.code_kind("young_adult") is None
    assert pg.code_kind("music_training_participant") is None
    assert pg.code_kind("healthy_control") == "control"
    assert pg.code_kind("frontotemporal_dementia") == "diagnosis"


# ── Columns ───────────────────────────────────────────────────────────────────

def test_ftd_dataset_gets_its_three_groups() -> None:
    plan = pg.plan_dataset(_rows("Group", ["A", "F", "C", "F", "A"]), FTD_SIDECAR)
    assert plan["column"] == "Group"
    assert plan["codes"] == {"A": "alzheimers_disease", "F": "frontotemporal_dementia",
                             "C": "healthy_control"}
    assert list(plan["assign"].values()).count("frontotemporal_dementia") == 2


def test_without_levels_the_values_themselves_are_read() -> None:
    plan = pg.plan_dataset(_rows("GROUP", ["PD", "Control", "PD"]), None)
    assert plan["codes"] == {"PD": "parkinsons_disease", "Control": "healthy_control"}


@pytest.mark.parametrize("column, values", [
    ("AgeGroup", ["Young", "Old"]),                  # not a group of patients
    ("Group", ["Control", "Intervention"]),          # controls only: no diagnosis
    ("Group", ["1", "2", "1"]),                      # codes without Levels
    ("other_diagnosis_description", ["Anxiety", "None"]),  # a second diagnosis (ds007182)
    ("adhd_diagnosis", ["yes", "no"]),               # yes/no column
])
def test_columns_that_are_not_decoded(column, values) -> None:
    assert pg.plan_dataset(_rows(column, values), None) is None


def test_the_column_coding_most_participants_wins() -> None:
    rows = [(f"sub-{i}", {"study_group": g, "dx": d}) for i, (g, d) in
            enumerate([("ADHD", "adhd"), ("ADHD", "x"), ("TD/NC", "y"), ("TD/NC", "z")])]
    sidecar = {"study_group": {"Levels": {"ADHD": "participants with ADHD",
                                          "TD/NC": "typically developing comparator participants"}}}
    plan = pg.plan_dataset(rows, sidecar)
    assert plan["column"] == "study_group" and len(plan["assign"]) == 4


# ── Run ───────────────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, rows=(), rowcount=0):
        self._rows, self.rowcount = list(rows), rowcount

    def all(self):
        return self._rows


class _Connection:
    def __init__(self):
        self.writes = []

    async def execute(self, statement, params=None):
        sql = str(statement)
        if sql.lstrip().startswith("SELECT d.id"):
            return _Result([("uuid-1", "ds004504")])
        if "FROM bids_participants WHERE dataset_id" in sql:
            return _Result(_rows("Group", ["A", "F", "C"]))
        if sql.lstrip().startswith("UPDATE"):
            self.writes.append(params)
            return _Result(rowcount=len(params["pids"]))
        raise AssertionError(sql)


def test_dry_run_writes_nothing_and_apply_writes_each_code_with_its_source() -> None:
    conn = _Connection()
    reports = asyncio.run(pg.run(conn, apply=False, fetch=lambda a: FTD_SIDECAR))
    assert conn.writes == [] and reports[0]["participants"]["frontotemporal_dementia"] == 1
    reports = asyncio.run(pg.run(conn, apply=True, fetch=lambda a: FTD_SIDECAR))
    assert {w["code"] for w in conn.writes} == {"alzheimers_disease", "frontotemporal_dementia",
                                                "healthy_control"}
    assert all(w["source_key"] == "_diagnosis_from" and w["column"] == "Group" for w in conn.writes)
    assert reports[0]["written"] == 3


def test_a_sidecar_that_cannot_be_read_skips_the_dataset() -> None:
    # Decoding without the sidecar's Levels would write raw values, and a
    # dataset with any diagnosis is never looked at again.
    def fetch(accession):
        raise pg.ParticipantSidecarError("timed out")

    conn = _Connection()
    assert asyncio.run(pg.run(conn, apply=True, fetch=fetch)) == []
    assert conn.writes == []


def test_fetch_sidecar_tells_missing_from_unreadable(monkeypatch) -> None:
    import urllib.error

    def urlopen(error):
        def fail(url, timeout):
            raise error
        return fail

    monkeypatch.setattr(pg.urllib.request, "urlopen",
                        urlopen(urllib.error.HTTPError("u", 404, "Not Found", {}, None)))
    assert pg.fetch_sidecar("ds004504") is None
    monkeypatch.setattr(pg.urllib.request, "urlopen", urlopen(urllib.error.URLError("timed out")))
    with pytest.raises(pg.ParticipantSidecarError):
        pg.fetch_sidecar("ds004504")
    monkeypatch.setattr(pg.urllib.request, "urlopen", urlopen(TypeError("a bug")))
    with pytest.raises(TypeError):
        pg.fetch_sidecar("ds004504")
