"""Participant sex as the vocabulary's standard codes (remote/participant_sex.py).

Uses the real vocabulary (RAG/value_mappings.yaml).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "crawlers" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

from remote import participant_groups as pg  # noqa: E402
from remote import participant_sex as ps  # noqa: E402

ISO_SIDECAR = {"sex": {"Description": "Sex of the participant", "Levels": {"1": "male", "2": "female"}}}
FLAG_SIDECAR = {"Gender": {"Levels": {"0": "Male", "1": "Female"}}}


# ── Values ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("typed, code", [
    ("F", "female"), ("f", "female"), ("Female", "female"), ("FEMALE", "female"),
    (" female ", "female"), ("woman", "female"), ("W", "female"),
    ("M", "male"), ("Male", "male"), ("MALE", "male"), ("man", "male"),
    ("O", "other_diverse"), ("other", "other_diverse"),
    ("U", "unknown_missing_sex"),
])
def test_typed_values_get_their_standard_code(typed, code) -> None:
    assert ps.standard_code(typed) == code


@pytest.mark.parametrize("typed", ["0", "1", "2", "-1", "3", "1.0", "yes", "x", "24"])
def test_numbers_and_unlisted_values_are_not_read_by_themselves(typed) -> None:
    # 1/2 and 0/1 are used both ways round; the vocabulary's own number
    # codes would make a 0 = male dataset all female.
    assert ps.standard_code(typed) is None


def test_a_number_counts_through_the_levels_of_participants_json() -> None:
    assert ps.standard_code("1", ps.sex_levels(ISO_SIDECAR)) == "male"
    assert ps.standard_code("2", ps.sex_levels(ISO_SIDECAR)) == "female"
    # The same numbers the other way round, in a column called Gender.
    assert ps.standard_code("0", ps.sex_levels(FLAG_SIDECAR)) == "male"
    assert ps.standard_code("1", ps.sex_levels(FLAG_SIDECAR)) == "female"
    # A level the sidecar does not explain stays unknown.
    assert ps.standard_code("3", ps.sex_levels(ISO_SIDECAR)) is None


def test_no_typed_value_stands_for_two_codes() -> None:
    import yaml

    entries = yaml.load(ps._vocabulary_file().read_text(encoding="utf-8"),
                        Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["sex"]
    seen: dict = {}
    for entry in entries.values():
        for value in (entry["standard_code"], *(entry.get("codes") or ())):
            key = str(value).strip().lower()
            assert seen.setdefault(key, entry["standard_code"]) == entry["standard_code"], key


def test_the_unknown_code_is_read_but_never_stored() -> None:
    table, stored = ps.sex_codes()
    assert set(stored) == {"male", "female", "other_diverse"}
    assert all(table[code] == code for code in stored)
    # Its spellings are still recognised, so that they become NULL and are
    # not reported as values nothing names.
    assert table["unknown_missing_sex"] == table["u"] == ps.UNKNOWN_CODE


# ── Run ───────────────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, rows=(), rowcount=0):
        self._rows, self.rowcount = list(rows), rowcount

    def all(self):
        return self._rows


class _Connection:
    """Two datasets: typed letters, and numbers that need participants.json."""

    ROWS = [("uuid-1", "ds000001", "F", 12), ("uuid-1", "ds000001", "Male", 9),
            ("uuid-1", "ds000001", "U", 2), ("uuid-1", "ds000001", "23.5", 1),
            ("uuid-2", "ds000002", "1", 20), ("uuid-2", "ds000002", "2", 22)]

    def __init__(self):
        self.writes = []
        self.selects = []

    async def execute(self, statement, params=None):
        sql = str(statement)
        if sql.lstrip().startswith("SELECT d.id"):
            self.selects.append((sql, params))
            return _Result(self.ROWS)
        if sql.lstrip().startswith("UPDATE"):
            self.writes.append(params)
            return _Result(rowcount=next(n for i, _a, v, n in self.ROWS
                                         if i == params["id"] and v == params["raw"]))
        raise AssertionError(sql)


def test_dry_run_writes_nothing_and_apply_writes_each_value_with_what_was_typed() -> None:
    conn = _Connection()
    reports = asyncio.run(ps.run(conn, apply=False, fetch=lambda accession: ISO_SIDECAR))
    assert conn.writes == []
    first, second = reports
    # 'U' is a sex that is not known; '23.5' is an age in the wrong column.
    assert first == {"accession": "ds000001", "written": 0,
                     "values": {"F": ["female", 12], "Male": ["male", 9], "U": [None, 2]},
                     "unknown": {"23.5": 1}}
    assert second["values"] == {"1": ["male", 20], "2": ["female", 22]} and not second["unknown"]

    reports = asyncio.run(ps.run(conn, apply=True, fetch=lambda accession: ISO_SIDECAR))
    assert [(w["id"], w["raw"], w["code"]) for w in conn.writes] == [
        ("uuid-1", "F", "female"), ("uuid-1", "Male", "male"),
        # Both become NULL; what was typed is kept in extra._sex_raw.
        ("uuid-1", "U", None), ("uuid-1", "23.5", None),
        ("uuid-2", "1", "male"), ("uuid-2", "2", "female")]
    assert all(w["raw_key"] == "_sex_raw" for w in conn.writes)
    assert [r["written"] for r in reports] == [24, 42]


def test_rows_that_hold_a_standard_code_are_not_read_again() -> None:
    conn = _Connection()
    asyncio.run(ps.run(conn, ["ds000001"], apply=False, fetch=lambda accession: None))
    sql, params = conn.selects[0]
    assert "NOT (p.sex = ANY(CAST(:standard AS text[])))" in sql
    # Rows that hold the unknown code are read again: they become NULL.
    assert set(params["standard"]) == {"male", "female", "other_diverse"}
    assert params["accessions"] == ["ds000001"]


def test_without_participants_json_the_listed_values_are_still_written() -> None:
    def fetch(accession):
        raise pg.ParticipantSidecarError("timed out")

    conn = _Connection()
    reports = asyncio.run(ps.run(conn, apply=True, fetch=fetch))
    assert [(w["raw"], w["code"]) for w in conn.writes] == [
        ("F", "female"), ("Male", "male"), ("U", None)]
    # The numbers stay as typed, not NULL, so the next run asks for the
    # sidecar again; so does the value nothing names.
    assert [r["accession"] for r in reports] == ["ds000001"] and reports[0]["unknown"] == {}


def test_the_sidecar_is_fetched_only_for_values_the_vocabulary_does_not_list() -> None:
    class _Listed(_Connection):
        ROWS = [("uuid-1", "ds000001", "F", 12), ("uuid-1", "ds000001", "M", 9)]

    asked = []
    asyncio.run(ps.run(_Listed(), apply=False, fetch=lambda accession: asked.append(accession)))
    assert asked == []
