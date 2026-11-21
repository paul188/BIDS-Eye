"""What the catalogue database holds, counted, and the startup schema check
(services/catalogue/catalogue.py, services/catalogue/tables.py)."""

from __future__ import annotations

import asyncio
import time

import pytest
from sqlalchemy.exc import ProgrammingError

from constants import OPTIONAL_DATASET_COLUMNS
from services.catalogue import tables
from services.catalogue.catalogue import (Catalogue, ColumnCoverage, MissingSchemaError, Summaries,
                                          ValueCounts, catalogue, check_schema)

# Coded values as load() fills them: value → datasets.
KNOWN = {
    "datatype": {"functional_mri": 40, "anatomical_mri": 55},
    "suffix": {"fmri_bold": 38, "t1_weighted_mri": 52},
    "task": {"resting_state": 20, "n_back": 4},
    "diagnosis": {"healthy_control": 155, "parkinsons_disease": 9},
}

# What the fake database answers.
_FACET_ROWS = [("datatype", "functional_mri", 3), ("suffix", "fmri_bold", 3), ("task", "n_back", 1)]
_PARTICIPANT_ROWS = [("diagnosis", "healthy_control", 155), ("diagnosis", "epilepsy", 7),
                     ("sex", "female", 300), ("handedness", "right_handed", 120)]


class _Result:
    def __init__(self, value=None, rows=()):
        self._value, self._rows = value, list(rows)

    def __iter__(self):
        return iter(self._rows)

    def scalar_one(self):
        return self._value

    def mappings(self):
        return self

    def one(self):
        return self._value

    def all(self):
        return self._rows


class _Connection:
    """Answers load() and check_schema() from canned data.

    *filled*: datasets per optional column, of *total*. *uncovered*: datasets
    with files that the facets table misses. *fails*: the reads ("coverage",
    "summaries", a table name) that raise a database error. *columns*: the
    (table, column) pairs information_schema lists.
    """

    def __init__(self, total=1000, filled=None, uncovered=0, subject_counts=1000,
                 fails=(), columns=()):
        self.total, self.filled = total, dict(filled or {})
        self.uncovered, self.subject_counts = uncovered, subject_counts
        self.fails, self.columns = fails, list(columns)
        self.savepoints: list[str] = []
        self.reads: list[str] = []

    def begin_nested(self):
        connection = self

        class _Savepoint:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                connection.savepoints.append("rolled back" if exc_type else "released")
                return False

        return _Savepoint()

    def _read(self, name, sql):
        self.reads.append(name)
        if name in self.fails:
            raise ProgrammingError(sql, None, Exception("permission denied"))

    async def execute(self, statement, params=None):
        sql = str(statement)
        if "information_schema.columns" in sql:
            return _Result(rows=self.columns)
        if "AS total" in sql:
            self._read("coverage", sql)
            return _Result({"total": self.total,
                            **{name: self.filled.get(name, 0) for name in OPTIONAL_DATASET_COLUMNS}})
        if "count(bids_datasets.n_subjects)" in sql:
            self._read("summaries", sql)
            return _Result(self.subject_counts)
        if "bids_dataset_facets.dataset_id = bids_datasets.id" in sql:
            return _Result(self.uncovered)
        if "UNION ALL" in sql:
            table = tables.dataset_facets.name if tables.dataset_facets.name in sql else tables.participants.name
            self._read(table, sql)
            return _Result(rows=_FACET_ROWS if table == tables.dataset_facets.name else _PARTICIPANT_ROWS)
        raise AssertionError(f"unexpected statement: {sql}")


def _load(connection: _Connection, into: Catalogue | None = None) -> Catalogue:
    target = into or Catalogue()
    asyncio.run(target.load(connection))
    return target


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_a_column_is_usable_once_five_percent_of_the_datasets_fill_it() -> None:
    coverage = _load(_Connection(total=1000, filled={"readme_text": 850, "dx_status": 49,
                                                     "institutions": 50})).coverage
    assert coverage.usable_columns() == {"readme_text", "institutions"}
    assert not coverage.usable("dx_status")


def test_an_empty_catalogue_has_no_usable_column() -> None:
    assert ColumnCoverage().usable_columns() == frozenset()
    assert _load(_Connection(total=0)).coverage.usable_columns() == frozenset()


def test_examples_naming_an_unusable_column_are_recognised() -> None:
    coverage = ColumnCoverage.full({"readme_text"})
    assert not coverage.uses_unusable_column("SELECT 1 FROM bids_datasets d WHERE d.readme_text ILIKE '%x%'")
    assert coverage.uses_unusable_column("SELECT 1 FROM unnest(d.institutions) AS inst")


def test_full_coverage_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        ColumnCoverage.full({"no_such_column"})


def test_every_optional_column_is_declared() -> None:
    assert set(OPTIONAL_DATASET_COLUMNS) <= set(tables.datasets.c.keys())


# ── Summaries ─────────────────────────────────────────────────────────────────

def test_complete_facets_and_filled_subject_counts_are_used() -> None:
    assert _load(_Connection()).summaries == Summaries(facets=True, subject_counts=True)


def test_facets_missing_a_dataset_are_not_used_until_a_later_load_finds_them_complete() -> None:
    state = _load(_Connection(uncovered=3))
    assert not state.summaries.facets
    _load(_Connection(uncovered=0), into=state)
    assert state.summaries.facets


# ── Coded values ──────────────────────────────────────────────────────────────

def test_values_and_counts_are_loaded_with_complete_facets() -> None:
    connection = _Connection()
    values = _load(connection).values
    # One query per table, every field at once.
    assert [r for r in connection.reads if r.startswith("bids_")] == [
        tables.dataset_facets.name, tables.participants.name]
    assert all(values.loaded(field) for field in Catalogue.FIELDS)
    assert "fmri_bold" in values.known_values("suffix") and "bold" not in values.known_values("suffix")
    assert values.dataset_count("diagnosis", "epilepsy") == 7
    assert values.dataset_count("diagnosis", "frontotemporal_dementia") == 0
    assert values.dataset_count("handedness", "right_handed") == 120


def test_incomplete_facets_leave_the_facet_fields_unread() -> None:
    # A value of a dataset missing from the facets would look unknown.
    state = Catalogue()
    state.values = ValueCounts(dict(KNOWN))
    connection = _Connection(uncovered=3)
    _load(connection, into=state)
    assert tables.dataset_facets.name not in connection.reads
    assert not any(state.values.loaded(field) for field in Catalogue.FACET_FIELDS)
    assert state.values.dataset_count("diagnosis", "healthy_control") == 155


# ── Failures ──────────────────────────────────────────────────────────────────

def test_a_failed_read_keeps_that_part_and_the_others_still_load() -> None:
    state = Catalogue()
    state.coverage = ColumnCoverage.full({"readme_text"})
    connection = _Connection(fails=("coverage", tables.dataset_facets.name))
    _load(connection, into=state)
    assert state.coverage.usable_columns() == {"readme_text"}
    assert state.summaries.facets
    assert not state.values.loaded("suffix") and state.values.loaded("diagnosis")
    # Each read in its own savepoint: only the failed ones are rolled back.
    assert connection.savepoints == ["rolled back", "released", "rolled back", "released"]


def test_a_failed_reload_keeps_the_old_values_and_waits_before_trying_again() -> None:
    state = Catalogue()
    state.summaries = Summaries(facets=True, subject_counts=True)
    state.values = ValueCounts(dict(KNOWN))
    state._last_attempt = time.monotonic() - Catalogue.REFRESH_SECONDS - 1
    connection = _Connection(fails=("summaries", tables.dataset_facets.name, tables.participants.name))
    asyncio.run(state.refresh_if_stale(connection))
    asyncio.run(state.refresh_if_stale(connection))
    assert connection.reads.count("coverage") == 1   # one load; the second call waits
    assert state.values.dataset_count("task", "resting_state") == 20
    assert state.values.dataset_count("diagnosis", "parkinsons_disease") == 9


def test_an_error_that_is_not_a_database_error_is_not_swallowed() -> None:
    class _Buggy(_Connection):
        async def execute(self, statement, params=None):
            raise TypeError("a bug in load()")

    state = Catalogue()
    with pytest.raises(TypeError):
        asyncio.run(state.refresh_if_stale(_Buggy()))
    assert state._last_attempt is None  # the next search tries again


def test_the_shared_instance_starts_empty() -> None:
    # conftest empties it before every test.
    assert catalogue.coverage == ColumnCoverage() and not catalogue.values.loaded("task")


# ── Schema ────────────────────────────────────────────────────────────────────

def _all_columns(*, without=()):
    return [(table.name, column.name) for table in tables.metadata.sorted_tables
            for column in table.columns if (table.name, column.name) not in without]


def test_a_complete_schema_passes(monkeypatch) -> None:
    monkeypatch.setenv(tables.EXACT_FLAG, "true")
    monkeypatch.setenv(tables.PAPERS_FLAG, "true")
    asyncio.run(check_schema(_Connection(columns=_all_columns())))


def test_a_missing_column_stops_startup_and_names_its_migration() -> None:
    connection = _Connection(columns=_all_columns(without={("bids_datasets", "readme_text")}))
    with pytest.raises(MissingSchemaError, match=r"bids_datasets\.readme_text \(crawlers/migrate_add_openneuro_metadata\.sql\)"):
        asyncio.run(check_schema(connection))


def test_a_missing_table_names_every_column() -> None:
    missing = {("bids_dataset_facets", name) for name in tables.dataset_facets.c.keys()}
    with pytest.raises(MissingSchemaError, match="bids_dataset_facets.dataset_id"):
        asyncio.run(check_schema(_Connection(columns=_all_columns(without=missing))))


def test_columns_behind_a_switch_are_needed_only_while_it_is_on(monkeypatch) -> None:
    exact = {("bids_datasets", "search_tsv_exact"), ("bids_datasets", "search_tsv_exact_papers")}
    exact |= {("bids_word_forms", name) for name in tables.word_forms.c.keys()}
    connection = _Connection(columns=_all_columns(without=exact))
    monkeypatch.delenv(tables.EXACT_FLAG, raising=False)
    asyncio.run(check_schema(connection))
    monkeypatch.setenv(tables.EXACT_FLAG, "true")
    with pytest.raises(MissingSchemaError, match="bids_word_forms.form"):
        asyncio.run(check_schema(connection))


def test_start_checks_the_schema_before_loading() -> None:
    class _Engine:
        def __init__(self, connection):
            self.connection = connection

        def connect(self):
            connection = self.connection

            class _Context:
                async def __aenter__(self):
                    return connection

                async def __aexit__(self, *exc):
                    return False

            return _Context()

    connection = _Connection(columns=_all_columns(without={("bids_datasets", "search_tsv")}))
    state = Catalogue()
    with pytest.raises(MissingSchemaError):
        asyncio.run(state.start(_Engine(connection)))
    assert connection.reads == []
