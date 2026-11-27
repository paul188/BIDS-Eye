"""What the catalogue database holds, counted, and the startup schema check
(services/database_stats/database_stats.py, services/schema/)."""

from __future__ import annotations

import asyncio
import time

import pytest
from sqlalchemy.exc import ProgrammingError

from services.database_stats.database_stats import ColumnCoverage, DatabaseStats, Names, Summaries, ValueCounts, database_stats
from services.schema import tables
from services.schema.check import MissingSchemaError, check_schema, sql_type

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
    "summaries", a table name, "institution names", "site names") that raise
    a database error. *columns*: the (table, column) pairs information_schema
    lists. *institutions*: bids_datasets.institutions entries. *places*:
    (city, region, country) of the sites; *site_institutions*: their names.
    """

    def __init__(self, total=1000, filled=None, uncovered=0, subject_counts=1000,
                 fails=(), columns=(), institutions=(), places=(), site_institutions=()):
        self.institutions, self.places = list(institutions), list(places)
        self.site_institutions = list(site_institutions)
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
                            **{name: self.filled.get(name, 0) for name in tables.OPTIONAL_DATASET_COLUMNS}})
        if "count(bids_datasets.n_subjects)" in sql:
            self._read("summaries", sql)
            return _Result(self.subject_counts)
        if "bids_dataset_facets.dataset_id = bids_datasets.id" in sql:
            return _Result(self.uncovered)
        if "UNION ALL" in sql:
            table = tables.dataset_facets.name if tables.dataset_facets.name in sql else tables.participants.name
            self._read(table, sql)
            return _Result(rows=_FACET_ROWS if table == tables.dataset_facets.name else _PARTICIPANT_ROWS)
        if "unnest(bids_datasets.institutions)" in sql:
            self._read("institution names", sql)
            return _Result(rows=[(name,) for name in self.institutions])
        if tables.site_places.name in sql:
            if "coalesce" in sql:
                self._read("site names", sql)
                return _Result(rows=self.places)
            return _Result(rows=[(name,) for name in self.site_institutions])
        raise AssertionError(f"unexpected statement: {sql}")


def _load(connection: _Connection, into: DatabaseStats | None = None) -> DatabaseStats:
    target = into or DatabaseStats()
    asyncio.run(target.load(connection))
    return target


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_coverage_counts_the_datasets_that_fill_each_optional_column() -> None:
    coverage = _load(_Connection(total=1000, filled={"readme_text": 850, "dx_status": 49})).coverage
    assert coverage.total == 1000
    assert coverage.filled["readme_text"] == 850 and coverage.filled["dx_status"] == 49
    assert coverage.filled["institutions"] == 0


def test_full_coverage_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        ColumnCoverage.full({"no_such_column"})


def test_every_optional_column_is_declared() -> None:
    assert set(tables.OPTIONAL_DATASET_COLUMNS) <= set(tables.datasets.c.keys())


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
    assert all(values.loaded(field) for field in DatabaseStats.FIELDS)
    assert "fmri_bold" in values.known_values("suffix") and "bold" not in values.known_values("suffix")
    assert values.dataset_count("diagnosis", "epilepsy") == 7
    assert values.dataset_count("diagnosis", "frontotemporal_dementia") == 0
    assert values.dataset_count("handedness", "right_handed") == 120


def test_incomplete_facets_leave_the_facet_fields_unread() -> None:
    # A value of a dataset missing from the facets would look unknown.
    state = DatabaseStats()
    state.values = ValueCounts(dict(KNOWN))
    connection = _Connection(uncovered=3)
    _load(connection, into=state)
    assert tables.dataset_facets.name not in connection.reads
    assert not any(state.values.loaded(field) for field in DatabaseStats.FACET_FIELDS)
    assert state.values.dataset_count("diagnosis", "healthy_control") == 155


# ── Names ─────────────────────────────────────────────────────────────────────

_PLACES = [("Bonn", "North Rhine-Westphalia", "Germany"), ("", "", "Mexico"), ("", "", "")]


def test_names_are_read_sorted_without_blank_ones() -> None:
    names = _load(_Connection(institutions=["University of Bonn", " ", None, "MIT"], places=_PLACES,
                              site_institutions=["University of Bonn"])).names
    assert names.dataset_institutions == ("MIT", "University of Bonn")
    assert names.site_places == (("", "", "Mexico"), ("Bonn", "North Rhine-Westphalia", "Germany"))
    assert names.site_institutions == ("University of Bonn",)


def test_new_names_are_handed_on_once_and_again_only_when_they_change() -> None:
    handed: list[tuple[Names, Names]] = []
    state = DatabaseStats(on_new_names=lambda names, before: handed.append((names, before)))
    _load(_Connection(institutions=["MIT"]), into=state)
    _load(_Connection(institutions=["MIT"]), into=state)
    assert len(handed) == 1 and handed[0][1] == Names()
    _load(_Connection(institutions=["MIT", "University of Bonn"]), into=state)
    assert len(handed) == 2
    assert handed[1][0].dataset_institutions == ("MIT", "University of Bonn")
    assert handed[1][1].dataset_institutions == ("MIT",)


def test_a_failed_read_of_names_keeps_them_and_unread_names_stay_unread() -> None:
    state = _load(_Connection(fails=("institution names",), places=_PLACES))
    assert state.names.dataset_institutions is None   # name_index.json stays the source
    assert state.names.site_places is not None
    _load(_Connection(institutions=["MIT"], places=_PLACES), into=state)
    _load(_Connection(fails=("institution names", "site names")), into=state)
    assert state.names.dataset_institutions == ("MIT",)
    assert len(state.names.site_places) == 2


def test_the_name_indexes_are_rebuilt_from_changed_lists_only(monkeypatch) -> None:
    import institution_index
    import place_index
    from services.database_stats.database_stats import fill_name_indexes

    built: list[str] = []
    monkeypatch.setattr(institution_index, "set_names", lambda names: built.append(f"institutions {len(names)}"))
    monkeypatch.setattr(place_index, "set_places",
                        lambda places, rows: built.append(f"places {len(places)} {len(rows)}") or ())
    first = Names(("MIT",), (("Bonn", "", "Germany"),), ("University of Bonn",))
    fill_name_indexes(first, Names())
    assert built == ["institutions 1", "places 1 1"]
    fill_name_indexes(Names(("MIT", "Yale"), first.site_places, first.site_institutions), first)
    assert built[2:] == ["institutions 2"]
    # Unread lists replace nothing.
    fill_name_indexes(Names(), first)
    assert len(built) == 3


# ── Failures ──────────────────────────────────────────────────────────────────

def test_a_failed_read_keeps_that_part_and_the_others_still_load() -> None:
    state = DatabaseStats()
    state.coverage = ColumnCoverage.full({"readme_text"})
    connection = _Connection(fails=("coverage", tables.dataset_facets.name))
    _load(connection, into=state)
    assert state.coverage.filled == {"readme_text": 1}
    assert state.summaries.facets
    assert not state.values.loaded("suffix") and state.values.loaded("diagnosis")
    # Each read in its own savepoint: only the failed ones are rolled back.
    assert connection.savepoints == ["rolled back", "released", "rolled back", "released",
                                     "released", "released"]


def test_a_failed_reload_keeps_the_old_values_and_waits_before_trying_again() -> None:
    state = DatabaseStats()
    state.summaries = Summaries(facets=True, subject_counts=True)
    state.values = ValueCounts(dict(KNOWN))
    state._last_attempt = time.monotonic() - DatabaseStats.REFRESH_SECONDS - 1
    connection = _Connection(fails=("summaries", tables.dataset_facets.name, tables.participants.name))
    asyncio.run(state.refresh_if_stale(connection))
    asyncio.run(state.refresh_if_stale(connection))
    assert connection.reads.count("coverage") == 1   # one load; the second call waits
    assert state.values.dataset_count("task", "resting_state") == 20
    assert state.values.dataset_count("diagnosis", "parkinsons_disease") == 9


def test_searches_arriving_during_a_reload_do_not_reload_too() -> None:
    class _Slow(_Connection):
        async def execute(self, statement, params=None):
            await asyncio.sleep(0)
            return await super().execute(statement, params)

    async def three_searches() -> None:
        await asyncio.gather(*(state.refresh_if_stale(connection) for _ in range(3)))

    state, connection = DatabaseStats(), _Slow(filled={"readme_text": 850})
    asyncio.run(three_searches())
    assert connection.reads.count("coverage") == 1
    assert state.coverage.filled["readme_text"] == 850
    # A reload that ends, by success or by error, lets the next stale search reload.
    state._last_attempt = time.monotonic() - DatabaseStats.REFRESH_SECONDS - 1
    asyncio.run(three_searches())
    assert connection.reads.count("coverage") == 2


def test_an_error_that_is_not_a_database_error_is_not_swallowed() -> None:
    class _Buggy(_Connection):
        async def execute(self, statement, params=None):
            raise TypeError("a bug in load()")

    state = DatabaseStats()
    with pytest.raises(TypeError):
        asyncio.run(state.refresh_if_stale(_Buggy()))
    assert state._last_attempt is None  # the next search tries again


def test_each_part_names_the_modules_that_use_it() -> None:
    # The "Used by" lists in database_stats.py are kept by hand; this keeps them complete.
    import re
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    own = backend / "services" / "database_stats" / "database_stats.py"
    folders = (backend, backend.parent / "LLM_preprocessor", backend.parent / "RAG")
    sources = [path for folder in folders for path in folder.rglob("*.py")
               if "tests" not in path.parts and path != own]
    for part, cls in (("coverage", ColumnCoverage), ("summaries", Summaries), ("values", ValueCounts)):
        users = {path.name for path in sources
                 if re.search(rf"database_stats\.{part}\b", path.read_text(encoding="utf-8"))}
        assert users, part
        missing = sorted(name for name in users if name not in cls.__doc__)
        assert missing == [], f"{cls.__name__} does not name {missing} under 'Used by'"


def test_the_shared_instance_starts_empty() -> None:
    # conftest empties it before every test.
    assert database_stats.coverage == ColumnCoverage() and not database_stats.values.loaded("task")


# ── Schema ────────────────────────────────────────────────────────────────────

def _all_columns(*, without=()):
    return [(table.name, column.name, sql_type(column)) for table in tables.MIGRATED_TABLES
            for column in table.columns if (table.name, column.name) not in without]


def test_a_complete_schema_passes(monkeypatch) -> None:
    monkeypatch.setenv(tables.EXACT_FLAG, "true")
    monkeypatch.setenv(tables.PAPERS_FLAG, "true")
    asyncio.run(check_schema(_Connection(columns=_all_columns())))


def test_a_missing_column_stops_startup_and_names_its_migration() -> None:
    connection = _Connection(columns=_all_columns(without={("bids_datasets", "readme_text")}))
    with pytest.raises(MissingSchemaError, match=r"bids_datasets\.readme_text \(revision 0001\)"):
        asyncio.run(check_schema(connection))


def test_a_column_with_another_type_stops_startup() -> None:
    columns = _all_columns(without={("bids_datasets", "downloads")})
    connection = _Connection(columns=columns + [("bids_datasets", "downloads", "text")])
    with pytest.raises(MissingSchemaError, match=r"bids_datasets\.downloads is text, declared integer"):
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
    state = DatabaseStats()
    with pytest.raises(MissingSchemaError):
        asyncio.run(state.start(_Engine(connection)))
    assert connection.reads == []


def test_only_columns_from_a_migration_are_checked() -> None:
    # name is created by BIDS-SQL itself; the same table object has it.
    assert "name" in tables.datasets.c and "search_tsv" in tables.datasets.c
    asyncio.run(check_schema(_Connection(columns=_all_columns(without={("bids_datasets", "name")}))))


def test_bids_sql_neither_maps_nor_creates_the_added_columns() -> None:
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    from db.models import BIDSDataset

    assert tables.datasets is BIDSDataset.__table__
    loads = str(select(BIDSDataset).compile(dialect=postgresql.dialect()))
    creates = str(CreateTable(tables.datasets).compile(dialect=postgresql.dialect()))
    assert "search_tsv" not in loads and "search_tsv" not in creates
    assert "accession_id" in loads and "accession_id" in creates
