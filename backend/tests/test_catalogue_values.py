"""The coded values and their dataset counts, loaded with the facets and the
participants (services/catalogue/catalogue_values.py, services/catalogue/schema_capabilities.py)."""

from __future__ import annotations

import time

import pytest

from services.catalogue.catalogue_values import catalogue_values as cv

# The cache as catalogue_values.load() fills it: value → datasets.
KNOWN = {
    "datatype": {"functional_mri": 40, "anatomical_mri": 55},
    "suffix": {"fmri_bold": 38, "t1_weighted_mri": 52},
    "task": {"resting_state": 20, "n_back": 4, "face_recognition": 3, "object_recognition_general": 1},
    "diagnosis": {"healthy_control": 155, "parkinsons_disease": 9},
}

# What the fake database answers, per table.
_FACET_ROWS = [("datatype", "functional_mri", 3), ("suffix", "fmri_bold", 3), ("task", "n_back", 1)]
_PARTICIPANT_ROWS = [("diagnosis", "healthy_control", 155), ("diagnosis", "epilepsy", 7),
                     ("sex", "female", 300), ("handedness", "right_handed", 120)]


@pytest.fixture(autouse=True)
def catalogue(monkeypatch):
    monkeypatch.setattr(cv, "_counts", dict(KNOWN))


class _Result:
    def __init__(self, value=None, rows=()):
        self._value, self._rows = value, list(rows)

    def scalar(self):
        return self._value

    def all(self):
        return self._rows


class _Savepoint:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.connection.savepoints.append("rolled back" if exc_type else "released")
        return False


class _Connection:
    """Answers the statements of _detect_summaries and catalogue_values.load.

    *fails*: the tables whose count query raises a database error.
    """

    def __init__(self, uncovered: int, fails: tuple[str, ...] = ()):
        self.uncovered = uncovered
        self.fails = fails
        self.savepoints: list[str] = []
        self.count_queries: list[str] = []

    def begin_nested(self):
        return _Savepoint(self)

    async def execute(self, statement, params=None):
        from sqlalchemy.exc import ProgrammingError

        sql = str(statement)
        if "to_regclass" in sql:
            return _Result(True)
        if "NOT EXISTS" in sql:
            return _Result(self.uncovered)
        if "count(DISTINCT dataset_id)" in sql:
            table = cv.FACETS_TABLE if cv.FACETS_TABLE in sql else cv.PARTICIPANTS_TABLE
            self.count_queries.append(table)
            if table in self.fails:
                raise ProgrammingError(sql, None, Exception("permission denied for table"))
            return _Result(rows=_FACET_ROWS if table == cv.FACETS_TABLE else _PARTICIPANT_ROWS)
        return _Result(1)


def _detect(uncovered: int, connection=None) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    connection = connection or _Connection(uncovered)
    asyncio.run(schema_capabilities._detect_summaries(connection, {"n_subjects"}))


def test_values_and_counts_are_loaded_with_complete_facets() -> None:
    connection = _Connection(uncovered=0)
    _detect(uncovered=0, connection=connection)
    # One query per table, every field at once.
    assert connection.count_queries == [cv.FACETS_TABLE, cv.PARTICIPANTS_TABLE]
    assert all(cv.loaded(field) for field in cv.FIELDS)
    assert "fmri_bold" in cv.known_values("suffix") and "bold" not in cv.known_values("suffix")
    assert cv.dataset_count("diagnosis", "epilepsy") == 7
    assert cv.dataset_count("diagnosis", "frontotemporal_dementia") == 0
    assert cv.dataset_count("handedness", "right_handed") == 120


def test_incomplete_facets_leave_the_facet_fields_uncached() -> None:
    # A value of a dataset missing from the facets would look unknown.
    connection = _Connection(uncovered=3)
    _detect(uncovered=3, connection=connection)
    assert connection.count_queries == [cv.PARTICIPANTS_TABLE]
    assert not any(cv.loaded(field) for field in cv.FACET_FIELDS)
    assert cv.dataset_count("diagnosis", "healthy_control") == 155


def test_a_failed_read_at_startup_leaves_that_table_uncached() -> None:
    from services.catalogue import schema_capabilities

    connection = _Connection(uncovered=0, fails=(cv.FACETS_TABLE,))
    _detect(uncovered=0, connection=connection)
    assert schema_capabilities.has_dataset_facets()
    assert not cv.loaded("suffix")
    assert cv.loaded("diagnosis")   # the other table still loads
    # The summary reads, then one savepoint per table: only the failed one is rolled back.
    assert connection.savepoints == ["released", "rolled back", "released"]


def test_a_failed_reload_keeps_the_old_values_and_waits_before_trying_again(monkeypatch) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    monkeypatch.setattr(schema_capabilities, "has_dataset_facets", lambda: True)
    monkeypatch.setattr(cv, "_last_attempt", time.monotonic() - cv.REFRESH_SECONDS - 1)
    session = _Connection(uncovered=0, fails=(cv.FACETS_TABLE, cv.PARTICIPANTS_TABLE))
    asyncio.run(cv.refresh_if_stale(session))
    asyncio.run(cv.refresh_if_stale(session))
    assert session.savepoints == ["rolled back", "rolled back"]  # one load; the second call waits
    assert "fmri_bold" in cv.known_values("suffix") and cv.dataset_count("task", "resting_state") == 20
    assert cv.dataset_count("diagnosis", "parkinsons_disease") == 9


def test_without_facets_a_reload_drops_the_facet_fields(monkeypatch) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    monkeypatch.setattr(schema_capabilities, "has_dataset_facets", lambda: False)
    asyncio.run(cv.load(_Connection(uncovered=0)))
    assert not cv.loaded("suffix") and cv.loaded("diagnosis") and cv.loaded("sex")


def test_an_error_that_is_not_a_database_error_is_not_swallowed(monkeypatch) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    class _Buggy(_Connection):
        async def execute(self, statement, params=None):
            raise TypeError("a bug in load()")

    monkeypatch.setattr(schema_capabilities, "has_dataset_facets", lambda: True)
    monkeypatch.setattr(cv, "_last_attempt", None)
    with pytest.raises(TypeError):
        asyncio.run(cv.refresh_if_stale(_Buggy(uncovered=0)))
    assert cv._last_attempt is None  # the next search tries again


def test_the_count_statement_names_only_its_own_fields() -> None:
    sql = cv._count_sql(cv.PARTICIPANTS_TABLE, cv.PARTICIPANT_FIELDS)
    assert sql.count("UNION ALL") == 2 and "bids_dataset_facets" not in sql
    assert "count(DISTINCT dataset_id)" in sql and "GROUP BY handedness" in sql
