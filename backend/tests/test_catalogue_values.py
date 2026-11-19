"""The known datatype / suffix / task values, loaded with the facets
(services/catalogue/catalogue_values.py, services/catalogue/schema_capabilities.py)."""

from __future__ import annotations

import pytest

from services.catalogue import catalogue_values as cv

KNOWN = {
    "datatype": {"functional_mri", "anatomical_mri"},
    "suffix": {"fmri_bold", "t1_weighted_mri"},
    "task": {"resting_state", "n_back", "face_recognition", "object_recognition_general"},
}


@pytest.fixture(autouse=True)
def catalogue():
    cv.set_known(KNOWN)
    yield
    cv.set_known({})


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
    """Answers the statements of _detect_summaries and catalogue_values.load."""

    def __init__(self, uncovered: int, load_fails: bool = False):
        self.uncovered = uncovered
        self.load_fails = load_fails
        self.loads = 0
        self.savepoints = []

    def begin_nested(self):
        return _Savepoint(self)

    async def execute(self, statement, params=None):
        from sqlalchemy.exc import ProgrammingError

        sql = str(statement)
        if "to_regclass" in sql:
            return _Result(True)
        if "NOT EXISTS" in sql:
            return _Result(self.uncovered)
        if "UNION ALL" in sql:
            self.loads += 1
            if self.load_fails:
                raise ProgrammingError(sql, None, Exception("permission denied for table"))
            return _Result(rows=[("suffix", "fmri_bold"), ("task", "n_back"), ("datatype", "functional_mri")])
        return _Result(1)


def _detect(uncovered: int, connection=None) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    connection = connection or _Connection(uncovered)
    asyncio.run(schema_capabilities._detect_summaries(connection, {"n_subjects"}))


def test_values_are_loaded_with_complete_facets() -> None:
    cv.set_known({})
    _detect(uncovered=0)
    assert cv.loaded() and cv.is_known("suffix", "fmri_bold") and not cv.is_known("suffix", "bold")


def test_incomplete_facets_switch_the_check_off() -> None:
    # A value of a dataset missing from the facets would look unknown.
    _detect(uncovered=3)
    assert not cv.loaded()


def test_a_failed_load_at_startup_keeps_the_facets_and_turns_the_check_off() -> None:
    from services.catalogue import schema_capabilities

    connection = _Connection(uncovered=0, load_fails=True)
    _detect(uncovered=0, connection=connection)
    assert schema_capabilities.has_dataset_facets()
    assert not cv.loaded()
    # The summary reads, then the failed load: only its savepoint is rolled back.
    assert connection.savepoints == ["released", "rolled back"]


def test_a_failed_reload_keeps_the_old_values_and_waits_before_trying_again(monkeypatch) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    monkeypatch.setattr(schema_capabilities, "has_dataset_facets", lambda: True)
    monkeypatch.setattr(cv, "_loaded_at", cv._loaded_at - cv.REFRESH_SECONDS - 1)
    session = _Connection(uncovered=0, load_fails=True)
    asyncio.run(cv.refresh_if_stale(session))
    asyncio.run(cv.refresh_if_stale(session))
    assert session.loads == 1 and session.savepoints == ["rolled back"]
    assert cv.is_known("suffix", "fmri_bold") and cv.is_known("task", "resting_state")


def test_an_error_that_is_not_a_database_error_is_not_swallowed(monkeypatch) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    async def broken(connection):
        raise TypeError("a bug in load()")

    monkeypatch.setattr(schema_capabilities, "has_dataset_facets", lambda: True)
    monkeypatch.setattr(cv, "_loaded_at", None)
    monkeypatch.setattr(cv, "load", broken)
    with pytest.raises(TypeError):
        asyncio.run(cv.refresh_if_stale(_Connection(uncovered=0)))
