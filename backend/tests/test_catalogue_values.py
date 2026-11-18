"""Generated datatype / suffix / task values checked against the catalogue (services/catalogue/catalogue_values.py)."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlglot")

from services.catalogue import catalogue_values as cv  # noqa: E402

KNOWN = {
    "datatype": {"functional_mri", "anatomical_mri"},
    "suffix": {"fmri_bold", "t1_weighted_mri"},
    "task": {"resting_state", "n_back", "face_recognition", "object_recognition_general"},
}
GROUPS = {"task": {"recognition": ["face_recognition", "object_recognition_general", "never_recorded"]}}
LEAVES = {"suffix": {"bold": "fmri_bold", "t1w": "t1_weighted_mri"},
          "task": {"recognition": "emotional_recognition_ibc"}}


@pytest.fixture(autouse=True)
def catalogue(monkeypatch):
    cv.set_known(KNOWN)
    monkeypatch.setattr(cv, "_vocabulary", lambda: (GROUPS, LEAVES))
    monkeypatch.delenv(cv.ENV, raising=False)
    yield
    cv.set_known({})


def _exists(condition: str) -> str:
    return (f"SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS o2 "
            f"WHERE o2.dataset_id = d.id AND {condition}) GROUP BY d.id")


def test_raw_bids_suffix_is_corrected() -> None:
    # Gold request qp05: suffix = 'bold' found 0 datasets.
    sql, notes = cv.check_sql(_exists("o2.suffix = 'bold' AND o2.datatype = 'functional_mri'"))
    assert "o2.suffix = 'fmri_bold'" in sql and "'bold'" not in sql
    assert notes == ["suffix 'bold' occurs in no dataset: replaced by 'fmri_bold'"]


def test_known_values_leave_the_sql_as_it_is() -> None:
    original = _exists("o2.task IN ('resting_state', 'n_back')")
    assert cv.check_sql(original) == (original, [])


def test_a_task_category_becomes_its_recorded_tasks() -> None:
    # The category, not the one narrow synonym; only tasks some dataset has.
    sql, _ = cv.check_sql(_exists("o2.task = 'recognition'"))
    assert "o2.task IN ('face_recognition', 'object_recognition_general')" in sql


def test_unknown_value_in_a_list_is_left_out() -> None:
    sql, notes = cv.check_sql(_exists("o2.task IN ('resting_state', 'social_cognition')"))
    assert "o2.task IN ('resting_state')" in sql
    assert notes == ["task 'social_cognition' occurs in no dataset: left out of the list"]


def test_unknown_required_value_drops_its_condition() -> None:
    sql, notes = cv.check_sql(_exists("o2.datatype = 'functional_mri' AND o2.task = 'social_cognition'"))
    assert "social_cognition" not in sql
    assert "o2.datatype = 'functional_mri'" in sql and "o2.dataset_id = d.id" in sql
    assert notes[0].endswith("condition dropped")


def test_unknown_alternative_of_an_or_changes_nothing() -> None:
    sql, notes = cv.check_sql(_exists("(o2.task = 'n_back' OR o2.task = 'social_cognition')"))
    assert "social_cognition" not in sql and "o2.task = 'n_back'" in sql
    assert notes[0].endswith("left out (one of several alternatives)")


def test_every_alternative_unknown_drops_the_whole_condition() -> None:
    sql, _ = cv.check_sql(_exists("o2.datatype = 'functional_mri' AND o2.task IN ('a_task', 'b_task')"))
    assert "a_task" not in sql and "o2.datatype = 'functional_mri'" in sql


def test_under_not_the_condition_stays() -> None:
    # "without X": no dataset has X, so the exclusion holds for all. Dropping the
    # value would turn it into "without any file".
    original = ("SELECT d.id FROM bids_datasets AS d WHERE NOT EXISTS (SELECT 1 FROM bids_objects AS o2 "
                "WHERE o2.dataset_id = d.id AND o2.task = 'social_cognition')")
    sql, notes = cv.check_sql(original)
    assert "o2.task = 'social_cognition'" in sql and "o2.dataset_id = d.id" in sql
    assert notes[0].endswith("kept (under NOT it excludes nothing)")


def test_not_in_loses_only_the_unknown_value() -> None:
    sql, _ = cv.check_sql(_exists("NOT o2.task IN ('n_back', 'social_cognition')"))
    assert "NOT o2.task IN ('n_back')" in sql


def test_value_written_on_the_left() -> None:
    sql, _ = cv.check_sql(_exists("'bold' = o2.suffix"))
    assert "o2.suffix = 'fmri_bold'" in sql


def test_pattern_that_matches_nothing_is_dropped() -> None:
    sql, notes = cv.check_sql(_exists("o2.datatype = 'functional_mri' AND o2.task ILIKE '%juggling%'"))
    assert "juggling" not in sql and notes[0].startswith("task pattern '%juggling%' matches no dataset")
    original = _exists("o2.task ILIKE '%RECOGNITION%'")
    assert cv.check_sql(original) == (original, [])


def test_off_without_known_values_or_by_switch(monkeypatch) -> None:
    original = _exists("o2.suffix = 'bold'")
    monkeypatch.setenv(cv.ENV, "false")
    assert cv.check_sql(original) == (original, [])
    monkeypatch.delenv(cv.ENV)
    cv.set_known({})
    assert cv.check_sql(original) == (original, [])


def test_placeholders_of_later_stages_survive() -> None:
    sql, _ = cv.check_sql(_exists("o2.suffix = 'bold'") .replace(
        "GROUP BY", "AND text_match('pediatric', 'short') GROUP BY"))
    assert "text_match('pediatric', 'short')" in sql.lower()


def test_rewriter_runs_the_check_and_reports_it() -> None:
    from services.translate.sql_rewriter import rewrite_generated_sql

    raw = ("[SQL]\nSELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
           "WHERE EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id AND o2.suffix = 'bold') "
           "GROUP BY d.id\n[/SQL]")
    result = rewrite_generated_sql(raw, "")
    assert "'fmri_bold'" in result.sql and "'bold'" not in result.sql
    assert result.value_fixes == ["suffix 'bold' occurs in no dataset: replaced by 'fmri_bold'"]


def test_real_vocabulary_knows_the_raw_suffixes(monkeypatch) -> None:
    monkeypatch.undo()  # the real vocabulary, RAG/value_mappings.yaml
    cv.set_known(KNOWN)
    try:
        cv._vocabulary()
    except Exception as exc:  # pragma: no cover - vocabulary not on the path
        pytest.skip(f"vocabulary not importable: {exc}")
    assert cv.correction("suffix", "bold") == ["fmri_bold"]
    assert cv.correction("suffix", "T1w") == ["t1_weighted_mri"]


# ── Loading with the facets (services/catalogue/schema_capabilities.py) ─────────────────

class _Result:
    def __init__(self, value=None, rows=()):
        self._value, self._rows = value, list(rows)

    def scalar(self):
        return self._value

    def all(self):
        return self._rows


class _Connection:
    """Answers the statements of _detect_summaries and catalogue_values.load."""

    def __init__(self, uncovered: int):
        self.uncovered = uncovered

    async def execute(self, statement, params=None):
        sql = str(statement)
        if "to_regclass" in sql:
            return _Result(True)
        if "NOT EXISTS" in sql:
            return _Result(self.uncovered)
        if "UNION ALL" in sql:
            return _Result(rows=[("suffix", "fmri_bold"), ("task", "n_back"), ("datatype", "functional_mri")])
        return _Result(1)


def _detect(uncovered: int) -> None:
    import asyncio
    from services.catalogue import schema_capabilities

    asyncio.run(schema_capabilities._detect_summaries(_Connection(uncovered), {"n_subjects"}))


def test_values_are_loaded_with_complete_facets() -> None:
    cv.set_known({})
    _detect(uncovered=0)
    assert cv.enabled() and cv.is_known("suffix", "fmri_bold") and not cv.is_known("suffix", "bold")


def test_incomplete_facets_switch_the_check_off() -> None:
    # A value of a dataset missing from the facets would look unknown.
    _detect(uncovered=3)
    assert not cv.enabled()
