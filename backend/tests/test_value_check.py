"""Generated datatype / suffix / task values checked against the catalogue (services/translate/value_check.py)."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlglot")

from services.database_stats.database_stats import ValueCounts, database_stats  # noqa: E402
from services.translate import value_check as vc  # noqa: E402

# The values as database_stats.load() reads them: value → datasets.
KNOWN = {
    "datatype": {"functional_mri": 40, "anatomical_mri": 55},
    "suffix": {"fmri_bold": 38, "t1_weighted_mri": 52},
    "task": {"resting_state": 20, "n_back": 4, "face_recognition": 3, "object_recognition_general": 1},
}
GROUPS = {"task": {"recognition": ["face_recognition", "object_recognition_general", "never_recorded"]}}
LEAVES = {"suffix": {"bold": "fmri_bold", "t1w": "t1_weighted_mri"},
          "task": {"recognition": "emotional_recognition_ibc"}}


@pytest.fixture(autouse=True)
def known_values(monkeypatch):
    monkeypatch.setattr(database_stats, "values", ValueCounts(dict(KNOWN)))
    monkeypatch.setattr(vc, "_vocabulary", lambda: (GROUPS, LEAVES))
    monkeypatch.delenv(vc.ENV, raising=False)


def _exists(condition: str) -> str:
    return (f"SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS o2 "
            f"WHERE o2.dataset_id = d.id AND {condition}) GROUP BY d.id")


def test_raw_bids_suffix_is_corrected() -> None:
    # Gold request qp05: suffix = 'bold' found 0 datasets.
    sql, notes = vc.check_sql(_exists("o2.suffix = 'bold' AND o2.datatype = 'functional_mri'"))
    assert "o2.suffix = 'fmri_bold'" in sql and "'bold'" not in sql
    assert notes == ["suffix 'bold' occurs in no dataset: replaced by 'fmri_bold'"]


def test_known_values_leave_the_sql_as_it_is() -> None:
    original = _exists("o2.task IN ('resting_state', 'n_back')")
    assert vc.check_sql(original) == (original, [])


def test_a_task_category_becomes_its_recorded_tasks() -> None:
    # The category, not the one narrow synonym; only tasks some dataset has.
    sql, _ = vc.check_sql(_exists("o2.task = 'recognition'"))
    assert "o2.task IN ('face_recognition', 'object_recognition_general')" in sql


def test_unknown_value_in_a_list_is_left_out() -> None:
    sql, notes = vc.check_sql(_exists("o2.task IN ('resting_state', 'social_cognition')"))
    assert "o2.task IN ('resting_state')" in sql
    assert notes == ["task 'social_cognition' occurs in no dataset: left out of the list"]


def test_unknown_required_value_drops_its_condition() -> None:
    sql, notes = vc.check_sql(_exists("o2.datatype = 'functional_mri' AND o2.task = 'social_cognition'"))
    assert "social_cognition" not in sql
    assert "o2.datatype = 'functional_mri'" in sql and "o2.dataset_id = d.id" in sql
    assert notes[0].endswith("condition dropped")


def test_unknown_alternative_of_an_or_changes_nothing() -> None:
    sql, notes = vc.check_sql(_exists("(o2.task = 'n_back' OR o2.task = 'social_cognition')"))
    assert "social_cognition" not in sql and "o2.task = 'n_back'" in sql
    assert notes[0].endswith("left out (one of several alternatives)")


def test_every_alternative_unknown_drops_the_whole_condition() -> None:
    sql, _ = vc.check_sql(_exists("o2.datatype = 'functional_mri' AND o2.task IN ('a_task', 'b_task')"))
    assert "a_task" not in sql and "o2.datatype = 'functional_mri'" in sql


def test_under_not_the_condition_stays() -> None:
    # "without X": no dataset has X, so the exclusion holds for all. Dropping the
    # value would turn it into "without any file".
    original = ("SELECT d.id FROM bids_datasets AS d WHERE NOT EXISTS (SELECT 1 FROM bids_objects AS o2 "
                "WHERE o2.dataset_id = d.id AND o2.task = 'social_cognition')")
    sql, notes = vc.check_sql(original)
    assert "o2.task = 'social_cognition'" in sql and "o2.dataset_id = d.id" in sql
    assert notes[0].endswith("kept (under NOT it excludes nothing)")


def test_not_in_loses_only_the_unknown_value() -> None:
    sql, _ = vc.check_sql(_exists("NOT o2.task IN ('n_back', 'social_cognition')"))
    assert "NOT o2.task IN ('n_back')" in sql


def test_value_written_on_the_left() -> None:
    sql, _ = vc.check_sql(_exists("'bold' = o2.suffix"))
    assert "o2.suffix = 'fmri_bold'" in sql


def test_pattern_that_matches_nothing_is_dropped() -> None:
    sql, notes = vc.check_sql(_exists("o2.datatype = 'functional_mri' AND o2.task ILIKE '%juggling%'"))
    assert "juggling" not in sql and notes[0].startswith("task pattern '%juggling%' matches no dataset")
    original = _exists("o2.task ILIKE '%RECOGNITION%'")
    assert vc.check_sql(original) == (original, [])


def test_off_without_known_values_or_by_switch(monkeypatch) -> None:
    original = _exists("o2.suffix = 'bold'")
    monkeypatch.setenv(vc.ENV, "false")
    assert vc.check_sql(original) == (original, [])
    monkeypatch.delenv(vc.ENV)
    database_stats.clear()
    assert vc.check_sql(original) == (original, [])


def test_off_while_only_the_participant_fields_are_cached(monkeypatch) -> None:
    # Facets table missing or incomplete: every suffix would look unknown.
    monkeypatch.setattr(database_stats, "values", ValueCounts({"diagnosis": {"epilepsy": 7}, "sex": {}, "handedness": {}}))
    original = _exists("o2.suffix = 'bold'")
    assert vc.check_sql(original) == (original, [])


def test_diagnosis_values_are_never_checked(monkeypatch) -> None:
    # A dropped required diagnosis would widen the search to everything else.
    monkeypatch.setattr(database_stats, "values", ValueCounts({**KNOWN, "diagnosis": {"epilepsy": 7}}))
    original = ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_participants AS p2 "
                "WHERE p2.dataset_id = d.id AND p2.diagnosis = 'frontotemporal_dementia') GROUP BY d.id")
    assert vc.check_sql(original) == (original, [])


def test_placeholders_of_later_stages_survive() -> None:
    sql, _ = vc.check_sql(_exists("o2.suffix = 'bold'") .replace(
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
    monkeypatch.setattr(database_stats, "values", ValueCounts(dict(KNOWN)))
    if vc._vocabulary() is None:  # pragma: no cover - vocabulary not on the path
        pytest.skip("vocabulary not importable")
    assert vc.correction("suffix", "bold") == ["fmri_bold"]
    assert vc.correction("suffix", "T1w") == ["t1_weighted_mri"]


def test_an_unloadable_vocabulary_is_tried_once(monkeypatch) -> None:
    import builtins

    monkeypatch.undo()
    monkeypatch.setattr(database_stats, "values", ValueCounts(dict(KNOWN)))
    real_import = builtins.__import__
    attempts = []

    def failing_import(name, *args, **kwargs):
        if name == "yaml_to_llamaindex":
            attempts.append(name)
            raise ImportError("no RAG on the path")
        return real_import(name, *args, **kwargs)

    vc._vocabulary.cache_clear()
    monkeypatch.setattr(builtins, "__import__", failing_import)
    try:
        assert vc.correction("suffix", "bold") == []
        assert vc.correction("task", "recognition") == []
        assert attempts == ["yaml_to_llamaindex"]
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        vc._vocabulary.cache_clear()
