"""BIDS filename entities as scan terms (backend/vocabulary/bids_entities.py)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from bids_entities import entity_condition

MIGRATION = Path(__file__).resolve().parents[2] / "migrations" / "versions" / "0004_entity_indexes.py"


def test_an_entity_term_is_a_condition_on_the_files() -> None:
    assert entity_condition("dir-AP") == (
        "EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id "
        "AND LOWER(o2.other_entities->>'direction') = 'ap')")
    assert "'acquisition') = 'highres'" in entity_condition("acq-highres")
    assert "'echo') = '2'" in entity_condition("_echo-2")
    assert "'hemisphere') = 'l'" in entity_condition("hemi-L")


@pytest.mark.parametrize("term", [
    "echo-planar", "part-time", "mt-something", "CE-MRI", "resting-state", "T1-weighted",
    "dir-AP scans", "sub-01", "task-rest", "dir-", "dir-A_P"])
def test_a_word_with_a_hyphen_is_not_an_entity(term: str) -> None:
    assert entity_condition(term) is None


def _migration():
    spec = importlib.util.spec_from_file_location("entity_indexes", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entity", _migration().INDEXED_ENTITIES)
def test_the_condition_is_the_indexed_expression(entity: str) -> None:
    # PostgreSQL uses an expression index only for the expression it was built on.
    key = {"direction": "dir", "acquisition": "acq"}[entity]
    condition = entity_condition(f"{key}-x")
    written = re.search(r"LOWER\(o2\.other_entities->>'(\w+)'\)", condition)
    assert written and written.group(1) == entity
    source = MIGRATION.read_text()
    assert "(LOWER(other_entities ->> '{entity}'), dataset_id)" in source


def _plan(**fields):
    from preprocess import QueryPlan
    # query_family goes away with PR #48; until then the plan requires it.
    family = {"query_family": "scan_filter"} if "query_family" in QueryPlan.model_fields else {}
    return QueryPlan(**family, **fields)


def test_entity_scan_terms_get_a_filter_line_and_no_text_search() -> None:
    from preprocess import ScanRequirement, _build_augmented_question

    plan = _plan(natural_language_summary="Resting-state fMRI with dir-AP and dir-PA scans.",
                     scan_requirements=[ScanRequirement(scan_terms=["dir-AP"]),
                                        ScanRequirement(scan_terms=["dir-PA"])])
    augmented = _build_augmented_question(plan, [], unresolved_terms={"scan": ["dir-AP", "dir-PA"]})
    assert ('  scan "dir-AP": EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id '
            "AND LOWER(o2.other_entities->>'direction') = 'ap')") in augmented
    assert "'direction') = 'pa')" in augmented
    assert "VOCABULARY MISS" not in augmented and "text_match('dir" not in augmented


def test_other_unknown_scan_terms_keep_their_text_search() -> None:
    from preprocess import ScanRequirement, _build_augmented_question

    plan = _plan(natural_language_summary="x",
                     scan_requirements=[ScanRequirement(scan_terms=["dir-AP", "eye tracking"])])
    augmented = _build_augmented_question(plan, [], unresolved_terms={"scan": ["dir-AP", "eye tracking"]})
    assert "'direction') = 'ap')" in augmented
    assert "VOCABULARY MISS" in augmented and "'eye tracking'" in augmented
    assert "'dir-AP'" not in augmented.split("VOCABULARY MISS")[1]
