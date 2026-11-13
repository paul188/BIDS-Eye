from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

import constants
from services import schema_capabilities as caps
from services import text_to_sql

FEW_SHOT_PATH = Path(__file__).resolve().parents[1] / "few_shot_examples.json"


@pytest.fixture(autouse=True)
def _reset_capabilities():
    caps.set_available_columns(())
    caps.set_display_columns(())
    caps.set_fts(False)
    yield
    caps.set_available_columns(())
    caps.set_display_columns(())
    caps.set_fts(False)


def _dataset_table(ddl: str) -> str:
    return ddl[: ddl.index("CREATE TABLE bids_objects")]


def test_schema_hides_optional_columns_by_default() -> None:
    table = _dataset_table(constants.build_schema_ddl(()))
    for column in constants.OPTIONAL_DATASET_COLUMNS:
        assert column not in table
    assert "description_text  TEXT " in table
    # The last column must not end with a comma.
    assert re.search(r"description_text\s+TEXT\s+--", table)


def test_schema_shows_only_available_optional_columns() -> None:
    table = _dataset_table(constants.build_schema_ddl({"readme_text", "size_bytes"}))
    assert "readme_text       TEXT," in table
    assert re.search(r"size_bytes\s+BIGINT\s+--", table)
    assert "institutions" not in table
    assert "dx_status" not in table


def test_few_shot_examples_only_use_known_dataset_columns() -> None:
    """Guards against examples drifting away from the schema the model sees."""
    known = {name for name, _, _ in constants._CORE_DATASET_COLUMNS}
    known |= set(constants.OPTIONAL_DATASET_COLUMNS)
    pool = json.loads(FEW_SHOT_PATH.read_text())
    used = {
        column
        for examples in pool.values()
        for example in examples
        for column in re.findall(r"\bd\.(\w+)", example["sql"])
    }
    assert used <= known, f"few-shot examples use unknown columns: {sorted(used - known)}"


def test_examples_using_unavailable_columns_are_never_selected() -> None:
    pool = {
        "description_search": [
            {"question": "datasets mentioning 7 Tesla scanners",
             "sql": "SELECT {{COLS}} FROM bids_datasets d WHERE d.readme_text ILIKE '%7T%'"},
            {"question": "datasets mentioning 7 Tesla scanners in the name",
             "sql": "SELECT {{COLS}} FROM bids_datasets d WHERE d.name ILIKE '%7T%'"},
        ],
        "institution_query": [
            {"question": "datasets scanned at MIT with 7 Tesla scanners",
             "sql": "SELECT {{COLS}} FROM bids_datasets d WHERE EXISTS "
                    "(SELECT 1 FROM unnest(d.institutions) AS inst WHERE inst ILIKE '%MIT%')"},
        ],
    }
    seed = "datasets mentioning 7 Tesla scanners"

    picked = text_to_sql._pick_examples(pool, "description_search", n=5, seed=seed)
    assert [ex["sql"] for ex in picked] == [pool["description_search"][1]["sql"]]

    caps.set_available_columns({"readme_text"})
    picked = text_to_sql._pick_examples(pool, "description_search", n=5, seed=seed)
    assert len(picked) == 2
    assert all("institutions" not in ex["sql"] for ex in picked)


def test_real_few_shot_pool_never_yields_readme_or_institutions_when_hidden() -> None:
    pool = json.loads(FEW_SHOT_PATH.read_text())
    for family in ("description_search", "paper_reference_query", "institution_query"):
        picked = text_to_sql._pick_examples(
            pool, family, n=50, seed="datasets scanned at MIT that mention a paper in the readme"
        )
        assert picked, family
        for example in picked:
            assert "readme_text" not in example["sql"]
            assert "institutions" not in example["sql"]


def test_disease_fallback_grows_with_available_columns() -> None:
    assert caps.disease_text_columns() == ["name", "description_text"]
    caps.set_available_columns({"dx_status", "study_domain", "readme_text"})
    assert caps.disease_text_columns() == ["name", "description_text", "dx_status", "study_domain"]


def test_set_available_columns_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        caps.set_available_columns({"no_such_column"})


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def mappings(self):
        return self

    def one(self):
        return self._rows[0]


class _FakeConnection:
    """Answers the two detection queries from canned data."""

    def __init__(self, columns, counts):
        self._columns = columns
        self._counts = counts
        self.statements = []

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "information_schema.columns" in sql:
            return _FakeResult([(name,) for name in self._columns])
        return _FakeResult([self._counts])


def test_detection_requires_column_to_exist_and_hold_data() -> None:
    connection = _FakeConnection(
        columns=["id", "name", "description_text", "institutions", "readme_text", "dx_status"],
        # institutions exists but is empty; dx_status is below the coverage floor
        counts={"total": 1000, "institutions": 0, "readme_text": 850, "dx_status": 49},
    )
    detected = asyncio.run(caps.detect_available_columns(connection))
    assert detected == frozenset({"readme_text"})
    assert not caps.has_display_column("deprecated_reason")
    # Columns that are not in the table are never queried.
    assert "size_bytes" not in connection.statements[-1]


def test_detection_failure_hides_every_optional_column() -> None:
    class _BrokenEngine:
        def connect(self):
            raise RuntimeError("database unreachable")

    caps.set_available_columns({"readme_text"})
    assert asyncio.run(caps.refresh_available_columns(_BrokenEngine())) == frozenset()
    assert caps.available_columns() == frozenset()


def test_display_column_needs_only_to_exist() -> None:
    connection = _FakeConnection(
        columns=["id", "name", "deprecated_reason"],
        counts={"total": 1000},
    )
    asyncio.run(caps.detect_available_columns(connection))
    assert caps.has_display_column("deprecated_reason")


def test_size_limit_becomes_a_filter_only_when_the_column_is_available() -> None:
    from preprocess import QueryPlan, _build_augmented_question

    plan = QueryPlan(
        query_family="scan_filter",
        natural_language_summary="Find fMRI datasets.",
        max_size_gb=5,
        min_size_gb=0.5,
    )
    # Database without size_bytes: the question passes through unchanged.
    assert _build_augmented_question(plan, []) == "Find fMRI datasets."

    augmented = _build_augmented_question(plan, [], available_columns={"size_bytes"})
    assert "dataset size: d.size_bytes <= 5368709120 AND d.size_bytes >= 536870912" in augmented
    assert augmented.endswith("Find fMRI datasets.")

    no_limit = QueryPlan(query_family="scan_filter", natural_language_summary="Find fMRI datasets.")
    assert _build_augmented_question(no_limit, [], available_columns={"size_bytes"}) == "Find fMRI datasets."
