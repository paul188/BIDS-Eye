"""What the SQL model is shown: the schema with the usable optional columns,
and few-shot examples that use only those (schema_ddl.build_schema_ddl,
examples.pick_examples; usable columns from services/translate/optional_columns.py)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from services.database_stats.database_stats import ColumnCoverage, database_stats
from services.translate import text_to_sql
from services.translate.prompt import examples, instructions, schema_ddl

FEW_SHOT_PATH = Path(schema_ddl.__file__).resolve().parent / "few_shot_examples.json"


def _dataset_table(ddl: str) -> str:
    return ddl[: ddl.index("CREATE TABLE bids_objects")]


def test_schema_hides_optional_columns_by_default() -> None:
    table = _dataset_table(schema_ddl.build_schema_ddl(()))
    for column in schema_ddl.OPTIONAL_DATASET_COLUMNS:
        assert column not in table
    assert "description_text  TEXT " in table
    # The last column must not end with a comma.
    assert re.search(r"description_text\s+TEXT\s+--", table)


def test_schema_shows_only_available_optional_columns() -> None:
    table = _dataset_table(schema_ddl.build_schema_ddl({"readme_text", "size_bytes"}))
    assert "readme_text       TEXT," in table
    assert re.search(r"size_bytes\s+BIGINT\s+--", table)
    assert "institutions" not in table
    assert "dx_status" not in table


def test_few_shot_examples_only_use_known_dataset_columns() -> None:
    """Guards against examples drifting away from the schema the model sees."""
    known = {name for name, _, _ in schema_ddl._CORE_DATASET_COLUMNS}
    known |= set(schema_ddl.OPTIONAL_DATASET_COLUMNS)
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

    picked = examples.pick_examples(pool, ["description_search"], n=5, seed=seed)
    assert [ex["sql"] for ex in picked] == [pool["description_search"][1]["sql"]]

    database_stats.coverage = ColumnCoverage.full({"readme_text"})
    picked = examples.pick_examples(pool, ["description_search"], n=5, seed=seed)
    assert len(picked) == 2
    assert all("institutions" not in ex["sql"] for ex in picked)


def test_real_few_shot_pool_never_yields_readme_or_institutions_when_hidden() -> None:
    pool = json.loads(FEW_SHOT_PATH.read_text())
    for family in ("description_search", "paper_reference_query", "institution_query"):
        picked = examples.pick_examples(
            pool, [family], n=50, seed="datasets scanned at MIT that mention a paper in the readme"
        )
        assert picked, family
        for example in picked:
            assert "readme_text" not in example["sql"]
            assert "institutions" not in example["sql"]


def test_example_tokens_normalize_irregular_clinical_plurals() -> None:
    for singular, plural in (
        ("metastasis", "metastases"),
        ("diagnosis", "diagnoses"),
        ("analysis", "analyses"),
        ("psychosis", "psychoses"),
        ("child", "children"),
    ):
        assert examples._example_tokens(singular) == examples._example_tokens(plural)


def test_empty_query_metadata_does_not_invent_intent_categories() -> None:
    empty_codes = {field: set() for field in ("datatype", "task", "diagnosis", "suffix")}
    intents = examples._intent_features(
        "datasets mentioning 7 Tesla scanners", "description_search", empty_codes
    )
    assert {"acquisition", "field_strength"} <= intents
    assert "clinical" not in intents
    assert "task" not in intents
    assert "modality" not in intents


def test_full_sql_prompt_budget_drops_only_whole_lower_ranked_examples(monkeypatch) -> None:
    monkeypatch.setattr(schema_ddl, "build_schema_ddl", lambda _columns: "schema")
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["system"] = kwargs["system"]
        return ("SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o "
                "ON o.dataset_id = d.id AND o.subject IS NOT NULL GROUP BY d.id")

    monkeypatch.setattr(text_to_sql, "llm_generate", fake_llm)
    examples = [
        {"question": f"query-{i}-" + ("x" * 20_000),
         "sql": "SELECT {{COLS}} FROM bids_datasets d GROUP BY d.id"}
        for i in range(3)
    ]
    text_to_sql._generate_sql("current request", examples)
    prompt = captured["prompt"]
    assert len(prompt) + len(captured["system"]) <= instructions.SQL_PROMPT_CHAR_BUDGET
    assert prompt.count("[QUESTION]") == prompt.count("[/QUESTION]")
    assert prompt.count("[SQL]") == prompt.count("[/SQL]")
    assert prompt.index("<current_request>") > prompt.index("<examples")
    assert "current request" in prompt


def test_size_limit_becomes_a_filter_only_when_the_column_is_available() -> None:
    from preprocess import QueryPlan, _build_augmented_question

    plan = QueryPlan(
        natural_language_summary="Find fMRI datasets.",
        max_size_gb=5,
        min_size_gb=0.5,
    )
    # Database without size_bytes: the question passes through unchanged.
    assert _build_augmented_question(plan, []) == "Find fMRI datasets."

    augmented = _build_augmented_question(plan, [], available_columns={"size_bytes"})
    assert "dataset size: d.size_bytes <= 5368709120 AND d.size_bytes >= 536870912" in augmented
    assert augmented.endswith("Find fMRI datasets.")

    no_limit = QueryPlan(natural_language_summary="Find fMRI datasets.")
    assert _build_augmented_question(no_limit, [], available_columns={"size_bytes"}) == "Find fMRI datasets."
