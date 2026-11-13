"""Broad vocabulary keys in generated SQL are expanded to the codes they stand for."""

from __future__ import annotations

import pytest

from constants import build_schema_ddl
from services.concept_keys import expand_concept_keys, group_codes
from services.sql_rewriter import SqlRewriteService


def test_category_key_becomes_its_codes() -> None:
    sql = expand_concept_keys("WHERE p.diagnosis = 'neurodevelopmental'")
    assert sql.startswith("WHERE p.diagnosis IN ('")
    assert "'adhd'" in sql and "'autism_spectrum_disorder'" in sql
    assert "neurodevelopmental" not in sql


def test_negation_and_in_lists() -> None:
    assert expand_concept_keys("WHERE o.datatype <> 'mri'").startswith("WHERE o.datatype NOT IN ('functional_mri'")
    sql = expand_concept_keys("WHERE o2.suffix IN ('eeg', 'mri_functional')")
    assert sql == "WHERE o2.suffix IN ('eeg', 'fmri_bold', 'bold_map', 'single_band_reference', 'bold_reference')"
    assert "NOT IN ('depression'" in expand_concept_keys("WHERE p.diagnosis NOT IN ('mood_disorders')")


def test_key_written_with_underscores_finds_the_group_label() -> None:
    assert "'n_back_working_memory'" in expand_concept_keys("WHERE o.task = 'working_memory'")


@pytest.mark.parametrize(
    "sql",
    [
        # A stored code stays as it is, even when it has narrower codes below it.
        "WHERE p.diagnosis = 'epilepsy'",
        "WHERE o2.task = 'resting_state'",
        "WHERE o2.task IN ('n_back_working_memory', 'one_back')",
        # Not a column filter.
        "WHERE o.other_entities->>'task' = 'memory'",
        "WHERE mc.field = 'datatype' AND mt.datatype = mc.code",
        "WHERE o.subtask = 'memory'",
        # Unknown value.
        "WHERE p.diagnosis = 'no_such_category'",
    ],
)
def test_everything_else_is_left_alone(sql: str) -> None:
    assert expand_concept_keys(sql) == sql


def test_rewriter_applies_the_expansion() -> None:
    raw = (
        "SELECT {{COLS}} FROM bids_datasets d "
        "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL "
        "WHERE EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id "
        "AND p.diagnosis = 'mood_disorders') GROUP BY d.id"
    )
    rewritten = SqlRewriteService().rewrite_generated_sql(raw, "Datasets with mood disorder patients")
    assert "mood_disorders" not in rewritten.sql
    assert "'major_depressive_disorder'" in rewritten.sql


def test_every_key_the_schema_prompt_names_can_be_expanded() -> None:
    # The prompt used to name 'epilepsy_spectrum' and 'psychiatric', which are
    # not in the vocabulary; a filter on them matched no row.
    ddl = build_schema_ddl()
    for field, keys in (("task", ("memory", "sleep", "language_speech", "working_memory")),
                        ("diagnosis", ("psychiatric_disorder", "neurodevelopmental", "neurodegenerative"))):
        for key in keys:
            assert f"'{key}'" in ddl, key
            assert group_codes(field, key), (field, key)
