"""join_registry.field_column: where a field is stored, from join_paths.yaml."""

from __future__ import annotations

import pytest

import join_registry
from join_registry import FieldColumn


def test_a_field_in_another_table_is_named_by_its_exists_alias() -> None:
    assert join_registry.field_column("task") == FieldColumn("bids_objects", "o2", "task")
    assert join_registry.field_column("task").ref == "o2.task"


def test_a_field_on_the_datasets_table_is_named_by_the_outer_alias() -> None:
    stored = join_registry.field_column("institution")
    assert stored == FieldColumn("bids_datasets", "d", "institutions")
    assert stored.ref == "d.institutions"


@pytest.mark.parametrize("field", ["no_such_field", "scan", ""])
def test_a_field_without_an_entry_raises_instead_of_guessing_a_column(field: str) -> None:
    # 'scan' is a RAGField but no column: datatype, suffix and task searched together.
    with pytest.raises(KeyError, match="join_paths.yaml"):
        join_registry.field_column(field)
