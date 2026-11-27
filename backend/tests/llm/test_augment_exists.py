"""augment._exists: one EXISTS per table, built from join_paths.yaml."""

from __future__ import annotations

import pytest

from preprocess import RAGField
from preprocess.augment import _exists


def test_fields_of_one_table_share_one_exists() -> None:
    fields = [RAGField.DATATYPE, RAGField.SUFFIX, RAGField.TASK]
    assert _exists(fields, "o2.task = 'rest'") == (
        "EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id AND o2.task = 'rest')"
    )


def test_fields_of_different_tables_are_refused() -> None:
    # A diagnosis is a participant row, a task a file row: one EXISTS cannot hold both.
    with pytest.raises(ValueError, match="not stored in one table"):
        _exists([RAGField.DIAGNOSIS, RAGField.TASK], "true")
