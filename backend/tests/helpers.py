"""States of database_stats the tests set up."""

from __future__ import annotations

from collections.abc import Collection

from services.schema.tables import OPTIONAL_DATASET_COLUMNS
from services.database_stats.database_stats import ColumnCoverage


def full_coverage(columns: Collection[str]) -> ColumnCoverage:
    """Coverage in which exactly *columns* are filled in every dataset."""
    unknown = set(columns) - set(OPTIONAL_DATASET_COLUMNS)
    if unknown:
        raise ValueError(f"not optional dataset columns: {sorted(unknown)}")
    return ColumnCoverage(total=1, filled=dict.fromkeys(columns, 1))
