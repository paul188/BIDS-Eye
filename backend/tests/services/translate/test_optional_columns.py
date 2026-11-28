"""Which optional bids_datasets columns the search uses
(services/translate/optional_columns.py)."""

from __future__ import annotations

import logging

from services.database_stats.database_stats import ColumnCoverage
from services.translate import optional_columns
from services.translate.optional_columns import usable, usable_columns, uses_unusable_column


def test_a_column_is_usable_once_five_percent_of_the_datasets_fill_it() -> None:
    coverage = ColumnCoverage(total=1000, filled={"readme_text": 850, "dx_status": 49,
                                                  "institutions": 50})
    assert usable_columns(coverage) == {"readme_text", "institutions"}
    assert not usable(coverage, "dx_status")


def test_an_empty_catalogue_has_no_usable_column() -> None:
    assert usable_columns(ColumnCoverage()) == frozenset()
    assert usable_columns(ColumnCoverage(total=0, filled={"readme_text": 0})) == frozenset()


def test_examples_naming_an_unusable_column_are_recognised() -> None:
    coverage = ColumnCoverage(total=1, filled={"readme_text": 1})
    assert not uses_unusable_column(coverage, "SELECT 1 FROM bids_datasets d WHERE d.readme_text ILIKE '%x%'")
    assert uses_unusable_column(coverage, "SELECT 1 FROM unnest(d.institutions) AS inst")


def test_the_columns_are_logged_only_when_the_usable_ones_change(monkeypatch, caplog) -> None:
    monkeypatch.setattr(optional_columns, "_logged", None)
    caplog.set_level(logging.INFO, logger=optional_columns.__name__)
    optional_columns.log_changes(ColumnCoverage(total=1, filled={"readme_text": 1}))
    first = len(caplog.records)
    assert first > 0
    assert any("readme_text" in r.getMessage() and "shown" in r.getMessage() for r in caplog.records)
    optional_columns.log_changes(ColumnCoverage(total=2, filled={"readme_text": 2}))
    assert len(caplog.records) == first
    optional_columns.log_changes(ColumnCoverage(total=1, filled={"readme_text": 1, "institutions": 1}))
    assert len(caplog.records) == 2 * first
