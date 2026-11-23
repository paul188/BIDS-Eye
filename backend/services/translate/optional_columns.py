"""
services/translate/optional_columns.py
--------------------------------------
Which optional bids_datasets columns the search uses.

The catalogue counts how many datasets fill each optional column
(services/catalogue/catalogue.py, ColumnCoverage). The rule here turns the
counts into a decision: a column filled in at least MIN_COVERAGE of the
datasets is usable. A filter on a column that is nearly empty would discard
almost every dataset.

The decision is applied in four places:

  the SQL model's schema    constants.build_schema_ddl renders only the usable
                            columns (text_to_sql.py)
  the few-shot examples     an example whose SQL names another optional column
                            is not shown (text_to_sql._pick_examples)
  the compiled conditions   preprocessing writes the size, species, study
                            structure and institution conditions only for
                            usable columns (available_columns in
                            LLM_preprocessor/preprocess/augment.py)
  the reranker              a dataset is described with the usable columns
                            only (services/results/rerank.py)
"""

from __future__ import annotations

import logging
import math
import re

from constants import OPTIONAL_DATASET_COLUMNS
from services.catalogue.catalogue import ColumnCoverage

log = logging.getLogger(__name__)

# A column counts as usable when at least this share of datasets fills it.
MIN_COVERAGE = 0.05

# The usable columns last written to the log, None before the first time.
_logged: frozenset[str] | None = None


def _needed(coverage: ColumnCoverage) -> int:
    """How many datasets must fill a column for it to be usable."""
    return max(1, math.ceil(MIN_COVERAGE * coverage.total))


def usable(coverage: ColumnCoverage, column: str) -> bool:
    """True when at least MIN_COVERAGE of the datasets fill *column*."""
    return coverage.filled.get(column, 0) >= _needed(coverage)


def usable_columns(coverage: ColumnCoverage) -> frozenset[str]:
    """The optional columns that are usable."""
    return frozenset(column for column in OPTIONAL_DATASET_COLUMNS if usable(coverage, column))


def uses_unusable_column(coverage: ColumnCoverage, sql: str) -> bool:
    """True if *sql* names an optional column that is not usable."""
    unusable = set(OPTIONAL_DATASET_COLUMNS) - usable_columns(coverage)
    if not unusable:
        return False
    pattern = r"\b(?:" + "|".join(sorted(map(re.escape, unusable))) + r")\b"
    return re.search(pattern, sql) is not None


def log_changes(coverage: ColumnCoverage) -> None:
    """Log every optional column and whether it is used, when the usable ones changed.

    Called after the catalogue was read: at startup and after each refresh.
    """
    global _logged
    columns = usable_columns(coverage)
    if columns == _logged:
        return
    _logged = columns
    for name in OPTIONAL_DATASET_COLUMNS:
        log.info("Optional dataset column %-18s %5d / %d datasets → %s", name,
                 coverage.filled.get(name, 0), coverage.total,
                 "shown to the SQL model" if name in columns else "hidden (too sparse)")
