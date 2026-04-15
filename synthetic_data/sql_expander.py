"""
sql_expander.py
---------------
Post-processing step applied to LLM-generated SQL at inference time.

The LLM is trained to emit broad concept keys in column filters when a user
asks about a general category:

    p.diagnosis = 'epilepsy_spectrum'
    o.task IN ('resting_state')
    o.suffix = 'mri_functional'

This module expands each concept key to the full list of specific standard_codes
that exist in the database:

    p.diagnosis IN ('epilepsy', 'watanabe_syndrome', 'hhe_syndrome', ...)
    o.task IN ('resting_state', 'resting_state_eyes_open', 'resting_state_eyes_closed', ...)
    o.suffix IN ('fmri_bold', 'single_band_reference', 'bold_reference', ...)

Expansion is recursive: concept keys can nest (e.g. 'neurological' → 'epilepsy_spectrum'
→ leaf codes). All descendant leaf standard_codes are collected.

Supported DB columns / table aliases:
    p.diagnosis, p.sex, p.handedness
    o.task, o.suffix, o.datatype
    d.dataset_type, d.source_type (passed through unchanged — no concept expansion)

The FIELD_CONCEPT_EXPANSION table is loaded from value_mappings.py (which reads
value_mappings.yaml).  In environments without PyYAML (e.g. a minimal Modal
container), copy both value_mappings.py and value_mappings.yaml into the image,
or run --regen to embed a static snapshot directly in this file.

Regenerate embedded snapshot:
    python training_data_generation/sql_expander.py --regen

Usage:
    from training_data_generation.sql_expander import expand_sql
    sql = expand_sql(raw_sql_from_llm)
"""
from __future__ import annotations

import re
import sys
from typing import Optional


# ── Load expansion tables ──────────────────────────────────────────────────────
# Try live import from value_mappings first (stays in sync with YAML).
# Fall back to the embedded snapshot below for environments without the module.

try:
    from value_mappings import FIELD_CONCEPT_EXPANSION as _LIVE
    FIELD_CONCEPT_EXPANSION: dict[str, dict[str, list[str]]] = _LIVE
except ImportError:
    try:
        from training_data_generation.value_mappings import FIELD_CONCEPT_EXPANSION as _LIVE
        FIELD_CONCEPT_EXPANSION = _LIVE
    except ImportError:
        raise ImportError(
            "sql_expander requires value_mappings.py and value_mappings.yaml "
            "to be on sys.path. Copy both files to the same directory or ensure "
            "the training_data_generation package is importable."
        )


# ── Column → section name mapping ─────────────────────────────────────────────
# Maps DB column names to the section key in FIELD_CONCEPT_EXPANSION.
# Columns not listed here are passed through unchanged.
_COLUMN_SECTION: dict[str, str] = {
    "diagnosis": "diagnosis",
    "task":      "task",
    "suffix":    "suffix",
    "datatype":  "datatype",
}


# ── SQL rewriting ──────────────────────────────────────────────────────────────

# Matches:  <alias>.<column> = 'value'
# Groups:   (1) full prefix up to and including '='  (2) alias  (3) column  (4) value
# Aliases: p (bids_participants), o (bids_objects main join), d (bids_datasets),
#          bo / bo2 / o1 / o2 (correlated bids_objects aliases in EXISTS subqueries)
_ALIAS_RE = r"(?:p|d|bo\d*|o\d*)"
_EQ_PATTERN = re.compile(
    rf"""(({_ALIAS_RE})\.(\w+)\s*=\s*)'([^']+)'""",
    re.IGNORECASE,
)

# Matches:  <alias>.<column> IN ('v1', 'v2', ...)
# Groups:   (1) alias  (2) column  (3) comma-separated quoted values
_IN_PATTERN = re.compile(
    rf"""({_ALIAS_RE})\.(\w+)\s+IN\s*\(([^)]+)\)""",
    re.IGNORECASE,
)

# Matches:  <alias>.<column> NOT IN ('v1', 'v2', ...)
_NOT_IN_PATTERN = re.compile(
    rf"""({_ALIAS_RE})\.(\w+)\s+NOT\s+IN\s*\(([^)]+)\)""",
    re.IGNORECASE,
)


def _get_expansion(column: str) -> Optional[dict[str, list[str]]]:
    """Return the concept expansion dict for a column, or None if not expandable."""
    section = _COLUMN_SECTION.get(column.lower())
    if section is None:
        return None
    return FIELD_CONCEPT_EXPANSION.get(section)


def _expand_value(v: str, expansion: dict[str, list[str]]) -> list[str]:
    """Expand a single concept key to its leaf standard_codes, or return [v] if leaf/unknown."""
    children = expansion.get(v)
    return list(children) if children else [v]


def _expand_values(values_str: str, expansion: dict[str, list[str]]) -> list[str]:
    """
    Parse a comma-separated SQL value list like "'resting_state', 'nback'"
    and expand any concept keys, deduplicating while preserving order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in values_str.split(","):
        v = raw.strip().strip("'\"")
        for expanded in _expand_value(v, expansion):
            if expanded not in seen:
                seen.add(expanded)
                result.append(expanded)
    return result


def _values_to_sql(vals: list[str]) -> str:
    return ", ".join(f"'{v}'" for v in vals)


def expand_sql(sql: str) -> str:
    """
    Expand any concept keys in column filters to their full standard_code lists.

    Handles:
      alias.column = 'concept_key'            → column IN ('leaf1', 'leaf2', ...)
      alias.column IN ('concept_key', ...)    → column IN ('leaf1', 'leaf2', ...)
      alias.column NOT IN ('concept_key', ...) → column NOT IN ('leaf1', 'leaf2', ...)

    Columns not in the expansion map (e.g. sex, handedness, extension) are untouched.
    Values that are already specific leaf standard_codes are passed through unchanged.
    """
    def replace_eq(m: re.Match) -> str:
        # Group layout after adding alias capture:
        # (1) full prefix "alias.column ="  (2) alias  (3) column  (4) value
        prefix = m.group(1)        # e.g. "p.diagnosis ="
        alias  = m.group(2)        # e.g. "p" or "bo"
        column = m.group(3)
        v      = m.group(4)
        expansion = _get_expansion(column)
        if expansion is None:
            return m.group(0)
        expanded = _expand_value(v, expansion)
        if len(expanded) == 1 and expanded[0] == v:
            return m.group(0)  # nothing to expand — keep equality form
        return f"{alias}.{column} IN ({_values_to_sql(expanded)})"

    def replace_in(m: re.Match) -> str:
        alias = m.group(1)
        column = m.group(2)
        values_str = m.group(3)
        expansion = _get_expansion(column)
        if expansion is None:
            return m.group(0)
        expanded = _expand_values(values_str, expansion)
        return f"{alias}.{column} IN ({_values_to_sql(expanded)})"

    def replace_not_in(m: re.Match) -> str:
        alias = m.group(1)
        column = m.group(2)
        values_str = m.group(3)
        expansion = _get_expansion(column)
        if expansion is None:
            return m.group(0)
        expanded = _expand_values(values_str, expansion)
        return f"{alias}.{column} NOT IN ({_values_to_sql(expanded)})"

    # Apply NOT IN before IN to avoid partial matches
    sql = _NOT_IN_PATTERN.sub(replace_not_in, sql)
    sql = _IN_PATTERN.sub(replace_in, sql)
    sql = _EQ_PATTERN.sub(replace_eq, sql)
    return sql


if __name__ == "__main__":
    # Smoke-test
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    tests = [
        # Diagnosis concept key (= form)
        "WHERE p.diagnosis = 'epilepsy_spectrum'",
        # Task concept key (IN form)
        "WHERE o.task IN ('resting_state')",
        # Suffix concept key
        "WHERE o.suffix = 'mri_anatomical'",
        # Leaf value — should be unchanged
        "WHERE p.diagnosis = 'epilepsy'",
        # NOT IN form
        "WHERE o.task NOT IN ('resting_state')",
        # Multiple concepts in IN list
        "WHERE p.diagnosis IN ('epilepsy_spectrum', 'psychiatric')",
    ]
    for sql in tests:
        print("IN :", sql)
        print("OUT:", expand_sql(sql))
        print()
