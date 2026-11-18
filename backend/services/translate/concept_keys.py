"""
services/translate/concept_keys.py
----------------------------------
Broad vocabulary keys in generated SQL → the codes they stand for.

The schema shown to the SQL model says a filter may name a broad category,

    p.diagnosis = 'neurodevelopmental'

and that the category is expanded at runtime. No row stores such a key: the
database holds the specific codes ('adhd', 'autism_spectrum_disorder', ...),
so without expansion the filter matches nothing. This module does the
expansion, from the same vocabulary the resolver uses (RAG/value_mappings.yaml).

Only a value that is NOT itself a stored code is expanded. 'epilepsy' is both a
code and the parent of other codes; a filter on it is left alone, because the
resolver may have chosen that one code on purpose. So expansion can only turn
a filter that matches nothing into one that matches its category.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Tuple

log = logging.getLogger(__name__)

# Column name = vocabulary category, on bids_participants and bids_objects.
FIELDS = ("diagnosis", "task", "suffix", "datatype")
_COLUMN = r"((?:\w+\.)?(" + "|".join(FIELDS) + r"))"
_LITERAL = r"'[^']*'"

_EQUALS_RE = re.compile(r"\b" + _COLUMN + r"\s*(=|!=|<>)\s*'([^']*)'", re.IGNORECASE)
_IN_RE = re.compile(
    r"\b" + _COLUMN + r"\s+((?:NOT\s+)?IN)\s*\(\s*(" + _LITERAL + r"(?:\s*,\s*" + _LITERAL + r")*)\s*\)",
    re.IGNORECASE,
)
_LITERAL_RE = re.compile(r"'([^']*)'")


@lru_cache(maxsize=1)
def _vocabulary() -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, FrozenSet[str]]]:
    """(group term → member codes, stored codes) per field."""
    from yaml_to_llamaindex import group_db, leaf_db

    groups = {field: group_db.get(field, {}) for field in FIELDS}
    stored = {field: frozenset(leaf_db.get(field, {}).values()) for field in FIELDS}
    return groups, stored


def group_codes(field: str, value: str) -> Optional[List[str]]:
    """Member codes if *value* names a category of *field* and is not a code itself."""
    try:
        groups, stored = _vocabulary()
    except Exception as exc:  # the vocabulary only widens a filter
        log.warning("Vocabulary not available for concept keys (%s)", exc)
        return None
    key = value.strip().lower()
    if not key or key in stored[field]:
        return None
    # 'working_memory' is written as a key; the vocabulary lists "working memory".
    for form in (key, key.replace("_", " ")):
        codes = groups[field].get(form)
        if codes:
            return list(codes)
    return None


def _quoted(codes: List[str]) -> str:
    return ", ".join("'" + code.replace("'", "''") + "'" for code in codes)


def expand_concept_keys(sql: str) -> str:
    """Replace category keys in diagnosis / task / suffix / datatype filters.

        p.diagnosis = 'mood_disorders'      → p.diagnosis IN ('depression', ...)
        o.datatype <> 'mri'                 → o.datatype NOT IN ('functional_mri', ...)
        o.suffix IN ('eeg', 'mri_anatomical') → o.suffix IN ('eeg', 't1_weighted_mri', ...)
    """
    def equals(match: re.Match) -> str:
        column, field, operator, value = match.group(1), match.group(2).lower(), match.group(3), match.group(4)
        codes = group_codes(field, value)
        if not codes:
            return match.group(0)
        log.info("Expanded %s key '%s' to %d codes", field, value, len(codes))
        return f"{column} {'IN' if operator == '=' else 'NOT IN'} ({_quoted(codes)})"

    def in_list(match: re.Match) -> str:
        column, field, operator = match.group(1), match.group(2).lower(), match.group(3)
        values = _LITERAL_RE.findall(match.group(4))
        expanded: List[str] = []
        changed = False
        for value in values:
            codes = group_codes(field, value)
            changed = changed or bool(codes)
            for code in codes or [value]:
                if code not in expanded:
                    expanded.append(code)
        if not changed:
            return match.group(0)
        log.info("Expanded %s keys in an IN list to %d codes", field, len(expanded))
        return f"{column} {operator} ({_quoted(expanded)})"

    return _IN_RE.sub(in_list, _EQUALS_RE.sub(equals, sql))
