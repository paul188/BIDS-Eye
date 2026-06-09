from __future__ import annotations

import re
from typing import Optional

from value_mappings import FIELD_CONCEPT_EXPANSION

_ALIAS_RE = r"(?:p|d|bo\d*|o\d*)"
_EQ_PATTERN = re.compile(rf"(({_ALIAS_RE})\.(\w+)\s*=\s*)'([^']+)'", re.IGNORECASE)
_IN_PATTERN = re.compile(rf"({_ALIAS_RE})\.(\w+)\s+IN\s*\(([^)]+)\)", re.IGNORECASE)
_NOT_IN_PATTERN = re.compile(rf"({_ALIAS_RE})\.(\w+)\s+NOT\s+IN\s*\(([^)]+)\)", re.IGNORECASE)

_COLUMN_SECTION: dict[str, str] = {
    "diagnosis": "diagnosis",
    "task": "task",
    "suffix": "suffix",
    "datatype": "datatype",
}


def _get_expansion(column: str) -> Optional[dict[str, list[str]]]:
    section = _COLUMN_SECTION.get(column.lower())
    if section is None:
        return None
    return FIELD_CONCEPT_EXPANSION.get(section)


def _expand_value(value: str, expansion: dict[str, list[str]]) -> list[str]:
    children = expansion.get(value)
    return list(children) if children else [value]


def _parse_values(values_str: str) -> list[str]:
    return [tok.strip().strip("'\"") for tok in values_str.split(",") if tok.strip().strip("'\"")]


def _format_values(values: list[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def expand_sql(sql: str) -> str:
    """Expand broad concept keys in a SQL string into concrete standard_code lists."""

    def repl_eq(match: re.Match) -> str:
        prefix, alias, column, value = match.groups()
        expansion = _get_expansion(column)
        if not expansion:
            return match.group(0)
        expanded = _expand_value(value, expansion)
        if len(expanded) == 1 and expanded[0] == value:
            return match.group(0)
        return f"{prefix}IN ({_format_values(expanded)})"

    def repl_in(match: re.Match, negate: bool = False) -> str:
        alias, column, values_str = match.groups()
        expansion = _get_expansion(column)
        if not expansion:
            return match.group(0)
        values = _parse_values(values_str)
        expanded: list[str] = []
        seen: set[str] = set()
        for value in values:
            for item in _expand_value(value, expansion):
                if item not in seen:
                    seen.add(item)
                    expanded.append(item)
        if len(expanded) == len(values) and all(v == e for v, e in zip(values, expanded)):
            return match.group(0)
        op = "NOT IN" if negate else "IN"
        return f"{alias}.{column} {op} ({_format_values(expanded)})"

    sql = _EQ_PATTERN.sub(repl_eq, sql)
    sql = _IN_PATTERN.sub(lambda m: repl_in(m, negate=False), sql)
    sql = _NOT_IN_PATTERN.sub(lambda m: repl_in(m, negate=True), sql)
    return sql
