"""
services/translate/value_check.py
---------------------------------
Every datatype, suffix and task value in generated SQL, checked against the
values the catalogue holds before the query runs (from sql_rewriter).

A value that occurs in no dataset makes its condition match nothing. The SQL
model wrote suffix = 'bold' (stored as 'fmri_bold') and the request found 0
datasets (gold request qp05, 2026-11-14). In the recorded benchmark runs, 63
of 7,878 searches held such a value: invented task codes ('social_cognition',
'cognitive_tasks'), raw BIDS suffixes ('bold', 't1w'), a suffix written as a
task ('recognition').

The known values are those of services/catalogue/catalogue_values.py; while
none are loaded the check is off.

An unknown value is
  corrected   to the stored code the vocabulary gives for it
              (RAG/value_mappings.yaml): a datatype or suffix by its synonym
              ('bold' → 'fmri_bold'), a task by its category
              ('recognition' → the recognition tasks), else by its synonym;
              only codes that occur in some dataset;
  dropped     otherwise. In an OR or an IN list it is left out, which changes
              nothing. As a required condition (AND) the condition goes, so
              the rest of the query decides. Under a NOT it stays: "without
              X" holds for every dataset when no dataset has X.
LIKE / ILIKE patterns that match no known value are dropped the same way.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from services.catalogue import catalogue_values

try:
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp
except ImportError:  # pragma: no cover - sqlglot is a backend dependency
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

ENV = "CATALOGUE_VALUE_CHECK"


# ── Corrections ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _vocabulary() -> Optional[Tuple[dict, dict]]:
    """(group_db, leaf_db) of the curated vocabulary (RAG/, on the backend's
    path), or None when it cannot be loaded. The failure is cached: a failed
    import is not kept in sys.modules, so every correction() would otherwise
    import and parse value_mappings.yaml again, and warn again."""
    import yaml
    try:
        from yaml_to_llamaindex import group_db, leaf_db
    except (ImportError, OSError, yaml.YAMLError) as exc:
        log.warning("Vocabulary not available for value corrections (%s)", exc)
        return None
    return group_db, leaf_db


def correction(field: str, value: str) -> List[str]:
    """Known codes the vocabulary gives for an unknown *value*; [] if none."""
    known = catalogue_values.known(field)
    forms = list(dict.fromkeys(f for f in (
        value, value.strip(), value.strip().lower(),
        value.strip().lower().replace("-", "_").replace(" ", "_"),
        value.strip().lower().replace("_", " ").replace("-", " "),
    ) if f))
    for form in forms:                      # spelling only: 'Fmri_Bold'
        if form in known:
            return [form]
    vocabulary = _vocabulary()
    if vocabulary is None:  # without the vocabulary an unknown value is only dropped
        return []
    group_db, leaf_db = vocabulary

    def synonym() -> List[str]:
        for form in forms:
            code = leaf_db.get(field, {}).get(form)
            if code and code in known:
                return [code]
        return []

    def category() -> List[str]:
        for form in forms:
            members = [c for c in group_db.get(field, {}).get(form) or [] if c in known]
            if members:
                return list(dict.fromkeys(members))
        return []

    # A task name is often a category the model shortened ('recognition'); its
    # one synonym can be a narrow task ('emotional_recognition_ibc').
    return (category() or synonym()) if field == "task" else (synonym() or category())


# ── The check ─────────────────────────────────────────────────────────────────
# On every translation, from sql_rewriter.

def enabled() -> bool:
    """True when known values are loaded and CATALOGUE_VALUE_CHECK is not 'false'.

    Without known values every value would look unknown and be dropped, so
    "nothing loaded" means "do not check".
    """
    return catalogue_values.loaded() and os.getenv(ENV, "true").strip().lower() not in ("false", "0", "no", "off")


def _field_of(node: Any) -> Optional[str]:
    if isinstance(node, exp.Column) and node.name.lower() in catalogue_values.FIELDS:
        return node.name.lower()
    return None


def _string(node: Any) -> Optional[str]:
    return node.this if isinstance(node, exp.Literal) and node.is_string else None


def _negations(node: Any) -> int:
    count, parent = 0, node.parent
    while parent is not None:
        if isinstance(parent, exp.Not):
            count += 1
        parent = parent.parent
    return count


def _remove(atom: Any) -> str:
    """Take a condition that matches nothing out of the query.

    Returns 'exact' (an alternative of an OR: nothing changes), 'dropped' (a
    required condition: the query no longer asks for it) or '' (left as it
    is: under a NOT, or not part of a WHERE).
    """
    target = atom
    while isinstance(target.parent, exp.Paren):
        target = target.parent
    parent = target.parent
    if isinstance(parent, (exp.Or, exp.And)):
        if isinstance(parent, exp.And) and _negations(target) % 2:
            return ""
        other = parent.expression if parent.this is target else parent.this
        parent.replace(other)
        return "exact" if isinstance(parent, exp.Or) else "dropped"
    if isinstance(parent, exp.Where) and not _negations(target) % 2:
        parent.pop()
        return "dropped"
    return ""


def _dropped_note(outcome: str) -> str:
    """The note for what _remove() did."""
    return {
        "exact": "left out (one of several alternatives)",
        "dropped": "condition dropped",
        "": "kept (under NOT it excludes nothing)",
    }[outcome]


def _quoted(value: str) -> str:
    return "'" + value + "'"


def _shown(codes: List[str]) -> str:
    return ", ".join(map(_quoted, codes[:5])) + (f" (+{len(codes) - 5})" if len(codes) > 5 else "")


def _like_matches(field: str, pattern: str, case_insensitive: bool) -> bool:
    regex = "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern)
    flags = re.IGNORECASE if case_insensitive else 0
    return any(re.fullmatch(regex, value, flags) for value in catalogue_values.known(field))


# One function per kind of condition. Each changes the tree in place and
# returns one note per value it corrected or dropped ([] when all are known).

def _check_in_list(node: Any, field: str) -> List[str]:
    """field IN ('a', 'b'): replace unknown values, leave out the rest; an
    empty list drops the condition."""
    literals = node.expressions
    values = [s for v in literals if (s := _string(v)) is not None]
    if not literals or len(values) < len(literals) or node.args.get("query"):
        return []
    unknown = [v for v in values if not catalogue_values.is_known(field, v)]
    if not unknown:
        return []
    fixes = {v: correction(field, v) for v in unknown}
    kept: List[str] = []
    for value in values:
        kept.extend(c for c in (fixes.get(value, [value])) if c not in kept)
    if not kept:
        return [f"{field} {', '.join(map(_quoted, unknown))} occur in no dataset: "
                + _dropped_note(_remove(node))]
    node.set("expressions", [exp.Literal.string(v) for v in kept])
    return [f"{field} '{v}' occurs in no dataset: "
            + (f"replaced by {_shown(fixes[v])}" if fixes[v] else "left out of the list")
            for v in unknown]


def _check_pattern(node: Any, field: str, pattern_node: Any) -> List[str]:
    """field LIKE / ILIKE 'pattern': dropped when it matches no known value."""
    pattern = _string(pattern_node)
    if pattern is None or _like_matches(field, pattern, isinstance(node, exp.ILike)):
        return []
    return [f"{field} pattern '{pattern}' matches no dataset: {_dropped_note(_remove(node))}"]


def _check_comparison(node: Any, field: str, column: Any, value_node: Any) -> List[str]:
    """field = 'x' / field != 'x': replace an unknown value; else drop an =
    (it matches nothing) and keep a != (it excludes nothing)."""
    value = _string(value_node)
    if value is None or catalogue_values.is_known(field, value):
        return []
    fixed = correction(field, value)
    if fixed:
        replacement: exp.Expression
        if len(fixed) == 1:
            replacement = (exp.EQ if isinstance(node, exp.EQ) else exp.NEQ)(
                this=column.copy(), expression=exp.Literal.string(fixed[0]))
        else:
            replacement = exp.In(this=column.copy(), expressions=[exp.Literal.string(v) for v in fixed])
            if isinstance(node, exp.NEQ):
                replacement = exp.Not(this=replacement)
        node.replace(replacement)
        return [f"{field} '{value}' occurs in no dataset: replaced by {_shown(fixed)}"]
    if isinstance(node, exp.EQ):
        return [f"{field} '{value}' occurs in no dataset: {_dropped_note(_remove(node))}"]
    return [f"{field} '{value}' occurs in no dataset: the exclusion excludes nothing"]


def check_sql(sql: str) -> Tuple[str, List[str]]:
    """*sql* with unknown datatype / suffix / task values corrected or dropped.

    Returns the SQL (unchanged, same string, when every value is known or the
    check is off) and one note per changed value.
    """
    if not enabled() or sqlglot is None or not re.search(r"\b(datatype|suffix|task)\b", sql, re.I):
        return sql, []
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:  # query_guard refuses it later
        return sql, []

    notes: List[str] = []
    for node in list(tree.find_all(exp.EQ, exp.NEQ, exp.In, exp.Like, exp.ILike)):
        if node.root() is not tree:
            continue  # taken out with an earlier condition
        left, right = node.this, node.args.get("expression")
        if isinstance(node, (exp.EQ, exp.NEQ)) and _field_of(right) and _string(left) is not None:
            left, right = right, left              # 'bold' = o.suffix
        field = _field_of(left)
        if field is None:
            continue
        if isinstance(node, exp.In):
            notes.extend(_check_in_list(node, field))
        elif isinstance(node, (exp.Like, exp.ILike)):
            notes.extend(_check_pattern(node, field, right))
        else:
            notes.extend(_check_comparison(node, field, left, right))

    if not notes:
        return sql, []
    for note in notes:
        log.warning("Catalogue check: %s", note)
    return tree.sql(dialect="postgres"), notes
