"""
services/catalogue/catalogue_values.py
--------------------------------------
Every datatype, suffix and task value in generated SQL, checked against the
catalogue before the query runs.

A value that occurs in no dataset makes its condition match nothing. The SQL
model wrote suffix = 'bold' (stored as 'fmri_bold') and the request found 0
datasets (gold request qp05, 2026-11-14). In the recorded benchmark runs, 63
of 7,878 searches held such a value: invented task codes ('social_cognition',
'cognitive_tasks'), raw BIDS suffixes ('bold', 't1w'), a suffix written as a
task ('recognition').

The known values come from bids_dataset_facets (one row per distinct dataset,
datatype, suffix and task; crawlers/migrate_add_dataset_facets.sql): three
DISTINCT reads of about 21,000 rows. Without that table the check is off.

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
import time
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover - sqlglot is a backend dependency
    sqlglot = None
    exp = None

log = logging.getLogger(__name__)

FIELDS = ("datatype", "suffix", "task")
FACETS_TABLE = "bids_dataset_facets"
ENV = "CATALOGUE_VALUE_CHECK"
# The facets are refreshed after each crawl (post_crawl.py); a new task code
# must not be dropped for long because this copy is older.
REFRESH_SECONDS = 600

_known: Dict[str, FrozenSet[str]] = {}
_loaded_at: float = 0.0


# ── Loading the known values ──────────────────────────────────────────────────
# At startup (schema_capabilities) and, at most every REFRESH_SECONDS, before
# a search (refresh_if_stale).

def set_known(values: Mapping[str, Iterable[str]]) -> None:
    """Replace the known values (load(), tests). Empty: the check is off."""
    global _known, _loaded_at
    _known = {field: frozenset(values.get(field, ())) for field in FIELDS} if values else {}
    _loaded_at = time.monotonic()


_LOAD_SQL = " UNION ALL ".join(
    f"SELECT '{field}' AS field, {field} AS value FROM {FACETS_TABLE} "
    f"WHERE {field} IS NOT NULL GROUP BY {field}"
    for field in FIELDS
)


async def load(connection: Any) -> None:
    """Read the known values from the facets table (AsyncConnection or AsyncSession)."""
    from sqlalchemy import text

    values: Dict[str, set] = {field: set() for field in FIELDS}
    for field, value in (await connection.execute(text(_LOAD_SQL))).all():
        values[field].add(value)
    set_known(values)
    log.info("Catalogue values: %s", ", ".join(f"{len(v)} {f}" for f, v in values.items()))


async def refresh_if_stale(session: Any) -> None:
    """Reload the known values when they are older than REFRESH_SECONDS.

    Only while the facets table is in use; a failed reload keeps the old values.
    """
    from services.catalogue import schema_capabilities

    if not schema_capabilities.has_dataset_facets():
        return
    if _known and time.monotonic() - _loaded_at < REFRESH_SECONDS:
        return
    try:
        await load(session)
    except Exception as exc:  # the check only removes errors; never fail a search
        log.warning("Could not reload catalogue values (%s)", exc)


# ── Looking a value up ────────────────────────────────────────────────────────

def is_known(field: str, value: str) -> bool:
    return value in _known.get(field, frozenset())


# ── Corrections ───────────────────────────────────────────────────────────────

def _vocabulary():
    from yaml_to_llamaindex import group_db, leaf_db  # RAG/, on the backend's path
    return group_db, leaf_db


def correction(field: str, value: str) -> List[str]:
    """Known codes the vocabulary gives for an unknown *value*; [] if none."""
    known = _known.get(field, frozenset())
    forms = list(dict.fromkeys(f for f in (
        value, value.strip(), value.strip().lower(),
        value.strip().lower().replace("-", "_").replace(" ", "_"),
        value.strip().lower().replace("_", " ").replace("-", " "),
    ) if f))
    for form in forms:                      # spelling only: 'Fmri_Bold'
        if form in known:
            return [form]
    try:
        group_db, leaf_db = _vocabulary()
    except Exception as exc:  # without the vocabulary an unknown value is only dropped
        log.warning("Vocabulary not available for value corrections (%s)", exc)
        return []

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
    return bool(_known) and os.getenv(ENV, "true").strip().lower() not in ("false", "0", "no", "off")


def _field_of(node: Any) -> Optional[str]:
    if isinstance(node, exp.Column) and node.name.lower() in FIELDS:
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


def _quoted(value: str) -> str:
    return "'" + value + "'"


def _shown(codes: List[str]) -> str:
    return ", ".join(map(_quoted, codes[:5])) + (f" (+{len(codes) - 5})" if len(codes) > 5 else "")


def _like_matches(field: str, pattern: str, case_insensitive: bool) -> bool:
    regex = "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern)
    flags = re.IGNORECASE if case_insensitive else 0
    return any(re.fullmatch(regex, value, flags) for value in _known.get(field, ()))


def check_sql(sql: str) -> Tuple[str, List[str]]:
    """*sql* with unknown datatype / suffix / task values corrected or dropped.

    Returns the SQL (unchanged, same string, when every value is known or the
    check is off) and one note per changed value.
    """
    if not enabled() or sqlglot is None or not re.search(r"\b(datatype|suffix|task)\b", sql, re.I):
        return sql, []
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
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
            literals = node.expressions
            values = [_string(v) for v in literals]
            if not literals or any(v is None for v in values) or node.args.get("query"):
                continue
            unknown = [v for v in values if not is_known(field, v)]
            if not unknown:
                continue
            fixes = {v: correction(field, v) for v in unknown}
            kept: List[str] = []
            for value in values:
                kept.extend(c for c in (fixes.get(value, [value])) if c not in kept)
            if kept:
                node.set("expressions", [exp.Literal.string(v) for v in kept])
                notes.extend(f"{field} '{v}' occurs in no dataset: "
                             + (f"replaced by {_shown(fixes[v])}" if fixes[v] else "left out of the list")
                             for v in unknown)
            else:
                notes.append(f"{field} {', '.join(map(_quoted, unknown))} occur in no dataset: "
                             + _dropped_note(_remove(node)))
            continue

        if isinstance(node, (exp.Like, exp.ILike)):
            pattern = _string(right)
            if pattern is None or _like_matches(field, pattern, isinstance(node, exp.ILike)):
                continue
            notes.append(f"{field} pattern '{pattern}' matches no dataset: {_dropped_note(_remove(node))}")
            continue

        value = _string(right)
        if value is None or is_known(field, value):
            continue
        fixed = correction(field, value)
        if fixed:
            column = left.copy()
            if len(fixed) == 1:
                replacement = (exp.EQ if isinstance(node, exp.EQ) else exp.NEQ)(
                    this=column, expression=exp.Literal.string(fixed[0]))
            else:
                replacement = exp.In(this=column, expressions=[exp.Literal.string(v) for v in fixed])
                if isinstance(node, exp.NEQ):
                    replacement = exp.Not(this=replacement)
            node.replace(replacement)
            notes.append(f"{field} '{value}' occurs in no dataset: replaced by {_shown(fixed)}")
        elif isinstance(node, exp.EQ):
            notes.append(f"{field} '{value}' occurs in no dataset: {_dropped_note(_remove(node))}")
        else:
            notes.append(f"{field} '{value}' occurs in no dataset: the exclusion excludes nothing")

    if not notes:
        return sql, []
    for note in notes:
        log.warning("Catalogue check: %s", note)
    return tree.sql(dialect="postgres"), notes


def _dropped_note(outcome: str) -> str:
    return {
        "exact": "left out (one of several alternatives)",
        "dropped": "condition dropped",
        "": "kept (under NOT it excludes nothing)",
    }[outcome]
