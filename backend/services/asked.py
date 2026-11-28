"""
services/asked.py
-----------------
What the conditions of a generated query ask for, read from the SQL that
runs: the age range, the sex, the counts, the species, the searched
phrases. Facts only, no wording: the chips (services/results/display/chips.py),
the highlights and the uncertain tier say them in their own words.

Built on the general readers: services/sql_tree.py (compared values and
bounds, JSON keys, ORDER BY and LIMIT) and services/fulltext/ (text
searches, Clause.queries and Clause.guard_forms, tsqueries).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from sqlglot import exp

import cohort_count
from participant_codes import sex_code  # backend/vocabulary is on the path wherever the backend runs
from services import sql_tree
from services.fulltext.text_search import Clause
from services.fulltext.tsquery import is_near, query_phrases
from services.schema import tables as _tables
from services.sql_tree import Bounds

# A participant's age, and the dataset's range of them.
AGE_COLUMNS = ("age", "age_min", "age_max")
# The stored codes that name one sex.
ONE_SEX = ("female", "male")
FIELD_STRENGTH_KEY = "MagneticFieldStrength"
# The strengths scanners are sold at; a range asks for the ones inside it.
NOMINAL_TESLA = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.7, 7.0, 9.4, 10.5, 11.7, 14.0)
# What a count can count, in the order a chip prefers them.
COUNTED_COLUMNS = ("subject", "participant_id", "session", "run")
PLACE_COLUMNS = ("city", "region", "country")
# Older plans search the description with ILIKE instead of full text.
_ILIKE_TEXT_COLUMNS = ("description_text", "readme_text", "readme")
# Join keys and row ids say nothing about what a condition asks.
KEY_COLUMNS = frozenset({"id", "dataset_id", "participant_id", "subject", "paper_id", "name_raw"})


# ── Shape ─────────────────────────────────────────────────────────────────────

def without_parentheses(node: Any) -> Any:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def unwrapped(node: Any) -> Tuple[Any, bool]:
    """The condition inside its parentheses and NOTs, and whether it is negated."""
    negated = False
    while isinstance(node, (exp.Paren, exp.Not)):
        if isinstance(node, exp.Not):
            negated = not negated
        node = node.this
    return node, negated


def dataset_alias(tree: Any) -> Optional[str]:
    """The alias of bids_datasets in FROM, None when the query reads another table."""
    source = tree.args.get("from") or tree.args.get("from_")
    table = getattr(source, "this", None)
    if isinstance(table, exp.Table) and table.name == _tables.datasets.name:
        return table.alias_or_name
    return None


# ── Text searches ─────────────────────────────────────────────────────────────

def searched_phrases(clause: Clause) -> List[str]:
    """The phrases the condition searches for. Words inside a NOT are what it
    excludes (excluded_phrases), not what it searches for."""
    phrases = [phrase for query in clause.queries(negated=False) for phrase in query_phrases(query)]
    phrases.extend(sql_tree.compared_values(clause.node, list(_ILIKE_TEXT_COLUMNS)))
    return list(dict.fromkeys(phrases))


def excluded_phrases(clause: Clause) -> List[str]:
    phrases = [phrase for query in clause.queries(negated=True) for phrase in query_phrases(query)]
    return list(dict.fromkeys(phrases))


def is_topic(clause: Clause) -> bool:
    """A topic condition: a 'near' search, or a search that excludes other meanings."""
    if any(is_near(q) for q in clause.queries(negated=False)):
        return True
    return bool(clause.queries(negated=True)) and bool(clause.queries(negated=False))


def excluded_places(clause: Clause) -> List[str]:
    """The place names a location's second condition leaves out, [] for any
    other condition. The shape is (a site in the place OR NOT text naming
    another place), written by LLM_preprocessor/preprocess/augment.py
    _location_lines for a name that is a word of another place's name:
    "Mexico" must not be found in "University of New Mexico"."""
    node = without_parentheses(clause.node)
    clause = clause.part(node)
    if not isinstance(node, exp.Or) or clause.queries(negated=False):
        return []
    if not sql_tree.compared_values(node, list(PLACE_COLUMNS)):
        return []
    return excluded_phrases(clause)


# ── Participants ──────────────────────────────────────────────────────────────

def age_range(node: Any) -> Tuple[float, float]:
    """The age range an age condition asks for, 0–120 when open."""
    bounds = sql_tree.compared_bounds(node, lambda side: sql_tree.compares(side, AGE_COLUMNS))
    if bounds is None:
        return 0.0, 120.0
    return (max(0.0, bounds.low) if bounds.low is not None else 0.0,
            min(120.0, bounds.high) if bounds.high is not None else 120.0)


def age_bounds(node: Any) -> Optional[Bounds]:
    """The participant ages the condition asks for, else the dataset-level
    bounds (d.age_min / d.age_max); None when it asks for no age."""
    bounds = sql_tree.compared_bounds(node, lambda side: sql_tree.compares(side, ["age"]))
    if bounds is not None and bounds.low is not None and bounds.low <= 0:
        # "age > 0" only leaves out placeholder ages (RAG/criteria.py)
        bounds = (Bounds(high=bounds.high, high_inclusive=bounds.high_inclusive)
                  if bounds.high is not None else None)
    if bounds is not None:
        return bounds
    if not {"age_min", "age_max"} & {c.name.lower() for c in node.find_all(exp.Column)}:
        return None
    low, high = age_range(node)
    bounds = Bounds(low if low > 0 else None, high if high < 120 else None)
    return None if bounds.low is None and bounds.high is None else bounds


def sex(node: Any) -> Optional[str]:
    """'female' or 'male' when a sex condition asks for one sex, else None."""
    # Which value stands for which sex: the vocabulary (RAG/participant_codes.py).
    values = {sex_code(v) for v in sql_tree.compared_values(node, ["sex"])} & set(ONE_SEX)
    return next(iter(values)) if len(values) == 1 else None


def species(node: Any) -> Optional[List[str]]:
    """The species a d.species condition asks for (['human'], ['mouse']), []
    for any animal, None for no species condition.

    Patterns as written by RAG/criteria.py; the label that says nothing
    ('other', 'n/a', 'unknown') is no species.
    """
    names: List[str] = []
    seen = False
    for match in node.find_all(exp.RegexpILike):
        column = match.this
        if not (isinstance(column, exp.Column) and column.name.lower() == "species"):
            continue
        pattern = str(getattr(match.expression, "this", "") or "")
        if "unknown" in pattern:
            continue
        seen = True
        if sql_tree.under_negation(match, node):
            continue
        first = re.search(r"\(([^|)]+)", pattern)
        if first:
            name = re.sub(r"\\[a-zA-Z]|.\?|[\^$*\\]", "", first.group(1)).strip()
            if name and name not in names:
                names.append(name)
    return names if seen else None


def patients(node: Any) -> Optional[Tuple[int, List[str]]]:
    """(25, ['frontotemporal_dementia']) for "at least N patients with X"
    (backend/vocabulary/cohort_count.py): a count of the participants with the
    diagnosis, not of all subjects."""
    found = cohort_count.find(node)
    if found is None:
        return None
    return found.minimum, list(found.codes)


@dataclass(frozen=True)
class Count:
    """COUNT(DISTINCT o.subject) >= 40: what is counted (one of
    COUNTED_COLUMNS), the bounds, and the ages the count is limited to."""
    column: str
    bounds: Bounds
    age: Optional[Bounds] = None


def count(node: Any) -> Optional[Count]:
    """The count a condition compares, as in COUNT(DISTINCT o.subject) >= 40.

    The count may sit inside the compared expression: "at least 20 adolescents"
    comes back as (SELECT COUNT(DISTINCT p.participant_id) FROM bids_participants
    p WHERE ... p.age >= 10 AND p.age <= 19) >= 20, and "at least 100
    participants" as GREATEST(COUNT(DISTINCT o.subject), (SELECT COUNT ...)) >= 100.
    """
    for counted in node.find_all(exp.Count):
        columns = [c.name.lower() for c in counted.find_all(exp.Column)]
        column = next((c for c in columns if c in COUNTED_COLUMNS), None)
        if not column:
            continue
        def is_this_count(left: exp.Expression, counted: exp.Count = counted) -> bool:
            return left is counted or any(inner is counted for inner in left.find_all(exp.Count))

        bounds = sql_tree.compared_bounds(node, is_this_count)
        if not bounds:
            continue
        age = None
        scope = counted.find_ancestor(exp.Select)
        if scope is not None and scope is not node and scope.find_ancestor(exp.Subquery) is not None:
            age = age_bounds(scope.args.get("where") or scope)
        return Count(column, bounds, age)
    return None


# ── Files and the dataset ─────────────────────────────────────────────────────

def field_strengths(node: Any) -> List[float]:
    """The nominal strengths a field-strength condition asks for: "> 6.5 AND
    < 7.5" → [7.0], "= 1.5" → [1.5], "≥ 3" → [3.0, 4.0, …]."""
    bounds = sql_tree.compared_bounds(
        node, lambda side: sql_tree.compares(side, json_key=FIELD_STRENGTH_KEY))
    if bounds is None:
        return []
    if bounds.exact is not None:
        return [bounds.exact]
    return [t for t in NOMINAL_TESLA if bounds.allows(t)]


def size_bounds(node: Any) -> Optional[Bounds]:
    """The bounds of a d.size_bytes comparison, in bytes."""
    return sql_tree.compared_bounds(node, lambda side: sql_tree.compares(side, ["size_bytes"]))


def json_key(node: Any) -> Optional[Tuple[str, Optional[Bounds]]]:
    """('MagneticFieldStrength', bounds) for other_entities ->> 'MagneticFieldStrength';
    the bounds are None when the condition only asks that the key is recorded."""
    name = next((key for key in sql_tree.json_keys(node) if key), None)
    if name is None:
        return None
    return name, sql_tree.compared_bounds(node, lambda side: True)


# ── Any other condition ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Parts:
    """What a condition no other reader describes is made of, as written."""
    tables: List[str]
    columns: List[str]       # without join keys and counted columns
    json_keys: List[str]
    literals: List[str]      # string literals, '%' stripped, none inside a function call
    tests_null: bool         # an IS NULL test


def parts(node: Any) -> Parts:
    columns = [column.name for column in node.find_all(exp.Column)
               if column.name and column.name.lower() not in KEY_COLUMNS
               and not column.find_ancestor(exp.Count)]
    keys = [key.expression.this for key in node.find_all(exp.JSONExtractScalar, exp.JSONExtract)
            if isinstance(key.expression, exp.Literal) and key.expression.is_string]
    literals = [lit.this.strip("%") for lit in node.find_all(exp.Literal)
                if lit.is_string and not lit.find_ancestor(exp.Anonymous)]
    return Parts(
        tables=[t.name for t in node.find_all(exp.Table) if t.name],
        columns=columns, json_keys=keys, literals=list(dict.fromkeys(literals)),
        tests_null=any(isinstance(n.expression, exp.Null) for n in node.find_all(exp.Is)),
    )
