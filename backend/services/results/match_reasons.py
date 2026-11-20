"""
services/results/match_reasons.py
---------------------------------
Why each result matched: for every filter chip that can match in more than
one way, the ways that hold for the dataset ("Age: 12 of 40 participants
aged 4–23 months").

A chip is one top-level condition of the generated SQL, and the OR branches
of that condition are its venues of matching: the participant list, the
dataset's own fields, its files, its text, a linked paper. "infant datasets"
matches recorded ages under 3, or the dataset's age range, or (with no ages
recorded) a name or description naming infants. A match on recorded ages
needs this line: the README need not say "infant" at all.

Generic over every query: the venues are read from the SQL, not listed per
criterion, so a new criterion with several venues is explained without code
here. One query per page evaluates each branch for the page's datasets; a
branch on the participant list also counts the participants it holds for,
with their ages when it reads ages.
"""

from __future__ import annotations

import logging
import re
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import DatasetSchema
from services.results import query_filters

try:
    from sqlglot import exp
except ImportError:  # pragma: no cover - sqlglot is a backend dependency
    exp = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Venue kinds, and how a reason names them.
PARTICIPANTS = "participants"
FILES = "files"
DERIVATIVES = "derivatives"
SITES = "sites"
PAPER = "paper"
TEXT = "text"              # search_tsv: name, description, study fields, README
SHORT_TEXT = "short_text"  # name or description only (ILIKE, or search_tsv words labelled :AB)
README = "readme"
FIELD = "field"            # the dataset's own columns
ASSUMED = "assumed"        # the column is empty and its default holds (_DEFAULT_PHRASES)

_PHRASES = {
    FILES: "in the dataset's files",
    DERIVATIVES: "in its derivatives",
    SITES: "at a listed site",
    PAPER: "named in a linked paper",
    TEXT: "named in the README or description",
    SHORT_TEXT: "named in the title or description",
    README: "named in the README",
}

_TABLE_VENUES = {
    "bids_participants": PARTICIPANTS,
    "bids_objects": FILES,
    "bids_files": FILES,
    "bids_dataset_facets": FILES,
    "bids_derivatives": DERIVATIVES,
    "bids_dataset_sites": SITES,
    "bids_site_places": SITES,
    "bids_dataset_papers": PAPER,
    "bids_papers": PAPER,
}
_TEXT_COLUMNS = {
    "search_tsv": TEXT, "search_tsv_exact": TEXT,
    "search_tsv_papers": PAPER, "search_tsv_exact_papers": PAPER,
    "readme_text": README,
    "name": SHORT_TEXT, "description_text": SHORT_TEXT,
}
_FIELD_LABELS = {
    "species": "species label",
    "dx_status": "diagnosis label",
    "study_design": "study design",
    "study_domain": "study domain",
}
_MAX_VALUE_CHARS = 60
# Columns with a default (RAG/criteria.py): a branch that holds when the
# column is NULL holds by assuming this.
_DEFAULT_PHRASES = {"species": "no species label, assumed human"}


@dataclass
class Venue:
    kind: str
    # FIELD: the dataset columns the branch reads.
    columns: Tuple[str, ...] = ()
    # PARTICIPANTS: the participant rows the branch holds for, as a SELECT
    # whose projection the query replaces (COUNT, MIN/MAX of the age).
    participant_rows: Optional[object] = None
    age_column: Optional[object] = None   # the age column that SELECT compares
    reads_diagnosis: bool = False

    @property
    def key(self) -> Tuple[str, Tuple[str, ...]]:
        return self.kind, self.columns


@dataclass
class ChipPlan:
    """One chip and its evaluable branches, in the condition's order."""
    field: str
    kind: str
    # dataclasses.field: the name 'field' is this class's own attribute.
    branches: List[Tuple[str, Venue]] = dataclasses.field(default_factory=list)


# ── Reading the SQL ───────────────────────────────────────────────────────────

def or_branches(node) -> list:
    """The OR branches of a condition, through parentheses (not through NOT)."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Or):
        return or_branches(node.left) + or_branches(node.right)
    return [node]


def _ancestors(node, root):
    current = node.parent
    while current is not None and current is not root:
        yield current
        current = current.parent


def _in_subquery(node, root) -> bool:
    return any(isinstance(a, exp.Select) for a in _ancestors(node, root))


def _under_not(node, root) -> bool:
    return any(isinstance(a, exp.Not) for a in _ancestors(node, root))


def _outermost_selects(branch) -> list:
    return [s for s in branch.find_all(exp.Select) if not _in_subquery(s, branch)]


_TS_OPERATOR_RE = re.compile(r"<[^>]*>")
_TS_TOKEN_RE = re.compile(r"[^\s&|!()]+")
_TS_SHORT_WEIGHTS = {"A", "B"}   # name; description and study fields (migrate_add_search_tsv.sql)


def short_fields_only(tsquery: str) -> bool:
    """True when every word of *tsquery* is limited to weights A and B
    ('parkinson:AB'): the name and description, never the README (D)."""
    tokens = _TS_TOKEN_RE.findall(_TS_OPERATOR_RE.sub(" ", tsquery))
    if not tokens:
        return False
    for token in tokens:
        labels = set(token.split(":", 1)[1]) - {"*"} if ":" in token else set()
        if not labels or labels - _TS_SHORT_WEIGHTS:
            return False
    return True


def _branch_tsqueries(branch) -> List[str]:
    found = []
    for call in branch.find_all(exp.Anonymous, exp.Func):
        name = (call.name if isinstance(call, exp.Anonymous) else call.sql_name()).lower()
        if name != "to_tsquery" or _in_subquery(call, branch) or _under_not(call, branch):
            continue
        literals = [e for e in call.expressions if isinstance(e, exp.Literal) and e.is_string]
        if literals and literals[0].this.lower() != "simple":   # not a word-form guard
            found.append(literals[-1].this)
    return found


def classify_branch(branch, alias: str) -> Optional[Venue]:
    """The venue one OR branch reads; None when it cannot be evaluated alone.

    Text first: "no ages recorded AND the description names infants" is a
    text venue whose other parts only say when it applies. Then a subquery
    on another table, then the dataset's own columns.
    """
    outer_columns = [c for c in branch.find_all(exp.Column) if not _in_subquery(c, branch)]
    if any(c.table and c.table != alias for c in outer_columns):
        return None   # reads a join of the outer query (o.task = …)

    text_kinds = []
    for column in outer_columns:
        kind = _TEXT_COLUMNS.get(column.name.lower())
        if kind is None or _under_not(column, branch):
            continue
        if kind == SHORT_TEXT and not isinstance(column.parent, (exp.ILike, exp.Like)):
            continue   # d.name = 'x' is a field, not a text search
        text_kinds.append(kind)
    if TEXT in text_kinds:
        queries = _branch_tsqueries(branch)
        if queries and all(short_fields_only(q) for q in queries):
            return Venue(SHORT_TEXT)
    for kind in (TEXT, SHORT_TEXT, README, PAPER):
        if kind in text_kinds:
            return Venue(kind)

    for select in _outermost_selects(branch):
        if _under_not(select, branch):
            continue
        tables = [t.name.lower() for t in select.find_all(exp.Table)]
        kinds = [_TABLE_VENUES[t] for t in tables if t in _TABLE_VENUES]
        if not kinds:
            continue
        if kinds[0] != PARTICIPANTS:
            return Venue(kinds[0])
        countable = (isinstance(select.parent, exp.Exists)
                     and not select.args.get("group") and not select.args.get("having"))
        where = select.args.get("where")
        columns = list(where.find_all(exp.Column)) if where is not None else []
        age = next((c for c in columns if c.name.lower() == "age"), None)
        return Venue(PARTICIPANTS, participant_rows=select if countable else None,
                     age_column=age,
                     reads_diagnosis=any(c.name.lower() == "diagnosis" for c in columns))

    # A negated text search left here only says when the branch applies
    # (the species default: no label AND no animal named); its vector is no value.
    names = tuple(dict.fromkeys(c.name.lower() for c in outer_columns
                                if c.name.lower() != "id" and c.name.lower() not in _TEXT_COLUMNS))
    if len(names) == 1 and names[0] in _DEFAULT_PHRASES and any(
            isinstance(node.expression, exp.Null) and not _under_not(node, branch)
            for node in branch.find_all(exp.Is)):
        return Venue(ASSUMED, columns=names)
    if {"age_min", "age_max"} & set(names):
        # Either bound alone ("d.age_min < 3") is shown as the whole range.
        names = tuple(dict.fromkeys(("age_min", "age_max") + names))
    if names:
        return Venue(FIELD, columns=names)
    return None


def _is_age_species_guard(node, age_nodes: list) -> bool:
    """True for the species check an age condition carries with it:
    "(<ages> AND <human species>)" (RAG/criteria.py age_condition).

    The search never asked about species, so no reason explains it. A
    species limit the search did ask for ("human infants") is its own
    condition in the WHERE chain, not paired with the ages in parentheses.
    """
    child, parent = node, node.parent
    while isinstance(parent, exp.Paren):
        child, parent = parent, parent.parent
    if not isinstance(parent, exp.And) or not isinstance(parent.parent, exp.Paren):
        return False
    other = parent.left if parent.right is child else parent.right
    return any(other is age for age in age_nodes)


def plan(sql: Optional[str]) -> Tuple[Optional[str], List[ChipPlan]]:
    """The dataset alias and the chips worth explaining: not negated, with
    at least two kinds of venue (a modality chip only ever matches files),
    and not the species check of an age condition."""
    if exp is None:
        return None, []
    alias, conditions = query_filters.where_conditions(sql)
    if alias is None:
        return None, []
    age_nodes = [node for chip, node in conditions if chip.field == "Age"]
    plans = []
    for chip, node in conditions:
        if chip.negated:
            continue
        if chip.field == "Species" and _is_age_species_guard(node, age_nodes):
            continue
        branches = []
        for branch in or_branches(node):
            venue = classify_branch(branch, alias)
            if venue is not None:
                branches.append((branch.sql(dialect="postgres"), venue))
        if len({venue.key for _, venue in branches}) >= 2:
            plans.append(ChipPlan(field=chip.field, kind=chip.kind, branches=branches))
    return alias, plans


def _aggregate(rows_select, function: str, argument) -> str:
    select = rows_select.copy()
    select.set("expressions", [exp.Anonymous(this=function, expressions=[argument.copy()])])
    for key in ("order", "limit", "offset", "distinct"):
        select.set(key, None)
    return f"({select.sql(dialect='postgres')})"


def build_sql(alias: str, plans: Sequence[ChipPlan]) -> str:
    """One row per dataset: whether each branch holds, plus its numbers."""
    columns = [f"{alias}.id AS dataset_id"]
    field_columns: Dict[str, str] = {}
    for c, chip in enumerate(plans):
        for b, (branch_sql, venue) in enumerate(chip.branches):
            columns.append(f"COALESCE(({branch_sql}), FALSE) AS h{c}_{b}")
            if venue.participant_rows is not None:
                columns.append(f"{_aggregate(venue.participant_rows, 'COUNT', exp.Star())} AS n{c}_{b}")
                if venue.age_column is not None:
                    columns.append(f"{_aggregate(venue.participant_rows, 'MIN', venue.age_column)} AS lo{c}_{b}")
                    columns.append(f"{_aggregate(venue.participant_rows, 'MAX', venue.age_column)} AS hi{c}_{b}")
            for name in venue.columns:
                field_columns.setdefault(name, f"CAST({alias}.{name} AS TEXT) AS f_{name}")
    columns.extend(field_columns.values())
    return (f"SELECT {', '.join(columns)} FROM bids_datasets AS {alias} "
            f"WHERE {alias}.id = ANY(:ids)")


# ── Reasons ───────────────────────────────────────────────────────────────────

def _number(value: float) -> str:
    return f"{round(value, 1):g}"


def age_span(low: float, high: float) -> str:
    """'4–23 months' below 2 years, else '6–11 years'; one value once."""
    if high < 2:
        months_lo, months_hi = round(low * 12), round(high * 12)
        return (f"{months_lo} months" if months_lo == months_hi
                else f"{months_lo}–{months_hi} months")
    lo, hi = _number(low), _number(high)
    return f"{lo} years" if lo == hi else f"{lo}–{hi} years"


def _participants_phrase(venue: Venue, row: dict, c: int, b: int, total: Optional[int]) -> str:
    count = row.get(f"n{c}_{b}")
    if count is None:
        return "recorded in the participant list"
    whom = f"{count} of {total} participants" if total and total >= count else f"{count} participants"
    low, high = row.get(f"lo{c}_{b}"), row.get(f"hi{c}_{b}")
    if venue.age_column is not None and low is not None and high is not None:
        return f"{whom} aged {age_span(float(low), float(high))}"
    return f"coded for {whom}" if venue.reads_diagnosis else f"recorded for {whom}"


def _field_phrase(venue: Venue, row: dict) -> Optional[str]:
    values = {name: row.get(f"f_{name}") for name in venue.columns}
    if {"age_min", "age_max"} & set(values):
        try:
            low, high = float(values.get("age_min") or 0), float(values.get("age_max") or 0)
        except ValueError:
            low = high = 0
        if low > 0 or high > 0:
            low, high = (low or high), (high or low)
            return f"dataset age range {age_span(low, high)}"
    shown = []
    for name, value in values.items():
        if name in ("age_min", "age_max") or value in (None, ""):
            continue
        value = str(value)
        if len(value) > _MAX_VALUE_CHARS:
            value = value[:_MAX_VALUE_CHARS].rstrip() + "…"
        shown.append(f"{_FIELD_LABELS.get(name, name.replace('_', ' '))} “{value}”")
    return ", ".join(shown) or None


def reasons_for(plans: Sequence[ChipPlan], row: dict, total: Optional[int]) -> List[str]:
    """'Age: 12 of 40 participants aged 4–23 months', one per chip that holds."""
    reasons = []
    for c, chip in enumerate(plans):
        phrases: List[str] = []
        phrase: Optional[str]
        for b, (_, venue) in enumerate(chip.branches):
            if not row.get(f"h{c}_{b}"):
                continue
            if venue.kind == PARTICIPANTS:
                phrase = _participants_phrase(venue, row, c, b, total)
            elif venue.kind == FIELD:
                phrase = _field_phrase(venue, row)
            elif venue.kind == ASSUMED:
                phrase = _DEFAULT_PHRASES[venue.columns[0]]
            else:
                phrase = _PHRASES[venue.kind]
            if phrase and phrase not in phrases:
                phrases.append(phrase)
        if phrases:
            reasons.append(f"{chip.field}: {'; '.join(phrases)}")
    return reasons


async def attach_match_reasons(session: AsyncSession, datasets: List[DatasetSchema],
                               filter_sql: Optional[str]) -> None:
    """Set match_reasons on the page's datasets; a failure leaves them empty."""
    if not datasets or not filter_sql:
        return
    # SQL that cannot be parsed gives (None, []) (query_filters._parse).
    alias, plans = plan(filter_sql)
    if alias is None or not plans:
        return
    sql = build_sql(alias, plans)
    try:
        async with session.begin_nested():
            rows = (await session.execute(text(sql), {"ids": [d.id for d in datasets]})).mappings().all()
    except SQLAlchemyError:
        log.warning("Match reasons unavailable for this page", exc_info=True)
        return
    by_id = {str(row["dataset_id"]): dict(row) for row in rows}
    for dataset in datasets:
        row = by_id.get(str(dataset.id))
        if row is None:
            continue
        total = dataset.participant_summary.total if dataset.participant_summary else None
        dataset.match_reasons = reasons_for(plans, row, total)
