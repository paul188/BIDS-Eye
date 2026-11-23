"""
services/results/match_reasons.py
---------------------------------
Why each result matched: for every filter chip that can match in more than
one way, the ways that hold for the dataset ("Age: 12 of 40 participants
aged 4–23 months").

A chip is one top-level condition of the generated SQL, and each OR branch
of that condition is one piece of evidence for it (services/evidence/): the
participant list, the dataset's own fields, its files, its text, a linked
paper. "infant datasets"
matches recorded ages under 3, or the dataset's age range, or (with no ages
recorded) a name or description naming infants. A match on recorded ages
needs this line: the README need not say "infant" at all.

Generic over every query: the evidence is read from the SQL, not listed per
criterion, so a new criterion with several branches is explained without
code here. What a kind of evidence shows on the card is written with the kind
(evidence/confirmed.py), the wording in evidence/reason_texts.py. One query per page evaluates each branch for the page's datasets; a
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
from services.evidence import reason_texts
from services.evidence.base import Evidence, EvidenceSource
from services.evidence.confirmed import Assumed, FieldValue, ParticipantRows, found
from services.results import query_filters
from services.schema import tables as _tables

try:
    from sqlglot import exp
except ImportError:  # pragma: no cover - sqlglot is a backend dependency
    exp = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# One OR branch of a condition is one piece of evidence (services/evidence/):
# where the value was found, and the text the card shows for it.
_TABLE_SOURCES = {
    _tables.participants.name: EvidenceSource.PARTICIPANTS,
    _tables.objects.name: EvidenceSource.FILES,
    "bids_files": EvidenceSource.FILES,  # not a catalogue table, kept as it was
    _tables.dataset_facets.name: EvidenceSource.FILES,
    _tables.derivatives.name: EvidenceSource.DERIVATIVES,
    _tables.dataset_sites.name: EvidenceSource.SITES,
    _tables.site_places.name: EvidenceSource.SITES,
    _tables.dataset_papers.name: EvidenceSource.PAPER,
    _tables.papers.name: EvidenceSource.PAPER,
}
_TEXT_COLUMNS = {
    "search_tsv": EvidenceSource.TEXT, "search_tsv_exact": EvidenceSource.TEXT,
    "search_tsv_papers": EvidenceSource.PAPER, "search_tsv_exact_papers": EvidenceSource.PAPER,
    "readme_text": EvidenceSource.README,
    "name": EvidenceSource.SHORT_TEXT, "description_text": EvidenceSource.SHORT_TEXT,
}


@dataclass
class ChipPlan:
    """One chip and the evidence of its evaluable branches, in the condition's order."""
    field: str
    kind: str
    # dataclasses.field: the name 'field' is this class's own attribute.
    branches: List[Evidence] = dataclasses.field(default_factory=list)


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


def classify_branch(branch, alias: str) -> Optional[Evidence]:
    """The evidence one OR branch is; None when it cannot be evaluated alone.

    Text first: "no ages recorded AND the description names infants" is
    evidence from the text whose other parts only say when it applies. Then
    a subquery on another table, then the dataset's own columns.
    """
    outer_columns = [c for c in branch.find_all(exp.Column) if not _in_subquery(c, branch)]
    if any(c.table and c.table != alias for c in outer_columns):
        return None   # reads a join of the outer query (o.task = …)
    sql = branch.sql(dialect="postgres")

    text_sources = []
    for column in outer_columns:
        source = _TEXT_COLUMNS.get(column.name.lower())
        if source is None or _under_not(column, branch):
            continue
        if (source == EvidenceSource.SHORT_TEXT
                and not isinstance(column.parent, (exp.ILike, exp.Like))):
            continue   # d.name = 'x' is a field, not a text search
        text_sources.append(source)
    if EvidenceSource.TEXT in text_sources:
        queries = _branch_tsqueries(branch)
        if queries and all(short_fields_only(q) for q in queries):
            return found(EvidenceSource.SHORT_TEXT, sql)
    for source in (EvidenceSource.TEXT, EvidenceSource.SHORT_TEXT, EvidenceSource.README,
                   EvidenceSource.PAPER):
        if source in text_sources:
            return found(source, sql)

    for select in _outermost_selects(branch):
        if _under_not(select, branch):
            continue
        tables = [t.name.lower() for t in select.find_all(exp.Table)]
        sources = [_TABLE_SOURCES[t] for t in tables if t in _TABLE_SOURCES]
        if not sources:
            continue
        if sources[0] != EvidenceSource.PARTICIPANTS:
            return found(sources[0], sql)
        countable = (isinstance(select.parent, exp.Exists)
                     and not select.args.get("group") and not select.args.get("having"))
        where = select.args.get("where")
        columns = list(where.find_all(exp.Column)) if where is not None else []
        age = next((c for c in columns if c.name.lower() == "age"), None)
        return ParticipantRows(sql, participant_rows=select if countable else None,
                               age_column=age,
                               reads_diagnosis=any(c.name.lower() == "diagnosis" for c in columns))

    # A negated text search left here only says when the branch applies
    # (the species default: no label AND no animal named); its vector is no value.
    names = tuple(dict.fromkeys(c.name.lower() for c in outer_columns
                                if c.name.lower() != "id" and c.name.lower() not in _TEXT_COLUMNS))
    if len(names) == 1 and names[0] in reason_texts.ASSUMED and any(
            isinstance(node.expression, exp.Null) and not _under_not(node, branch)
            for node in branch.find_all(exp.Is)):
        return Assumed(sql, columns=names)
    if {"age_min", "age_max"} & set(names):
        # Either bound alone ("d.age_min < 3") is shown as the whole range.
        names = tuple(dict.fromkeys(("age_min", "age_max") + names))
    if names:
        return FieldValue(sql, columns=names)
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


def _assumed_absent(chip, alias: str) -> Optional[ChipPlan]:
    """The one thing worth saying about an excluded diagnosis: the dataset
    passes because it records no diagnosis at all, not because a record
    rules the diagnosis out. Other negated chips are not explained."""
    if chip.kind != "diagnosis":
        return None
    try:
        from criteria import diagnosis_unknown
    except ImportError:  # the RAG package is not on the path
        return None
    return ChipPlan(field=f"No {chip.label}", kind=chip.kind,
                    branches=[found(EvidenceSource.ASSUMED_ABSENT, diagnosis_unknown(alias))])


def plan(sql: Optional[str]) -> Tuple[Optional[str], List[ChipPlan]]:
    """The dataset alias and the chips worth explaining: not negated, with
    at least two kinds of evidence (a modality chip only ever matches files),
    and not the species check of an age condition. An excluded diagnosis is
    explained where it holds by default (_assumed_absent)."""
    if exp is None:
        return None, []
    alias, conditions = query_filters.where_conditions(sql)
    if alias is None:
        return None, []
    age_nodes = [node for chip, node in conditions if chip.field == "Age"]
    plans = []
    for chip, node in conditions:
        if chip.negated:
            assumed = _assumed_absent(chip, alias)
            if assumed is not None:
                plans.append(assumed)
            continue
        if chip.field == "Species" and _is_age_species_guard(node, age_nodes):
            continue
        branches = [evidence for branch in or_branches(node)
                    if (evidence := classify_branch(branch, alias)) is not None]
        if len({evidence.key for evidence in branches}) >= 2:
            plans.append(ChipPlan(field=chip.field, kind=chip.kind, branches=branches))
    return alias, plans


def build_sql(alias: str, plans: Sequence[ChipPlan]) -> str:
    """One row per dataset: whether each piece of evidence holds, plus its numbers."""
    columns = [f"{alias}.id AS dataset_id"]
    field_columns: Dict[str, str] = {}
    for c, chip in enumerate(plans):
        for b, evidence in enumerate(chip.branches):
            columns.append(f"COALESCE(({evidence.sql}), FALSE) AS h{c}_{b}")
            columns.extend(evidence.number_columns(f"{c}_{b}"))
            for name in evidence.read_columns():
                field_columns.setdefault(name, f"CAST({alias}.{name} AS TEXT) AS f_{name}")
    columns.extend(field_columns.values())
    return (f"SELECT {', '.join(columns)} FROM {_tables.datasets.name} AS {alias} "
            f"WHERE {alias}.id = ANY(:ids)")


# ── Reasons ───────────────────────────────────────────────────────────────────

def reasons_for(plans: Sequence[ChipPlan], row: dict, total: Optional[int]) -> List[str]:
    """'Age: 12 of 40 participants aged 4–23 months', one per chip that holds."""
    reasons = []
    for c, chip in enumerate(plans):
        phrases: List[str] = []
        for b, evidence in enumerate(chip.branches):
            if not row.get(f"h{c}_{b}"):
                continue
            phrase = evidence.phrase(row, f"{c}_{b}", total)
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
