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
(evidence/confirmed.py), with its wording. One query per page evaluates each branch for the page's datasets; a
branch on the participant list also counts the participants it holds for,
with their ages when it reads ages.
"""

from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from routers.schemas import DatasetSchema
from services.evidence.base import Source
from services.evidence.branches import confirming_evidence
from services.evidence.confirmed import CardColumn, ConfirmedByDefaultEvidence, ConfirmedEvidence
from services.results import query_filters
from services.schema import tables as _tables

try:
    from sqlglot import exp
except ImportError:  # pragma: no cover - sqlglot is a backend dependency
    exp = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass
class ChipPlan:
    """One chip and the evidence of its evaluable branches, in the condition's order."""
    field: str
    kind: str
    # dataclasses.field: the name 'field' is this class's own attribute.
    branches: List[ConfirmedEvidence] = dataclasses.field(default_factory=list)


# ── The chips worth explaining ───────────────────────────────────────────────

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
                    branches=[ConfirmedByDefaultEvidence(diagnosis_unknown(alias), source=Source.ASSUMED_ABSENT)])


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
    age_nodes = [clause.node for chip, clause in conditions if chip.field == "Age"]
    plans = []
    for chip, clause in conditions:
        if chip.negated:
            assumed = _assumed_absent(chip, alias)
            if assumed is not None:
                plans.append(assumed)
            continue
        if chip.field == "Species" and _is_age_species_guard(clause.node, age_nodes):
            continue
        branches = list(confirming_evidence(clause, alias))
        if len({evidence.key for evidence in branches}) >= 2:
            plans.append(ChipPlan(field=chip.field, kind=chip.kind, branches=branches))
    return alias, plans


def _suffix(chip: int, branch: int) -> str:
    """What names the card columns of one branch of one chip (CardColumn)."""
    return f"{chip}_{branch}"


def build_sql(alias: str, plans: Sequence[ChipPlan]) -> str:
    """One row per dataset: whether each piece of evidence holds, plus its numbers."""
    columns = [f"{alias}.id AS dataset_id"]
    field_columns: Dict[str, str] = {}
    for c, chip in enumerate(plans):
        for b, evidence in enumerate(chip.branches):
            holds = CardColumn.HOLDS.of(_suffix(c, b))
            columns.append(f"COALESCE(({evidence.sql()}), FALSE) AS {holds}")
            columns.extend(evidence.number_columns(_suffix(c, b)))
            for name in evidence.read_columns():
                field_columns.setdefault(
                    name, f"CAST({alias}.{name} AS TEXT) AS {CardColumn.FIELD.of(name)}")
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
            if not row.get(CardColumn.HOLDS.of(_suffix(c, b))):
                continue
            phrase = evidence.reason(row, _suffix(c, b), total)
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
    # SQL that cannot be parsed gives (None, []) (sql_tree.parse_select).
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
