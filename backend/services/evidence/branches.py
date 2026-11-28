"""
services/evidence/branches.py
-----------------------------
The evidence that confirms a condition of the generated query, read from its
SQL: each OR branch of the condition is one way it holds (confirming_evidence).

Used by results/display/match_reasons.py (what a result card says) and by
services/uncertain/ (ClauseEvidence.confirming).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sqlglot import exp

from services import sql_tree
from services.evidence.base import Source
from services.evidence.confirmed import (ConfirmedByDefaultEvidence, ConfirmedEvidence,
                                         ConfirmedInDatasetColumnEvidence, ConfirmedInDatasetTextEvidence,
                                         ConfirmedInFileTablesEvidence, ConfirmedInPaperEvidence,
                                         ConfirmedInParticipantListEvidence)
from services.fulltext.text_search import Clause
from services.fulltext.tsquery import short_fields_only
from services.fulltext.vectors import EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN, FTS_COLUMN, PAPER_FTS_COLUMN
from services.schema.tables import COLUMN_TEXT, LABELS_IN_TEXT, TABLE_ABOUT, About, TextKind

log = logging.getLogger(__name__)

# One OR branch of a condition is one piece of evidence (services/evidence/):
# where the value was found, and the text the card shows for it.
# A subquery on bids_datasets or bids_word_forms is no source of its own.
_ABOUT_SOURCES = {
    About.PARTICIPANTS: Source.PARTICIPANTS,
    About.FILES: Source.FILES,
    About.DERIVATIVES: Source.DERIVATIVES,
    About.SITES: Source.SITES,
    About.PAPERS: Source.PAPERS,
}
_KIND_SOURCES = {
    TextKind.TEXT: Source.TEXT,
    TextKind.TEXT_AND_PAPERS: Source.PAPERS,
    TextKind.README: Source.README,
    TextKind.SHORT_TEXT: Source.SHORT_TEXT,
}
_TABLE_SOURCES = {table: _ABOUT_SOURCES[about] for table, about in TABLE_ABOUT.items()
                  if about in _ABOUT_SOURCES}
_TEXT_COLUMNS = {column: _KIND_SOURCES[kind] for column, kind in COLUMN_TEXT.items()}
# A vector that holds the papers too -> the same vector without them.
_WITHOUT_PAPERS = {PAPER_FTS_COLUMN: FTS_COLUMN, EXACT_PAPER_FTS_COLUMN: EXACT_FTS_COLUMN}


# ── Reading the SQL ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvidenceBranch:
    """One OR branch of a SQL condition, with what reading its evidence needs."""
    sql_branch: sql_tree.SqlBranch
    alias: str
    clause: Clause   # the branch with its searches; of the original when sql_branch is a copy

    @property
    def reads_a_join(self) -> bool:
        """It reads a join of the outer query (o.task = …)."""
        return self.sql_branch.reads_a_join(self.alias)

    @property
    def searches_short_fields_only(self) -> bool:
        """Its text searches, without the word-form guards, read only the name and description."""
        queries = [search.query for search in self.clause.find(negated=False, outer=True)]
        return bool(queries) and all(short_fields_only(q) for q in queries)

    def without_papers(self) -> Optional[EvidenceBranch]:
        """The branch with its text searches on the dataset's own text alone;
        None when it searches no papers."""
        from_dataset = sql_tree.SqlBranch(self.sql_branch.node.copy())
        columns = [c for c in from_dataset.outer_columns
                   if c.name.lower() in _WITHOUT_PAPERS and not from_dataset.negated(c)]
        for column in columns:
            column.set("this", exp.to_identifier(_WITHOUT_PAPERS[column.name.lower()]))
        return EvidenceBranch(from_dataset, self.alias, self.clause) if columns else None


# ── The evidence one branch is ────────────────────────────────────────────────

def classify_branch(branch: EvidenceBranch) -> Optional[ConfirmedEvidence]:
    """The evidence one OR branch is; None when it cannot be evaluated alone.
    The branch searches no papers (branch_evidence).

    Text first: "no ages recorded AND the description names infants" is
    evidence from the text whose other parts only say when it applies. Then
    a subquery on another table, then the dataset's own columns.
    """
    if branch.reads_a_join:
        return None
    found = (_from_dataset_text(branch) or _from_subquery(branch)
             or _from_dataset_columns(branch.sql_branch))
    if found is None and not branch.sql_branch.has_negation():
        log.warning("No kind of evidence for the branch: %s", sql_tree.without_literals(branch.sql_branch.node))
    return found


def _from_dataset_text(branch: EvidenceBranch) -> Optional[ConfirmedEvidence]:
    """The dataset's own text names the value."""
    sql_branch = branch.sql_branch
    sources = []
    for column in sql_branch.outer_columns:
        source = _TEXT_COLUMNS.get(column.name.lower())
        if source is None or sql_branch.negated(column):
            continue
        # d.name = 'x' is a field, d.readme_text IS NULL names nothing.
        if any(isinstance(a, sql_tree.TEXT_MATCHES) for a in sql_tree.ancestors(column, sql_branch.node)):
            sources.append(source)
    if Source.TEXT in sources and branch.searches_short_fields_only:
        return ConfirmedInDatasetTextEvidence(sql_branch.sql, source=Source.SHORT_TEXT)
    for source in (Source.TEXT, Source.SHORT_TEXT, Source.README):
        if source in sources:
            return ConfirmedInDatasetTextEvidence(sql_branch.sql, source=source)
    return None


def _from_subquery(branch: EvidenceBranch) -> Optional[ConfirmedEvidence]:
    """Rows of another table hold the value."""
    sql_branch = branch.sql_branch
    for select in sql_branch.outer_selects:
        if sql_branch.negated(select):
            continue
        tables = [t.name.lower() for t in select.find_all(exp.Table)]
        sources = [_TABLE_SOURCES[t] for t in tables if t in _TABLE_SOURCES]
        if not sources:
            continue
        if sources[0] == Source.PAPERS:
            return ConfirmedInPaperEvidence(sql_branch.sql)
        if sources[0] != Source.PARTICIPANTS:
            return ConfirmedInFileTablesEvidence(sql_branch.sql, source=sources[0])
        # One row is one participant only while the SELECT reads the
        # participant list alone (unique per dataset and participant_id): a
        # join would count a participant once per joined row.
        countable = (isinstance(select.parent, exp.Exists) and len(tables) == 1
                     and not select.args.get("joins")
                     and not select.args.get("group") and not select.args.get("having"))
        where = select.args.get("where")
        columns = list(where.find_all(exp.Column)) if where is not None else []
        age = next((c for c in columns if c.name.lower() == "age"), None)
        return ConfirmedInParticipantListEvidence(sql_branch.sql, participant_rows=select if countable else None,
                               age_column=age,
                               reads_diagnosis=any(c.name.lower() == "diagnosis" for c in columns))
    return None


def _from_dataset_columns(sql_branch: sql_tree.SqlBranch) -> Optional[ConfirmedEvidence]:
    """The dataset's own columns hold the value, or a default does."""
    # A negated text search left here only says when the branch applies
    # (the species default: no label AND no animal named); its vector is no value.
    names = tuple(dict.fromkeys(c.name.lower() for c in sql_branch.outer_columns
                                if c.name.lower() != "id" and c.name.lower() not in _TEXT_COLUMNS))
    if len(names) == 1 and names[0] in ConfirmedByDefaultEvidence.TEXTS and any(
            isinstance(node.expression, exp.Null) and not sql_branch.negated(node)
            for node in sql_branch.node.find_all(exp.Is)):
        return ConfirmedByDefaultEvidence(sql_branch.sql, columns=names)
    if {"age_min", "age_max"} & set(names):
        # Either bound alone ("d.age_min < 3") is shown as the whole range.
        names = tuple(dict.fromkeys(("age_min", "age_max") + names))
    if not names:
        return None
    source = Source.LABEL if LABELS_IN_TEXT.intersection(names) else Source.FIELD
    return ConfirmedInDatasetColumnEvidence(sql_branch.sql, columns=names, source=source)


def branch_evidence(branch: EvidenceBranch) -> Tuple[ConfirmedEvidence, ...]:
    """The evidence of one OR branch; none when it cannot be evaluated alone.

    A vector with the papers also holds the dataset's own text: a search on
    it is two ways to match, in the own text and only in a paper.
    """
    from_dataset = branch.without_papers()
    found = classify_branch(branch if from_dataset is None else from_dataset)
    if found is None:
        return ()
    if from_dataset is None or branch.searches_short_fields_only:
        return (found,)   # paper text has the lowest weight: such a search never reaches it
    only_paper = f"({branch.sql_branch.sql}) AND NOT ({from_dataset.sql_branch.sql})"
    return (found, ConfirmedInPaperEvidence(only_paper))


def confirming_evidence(clause: Clause, alias: str) -> Tuple[ConfirmedEvidence, ...]:
    """The ways the condition *clause* can hold, in its order, each once. A branch
    that cannot be evaluated alone is left out, so these need not cover the condition."""
    ways: List[ConfirmedEvidence] = []
    for branch in sql_tree.Condition(clause.node).branches:
        evidence = branch_evidence(EvidenceBranch(branch, alias, clause.part(branch.node)))
        ways.extend(way for way in evidence if way not in ways)
    return tuple(ways)
