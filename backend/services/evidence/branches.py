"""
services/evidence/branches.py
-----------------------------
The evidence that confirms a condition of the generated query, read from its
SQL: each OR branch of the condition is one way it holds (confirming_evidence).

Used by results/match_reasons.py (what a result card says) and by
services/uncertain/ (ClauseEvidence.confirming).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from sqlglot import exp

from services import sql_tree
from services.evidence.base import Source
from services.evidence.confirmed import Assumed, ConfirmedMatch, FieldValue, Found, ParticipantRows
from services.fulltext.text_search import text_searches
from services.fulltext.vectors import EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN, FTS_COLUMN, PAPER_FTS_COLUMN
from services.schema.tables import COLUMN_TEXT, TABLE_ABOUT, About, TextKind

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


@dataclass(frozen=True)
class EvidenceBranch:
    """Evidence interpretation of one branch from a SQL condition."""
    sql_branch: sql_tree.SqlBranch
    alias: str

    @property
    def node(self) -> Any:
        return self.sql_branch.node

    @property
    def sql(self) -> str:
        return self.sql_branch.sql

    @property
    def outer_columns(self) -> list:
        return self.sql_branch.outer_columns

    @property
    def reads_a_join(self) -> bool:
        """It reads a join of the outer query (o.task = …)."""
        return self.sql_branch.reads_a_join(self.alias)

    def negated(self, node) -> bool:
        return self.sql_branch.negated(node)

    @property
    def selects(self) -> list:
        """Its outermost subqueries."""
        return self.sql_branch.outer_selects

    @property
    def tsqueries(self) -> List[str]:
        """The query strings of its text searches, without the word-form guards."""
        return [search.query for search in text_searches(self.node)
                if not search.is_guard and not sql_tree.in_subquery(search.call, self.node)
                and not self.negated(search.call)]

    @property
    def searches_short_fields_only(self) -> bool:
        queries = self.tsqueries
        return bool(queries) and all(short_fields_only(q) for q in queries)

    def text_evidence(self) -> Optional[ConfirmedMatch]:
        """The dataset's own text names the value."""
        sources = []
        for column in self.outer_columns:
            source = _TEXT_COLUMNS.get(column.name.lower())
            if source is None or self.negated(column):
                continue
            # d.name = 'x' is a field, d.readme_text IS NULL names nothing.
            if any(isinstance(a, sql_tree.TEXT_MATCHES) for a in sql_tree.ancestors(column, self.node)):
                sources.append(source)
        if Source.TEXT in sources and self.searches_short_fields_only:
            return Found(self.sql, source=Source.SHORT_TEXT)
        for source in (Source.TEXT, Source.SHORT_TEXT, Source.README):
            if source in sources:
                return Found(self.sql, source=source)
        return None

    def subquery_evidence(self) -> Optional[ConfirmedMatch]:
        """Rows of another table hold the value."""
        for select in self.selects:
            if self.negated(select):
                continue
            tables = [t.name.lower() for t in select.find_all(exp.Table)]
            sources = [_TABLE_SOURCES[t] for t in tables if t in _TABLE_SOURCES]
            if not sources:
                continue
            if sources[0] != Source.PARTICIPANTS:
                return Found(self.sql, source=sources[0])
            # One row is one participant only while the SELECT reads the
            # participant list alone (unique per dataset and participant_id): a
            # join would count a participant once per joined row.
            countable = (isinstance(select.parent, exp.Exists) and len(tables) == 1
                         and not select.args.get("joins")
                         and not select.args.get("group") and not select.args.get("having"))
            where = select.args.get("where")
            columns = list(where.find_all(exp.Column)) if where is not None else []
            age = next((c for c in columns if c.name.lower() == "age"), None)
            return ParticipantRows(self.sql, participant_rows=select if countable else None,
                                   age_column=age,
                                   reads_diagnosis=any(c.name.lower() == "diagnosis" for c in columns))
        return None

    def field_evidence(self) -> Optional[ConfirmedMatch]:
        """The dataset's own columns hold the value, or a default does."""
        # A negated text search left here only says when the branch applies
        # (the species default: no label AND no animal named); its vector is no value.
        names = tuple(dict.fromkeys(c.name.lower() for c in self.outer_columns
                                    if c.name.lower() != "id" and c.name.lower() not in _TEXT_COLUMNS))
        if len(names) == 1 and names[0] in Assumed.TEXTS and any(
                isinstance(node.expression, exp.Null) and not self.negated(node)
                for node in self.node.find_all(exp.Is)):
            return Assumed(self.sql, columns=names)
        if {"age_min", "age_max"} & set(names):
            # Either bound alone ("d.age_min < 3") is shown as the whole range.
            names = tuple(dict.fromkeys(("age_min", "age_max") + names))
        return FieldValue(self.sql, columns=names) if names else None

    def without_papers(self) -> Optional[EvidenceBranch]:
        """The branch with its text searches on the dataset's own text alone;
        None when it searches no papers."""
        own = EvidenceBranch(sql_tree.SqlBranch(self.node.copy()), self.alias)
        columns = [c for c in own.outer_columns
                   if c.name.lower() in _WITHOUT_PAPERS and not own.negated(c)]
        for column in columns:
            column.set("this", exp.to_identifier(_WITHOUT_PAPERS[column.name.lower()]))
        return own if columns else None

    def has_negation(self) -> bool:
        return self.sql_branch.has_negation()


def classify_branch(branch: EvidenceBranch) -> Optional[ConfirmedMatch]:
    """The evidence one OR branch is; None when it cannot be evaluated alone.
    The branch searches no papers (branch_evidence).

    Text first: "no ages recorded AND the description names infants" is
    evidence from the text whose other parts only say when it applies. Then
    a subquery on another table, then the dataset's own columns.
    """
    if branch.reads_a_join:
        return None
    found = branch.text_evidence() or branch.subquery_evidence() or branch.field_evidence()
    if found is None and not branch.has_negation():
        log.warning("No kind of evidence for the branch: %s", sql_tree.without_literals(branch.node))
    return found


def branch_evidence(branch: EvidenceBranch) -> Tuple[ConfirmedMatch, ...]:
    """The evidence of one OR branch; none when it cannot be evaluated alone.

    A vector with the papers also holds the dataset's own text: a search on
    it is two ways to match, in the own text and only in a paper.
    """
    own = branch.without_papers()
    found = classify_branch(branch if own is None else own)
    if found is None:
        return ()
    if own is None or branch.searches_short_fields_only:
        return (found,)   # paper text has the lowest weight: such a search never reaches it
    only_paper = f"({branch.sql}) AND NOT ({own.sql})"
    return (found, Found(only_paper, source=Source.PAPERS))


def confirming_evidence(node, alias: str) -> Tuple[ConfirmedMatch, ...]:
    """The ways the condition *node* can hold, in its order, each once. A branch
    that cannot be evaluated alone is left out, so these need not cover the condition."""
    ways: List[ConfirmedMatch] = []
    for branch in sql_tree.Condition(node).branches:
        ways.extend(way for way in branch_evidence(EvidenceBranch(branch, alias)) if way not in ways)
    return tuple(ways)
