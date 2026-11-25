"""
services/evidence/branches.py
-----------------------------
The evidence that confirms a condition of the generated query, read from its
SQL: each OR branch of the condition is one way it holds (confirming_evidence).

Used by results/match_reasons.py (what a result card says) and by
services/uncertain/ (ClauseEvidence.confirming).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from services.evidence.base import Source
from services.evidence.confirmed import Assumed, ConfirmedMatch, FieldValue, Found, ParticipantRows
from services.fulltext.vectors import EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN, FTS_COLUMN, PAPER_FTS_COLUMN
from services.schema.tables import COLUMN_TEXT, TABLE_ABOUT, About, TextKind

try:
    from sqlglot import exp
except ImportError:  # pragma: no cover - sqlglot is a backend dependency
    exp = None  # type: ignore[assignment]

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

def or_branches(node) -> list:
    """The OR branches of a condition, through parentheses (not through NOT)."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Or):
        return or_branches(node.left) + or_branches(node.right)
    return [node]


def _ancestors(node, root):
    """What *node* stands in, up to and including *root*; nothing for *root* itself."""
    while node is not root:
        node = node.parent
        yield node


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


def classify_branch(branch, alias: str) -> Optional[ConfirmedMatch]:
    """The evidence one OR branch is; None when it cannot be evaluated alone.
    The branch searches no papers (branch_evidence).

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
        if (source == Source.SHORT_TEXT
                and not isinstance(column.parent, (exp.ILike, exp.Like))):
            continue   # d.name = 'x' is a field, not a text search
        text_sources.append(source)
    if Source.TEXT in text_sources:
        queries = _branch_tsqueries(branch)
        if queries and all(short_fields_only(q) for q in queries):
            return Found(sql, source=Source.SHORT_TEXT)
    for source in (Source.TEXT, Source.SHORT_TEXT, Source.README):
        if source in text_sources:
            return Found(sql, source=source)

    for select in _outermost_selects(branch):
        if _under_not(select, branch):
            continue
        tables = [t.name.lower() for t in select.find_all(exp.Table)]
        sources = [_TABLE_SOURCES[t] for t in tables if t in _TABLE_SOURCES]
        if not sources:
            continue
        if sources[0] != Source.PARTICIPANTS:
            return Found(sql, source=sources[0])
        # One row is one participant only while the SELECT reads the
        # participant list alone (unique per dataset and participant_id): a
        # join would count a participant once per joined row.
        countable = (isinstance(select.parent, exp.Exists) and len(tables) == 1
                     and not select.args.get("joins")
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
    if len(names) == 1 and names[0] in Assumed.TEXTS and any(
            isinstance(node.expression, exp.Null) and not _under_not(node, branch)
            for node in branch.find_all(exp.Is)):
        return Assumed(sql, columns=names)
    if {"age_min", "age_max"} & set(names):
        # Either bound alone ("d.age_min < 3") is shown as the whole range.
        names = tuple(dict.fromkeys(("age_min", "age_max") + names))
    if names:
        return FieldValue(sql, columns=names)
    return None


def _without_papers(branch):
    """The branch with its text searches on the dataset's own text alone;
    None when it searches no papers."""
    own = branch.copy()
    columns = [c for c in own.find_all(exp.Column)
               if c.name.lower() in _WITHOUT_PAPERS and not _in_subquery(c, own) and not _under_not(c, own)]
    for column in columns:
        column.set("this", exp.to_identifier(_WITHOUT_PAPERS[column.name.lower()]))
    return own if columns else None


def branch_evidence(branch, alias: str) -> Tuple[ConfirmedMatch, ...]:
    """The evidence of one OR branch; none when it cannot be evaluated alone.

    A vector with the papers also holds the dataset's own text: a search on
    it is two ways to match, in the own text and only in a paper.
    """
    own = _without_papers(branch)
    found = classify_branch(branch if own is None else own, alias)
    if found is None:
        return ()
    if own is None:
        return (found,)
    queries = _branch_tsqueries(branch)
    if queries and all(short_fields_only(q) for q in queries):
        return (found,)   # paper text has the lowest weight: such a search never reaches it
    only_paper = f"({branch.sql(dialect='postgres')}) AND NOT ({own.sql(dialect='postgres')})"
    return (found, Found(only_paper, source=Source.PAPERS))


def confirming_evidence(node, alias: str) -> Tuple[ConfirmedMatch, ...]:
    """The ways the condition *node* can hold, in its order, each once. A branch
    that cannot be evaluated alone is left out, so these need not cover the condition."""
    ways: List[ConfirmedMatch] = []
    for branch in or_branches(node):
        ways.extend(way for way in branch_evidence(branch, alias) if way not in ways)
    return tuple(ways)
