"""
services/uncertain/classify.py
----------------------------------
Which evidence a top-level AND condition can have (classify): text, ages,
diagnosis, sex, scanner fields, matches only in a paper the dataset names
(paper_atoms), and "at least N participants with a diagnosis"
(patient_count).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Optional, Tuple

try:  # sqlglot is a backend dependency; keep the module importable without it
    from sqlglot import exp
except ImportError:  # pragma: no cover
    exp = None  # type: ignore[assignment]

from services import sql_tree
from services.fulltext.text_search import Clause
from services.uncertain.conditions import LABEL_RE, ClauseEvidence
from services.uncertain.text import query_words, relax_text_condition
from services.evidence.base import Source, Strength, UncertainEvidence
from services.evidence.branches import confirming_evidence
from services.evidence.confirmed import ConfirmedEvidence
from services.evidence.possible import PossibleNotRecordedEvidence
from services.evidence.probable import ProbableCoreOfTermEvidence, ProbablePaperOnlyEvidence, TextSearch
from services.schema.tables import SEARCH_VECTORS
from services.uncertain import fields
from services.uncertain.scanner import sidecar


# ── Which conditions may be uncertain ───────────────────────────────────────
# This block and soft_kind() are the one place that decides it. classify()
# below only builds the weaker forms for what soft_kind() let through.
#
# A condition may be uncertain when a dataset can fail it without
# contradicting it: it reads a field a dataset may not record, or it is a
# text search and the dataset uses other words. Which table holds the field
# plays no part in that.

# Fields a dataset may not record: one file each in fields/, listed in
# fields.FIELDS. A column of one of them makes a condition of that field's
# kind (fields.KIND_OF_COLUMN).

# Sidecar values (field strength, echo times …) are recorded with the files.
# Field strength and scanner make or model can be missing there and named in
# the README instead: scanner.sidecar makes those three uncertain and leaves
# the other keys hard.
SIDECAR_COLUMN = "other_entities"

# Fields that stay hard although a README could name the value.
#   institutions  A resolved institution filter is evidence from the scanner
#                 headers. Its weaker form would be the name anywhere in the
#                 README or a paper, and those mentions are mostly suppliers,
#                 software and citations ('kindly provided by … Harvard
#                 Medical School', surfer.nmr.mgh.harvard.edu).
#   species       The README of a human study says 'mouse click', and the
#                 title/description fallback for unlabelled datasets is
#                 already part of the condition.
HARD_FIELDS = {"institutions", "species"}

# Every field neither in fields/ nor here is hard: modality, file types, size, subject
# counts and the like come from the files themselves, so a dataset that
# fails them is contradicted.


def _joined_row_tables(node, alias: str) -> set:
    """The joined tables whose rows *node* reads directly, not in a subquery.

    "p.age < 12" on a joined participant row is true or false per row; the
    tier needs one answer per dataset. Inside EXISTS(…) there is one.
    """
    tables = {column.table or "" for column in node.find_all(exp.Column)
              if column.find_ancestor(exp.Subquery, exp.Exists) is None}
    return tables - {alias, ""}


def _patient_count_parts(sql: str):
    """(codes, minimum) of "at least N participants with a diagnosis", else None."""
    try:
        import cohort_count
    except ImportError:  # the RAG package is not on the path
        return None
    return cohort_count.parse_expanded(sql)


def _denies_a_sidecar_value(node) -> bool:
    """A NOT stands around a sidecar value; "IS NOT NULL" only asks that it is recorded."""
    def asks_if_recorded(negation) -> bool:
        inner = negation.this.unnest() if isinstance(negation, exp.Not) else None
        return isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null)

    return any(sql_tree.is_negation(a) and not asks_if_recorded(a)
               for column in node.find_all(exp.Column) if column.name.lower() == SIDECAR_COLUMN
               for a in sql_tree.ancestors(column, node))


def soft_kind(clause: Clause, alias: str) -> Optional[str]:
    """The kind of uncertain condition *clause* may become; None: a hard filter.

    The rules, in this order:
      1. Negated: hard.
      2. Reads a sidecar value: "sidecar"; hard when it denies one
         ("not GE"): a text naming the value is no evidence for it.
      3. Reads a HARD_FIELDS column: hard.
      4. Reads a joined table's row outside a subquery: hard.
      5. "At least N participants with a diagnosis": "patient_count".
      6. Reads columns of one field in fields/: that field's kind ("age",
         "diagnosis", "sex"). Of two fields in one condition: hard.
      7. Holds a text search: "text".
      8. Anything else: hard.

    A kind is permission, not a promise: classify() still returns None when
    the condition has no weaker form (a one-word text search, a sidecar key
    other than field strength, make or model).
    """
    node = clause.node
    if isinstance(node, exp.Not):
        return None
    columns = {column.name.lower() for column in node.find_all(exp.Column)}
    if SIDECAR_COLUMN in columns:
        return None if _denies_a_sidecar_value(node) else "sidecar"
    if columns & HARD_FIELDS:
        return None
    if _joined_row_tables(node, alias):
        return None
    sql = node.sql(dialect="postgres")
    if _patient_count_parts(sql) is not None:
        return "patient_count"
    kinds = {fields.KIND_OF_COLUMN[column] for column in columns & fields.KIND_OF_COLUMN.keys()}
    if len(kinds) > 1:
        return None
    if kinds:
        return kinds.pop()
    if clause.find(stemmed=True):
        return "text"
    return None


def paper_searches(clause: Clause) -> Tuple[TextSearch, ...]:
    """The positive text searches of the condition *clause*, as they are run on paper text.

    Without labels: paper text carries the lowest weight, so a search scoped
    to the name and short fields could never reach it.
    """
    return tuple(TextSearch(LABEL_RE.sub("", search.query),
                            LABEL_RE.sub("", search.guard.query) if search.guard else None)
                 for search in clause.find(stemmed=True, negated=False))


_DISTANCE_RE = re.compile(r"<\d+>")


def _positive_words(clause: Clause) -> set:
    return {word for search in clause.find(stemmed=True, negated=False)
            for word in query_words(search.query)}


def loose_term(clause: Clause, alias: str, papers: bool = False) -> Optional[ClauseEvidence]:
    """The ClauseEvidence of a loose term that lost words, else None.

    services/fulltext/text_search.py writes such a term as (every word OR
    core): "physical exercise", found in 1 dataset, searches "exercise". The
    datasets with every word are confirmed; those with the core alone are
    probable ("missing: physical"). Recognised as an OR of two text searches
    whose right side uses only some of the left side's words.
    """
    node = clause.node
    if not isinstance(node, exp.Or):
        return None
    if any(column.name.lower() not in SEARCH_VECTORS for column in node.find_all(exp.Column)):
        return None
    every_word = node.left.sql(dialect="postgres")
    core = node.right.sql(dialect="postgres")
    # A topic search (phrase OR alternatives AND NOT unwanted meanings, or a
    # 'near' search) has its own weaker forms (text.relax_text_condition).
    if node.find(exp.Not) is not None or _DISTANCE_RE.search(every_word):
        return None
    left, right = clause.part(node.left), clause.part(node.right)
    words, core_words = _positive_words(left), _positive_words(right)
    if not core_words or not core_words < words:
        return None
    return ClauseEvidence(kind="text", sql=every_word,
                          confirming=confirming_evidence(left, alias), evidence=(
        # The core only in a paper: the rows the core's own paper form had before.
        *_paper_only(right, alias, papers),
        ProbableCoreOfTermEvidence(core, missing=tuple(sorted(words - core_words))),
    ))


def _paper_only(clause: Clause, alias: str, papers: bool) -> Tuple[UncertainEvidence, ...]:
    """The evidence that the text searches of the condition *clause* match only
    in a paper; none without paper text or without a positive text search."""
    searches = paper_searches(clause) if papers else ()
    return (ProbablePaperOnlyEvidence(searches, alias=alias),) if searches else ()


def classify(clause: Clause, alias: str, papers: bool = False) -> Optional[ClauseEvidence]:
    """A ClauseEvidence for a condition that may be uncertain, else None.

    *papers*: the database has paper text and it may serve as evidence.
    """
    if isinstance(clause.node, exp.Paren):
        clause = clause.part(clause.node.this)
    node = clause.node
    # Whether the condition may be uncertain is decided in soft_kind() alone;
    # from here on only its weaker forms are built.
    kind = soft_kind(clause, alias)
    if kind is None:
        return None
    sql = node.sql(dialect="postgres")
    if kind == "sidecar":
        return sidecar(clause, alias)
    confirming = confirming_evidence(clause, alias)
    if kind == "patient_count":
        return _patient_count(sql, alias, confirming)
    if kind == "text":
        loose = loose_term(clause, alias, papers)
        if loose is not None:
            return loose
    # Part of a text search the condition holds, anywhere in the text.
    relaxed = relax_text_condition(clause, alias)
    weaker_search: Tuple[UncertainEvidence, ...] = (relaxed,) if relaxed else ()
    paper = _paper_only(clause, alias, papers)
    if kind != "text":
        # A field a dataset may not record. What is particular to the field
        # is in its file (fields/); the steps are the same for all.
        field = fields.BY_KIND[kind]
        # The dataset's own text names the asked value: counts in place of a
        # part of the search.
        named = field.named_in_text(node, alias) if field.named_in_text else None
        weaker: Tuple[UncertainEvidence, ...] = (named,) if named is not None else weaker_search
        if field.paper_evidence:
            # Without it a paper-only match is 'possible' like any other
            # dataset that records nothing.
            weaker = paper + weaker
        records = field.recorded(alias)
        not_recorded = PossibleNotRecordedEvidence(field.label, field.unrecorded_score)
        evidence = ((*weaker, *records, not_recorded) if field.text_beats_recorded
                    else (*records, *weaker, not_recorded))
        return ClauseEvidence(kind=kind, sql=sql, evidence=evidence, confirming=confirming)
    if relaxed or paper:
        return ClauseEvidence(kind="text", sql=sql, evidence=(*paper, *weaker_search),
                              confirming=confirming)
    return None


@dataclass(frozen=True)
class ProbablePatientsNotCountedEvidence(UncertainEvidence):
    """Probable, for "at least *minimum* participants with one of *codes*":
    the dataset does not record who has the diagnosis, and it has enough
    participants to hold them."""
    codes: Tuple[str, ...]
    minimum: int
    stored_subject_counts: bool   # bids_datasets.n_subjects is filled
    alias: str = dataclasses.field(kw_only=True)   # the dataset row of the query
    source: Source = dataclasses.field(default=Source.NOT_RECORDED, kw_only=True)
    strength: Strength = dataclasses.field(default=Strength.PROBABLE, kw_only=True)

    def sql(self) -> Optional[str]:
        import cohort_count
        from services.schema.tables import datasets
        stored = (f"COALESCE({self.alias}.{datasets.c.n_subjects.name}, 0)"
                  if self.stored_subject_counts else None)
        return cohort_count.unknown_sql(list(self.codes), self.minimum, self.alias, subject_count=stored)

    def reason(self) -> str:
        return "how many participants have the diagnosis is not recorded"


def _patient_count(sql: str, alias: str,
                   confirming: Tuple[ConfirmedEvidence, ...] = ()) -> Optional[ClauseEvidence]:
    """"At least N participants with the diagnosis" (RAG/cohort_count.py).

    Confirmed when that many are coded or labelled with it. Uncertain when the
    dataset does not record who has the diagnosis and is large enough to hold
    them. Excluded when its participant rows show fewer.
    """
    try:
        import cohort_count
    except ImportError:  # the RAG package is not on the path
        return None
    parsed = cohort_count.parse_expanded(sql)
    if parsed is None:
        return None
    codes, minimum = parsed
    stored = False
    try:
        from services.database_stats.database_stats import database_stats
        stored = bool(database_stats.summaries.subject_counts)
    except ImportError:
        pass
    return ClauseEvidence(kind="patient_count", sql=sql, confirming=confirming, evidence=(
        ProbablePatientsNotCountedEvidence(tuple(codes), minimum, stored, alias=alias),
    ))
