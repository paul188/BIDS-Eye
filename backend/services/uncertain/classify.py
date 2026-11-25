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

from services.uncertain.conditions import LABEL_RE, ClauseEvidence, TSQUERY_RE
from services.uncertain.text import negated_spans, query_words, relax_text_condition
from services.evidence.base import PieceOfEvidence, Source, Strength
from services.evidence.branches import confirming_evidence
from services.evidence.confirmed import ConfirmedMatch
from services.evidence.contradicted import OtherValuesRecorded
from services.evidence.possible import NotRecorded
from services.evidence.probable import CoreOfTerm, PaperOnly, TextSearch
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


def soft_kind(node, alias: str) -> Optional[str]:
    """The kind of uncertain condition *node* may become; None: a hard filter.

    The rules, in this order:
      1. Negated: hard.
      2. Reads a sidecar value: "sidecar".
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
    if isinstance(node, exp.Not):
        return None
    columns = {column.name.lower() for column in node.find_all(exp.Column)}
    if SIDECAR_COLUMN in columns:
        return "sidecar"
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
    if TSQUERY_RE.search(sql):
        return "text"
    return None


def paper_searches(sql: str) -> Tuple[TextSearch, ...]:
    """The condition's positive text searches, as they are run on paper text.

    Without labels: paper text carries the lowest weight, so a search scoped
    to the name and short fields could never reach it.
    """
    spans = negated_spans(sql)
    searches = []
    for match in TSQUERY_RE.finditer(sql):
        if match.group("negated") or any(s < match.start() < e for s, e in spans):
            continue
        guard = match.group("guard")
        searches.append(TextSearch(LABEL_RE.sub("", match.group("query")),
                                   LABEL_RE.sub("", guard) if guard else None))
    return tuple(searches)


_DISTANCE_RE = re.compile(r"<\d+>")


def _positive_words(sql: str) -> set:
    spans = negated_spans(sql)
    return {word for match in TSQUERY_RE.finditer(sql)
            if not (match.group("negated") or any(s < match.start() < e for s, e in spans))
            for word in query_words(match.group("query"))}


def loose_term(node, alias: str, papers: bool = False) -> Optional[ClauseEvidence]:
    """The ClauseEvidence of a loose term that lost words, else None.

    services/fulltext/text_search.py writes such a term as (every word OR
    core): "physical exercise", found in 1 dataset, searches "exercise". The
    datasets with every word are confirmed; those with the core alone are
    probable ("missing: physical"). Recognised as an OR of two text searches
    whose right side uses only some of the left side's words.
    """
    if not isinstance(node, exp.Or):
        return None
    if any(not column.name.lower().startswith("search_tsv") for column in node.find_all(exp.Column)):
        return None
    every_word = node.left.sql(dialect="postgres")
    core = node.right.sql(dialect="postgres")
    # A topic search (phrase OR alternatives AND NOT unwanted meanings, or a
    # 'near' search) has its own weaker forms (text.relax_text_condition).
    if node.find(exp.Not) is not None or _DISTANCE_RE.search(every_word):
        return None
    words, core_words = _positive_words(every_word), _positive_words(core)
    if not core_words or not core_words < words:
        return None
    return ClauseEvidence(kind="text", alias=alias, sql=every_word,
                          confirming=confirming_evidence(node.left, alias), evidence=(
        # The core only in a paper: the rows the core's own paper form had before.
        *_paper_only(core, papers),
        CoreOfTerm(core, missing=tuple(sorted(words - core_words))),
    ))


def _paper_only(sql: str, papers: bool) -> Tuple[PieceOfEvidence, ...]:
    """The evidence that the text searches of *sql* match only in a paper;
    none without paper text or without a positive text search."""
    searches = paper_searches(sql) if papers else ()
    return (PaperOnly(searches),) if searches else ()


def classify(node, alias: str, papers: bool = False) -> Optional[ClauseEvidence]:
    """A ClauseEvidence for a condition that may be uncertain, else None.

    *papers*: the database has paper text and it may serve as evidence.
    """
    if isinstance(node, exp.Paren):
        node = node.this
    # Whether the condition may be uncertain is decided in soft_kind() alone;
    # from here on only its weaker forms are built.
    kind = soft_kind(node, alias)
    if kind is None:
        return None
    sql = node.sql(dialect="postgres")
    if kind == "sidecar":
        return sidecar(node, alias)
    confirming = confirming_evidence(node, alias)
    if kind == "patient_count":
        return _patient_count(sql, alias, confirming)
    if kind == "text":
        loose = loose_term(node, alias, papers)
        if loose is not None:
            return loose
    # Part of a text search the condition holds, anywhere in the text.
    relaxed = relax_text_condition(sql)
    weaker_search: Tuple[PieceOfEvidence, ...] = (relaxed,) if relaxed else ()
    paper = _paper_only(sql, papers)
    if kind != "text":
        # A field a dataset may not record. What is particular to the field
        # is in its file (fields/); the steps are the same for all.
        field = fields.BY_KIND[kind]
        # The dataset's own text names the asked value: counts in place of a
        # part of the search.
        named = field.named_in_text(node) if field.named_in_text else None
        weaker: Tuple[PieceOfEvidence, ...] = (named,) if named is not None else weaker_search
        if field.paper_evidence:
            # Without it a paper-only match is 'possible' like any other
            # dataset that records nothing.
            weaker = paper + weaker
        record = OtherValuesRecorded(field.recorded)
        not_recorded = NotRecorded(field.label, field.unrecorded_score)
        evidence = ((*weaker, record, not_recorded) if field.text_beats_recorded
                    else (record, *weaker, not_recorded))
        return ClauseEvidence(kind=kind, alias=alias, sql=sql, evidence=evidence, confirming=confirming)
    if relaxed or paper:
        return ClauseEvidence(kind="text", alias=alias, sql=sql, evidence=(*paper, *weaker_search),
                              confirming=confirming)
    return None


@dataclass(frozen=True)
class PatientsNotCounted(PieceOfEvidence):
    """"At least *minimum* participants with one of *codes*": the dataset
    does not record who has the diagnosis and is large enough to hold them."""
    codes: Tuple[str, ...]
    minimum: int
    stored_subject_counts: bool   # bids_datasets.n_subjects is filled
    source: Source = dataclasses.field(default=Source.NOT_RECORDED, kw_only=True)
    strength: Strength = dataclasses.field(default=Strength.PROBABLE, kw_only=True)

    def sql(self, alias: str) -> Optional[str]:
        import cohort_count
        from services.schema.tables import datasets
        stored = (f"COALESCE({alias}.{datasets.c.n_subjects.name}, 0)"
                  if self.stored_subject_counts else None)
        return cohort_count.unknown_sql(list(self.codes), self.minimum, alias, subject_count=stored)

    def reason(self) -> str:
        return "how many participants have the diagnosis is not recorded"


def _patient_count(sql: str, alias: str,
                   confirming: Tuple[ConfirmedMatch, ...] = ()) -> Optional[ClauseEvidence]:
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
    return ClauseEvidence(kind="patient_count", alias=alias, sql=sql, confirming=confirming, evidence=(
        PatientsNotCounted(tuple(codes), minimum, stored),
    ))
