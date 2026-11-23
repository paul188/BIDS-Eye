"""
services/uncertain/classify.py
----------------------------------
Which evidence a top-level AND condition can have (classify): text, ages,
diagnosis, sex, scanner fields, matches only in a paper the dataset names
(paper_atoms), and "at least N participants with a diagnosis"
(patient_count).
"""

from __future__ import annotations

from typing import Optional, Tuple

try:  # sqlglot is a backend dependency; keep the module importable without it
    from sqlglot import exp
except ImportError:  # pragma: no cover
    exp = None  # type: ignore[assignment]

from services.uncertain.conditions import LABEL_RE, SoftCondition, TSQUERY_RE
from services.uncertain.text import (
    negated_spans,
    partial_reason,
    relax_text_condition,
)
from services.evidence import reason_texts
from services.evidence.base import Evidence, Strength
from services.evidence.weaker import (AsAsked, NamedInText, NotRecorded, OtherValuesRecorded,
                                      PaperOnly, WeakerSearch)
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


def paper_atoms(sql: str, alias: str) -> Optional[str]:
    """The condition's positive text searches run on the paper vector, ORed.

    Labels are dropped: paper text carries the lowest weight, so a search
    scoped to the name and short fields could never reach it.
    """
    from services.fulltext.vectors import (EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN, FTS_COLUMN,
                                           PAPER_FTS_COLUMN)
    spans = negated_spans(sql)
    atoms = []
    for match in TSQUERY_RE.finditer(sql):
        if match.group("negated") or any(s < match.start() < e for s, e in spans):
            continue
        query = LABEL_RE.sub("", match.group("query"))
        paper = f"{alias}.{PAPER_FTS_COLUMN} @@ TO_TSQUERY('english', '{query}')"
        own = f"{alias}.{FTS_COLUMN} @@ TO_TSQUERY('english', '{query}')"
        if match.group("guard"):
            # Exact word forms count in the paper as in the dataset's text.
            guard = LABEL_RE.sub("", match.group("guard"))
            paper = f"({paper} AND {alias}.{EXACT_PAPER_FTS_COLUMN} @@ TO_TSQUERY('simple', '{guard}'))"
            own = f"({own} AND {alias}.{EXACT_FTS_COLUMN} @@ TO_TSQUERY('simple', '{guard}'))"
        # The paper vector also holds the dataset's own text; "only in a
        # paper" means it matches there and not in the dataset's text.
        atoms.append(f"({paper} AND NOT {own})")
    return " OR ".join(dict.fromkeys(atoms)) or None


def _paper_only(paper: Optional[str]) -> Tuple[Evidence, ...]:
    return (PaperOnly(paper),) if paper else ()


def classify(node, alias: str, papers: bool = False) -> Optional[SoftCondition]:
    """A SoftCondition for a condition that may be uncertain, else None.

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
    if kind == "patient_count":
        return _patient_count(sql, alias)
    relaxed, words = relax_text_condition(sql)
    # Part of a text search the condition holds, anywhere in the text.
    weaker_search = (WeakerSearch(relaxed, partial_reason(words)),) if relaxed else ()
    paper = paper_atoms(sql, alias) if papers else None
    if kind != "text":
        # A field a dataset may not record. What is particular to the field
        # is in its file (fields/); the steps are the same for all.
        field = fields.BY_KIND[kind]
        # The dataset's own text names the asked value: counts in place of a
        # part of the search.
        named = field.named_in_text(node, alias) if field.named_in_text else None
        weaker: Tuple[Evidence, ...] = (NamedInText(*named),) if named is not None else weaker_search
        if field.paper_evidence:
            # Without it a paper-only match is 'possible' like any other
            # dataset that records nothing.
            weaker = _paper_only(paper) + weaker
        record = OtherValuesRecorded(field.recorded(alias))
        not_recorded = NotRecorded(reason=reason_texts.NOT_RECORDED[kind])
        if field.text_beats_recorded:
            return SoftCondition(kind=kind, evidence=(AsAsked(sql), *weaker, record, not_recorded))
        return SoftCondition(kind=kind, evidence=(AsAsked(sql), record, *weaker, not_recorded))
    if relaxed or paper:
        return SoftCondition(kind="text", evidence=(
            AsAsked(sql), *_paper_only(paper), *weaker_search))
    return None


def _patient_count(sql: str, alias: str) -> Optional[SoftCondition]:
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
    stored = None
    try:
        from services.database_stats.database_stats import database_stats
        from services.schema.tables import datasets
        if database_stats.summaries.subject_counts:
            stored = f"COALESCE({alias}.{datasets.c.n_subjects.name}, 0)"
    except ImportError:
        pass
    # Not recorded, in a dataset large enough to hold the patients.
    unknown = cohort_count.unknown_sql(codes, minimum, alias, subject_count=stored)
    return SoftCondition(kind="patient_count", evidence=(
        AsAsked(sql),
        NotRecorded(unknown, reason_texts.PATIENT_COUNT_NOT_RECORDED, strength=Strength.PROBABLE),
    ))
