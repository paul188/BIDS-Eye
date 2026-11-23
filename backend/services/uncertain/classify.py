"""
services/uncertain/classify.py
----------------------------------
Which states a top-level AND condition can take (classify): text, ages,
diagnosis, sex, scanner fields, matches only in a paper the dataset names
(paper_atoms), and "at least N participants with a diagnosis"
(patient_count).
"""

from __future__ import annotations

from typing import Optional

try:  # sqlglot is a backend dependency; keep the module importable without it
    from sqlglot import exp
except ImportError:  # pragma: no cover
    exp = None  # type: ignore[assignment]

from services.uncertain.conditions import LABEL_RE, SoftCondition, TSQUERY_RE
from services.uncertain.text import (
    age_group_words,
    asked_age_range,
    known_sql,
    negated_spans,
    partial_reason,
    relax_text_condition,
    tsquery_for_words,
)
from services.uncertain import reason_texts
from services.uncertain.sex import sex_text_condition
from services.uncertain.scanner import sidecar


def _columns_outside_subqueries(node) -> set:
    found = set()
    for column in node.find_all(exp.Column):
        if column.find_ancestor(exp.Subquery, exp.Exists) is None:
            found.add(column.table or "")
    return found


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


def classify(node, alias: str, papers: bool = False) -> Optional[SoftCondition]:
    """A SoftCondition for a condition that may be uncertain, else None.

    *papers*: the database has paper text and it may serve as evidence.
    """
    if isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Not):
        return None
    sql = node.sql(dialect="postgres")
    columns = {column.name.lower() for column in node.find_all(exp.Column)}
    # Sidecar values (field strength, echo times …) are recorded with the
    # files. Field strength and scanner make can be missing there and named
    # in the README instead (sidecar); the rest stay hard filters.
    if "other_entities" in columns:
        return sidecar(node, alias)
    # A resolved institution filter is evidence from the scanner headers. Its
    # weaker form would be the name anywhere in the README or a paper, and
    # those mentions are mostly suppliers, software and citations ('kindly
    # provided by … Harvard Medical School', surfer.nmr.mgh.harvard.edu).
    # Species likewise: the README of a human study says 'mouse click', and
    # its title/description fallback for unlabelled datasets is already
    # part of the condition.
    if columns & {"institutions", "species"}:
        return None
    # A condition on the outer join's rows (o.datatype …) is about files.
    if _columns_outside_subqueries(node) - {alias, ""}:
        return None
    counted = _patient_count(sql, alias)
    if counted is not None:
        return counted
    kinds = []
    if columns & {"age", "age_min", "age_max"}:
        kinds.append("age")
    if columns & {"diagnosis", "dx_status"}:
        kinds.append("diagnosis")
    if "sex" in columns:
        kinds.append("sex")
    if len(kinds) > 1:
        return None
    relaxed, words = relax_text_condition(sql)
    paper = paper_atoms(sql, alias) if papers else None
    if kinds:
        kind = kinds[0]
        probable = relaxed
        reason_probable = partial_reason(words) if relaxed else ""
        if kind == "age":
            group_words = age_group_words(*asked_age_range(sql))
            paper = None
            if group_words:
                # The dataset's own text only. Of 63 datasets without ages whose
                # only child word was in a paper (live database, 2026-11-17),
                # about 9 studied children; the rest were adult studies citing
                # child work (Sherlock, BCIT) and two mouse datasets. Paper-only
                # matches fall to 'possible' instead.
                vector = "search_tsv"
                probable = (f"{alias}.{vector} @@ TO_TSQUERY('english', "
                            f"'{tsquery_for_words(group_words)}')")
                reason_probable = reason_texts.AGE_GROUP_NAMED
        if kind == "sex":
            text = sex_text_condition(node, alias)
            if text is not None:
                probable, reason_probable = text
        # The card text for each state. Possible: the field is not recorded
        # ("ages not recorded"). Probable: the more specific reason found
        # above (the text names the age group or sex, or matches in part),
        # else the same "not recorded".
        return SoftCondition(
            sql=sql, kind=kind, probable=probable, known=known_sql(kind, alias),
            reason_probable=reason_probable or reason_texts.NOT_RECORDED[kind],
            reason_possible=reason_texts.NOT_RECORDED[kind], paper=paper,
        )
    if relaxed or paper:
        return SoftCondition(
            sql=sql, kind="text", probable=relaxed, known=None,
            reason_probable=partial_reason(words) if relaxed else "",
            reason_possible="", paper=paper,
        )
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
        from services.catalogue.catalogue import catalogue
        from services.schema.tables import datasets
        if catalogue.summaries.subject_counts:
            stored = f"COALESCE({alias}.{datasets.c.n_subjects.name}, 0)"
    except ImportError:
        pass
    return SoftCondition(
        sql=sql, kind="patient_count",
        probable=cohort_count.unknown_sql(codes, minimum, alias, subject_count=stored), known=None,
        reason_probable=reason_texts.PATIENT_COUNT_NOT_RECORDED, reason_possible="",
    )
