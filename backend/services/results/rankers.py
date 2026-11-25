"""
services/results/rankers.py
---------------------------
The order of the results. A Ranker turns what a result row carries into the
sort key of the page; services/translate/sql_rewriter.build_page_sql writes
that key as the ORDER BY. The rewriter decides nothing about the order
itself, so another ranker is one class here and one argument there
(build_page_sql(..., ranker=...), or DEFAULT_RANKER for every search).

A ranker returns a sort key, not a comparison of two results: the page is cut
in the database (LIMIT/OFFSET), so the order has to be one the database can
compute for every row, and the same on every page.

What a row carries (ResultColumns):
  weakest_evidence   the weakest evidence any condition has for the dataset
                     (evidence.base.Strength; services/uncertain/tier.py)
  weakest_evidence_for(kind)
                     the same among the conditions of one kind ("age",
                     "diagnosis", "text", …), to tell apart which condition
                     a dataset is unsure about
  evidence_score     0 to 1: the scores of the query's conditions multiplied,
                     each the score of the piece of evidence that decided it
                     (services/uncertain/conditions.QueryEvidence)
  judged_order       where the language model's judgment of the first
                     candidates put it (services/results/rerank.py)
  other_words_only   it matches only the words a search was widened with,
                     not the user's own (services/fulltext/ranking.py)
  score              the matched vocabulary codes plus the text rank
  subjects           its number of subjects
  dataset_id         last in every key, so no two rows tie

A ranker that needs more than these (a learned score, say) adds a column
where the page query is built and names it in ResultColumns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class SortTerm:
    """One part of a sort key: an SQL expression and its direction."""
    sql: str
    descending: bool = False
    nulls_last: bool = False

    def render(self) -> str:
        return self.sql + (" DESC" if self.descending else "") + (" NULLS LAST" if self.nulls_last else "")


@dataclass(frozen=True)
class ResultColumns:
    """What the rows of one result carry, as sort terms a ranker may use.

    The flags say which of them this result has; a ranker asks before it
    uses one (a search without text has no scores, a result without an
    uncertain tier no weakest evidence).
    """
    row: str                      # SQL alias of the result row
    has_evidence: bool = False    # the uncertain tier ran
    has_judgment: bool = False    # the first candidates were judged
    has_other_words: bool = False  # the search was widened with other words
    has_scores: bool = False
    has_subjects: bool = False
    # (kind, column) for each kind of condition the search has
    # (ClauseEvidence.kind; services/uncertain/tier.kind_level_columns).
    evidence_columns: Tuple[Tuple[str, str], ...] = ()

    @property
    def weakest_evidence(self) -> SortTerm:
        """Confirmed results first, then probable, then possible."""
        return SortTerm(f"{self.row}.uncertain_level")

    @property
    def evidence_score(self) -> SortTerm:
        """The dataset whose evidence counts for more first."""
        return SortTerm(f"{self.row}.evidence_score", descending=True)

    def weakest_evidence_for(self, kind: str) -> Optional[SortTerm]:
        """Datasets whose conditions of *kind* are confirmed first; None when
        the search has no condition of that kind."""
        column = dict(self.evidence_columns).get(kind)
        return SortTerm(f"{self.row}.{column}") if column else None

    @property
    def judged_order(self) -> Tuple[SortTerm, ...]:
        """The datasets judged relevant first, in the judged order; those
        judged irrelevant last; all others between, in the order of the
        terms that follow."""
        return (
            SortTerm(f"CASE WHEN {self.row}.id = ANY(CAST(:_rr_top AS uuid[])) THEN 0 "
                     f"WHEN {self.row}.id = ANY(CAST(:_rr_low AS uuid[])) THEN 2 ELSE 1 END"),
            SortTerm(f"array_position(CAST(:_rr_top AS uuid[]), {self.row}.id)"),
        )

    @property
    def other_words_only(self) -> SortTerm:
        """Datasets with the user's own words before those that have only
        the words the search was widened with."""
        return SortTerm(f"{self.row}.other_words_only")

    @property
    def score(self) -> SortTerm:
        return SortTerm(f"{self.row}.relevance_score + {self.row}.text_rank", descending=True)

    @property
    def subjects(self) -> SortTerm:
        return SortTerm(f"{self.row}.subject_count", descending=True, nulls_last=True)

    @property
    def dataset_id(self) -> SortTerm:
        return SortTerm(f"{self.row}.id")


class Ranker(Protocol):
    """Decides the order of a result."""

    def sort_key(self, columns: ResultColumns) -> Sequence[SortTerm]:
        """The sort key, most important term first. The dataset id is added
        at the end when the key does not end with it (order_by)."""


class EvidenceFirst:
    """The order of every search so far.

    1. The strength of the evidence: a result every condition confirms
       comes before one that is only probable.
    2. The language model's judgment of the first candidates.
    3. The user's own words before words the search was widened with.
    4. The score: matched vocabulary codes plus the text rank.
    5. The larger dataset.
    """

    def sort_key(self, columns: ResultColumns) -> List[SortTerm]:
        key: List[SortTerm] = []
        if columns.has_evidence:
            key.append(columns.weakest_evidence)
        if columns.has_judgment:
            key.extend(columns.judged_order)
        if columns.has_other_words:
            key.append(columns.other_words_only)
        if columns.has_scores:
            key.append(columns.score)
        if columns.has_subjects:
            key.append(columns.subjects)
        return key


# The ranker of every search that names no other.
DEFAULT_RANKER: Ranker = EvidenceFirst()


def order_by(ranker: Ranker, columns: ResultColumns) -> str:
    """The ORDER BY clause *ranker* gives a result with *columns*.

    Always ends with the dataset id: LIMIT/OFFSET over a tied result can
    repeat or skip rows between pages.
    """
    key = list(ranker.sort_key(columns))
    if not key or key[-1] != columns.dataset_id:
        key.append(columns.dataset_id)
    return "ORDER BY " + ", ".join(term.render() for term in key)
