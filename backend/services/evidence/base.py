"""
services/evidence/base.py
-------------------------
PieceOfEvidence: one way a dataset can meet a condition of a search. What the kinds
of evidence are and who uses them: services/evidence/__init__.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


# ── Where evidence is found ───────────────────────────────────────────────────

class SourceGroup(str, Enum):
    """Sources that repeat each other: what the title says the README says too."""
    RECORDS = "records"     # the participant list, the dataset's columns, its files
    OWN_TEXT = "own_text"   # title, description and README, written by the dataset's authors
    PAPER = "paper"         # a paper the dataset names


class Source(str, Enum):
    """Where a piece of evidence was found."""
    PARTICIPANTS = "participants"  # rows of the participant list
    FIELD = "field"                # the dataset's own columns
    FILES = "files"
    DERIVATIVES = "derivatives"
    SITES = "sites"
    SHORT_TEXT = "short_text"      # name or description only (ILIKE, or search_tsv words labelled :AB)
    TEXT = "text"                  # search_tsv: name, description, study fields, README
    README = "readme"
    PAPERS = "papers"              # a paper the dataset names
    ASSUMED = "assumed"            # the column is empty and its default holds
    ASSUMED_ABSENT = "assumed_absent"  # an excluded diagnosis, and the dataset records none
    NOT_RECORDED = "not_recorded"  # the dataset does not record the field

    def __str__(self) -> str:
        # The value, on every Python version (format() of a str enum changed in 3.11).
        return self.value

    @property
    def group(self) -> SourceGroup:
        """Whose statement it is. Evidence of one group says one thing once
        (services/evidence/SCORING.md)."""
        group = _GROUPS.get(self)
        if group is None:
            raise ValueError(f"the source {self} has no source group")
        return group


# Not listed: ASSUMED, ASSUMED_ABSENT and NOT_RECORDED stand for an empty
# column, which states nothing.
_GROUPS = {
    Source.PARTICIPANTS: SourceGroup.RECORDS,
    Source.FIELD: SourceGroup.RECORDS,
    Source.FILES: SourceGroup.RECORDS,
    Source.DERIVATIVES: SourceGroup.RECORDS,
    Source.SITES: SourceGroup.RECORDS,
    Source.SHORT_TEXT: SourceGroup.OWN_TEXT,
    Source.TEXT: SourceGroup.OWN_TEXT,
    Source.README: SourceGroup.OWN_TEXT,
    Source.PAPERS: SourceGroup.PAPER,
}


# ── What evidence proves ──────────────────────────────────────────────────────

class Strength(IntEnum):
    """What the evidence says about one AND condition of a search. A dataset
    is as strong as its weakest condition (the uncertain_level column).

    The numbers are an order, best first, not scores. Cards and benchmarks
    read 0 to 2: do not renumber them."""
    CONFIRMED = 0      # the condition holds
    PROBABLE = 1       # a weaker form of it holds
    POSSIBLE = 2       # what it asks about is not recorded, and nothing hints either way
    CONTRADICTED = 3   # recorded, and it does not fit


# The score of a piece of evidence whose kind has not been measured
# (PieceOfEvidence.score). Hand-set, not measured: probable lies between the
# two measured probable kinds (0.37 and 0.53), possible between the two
# measured unrecorded fields (0.04 and 0.33).
DEFAULT_SCORES = {
    Strength.CONFIRMED: 1.0,
    Strength.PROBABLE: 0.4,
    Strength.POSSIBLE: 0.1,
    Strength.CONTRADICTED: 0.0,
}


# ── One piece of evidence ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PieceOfEvidence(ABC):
    """One way a dataset can meet a condition.

    A kind of evidence is a subclass. It holds what it is about (the words
    missing from a search, the sex a text names, the participant rows a
    branch counts) and writes its SQL and its card text from that. It is
    built for one query: a kind that writes its SQL holds the alias that
    query gives the dataset row.
    """
    source: Source = field(kw_only=True)
    strength: Strength = field(kw_only=True)

    @abstractmethod
    def sql(self) -> Optional[str]:
        """SQL that is true for the dataset rows this evidence holds for.
        None: it holds for whatever the evidence listed before it left over
        (possible.NotRecorded)."""

    def reason(self) -> str:
        """What the result card says when this evidence decides; "" says nothing."""
        return ""

    def score(self) -> float:
        """How much a condition met this way counts, from 0 to 1: the share
        of such datasets that are relevant, relative to a condition that
        holds as generated. A kind that was measured says so itself;
        the others have the default of their strength."""
        return DEFAULT_SCORES[self.strength]
