"""
services/evidence/base.py
-------------------------
Evidence: one way a dataset can meet a condition of a search. What the kinds
of evidence are and who uses them: services/evidence/__init__.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


# ── Where evidence is found ───────────────────────────────────────────────────

class Source(str, Enum):
    """Where a piece of evidence was found."""
    CONDITION = "condition"        # the condition as the search wrote it, whatever it reads
    PARTICIPANTS = "participants"  # rows of the participant list
    FIELD = "field"                # the dataset's own columns
    FILES = "files"
    DERIVATIVES = "derivatives"
    SITES = "sites"
    SHORT_TEXT = "short_text"      # name or description only (ILIKE, or search_tsv words labelled :AB)
    TEXT = "text"                  # search_tsv: name, description, study fields, README
    README = "readme"
    PAPER = "paper"                # a paper the dataset names
    ASSUMED = "assumed"            # the column is empty and its default holds
    ASSUMED_ABSENT = "assumed_absent"  # an excluded diagnosis, and the dataset records none
    NOT_RECORDED = "not_recorded"  # the dataset does not record the field

    def __str__(self) -> str:
        # The value, on every Python version (format() of a str enum changed in 3.11).
        return self.value


# ── What evidence proves ──────────────────────────────────────────────────────

class Strength(IntEnum):
    """What the evidence says about one AND condition of a search. A dataset
    is as strong as its weakest condition (the uncertain_level column).

    The numbers leave the backend in that column (the result cards and the
    benchmarks read them): do not renumber."""
    CONFIRMED = 0      # the condition holds
    PROBABLE = 1       # a weaker form of it holds
    POSSIBLE = 2       # what it asks about is not recorded, and nothing hints either way
    CONTRADICTED = 9   # recorded, and it does not fit


# ── One piece of evidence ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evidence(ABC):
    """One way a dataset can meet a condition.

    A kind of evidence is a subclass. It holds what it is about (the words
    missing from a search, the sex a text names, the participant rows a
    branch counts) and writes its SQL and its card text from that.
    """
    source: Source = field(kw_only=True)
    strength: Strength = field(kw_only=True)

    @abstractmethod
    def sql(self, alias: str) -> Optional[str]:
        """SQL that is true for the dataset row *alias* this evidence holds
        for. None: it holds for whatever the evidence listed before it left
        over (possible.NotRecorded)."""

    def reason(self) -> str:
        """What the result card says when this evidence decides; "" says nothing."""
        return ""
