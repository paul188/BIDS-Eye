"""
services/evidence/base.py
-------------------------
Evidence: one way a dataset can meet a condition of a search. What the kinds
of evidence are and who uses them: services/evidence/__init__.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


# ── Where evidence is found ───────────────────────────────────────────────────

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


# ── What evidence proves ──────────────────────────────────────────────────────

class Strength(IntEnum):
    """What one piece of evidence says about a condition, strongest first.

    The numbers are the values of the uncertain_level column
    (services/uncertain/tier.py) and sort results: lower comes first.
    """
    CONFIRMED = 0      # the condition holds
    PROBABLE = 1       # a weaker form holds
    POSSIBLE = 2       # the field is not recorded and nothing hints either way
    CONTRADICTED = 9   # the dataset records the field and it does not fit


# ── One piece of evidence ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evidence:
    """One way a dataset can meet a condition.

    The subclasses set the source and strength of their kind and add what
    only they have (confirmed.py: the participants counted, the value of a
    column). A kind that needs neither is an Evidence itself.
    """
    # SQL that is true for a dataset this evidence holds for. None: it holds
    # for whatever the evidence listed before it left over (weaker.NotRecorded).
    sql: Optional[str] = None
    # What the result card says when this evidence decides; "" says nothing.
    # The wordings: reason_texts.py.
    reason: str = ""
    source: str = field(kw_only=True)
    strength: Strength = field(kw_only=True)

    # ── Evidence that shows numbers or values on the card (confirmed.py) ──────

    def read_columns(self) -> Tuple[str, ...]:
        """The dataset columns whose values the card text shows."""
        return ()

    def number_columns(self, name: str) -> List[str]:
        """SQL columns for the numbers the card text shows, named after *name*."""
        return []

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        """The card text for one dataset. *row* holds the columns asked for
        above, *total* the dataset's number of participants."""
        return self.reason or None

    @property
    def key(self) -> Tuple[str, Tuple[str, ...]]:
        """Two pieces of evidence with one key are the same way of matching."""
        return self.source, self.read_columns()
