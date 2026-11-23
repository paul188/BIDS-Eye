"""
services/evidence/weaker.py
---------------------------
The evidence of the uncertain tier: what can be said about a dataset that a
condition does not hold for (services/uncertain/). A condition lists them in
the order they count (conditions.SoftCondition); the first that holds for a
dataset gives the condition's state and its card text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from services.evidence import reason_texts
from services.evidence.base import Evidence, EvidenceSource, Strength


@dataclass(frozen=True)
class AsAsked(Evidence):
    """The condition holds as the search wrote it."""
    source: EvidenceSource = field(default=EvidenceSource.CONDITION, kw_only=True)
    strength: Strength = field(default=Strength.CONFIRMED, kw_only=True)


@dataclass(frozen=True)
class PaperOnly(Evidence):
    """The search words match a paper the dataset names, not its own text."""
    reason: str = reason_texts.PAPER
    source: EvidenceSource = field(default=EvidenceSource.PAPER, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)


@dataclass(frozen=True)
class WeakerSearch(Evidence):
    """A text search of the condition matches in a weaker form: some of its
    words, the words apart, only further down in the README
    (services/uncertain/text.py)."""
    source: EvidenceSource = field(default=EvidenceSource.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)


@dataclass(frozen=True)
class NamedInText(Evidence):
    """The dataset's own text names the value the condition asks of a field:
    "22 women", "infants", "7T" (services/uncertain/fields/, scanner.py)."""
    source: EvidenceSource = field(default=EvidenceSource.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)


@dataclass(frozen=True)
class OtherValuesRecorded(Evidence):
    """The dataset records the field, and the condition does not hold: a
    contradiction. *sql* is true when the field is recorded.

    Its place in a condition's list says what a record outweighs. Evidence
    listed after it counts only for a dataset that records nothing; evidence
    listed before it counts although other values are recorded (so far only
    the text evidence for a diagnosis: coded diagnoses are often incomplete).
    """
    source: EvidenceSource = field(default=EvidenceSource.FIELD, kw_only=True)
    strength: Strength = field(default=Strength.CONTRADICTED, kw_only=True)


@dataclass(frozen=True)
class NotRecorded(Evidence):
    """The dataset does not record the field.

    Without *sql* it is the last entry of a list: what is left when nothing
    hints either way. It keeps a dataset only while the 'possible' level is
    on (conditions.possible_enabled). With *sql* and a strength of its own it
    is a field known to be missing where that alone is a hint (the patient
    count of a dataset large enough to hold the patients).
    """
    sql: Optional[str] = None
    source: EvidenceSource = field(default=EvidenceSource.NOT_RECORDED, kw_only=True)
    strength: Strength = field(default=Strength.POSSIBLE, kw_only=True)
