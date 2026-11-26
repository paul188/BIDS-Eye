"""
services/evidence/possible.py
-----------------------------
Evidence that a condition possibly holds: the dataset does not record what
the condition asks about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from services.evidence.base import PieceOfEvidence, Source, Strength


@dataclass(frozen=True)
class NotRecordedEvidence(PieceOfEvidence):
    """Possible: the field is not recorded and nothing hints either way:
    "ages not recorded".

    The last entry of a list: what is left when no other evidence holds. It
    keeps a dataset only while the 'possible' level is on
    (conditions.possible_enabled).
    """
    what: str   # the field as a card names it: "ages", "participant sex"
    # The score where it was measured for the field (SoftField.unrecorded_score).
    measured_score: Optional[float] = None
    source: Source = field(default=Source.NOT_RECORDED, kw_only=True)
    strength: Strength = field(default=Strength.POSSIBLE, kw_only=True)

    def sql(self) -> Optional[str]:
        return None

    def reason(self) -> str:
        return f"{self.what} not recorded"

    def score(self) -> float:
        return super().score() if self.measured_score is None else self.measured_score
