"""
services/evidence/contradicted.py
---------------------------------
Evidence that a condition does not hold: the dataset records what it asks
about, with other values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from services.evidence.base import PieceOfEvidence, Source, Strength


@dataclass(frozen=True)
class ContradictedByRecordedValuesEvidence(PieceOfEvidence):
    """Contradicted: the dataset records the field with values that do not
    fit the condition.

    Its place in a condition's list says what a record outweighs. Evidence
    listed after it counts only for a dataset that records nothing; evidence
    listed before it counts although other values are recorded (so far only
    the text evidence for a diagnosis: coded diagnoses are often incomplete).
    """
    recorded: str   # SQL: the dataset records the field at all
    source: Source = field(default=Source.FIELD, kw_only=True)
    strength: Strength = field(default=Strength.CONTRADICTED, kw_only=True)

    def sql(self) -> Optional[str]:
        return self.recorded
