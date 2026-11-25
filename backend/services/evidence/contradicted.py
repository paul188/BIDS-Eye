"""
services/evidence/contradicted.py
---------------------------------
Evidence that a condition does not hold: the dataset records what it asks
about, with other values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from services.evidence.base import Evidence, Source, Strength


@dataclass(frozen=True)
class OtherValuesRecorded(Evidence):
    """The dataset records the field, and the condition does not hold: a
    contradiction.

    Its place in a condition's list says what a record outweighs. Evidence
    listed after it counts only for a dataset that records nothing; evidence
    listed before it counts although other values are recorded (so far only
    the text evidence for a diagnosis: coded diagnoses are often incomplete).
    """
    recorded: Callable[[str], str]   # alias → SQL: the dataset records the field at all
    source: Source = field(default=Source.FIELD, kw_only=True)
    strength: Strength = field(default=Strength.CONTRADICTED, kw_only=True)

    def sql(self, alias: str) -> Optional[str]:
        return self.recorded(alias)
