"""
services/evidence/possible.py
-----------------------------
Evidence that a condition possibly holds: the dataset does not record what
the condition asks about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from services.evidence.base import Evidence, Source, Strength


@dataclass(frozen=True)
class NotRecorded(Evidence):
    """The dataset does not record the field, and nothing hints either way.

    The last entry of a list: what is left when no other evidence holds. It
    keeps a dataset only while the 'possible' level is on
    (conditions.possible_enabled).
    """
    what: str   # the field as a card names it: "ages", "participant sex"
    source: Source = field(default=Source.NOT_RECORDED, kw_only=True)
    strength: Strength = field(default=Strength.POSSIBLE, kw_only=True)

    def sql(self, alias: str) -> Optional[str]:
        return None

    def reason(self) -> str:
        return f"{self.what} not recorded"
