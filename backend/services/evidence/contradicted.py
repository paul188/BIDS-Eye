"""
services/evidence/contradicted.py
---------------------------------
Evidence that a condition does not hold: the dataset records what it asks
about, with other values. One kind per place the record is kept, as for
confirmed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from services.evidence.base import Source, Strength, UncertainEvidence


@dataclass(frozen=True)
class ContradictedEvidence(UncertainEvidence):
    """Contradicted: the dataset records the field, and the condition did not
    hold for it, so the recorded values do not fit. Base of the contradicted
    kinds; each says where the record is kept.

    Its place in a condition's list says what a record outweighs. Evidence
    listed after it counts only for a dataset that records nothing; evidence
    listed before it counts although other values are recorded (so far only
    the text evidence for a diagnosis: coded diagnoses are often incomplete).
    A field kept in two places has two, listed together.
    """
    recorded: str   # SQL: the dataset records the field here
    strength: Strength = field(default=Strength.CONTRADICTED, kw_only=True)

    def sql(self) -> Optional[str]:
        return self.recorded

    def reason(self) -> str:
        # A contradicted dataset is left out, so no card shows it.
        return ""


@dataclass(frozen=True)
class ContradictedInParticipantListEvidence(ContradictedEvidence):
    """Contradicted by the participant list: it records the field (sex,
    diagnosis, age) with other values."""
    source: Source = field(default=Source.PARTICIPANTS, kw_only=True)


@dataclass(frozen=True)
class ContradictedInDatasetColumnEvidence(ContradictedEvidence):
    """Contradicted by the dataset's own columns: age_min/age_max hold another range."""
    source: Source = field(default=Source.FIELD, kw_only=True)


@dataclass(frozen=True)
class ContradictedInFileTablesEvidence(ContradictedEvidence):
    """Contradicted by the dataset's file metadata: a sidecar records another
    scanner value (field strength, manufacturer, model)."""
    source: Source = field(default=Source.FILES, kw_only=True)
