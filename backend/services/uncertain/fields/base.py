"""
services/uncertain/fields/base.py
-------------------------------------
What every field a dataset may not record has in common: SoftField.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from services.evidence.probable import NamedInOwnText
from services.schema.tables import participants as _participants

# (condition, alias) → the evidence that a dataset's own text names the value the
# condition asks for. None when it asks for nothing a text could name.
TextForm = Callable[[object, str], Optional[NamedInOwnText]]


@dataclass(frozen=True)
class SoftField:
    """A field a dataset may not record, and what the uncertain tier does then.

    A condition on such a field is confirmed when the recorded values fit,
    contradicted when recorded values do not fit, and uncertain when nothing
    is recorded. The fields differ only in what is written here.
    """
    # Names the field in ClauseEvidence.kind.
    kind: str
    # How a result card names it: "ages not recorded".
    label: str
    # A condition that reads one of these columns is a condition on the field.
    columns: Tuple[str, ...]
    # alias → SQL: true when the dataset records the field at all.
    recorded: Callable[[str], str]
    # How the dataset's own text names the asked value ("22 women", "infants").
    # None: only the words of a text search in the same condition count.
    named_in_text: Optional[TextForm] = None
    # A match only in a paper the dataset names counts as evidence.
    paper_evidence: bool = True
    # Text evidence counts although the field is recorded with other values.
    text_beats_recorded: bool = False
    # PieceOfEvidence.score of a dataset that records nothing and gives no
    # hint, where it was measured; None: the default of that strength.
    unrecorded_score: Optional[float] = None


def participant_rows_record(column: str) -> Callable[[str], str]:
    """recorded= for a participant column: some row of the dataset has a value."""
    def recorded(alias: str) -> str:
        return (f"EXISTS(SELECT 1 FROM {_participants.name} AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND uk.{column} IS NOT NULL)")
    return recorded
