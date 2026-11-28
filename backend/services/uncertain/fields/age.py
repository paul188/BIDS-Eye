"""
services/uncertain/fields/age.py
------------------------------------
Participant ages as a field a dataset may not record (FIELD, at the end).
About half the datasets record none (tier.py).

Its text form: the dataset's own text names the age group the condition asks
for ("infants", "older adults").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from services import asked
from services.evidence.contradicted import ContradictedEvidence, ContradictedInDatasetColumnEvidence
from services.evidence.possible import PossibleNotRecordedEvidence
from services.evidence.probable import ProbableNamedInTextEvidence
from services.uncertain.conditions import text_from_dataset_matches
from services.uncertain.fields.base import SoftField, participant_rows_record
import text_evidence  # backend/vocabulary/ is on the path wherever the backend runs


LABEL = "ages"   # how a result card names the field

def age_group_words(low: float, high: float) -> List[str]:
    """README words for the age groups that overlap [low, high]."""
    if low <= 0 and high >= 120:
        return []
    words: List[str] = []
    for group_low, group_high, group_words in text_evidence.age_groups():
        # The group must cover a good part of the asked range, or the asked
        # range a good part of the group: "60 and over" is not about adults
        # in general, "18 to 65" not about older adults.
        overlap = min(group_high, high) - max(group_low, low)
        needed = min((high - low) / 2, (group_high - group_low) / 2)
        if overlap > 0 and overlap >= needed:
            words.extend(group_words)
    return list(dict.fromkeys(words))


def tsquery_for_words(words: Sequence[str]) -> str:
    parts = []
    for word in words:
        pieces = word.split("-")
        parts.append("(" + " <-> ".join(pieces) + ")" if len(pieces) > 1 else word)
    return " | ".join(parts)


@dataclass(frozen=True)
class ProbableAgeGroupNamedEvidence(ProbableNamedInTextEvidence):
    """Probable, for age: ages not recorded, and the dataset's own text names
    the age group (one of *words*)."""
    words: Tuple[str, ...] = ()

    def sql(self) -> Optional[str]:
        return text_from_dataset_matches(self.alias, tsquery_for_words(self.words))

    def score(self) -> float:
        # 45.7% of 92 judged rows relevant, against 86.1% of confirmed rows
        # (recall benchmark, LLM judge, 2026-11-13).
        return 0.53

    def reason(self) -> str:
        return f"{PossibleNotRecordedEvidence(LABEL).reason()}; the title, description or README names the age group"


def age_group_named(node, alias: str) -> Optional[ProbableAgeGroupNamedEvidence]:
    """The age group an age condition asks for, as a text would name it; None
    if the asked range fits no group."""
    group_words = age_group_words(*asked.age_range(node))
    return ProbableAgeGroupNamedEvidence(words=tuple(group_words), alias=alias) if group_words else None


def _recorded(alias: str) -> Tuple[ContradictedEvidence, ...]:
    """Ages are kept as the dataset's range and in the participant list."""
    in_columns = ContradictedInDatasetColumnEvidence(
        f"{alias}.age_min IS NOT NULL OR {alias}.age_max IS NOT NULL")
    return (in_columns, *participant_rows_record("age")(alias))


FIELD = SoftField(
    kind="age",
    label=LABEL,
    columns=("age", "age_min", "age_max"),
    recorded=_recorded,
    named_in_text=age_group_named,
    # The dataset's own text only. Of 63 datasets without ages whose only
    # child word was in a paper (live database, 2026-11-17), about 9 studied
    # children; the rest were adult studies citing child work (Sherlock, BCIT)
    # and two mouse datasets.
    paper_evidence=False,
    # 28.6% of 21 judged rows relevant, against 86.1% of confirmed rows
    # (recall benchmark, LLM judge, 2026-11-13).
    unrecorded_score=0.33,
)
