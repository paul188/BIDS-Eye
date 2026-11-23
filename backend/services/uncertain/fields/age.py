"""
services/uncertain/fields/age.py
------------------------------------
Participant ages as a field a dataset may not record (FIELD, at the end).
About half the datasets record none (tier.py).

Its text form: the dataset's own text names the age group the condition asks
for ("infants", "older adults").
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from services.evidence import reason_texts
from services.uncertain.fields.base import SoftField, participant_rows_record
import text_evidence  # RAG/ is on the path wherever the backend runs


_NUMBER_COMPARISON_RE = re.compile(
    r"\bage(?:_min|_max)?\s*(?P<op><=|>=|<|>|=)\s*(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE)
_BETWEEN_RE = re.compile(
    r"\bage(?:_min|_max)?\s+BETWEEN\s+(?P<low>\d+(?:\.\d+)?)\s+AND\s+(?P<high>\d+(?:\.\d+)?)",
    re.IGNORECASE)


def asked_age_range(sql: str) -> Tuple[float, float]:
    """The age range an age condition asks for, read from its comparisons."""
    low, high = 0.0, 120.0
    for match in _BETWEEN_RE.finditer(sql):
        low = max(low, float(match.group("low")))
        high = min(high, float(match.group("high")))
    for match in _NUMBER_COMPARISON_RE.finditer(sql):
        value = float(match.group("value"))
        op = match.group("op")
        if op in (">", ">="):
            low = max(low, value)
        elif op in ("<", "<="):
            high = min(high, value)
    return low, high


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


def age_text_condition(node, alias: str) -> Optional[Tuple[str, str]]:
    """(SQL, reason) for a dataset whose text names the asked age group; None
    if the asked range fits no group."""
    group_words = age_group_words(*asked_age_range(node.sql(dialect="postgres")))
    if not group_words:
        return None
    return (f"{alias}.search_tsv @@ TO_TSQUERY('english', '{tsquery_for_words(group_words)}')",
            reason_texts.AGE_GROUP_NAMED)


def _recorded(alias: str) -> str:
    return (f"({alias}.age_min IS NOT NULL OR {alias}.age_max IS NOT NULL OR "
            f"{participant_rows_record('age')(alias)})")


FIELD = SoftField(
    kind="age",
    columns=("age", "age_min", "age_max"),
    recorded=_recorded,
    named_in_text=age_text_condition,
    # The dataset's own text only. Of 63 datasets without ages whose only
    # child word was in a paper (live database, 2026-11-17), about 9 studied
    # children; the rest were adult studies citing child work (Sherlock, BCIT)
    # and two mouse datasets.
    paper_evidence=False,
)
