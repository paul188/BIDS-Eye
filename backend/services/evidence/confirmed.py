"""
services/evidence/confirmed.py
------------------------------
Evidence of a confirmed match that shows numbers or values of its own on the
result card: the participants a condition holds for, the value of a dataset
column, a default that was assumed. Every other confirmed match says only
where it was found (found()).

Built by results/match_reasons.py, one per OR branch of a condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sqlglot import exp

from services.evidence import base, reason_texts
from services.evidence.base import Evidence, Strength

_MAX_VALUE_CHARS = 60


def found(source: str, sql: str) -> Evidence:
    """Confirmed evidence that says only where the value was found."""
    return Evidence(sql, reason_texts.FOUND_IN[source], source=source, strength=Strength.CONFIRMED)


@dataclass(frozen=True)
class ParticipantRows(Evidence):
    """Rows of the participant list hold the asked value: "12 of 40
    participants aged 4–23 months"."""
    # The participant rows the evidence holds for, as a SELECT whose
    # projection number_columns replaces (COUNT, MIN/MAX of the age). None:
    # the rows cannot be counted (a GROUP BY, not an EXISTS).
    participant_rows: Optional[exp.Select] = None
    age_column: Optional[exp.Column] = None   # the age column that SELECT compares
    reads_diagnosis: bool = False
    source: str = field(default=base.PARTICIPANTS, kw_only=True)
    strength: Strength = field(default=Strength.CONFIRMED, kw_only=True)

    def number_columns(self, name: str) -> List[str]:
        if self.participant_rows is None:
            return []
        columns = [f"{self._aggregate('COUNT', exp.Star())} AS n{name}"]
        if self.age_column is not None:
            columns.append(f"{self._aggregate('MIN', self.age_column)} AS lo{name}")
            columns.append(f"{self._aggregate('MAX', self.age_column)} AS hi{name}")
        return columns

    def _aggregate(self, function: str, argument) -> str:
        assert self.participant_rows is not None
        select = self.participant_rows.copy()
        select.set("expressions", [exp.Anonymous(this=function, expressions=[argument.copy()])])
        for key in ("order", "limit", "offset", "distinct"):
            select.set(key, None)
        return f"({select.sql(dialect='postgres')})"

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        count = row.get(f"n{name}")
        if count is None:
            return reason_texts.PARTICIPANTS_LISTED
        low, high = row.get(f"lo{name}"), row.get(f"hi{name}")
        ages = (reason_texts.age_span(float(low), float(high))
                if self.age_column is not None and low is not None and high is not None else None)
        return reason_texts.participants_counted(count, total, ages, self.reads_diagnosis)


@dataclass(frozen=True)
class FieldValue(Evidence):
    """Columns of the dataset itself hold the asked value; the card shows it:
    "dataset age range 6–11 years", "species label “mouse”"."""
    columns: Tuple[str, ...] = ()
    source: str = field(default=base.FIELD, kw_only=True)
    strength: Strength = field(default=Strength.CONFIRMED, kw_only=True)

    def read_columns(self) -> Tuple[str, ...]:
        return self.columns

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        values = {column: row.get(f"f_{column}") for column in self.columns}
        if {"age_min", "age_max"} & set(values):
            try:
                low, high = float(values.get("age_min") or 0), float(values.get("age_max") or 0)
            except ValueError:
                low = high = 0
            if low > 0 or high > 0:
                low, high = (low or high), (high or low)
                return reason_texts.dataset_age_range(reason_texts.age_span(low, high))
        shown = []
        for column, value in values.items():
            if column in ("age_min", "age_max") or value in (None, ""):
                continue
            value = str(value)
            if len(value) > _MAX_VALUE_CHARS:
                value = value[:_MAX_VALUE_CHARS].rstrip() + "…"
            shown.append(reason_texts.field_value(column, value))
        return ", ".join(shown) or None


@dataclass(frozen=True)
class Assumed(Evidence):
    """The column is empty and its default holds: "no species label, assumed
    human" (the defaults: RAG/criteria.py; their wording: reason_texts.ASSUMED)."""
    columns: Tuple[str, ...] = ()
    source: str = field(default=base.ASSUMED, kw_only=True)
    strength: Strength = field(default=Strength.CONFIRMED, kw_only=True)

    def read_columns(self) -> Tuple[str, ...]:
        return self.columns

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        return reason_texts.ASSUMED[self.columns[0]]
