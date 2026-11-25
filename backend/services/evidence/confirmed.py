"""
services/evidence/confirmed.py
------------------------------
Evidence that a condition of the generated query holds for a dataset: one way
it holds. A condition is an OR of such ways (the participant list records the
value, a dataset column holds it, the title names it), or a single one. Each
kind writes what a result card says about it.

Built by branches.py, one per OR branch of a condition.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

from sqlglot import exp

from services.evidence.base import PieceOfEvidence, Source, Strength

_MAX_VALUE_CHARS = 60


def age_span(low: float, high: float) -> str:
    """'4–23 months' below 2 years, else '6–11 years'; one value once."""
    if high < 2:
        months_lo, months_hi = round(low * 12), round(high * 12)
        return (f"{months_lo} months" if months_lo == months_hi
                else f"{months_lo}–{months_hi} months")
    lo, hi = f"{round(low, 1):g}", f"{round(high, 1):g}"
    return f"{lo} years" if lo == hi else f"{lo}–{hi} years"


def _age(value: object) -> Optional[float]:
    """An age read back as text; None when the column is empty."""
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ConfirmedMatch(PieceOfEvidence):
    """One OR branch of a condition of the generated query. What the card
    says about it may need numbers or values from the database; a subclass
    names the columns it needs and writes the text from them."""
    branch: str   # its SQL, as generated: it names the dataset row by the query's own alias
    strength: Strength = field(default=Strength.CONFIRMED, kw_only=True)

    def sql(self) -> Optional[str]:
        return self.branch

    def read_columns(self) -> Tuple[str, ...]:
        """The dataset columns whose values the card text shows."""
        return ()

    def number_columns(self, name: str) -> List[str]:
        """SQL columns for the numbers the card text shows, named after *name*."""
        return []

    @abstractmethod
    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        """The card text for one dataset; None says nothing. *row* holds the
        columns asked for above, *total* the dataset's number of participants."""

    @property
    def key(self) -> Tuple[Source, Tuple[str, ...]]:
        """Two branches with one key are the same way of matching."""
        return self.source, self.read_columns()


@dataclass(frozen=True)
class Found(ConfirmedMatch):
    """A match that says only where the value was found: its source."""

    # What the card says, by the source.
    TEXTS: ClassVar[Dict[Source, str]] = {
        Source.FILES: "in the dataset's files",
        Source.DERIVATIVES: "in its derivatives",
        Source.SITES: "at a listed site",
        Source.PAPER: "named in a linked paper",
        Source.TEXT: "named in the README or description",
        Source.SHORT_TEXT: "named in the title or description",
        Source.README: "named in the README",
        # An excluded diagnosis the dataset passes because it records none.
        Source.ASSUMED_ABSENT: "no diagnosis recorded, assumed absent",
    }

    def __post_init__(self) -> None:
        if self.source not in self.TEXTS:
            raise ValueError(f"no card text for a match found in {self.source}")

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        return self.TEXTS[self.source]


@dataclass(frozen=True)
class ParticipantRows(ConfirmedMatch):
    """Rows of the participant list hold the asked value: "12 of 40
    participants aged 4–23 months"."""
    # The participant rows the evidence holds for, as a SELECT whose
    # projection number_columns replaces (COUNT, MIN/MAX of the age). None:
    # the rows cannot be counted (a GROUP BY, not an EXISTS).
    participant_rows: Optional[exp.Select] = None
    age_column: Optional[exp.Column] = None   # the age column that SELECT compares
    reads_diagnosis: bool = False
    source: Source = field(default=Source.PARTICIPANTS, kw_only=True)

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
        # One number per dataset: grouped rows would give one per group.
        assert not (self.participant_rows.args.get("group") or self.participant_rows.args.get("having"))
        select = self.participant_rows.copy()
        select.set("expressions", [exp.Anonymous(this=function, expressions=[argument.copy()])])
        for key in ("order", "limit", "offset", "distinct"):
            select.set(key, None)
        return f"({select.sql(dialect='postgres')})"

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        count = row.get(f"n{name}")
        if count is None:
            return "recorded in the participant list"
        whom = (f"{count} of {total} participants" if total and total >= count
                else f"{count} participants")
        low, high = row.get(f"lo{name}"), row.get(f"hi{name}")
        if self.age_column is not None and low is not None and high is not None:
            return f"{whom} aged {age_span(float(low), float(high))}"
        return f"coded for {whom}" if self.reads_diagnosis else f"recorded for {whom}"


@dataclass(frozen=True)
class FieldValue(ConfirmedMatch):
    """Columns of the dataset itself hold the asked value; the card shows it:
    "dataset age range 6–11 years", "species label “mouse”"."""
    columns: Tuple[str, ...] = ()
    source: Source = field(default=Source.FIELD, kw_only=True)
    # How a card names a column; others by their own name.
    LABELS: ClassVar[Dict[str, str]] = {
        "species": "species label",
        "dx_status": "diagnosis label",
        "study_design": "study design",
        "study_domain": "study domain",
    }

    def read_columns(self) -> Tuple[str, ...]:
        return self.columns

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        values = {column: row.get(f"f_{column}") for column in self.columns}
        if {"age_min", "age_max"} & set(values):
            low, high = _age(values.get("age_min")), _age(values.get("age_max"))
            # 0 to 0 is a placeholder, not an age: the four datasets that
            # have it are adult studies (live database, 2026-11-23).
            if (low is not None or high is not None) and (low or high):
                low = high if low is None else low
                high = low if high is None else high
                assert low is not None and high is not None
                return f"dataset age range {age_span(low, high)}"
        shown = []
        for column, value in values.items():
            if column in ("age_min", "age_max") or value in (None, ""):
                continue
            value = str(value)
            if len(value) > _MAX_VALUE_CHARS:
                value = value[:_MAX_VALUE_CHARS].rstrip() + "…"
            shown.append(f"{self.LABELS.get(column, column.replace('_', ' '))} “{value}”")
        return ", ".join(shown) or None


@dataclass(frozen=True)
class Assumed(ConfirmedMatch):
    """The column is empty and its default holds: "no species label, assumed
    human" (the defaults themselves: RAG/criteria.py)."""
    columns: Tuple[str, ...] = ()
    source: Source = field(default=Source.ASSUMED, kw_only=True)
    # What the card says, by the column that has a default.
    TEXTS: ClassVar[Dict[str, str]] = {"species": "no species label, assumed human"}

    def __post_init__(self) -> None:
        if len(self.columns) != 1 or self.columns[0] not in self.TEXTS:
            raise ValueError(f"no default to assume for {self.columns}")

    def read_columns(self) -> Tuple[str, ...]:
        return self.columns

    def phrase(self, row: dict, name: str, total: Optional[int] = None) -> Optional[str]:
        return self.TEXTS[self.columns[0]]
