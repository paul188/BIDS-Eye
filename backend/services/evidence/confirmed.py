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
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple

from sqlglot import exp

from services.evidence.base import PieceOfEvidence, Source, Strength

_MAX_VALUE_CHARS = 60


class CardColumn(str, Enum):
    """Columns of the card query (results/match_reasons.build_sql),
    by what they hold. Written there and read by reason() here."""
    HOLDS = "h"       # the branch holds; per suffix
    COUNT = "n"       # participant rows it holds for; per suffix
    YOUNGEST = "lo"   # their lowest age; per suffix
    OLDEST = "hi"     # their highest age; per suffix
    FIELD = "f_"      # a dataset column as text; per column name

    def of(self, key: str) -> str:
        """The column for *key*: a branch's suffix ("0_1") or a column name."""
        return f"{self.value}{key}"


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
class ConfirmedEvidence(PieceOfEvidence):
    """Confirmed: one OR branch of the generated condition holds. Base of
    the confirmed kinds; each subclass names the database columns its card
    text needs and writes the text from them."""
    branch: str   # its SQL, as generated: it names the dataset row by the query's own alias
    strength: Strength = field(default=Strength.CONFIRMED, kw_only=True)

    def sql(self) -> Optional[str]:
        return self.branch

    def read_columns(self) -> Tuple[str, ...]:
        """The dataset columns whose values the card text shows."""
        return ()

    def number_columns(self, suffix: str) -> List[str]:
        """SQL columns for the numbers the card text shows, named with *suffix*."""
        return []

    @abstractmethod
    def reason(self, row: dict, suffix: str, total: Optional[int] = None) -> Optional[str]:
        """The card text for one dataset; None says nothing. *row* holds the
        columns asked for above (CardColumn), *total* the dataset's number of
        participants."""

    @property
    def key(self) -> Tuple[Source, Tuple[str, ...]]:
        """Two branches with one key are the same way of matching."""
        return self.source, self.read_columns()


@dataclass(frozen=True)
class _ConfirmedWhereFound(ConfirmedEvidence):
    """A confirmed kind whose card says only where the value was found."""
    TEXTS: ClassVar[Dict[Source, str]] = {}   # card text by source

    def __post_init__(self) -> None:
        if self.source not in self.TEXTS:
            raise ValueError(f"{type(self).__name__} has no card text for {self.source}")

    def reason(self, row: dict, suffix: str, total: Optional[int] = None) -> Optional[str]:
        return self.TEXTS[self.source]


@dataclass(frozen=True)
class ConfirmedInDatasetTextEvidence(_ConfirmedWhereFound):
    """Confirmed by a text search in the dataset's name, description or README."""
    TEXTS: ClassVar[Dict[Source, str]] = {
        Source.TEXT: "named in the README or description",
        Source.SHORT_TEXT: "named in the title or description",
        Source.README: "named in the README",
    }


@dataclass(frozen=True)
class ConfirmedInPaperEvidence(_ConfirmedWhereFound):
    """Confirmed by the text of a paper the dataset names."""
    source: Source = field(default=Source.PAPERS, kw_only=True)
    TEXTS: ClassVar[Dict[Source, str]] = {Source.PAPERS: "named in a linked paper"}


@dataclass(frozen=True)
class ConfirmedInFileTablesEvidence(_ConfirmedWhereFound):
    """Confirmed by rows of the dataset's files, derivatives or sites."""
    TEXTS: ClassVar[Dict[Source, str]] = {
        Source.FILES: "in the dataset's files",
        Source.DERIVATIVES: "in its derivatives",
        Source.SITES: "at a listed site",
    }


@dataclass(frozen=True)
class ConfirmedInParticipantListEvidence(ConfirmedEvidence):
    """Confirmed by rows of the participant list; card counts them: "12 of
    40 participants aged 4–23 months"."""
    # The participant rows the evidence holds for, as a SELECT whose
    # projection number_columns replaces (COUNT, MIN/MAX of the age). None:
    # the rows cannot be counted (a GROUP BY, not an EXISTS).
    participant_rows: Optional[exp.Select] = None
    age_column: Optional[exp.Column] = None   # the age column that SELECT compares
    reads_diagnosis: bool = False
    source: Source = field(default=Source.PARTICIPANTS, kw_only=True)

    def number_columns(self, suffix: str) -> List[str]:
        if self.participant_rows is None:
            return []
        aggregates: List[Tuple[str, exp.Expression, CardColumn]] = [
            ("COUNT", exp.Star(), CardColumn.COUNT)]
        if self.age_column is not None:
            aggregates += [("MIN", self.age_column, CardColumn.YOUNGEST),
                           ("MAX", self.age_column, CardColumn.OLDEST)]
        return [f"{self._aggregate(function, argument)} AS {column.of(suffix)}"
                for function, argument, column in aggregates]

    def _aggregate(self, function: str, argument) -> str:
        assert self.participant_rows is not None
        # One number per dataset: grouped rows would give one per group.
        assert not (self.participant_rows.args.get("group") or self.participant_rows.args.get("having"))
        select = self.participant_rows.copy()
        select.set("expressions", [exp.Anonymous(this=function, expressions=[argument.copy()])])
        for key in ("order", "limit", "offset", "distinct"):
            select.set(key, None)
        return f"({select.sql(dialect='postgres')})"

    def reason(self, row: dict, suffix: str, total: Optional[int] = None) -> Optional[str]:
        count = row.get(CardColumn.COUNT.of(suffix))
        if count is None:
            return "recorded in the participant list"
        whom = (f"{count} of {total} participants" if total and total >= count
                else f"{count} participants")
        low = row.get(CardColumn.YOUNGEST.of(suffix))
        high = row.get(CardColumn.OLDEST.of(suffix))
        if self.age_column is not None and low is not None and high is not None:
            return f"{whom} aged {age_span(float(low), float(high))}"
        return f"coded for {whom}" if self.reads_diagnosis else f"recorded for {whom}"


@dataclass(frozen=True)
class ConfirmedInDatasetColumnEvidence(ConfirmedEvidence):
    """Confirmed by the dataset's own columns; card shows the value:
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

    def reason(self, row: dict, suffix: str, total: Optional[int] = None) -> Optional[str]:
        values = {column: row.get(CardColumn.FIELD.of(column)) for column in self.columns}
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
class ConfirmedByDefaultEvidence(ConfirmedEvidence):
    """Confirmed by a default: an empty column whose default fits ("no species
    label, assumed human"; defaults: backend/vocabulary/criteria.py), or an excluded
    diagnosis the dataset records none of (source ASSUMED_ABSENT)."""
    columns: Tuple[str, ...] = ()
    source: Source = field(default=Source.ASSUMED, kw_only=True)
    # What the card says, by the column that has a default.
    TEXTS: ClassVar[Dict[str, str]] = {"species": "no species label, assumed human"}
    ABSENT_DIAGNOSIS: ClassVar[str] = "no diagnosis recorded, assumed absent"

    def __post_init__(self) -> None:
        if self.source == Source.ASSUMED_ABSENT:
            if self.columns:
                raise ValueError(f"an absent diagnosis reads no columns, not {self.columns}")
        elif len(self.columns) != 1 or self.columns[0] not in self.TEXTS:
            raise ValueError(f"no default to assume for {self.columns}")

    def read_columns(self) -> Tuple[str, ...]:
        return self.columns

    def reason(self, row: dict, suffix: str, total: Optional[int] = None) -> Optional[str]:
        if self.source == Source.ASSUMED_ABSENT:
            return self.ABSENT_DIAGNOSIS
        return self.TEXTS[self.columns[0]]
