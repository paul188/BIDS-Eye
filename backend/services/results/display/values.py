"""
services/results/display/values.py
----------------------------------
What a condition asks for, in words: one class per kind of value, each with
format(). services/asked.py reads the values out of the SQL; chips.py picks
the class and the chip kind.

    Codes("diagnosis", ["adhd", "autism"]).format()  -> "ADHD or Autism"
    Quantity(Bounds(low=40)).format()                -> "≥ 40"
    Age(Bounds(high=2, high_inclusive=False))        -> "under 2"
    Phrases(["sleep deprivation"]).format()          -> "“sleep deprivation”"

format() gives a Label: the text, and for a list all its values and how
many the text names (the page adds "+N" for the rest).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from code_names import code_name  # RAG/ is on the path wherever the backend runs
from sqlglot import exp

from services import asked
from services.schema import tables as _tables
from services.sql_tree import Bounds, OrderKey

# How many values a label names before "+N".
MAX_VALUES_SHOWN = 3
# Searched phrases are longer: a label names two.
PHRASES_SHOWN = 2


@dataclass(frozen=True)
class Label:
    """A chip's label (QueryFilterSchema.label, .values, .shown)."""
    text: str
    values: Tuple[str, ...] = ()
    shown: int = 0

    def chip_fields(self) -> Dict[str, Any]:
        return {"label": self.text, "values": list(self.values), "shown": self.shown}

    def after(self, prefix: str) -> Label:
        """The same label with *prefix* before its text ("≥ 40 with ")."""
        return Label(prefix + self.text, self.values, self.shown)


class Value(ABC):
    """One kind of asked value."""

    @abstractmethod
    def format(self) -> Label:
        ...


def capitalised(name: str) -> str:
    """*name* with a capital first letter, unless its first word is already
    written with capitals ("fMRI", "iEEG")."""
    first = name.split(" ", 1)[0]
    return name[:1].upper() + name[1:] if first.islower() else name


def words_of(identifier: str) -> str:
    """'SubjectsWithFewSessions' / 'other_entities' / 'bids_derivatives' -> plain words."""
    name = re.sub(r"^bids_", "", identifier)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"[_\s]+", " ", name).strip().lower()


def number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


# ── Lists ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Listed(Value):
    """Names as they are: "Mexico or Peru", "Name, Authors"."""
    names: Sequence[str]
    joiner: str = " or "
    shown: int = MAX_VALUES_SHOWN

    def quote(self, name: str) -> str:
        return name

    def format(self) -> Label:
        names = tuple(dict.fromkeys(n for n in self.names if n))
        shown = min(self.shown, len(names))
        return Label(self.joiner.join(self.quote(n) for n in names[:shown]), names, shown)


@dataclass(frozen=True)
class Phrases(Listed):
    """Words searched in the dataset text, quoted: "“drawings” or “art”"."""

    def quote(self, name: str) -> str:
        return f"“{name}”"


@dataclass(frozen=True)
class Codes(Value):
    """Vocabulary codes of one field (CodedField or a value_mappings section),
    shown under their names (RAG/code_names.py)."""
    section: str
    codes: Sequence[str]
    shown: int = MAX_VALUES_SHOWN
    capitals: bool = False   # names start with a capital ("Parkinson's disease")

    def names(self) -> List[str]:
        names = [code_name(self.section, code) for code in self.codes]
        return [capitalised(n) for n in names] if self.capitals else names

    def format(self) -> Label:
        return Listed(self.names(), shown=self.shown).format()


# ── Numbers ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Quantity(Value):
    """A number range: "≥ 40", "6.5–7.5", "= 1.5"."""
    bounds: Bounds

    def number(self, value: float) -> str:
        return number_text(value)

    def format(self) -> Label:
        b, number = self.bounds, self.number
        if b.exact is not None:
            return Label(f"= {number(b.exact)}")
        if b.low is not None and b.high is not None:
            return Label(f"{number(b.low)}–{number(b.high)}")
        if b.low is not None:
            return Label(f"{'≥' if b.low_inclusive else '>'} {number(b.low)}")
        assert b.high is not None  # Bounds without either bound is not read as one
        return Label(f"{'≤' if b.high_inclusive else '<'} {number(b.high)}")


@dataclass(frozen=True)
class Size(Quantity):
    """A size in bytes, shown in GB: "≥ 1.5 GB"."""

    def number(self, value: float) -> str:
        return f"{value / 1024 ** 3:.3g}"

    def format(self) -> Label:
        return Label(super().format().text + " GB")


@dataclass(frozen=True)
class Age(Value):
    """Ages in years: "18+", "under 18", "up to 12", "10–19", "6 months+"."""
    bounds: Bounds

    @staticmethod
    def years(value: float) -> str:
        return f"{round(value * 12)} months" if 0 < value < 1 else f"{value:g}"

    def format(self) -> Label:
        low, high, years = self.bounds.low, self.bounds.high, self.years
        if low is not None and high is not None:
            return Label(f"{years(low)}–{years(high)}")
        if low is not None:
            return Label(f"{years(low)}+" if self.bounds.low_inclusive else f"over {years(low)}")
        assert high is not None  # asked.age_bounds gives at least one bound
        return Label(f"up to {years(high)}" if self.bounds.high_inclusive else f"under {years(high)}")


@dataclass(frozen=True)
class Counted(Value):
    """A count of subjects, sessions or runs: "≥ 40", "≥ 20 aged 10–19"."""
    count: asked.Count

    def format(self) -> Label:
        text = Quantity(self.count.bounds).format().text
        if self.count.age:
            text += f" aged {Age(self.count.age).format().text}"
        return Label(text)


@dataclass(frozen=True)
class Patients(Value):
    """Participants with a diagnosis: "≥ 25 with Frontotemporal dementia"."""
    minimum: int
    diagnoses: Sequence[str]   # codes

    def format(self) -> Label:
        return Codes("diagnosis", self.diagnoses, shown=1, capitals=True).format().after(
            f"≥ {self.minimum} with ")


# ── Other ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Plain(Value):
    """A label that is already words: "recorded", "any animal"."""
    text: str

    def format(self) -> Label:
        return Label(self.text)



@dataclass(frozen=True)
class Ordering(Value):
    """ORDER BY and LIMIT: "most participants first · top 1", "first 10"."""
    keys: Sequence[OrderKey] = field(default_factory=tuple)
    top: int | None = None

    @staticmethod
    def key_words(key: OrderKey) -> str:
        """'most participants first', from the words of its column (schema/tables.py)."""
        for column in key.columns:
            words = _tables.COLUMN_SORTED_AS.get(column)
            if words:
                return words.descending if key.descending else words.ascending
        # Any other key in words, never SQL: 'by age max', 'by session count'.
        if any(column in _tables.SEARCH_VECTORS for column in key.columns):
            return "best text match first"
        columns = [_tables.COLUMN_SHOWN_AS.get(column, words_of(column)).lower()
                   for column in key.columns if column not in asked.KEY_COLUMNS]
        if not columns:
            return "in a computed order"
        counted = " count" if key.expression.find(exp.Count) is not None else ""
        return f"by {', '.join(columns[:2])}{counted}" + (", highest first" if key.descending else "")

    def orders(self) -> List[str]:
        return list(dict.fromkeys(self.key_words(key) for key in self.keys))

    def format(self) -> Label:
        orders = self.orders()
        if not orders:
            return Label(f"first {self.top}")
        text = ", then ".join(orders[:2])
        return Label(text + (f" · top {self.top}" if self.top is not None else ""))
