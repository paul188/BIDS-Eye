"""
services/evidence/probable.py
-----------------------------
Evidence that a condition probably holds for a dataset it does not hold for
as generated: its words match only in a paper, a weaker form of its text
search matches, or the dataset's own text names the asked value.

The kinds of one field are written with the field: the age group
(uncertain/fields/age.py), participant sex (fields/sex.py), scanner values
(uncertain/scanner.py), the patient count (uncertain/classify.py).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from services.evidence.base import Source, Strength, UncertainEvidence


@dataclass(frozen=True)
class TextSearch:
    """One text search of a condition."""
    query: str                   # tsquery on the stemmed vector
    guard: Optional[str] = None  # tsquery on the exact vector: the word forms asked for


@dataclass(frozen=True)
class ProbablePaperOnlyEvidence(UncertainEvidence):
    """Probable: the text searches match only a paper the dataset names,
    not its own text: "found only in a paper the dataset names". *searches* are without weight labels: paper text carries the
    lowest weight, so a search scoped to the name could never reach it."""
    searches: Tuple[TextSearch, ...]
    alias: str = field(kw_only=True)   # the dataset row of the query
    source: Source = field(default=Source.PAPERS, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)

    def sql(self) -> Optional[str]:
        from services.fulltext.vectors import (EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN, FTS_COLUMN,
                                               PAPER_FTS_COLUMN)
        atoms = []
        for search in self.searches:
            paper = f"{self.alias}.{PAPER_FTS_COLUMN} @@ TO_TSQUERY('english', '{search.query}')"
            own = f"{self.alias}.{FTS_COLUMN} @@ TO_TSQUERY('english', '{search.query}')"
            if search.guard:
                # Exact word forms count in the paper as in the dataset's text.
                paper = (f"({paper} AND {self.alias}.{EXACT_PAPER_FTS_COLUMN} @@ "
                         f"TO_TSQUERY('simple', '{search.guard}'))")
                own = (f"({own} AND {self.alias}.{EXACT_FTS_COLUMN} @@ "
                       f"TO_TSQUERY('simple', '{search.guard}'))")
            # The paper vector also holds the dataset's own text; "only in a
            # paper" means it matches there and not in the dataset's text.
            atoms.append(f"({paper} AND NOT {own})")
        return " OR ".join(dict.fromkeys(atoms))

    def reason(self) -> str:
        return "found only in a paper the dataset names"


def _listed(terms: Sequence[str], limit: int) -> str:
    return ", ".join(terms[:limit]) + (" …" if len(terms) > limit else "")


def _words_missing(words: Sequence[str]) -> str:
    return f"matches only part of the search; missing: {_listed(words, 4)}"


def _readme_only(terms: Sequence[str]) -> str:
    """*terms*: the words whose search was widened to the whole README; may be empty."""
    if not terms:
        return "the search term appears only further down in the README"
    shown = " / ".join(f'"{t}"' for t in terms[:2]) + (" …" if len(terms) > 2 else "")
    return f"{shown} appears only further down in the README"


class Weakening(str, Enum):
    """How a text search was made weaker."""
    WORDS_MISSING = "words_missing"        # some of its words were dropped
    WORDS_APART = "words_apart"            # a 'near' search: its words anywhere
    EXCLUDED_MEANING = "excluded_meaning"  # a meaning the search excluded is let in
    README_ONLY = "readme_only"            # every word, but anywhere in the text


@dataclass(frozen=True)
class WeakenedSearch:
    """One text search in a weaker form (services/uncertain/text.py)."""
    search: TextSearch
    column: str                  # the stemmed vector it runs on
    guard_column: Optional[str]  # the exact vector of its guard
    weakening: Weakening
    # WORDS_MISSING: the dropped words. WORDS_APART: the words searched
    # apart. EXCLUDED_MEANING: the words of the excluded meanings.
    # README_ONLY: the words of the search.
    words: Tuple[str, ...]

    def sql(self, alias: str) -> str:
        atom = f"{alias}.{self.column} @@ TO_TSQUERY('english', '{self.search.query}')"
        if self.search.guard:
            # The weaker form still needs the word forms that were asked for:
            # "ageing" relaxed to its stem would be every README with an age.
            atom = (f"({atom} AND {alias}.{self.guard_column} @@ "
                    f"TO_TSQUERY('simple', '{self.search.guard}'))")
        return atom


@dataclass(frozen=True)
class ProbableWeakerSearchEvidence(UncertainEvidence):
    """Probable: a text search matches only in a weaker form (Weakening):
    "matches only part of the search; missing: …"."""
    searches: Tuple[WeakenedSearch, ...]
    alias: str = field(kw_only=True)   # the dataset row of the query
    source: Source = field(default=Source.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)

    def sql(self) -> Optional[str]:
        return " OR ".join(dict.fromkeys(search.sql(self.alias) for search in self.searches))

    def score(self) -> float:
        # 31.7% of 555 judged rows relevant, against 86.1% of confirmed rows
        # (recall benchmark, LLM judge, 2026-11-13;
        # reports/Probabilistic ranking for Bids Eye search.md).
        return 0.37

    def _words(self, weakening: Weakening) -> List[str]:
        return list(dict.fromkeys(word for search in self.searches
                                  if search.weakening == weakening for word in search.words))

    def reason(self) -> str:
        # One text serves every dataset at this level, so the weaker forms of
        # a topic search are named as alternatives: either may be why a
        # dataset is here.
        loose = []
        if apart := self._words(Weakening.WORDS_APART):
            loose.append(f"the words {_listed(apart, 3)} occur, but not together")
        if excluded := self._words(Weakening.EXCLUDED_MEANING):
            loose.append("a related term occurs next to a meaning the search excludes "
                         f"({_listed(excluded, 3)})")
        parts = [", or ".join(loose)] if loose else []
        shortened = any(search.weakening in (Weakening.WORDS_MISSING, Weakening.README_ONLY)
                        for search in self.searches)
        if shortened or not parts:
            missing = self._words(Weakening.WORDS_MISSING)
            parts.append(_words_missing(missing) if missing
                         else _readme_only(self._words(Weakening.README_ONLY)))
        return "; ".join(parts)


@dataclass(frozen=True)
class ProbableCoreOfTermEvidence(UncertainEvidence):
    """Probable: the generated query shortened a term to its core ("physical
    exercise" searched as "exercise", services/fulltext/text_search.py), and
    the dataset has the core but not the *missing* words."""
    core: str   # the SQL of the core search, as generated
    missing: Tuple[str, ...] = ()
    source: Source = field(default=Source.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)

    def sql(self) -> Optional[str]:
        return self.core

    def reason(self) -> str:
        return _words_missing(self.missing)


@dataclass(frozen=True)
class ProbableNamedInTextEvidence(UncertainEvidence):
    """Probable: the field is not recorded, but the dataset's name,
    description or README names the asked value: "22 women", "infants", "7T". Each field has its subclass,
    which holds the asked value and writes sql() and reason()."""
    alias: str = field(kw_only=True)   # the dataset row of the query
    source: Source = field(default=Source.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)

    @abstractmethod
    def reason(self) -> str:
        """What the result card says: the text depends on the field."""
