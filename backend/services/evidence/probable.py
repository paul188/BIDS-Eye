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

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from services.evidence.base import Evidence, Source, Strength


@dataclass(frozen=True)
class TextSearch:
    """One text search of a condition."""
    query: str                   # tsquery on the stemmed vector
    guard: Optional[str] = None  # tsquery on the exact vector: the word forms asked for


@dataclass(frozen=True)
class PaperOnly(Evidence):
    """The condition's text searches match a paper the dataset names, not its
    own text. *searches* are without weight labels: paper text carries the
    lowest weight, so a search scoped to the name could never reach it."""
    searches: Tuple[TextSearch, ...]
    source: Source = field(default=Source.PAPER, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)

    def sql(self, alias: str) -> Optional[str]:
        from services.fulltext.vectors import (EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN, FTS_COLUMN,
                                               PAPER_FTS_COLUMN)
        atoms = []
        for search in self.searches:
            paper = f"{alias}.{PAPER_FTS_COLUMN} @@ TO_TSQUERY('english', '{search.query}')"
            own = f"{alias}.{FTS_COLUMN} @@ TO_TSQUERY('english', '{search.query}')"
            if search.guard:
                # Exact word forms count in the paper as in the dataset's text.
                paper = (f"({paper} AND {alias}.{EXACT_PAPER_FTS_COLUMN} @@ "
                         f"TO_TSQUERY('simple', '{search.guard}'))")
                own = f"({own} AND {alias}.{EXACT_FTS_COLUMN} @@ TO_TSQUERY('simple', '{search.guard}'))"
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
class WeakerSearch(Evidence):
    """Some text search of the condition matches in a weaker form."""
    searches: Tuple[WeakenedSearch, ...]
    source: Source = field(default=Source.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)

    def sql(self, alias: str) -> Optional[str]:
        return " OR ".join(dict.fromkeys(search.sql(alias) for search in self.searches))

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
class CoreOfTerm(Evidence):
    """A term the generated query shortened to its core ("physical exercise"
    searched as "exercise", services/fulltext/text_search.py): the dataset
    has the core and not the *missing* words."""
    core: str   # the SQL of the core search, as generated
    missing: Tuple[str, ...] = ()
    source: Source = field(default=Source.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)

    def sql(self, alias: str) -> Optional[str]:
        return self.core

    def reason(self) -> str:
        return _words_missing(self.missing)


@dataclass(frozen=True)
class NamedInText(Evidence):
    """The dataset's own text names the value the condition asks of a field:
    "22 women", "infants", "7T". Each field has its subclass, which holds the
    asked value and writes sql() and reason()."""
    source: Source = field(default=Source.TEXT, kw_only=True)
    strength: Strength = field(default=Strength.PROBABLE, kw_only=True)
