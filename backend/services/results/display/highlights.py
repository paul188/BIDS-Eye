"""
services/results/display/highlights.py
--------------------------------------
The words a search marks in a dataset's README, read from its chips
(services/results/display/chips.py). frontend/src/highlight.ts mirrors them.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Iterable, List, Optional, Tuple

from sqlglot import exp

from services import sql_tree
from services.fulltext.text_search import Clause, read_clauses
from routers.schemas import ChipKind
from services import asked
from services.results.display.chips import describe
from services.uncertain.scanner import field_strength_regex

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# Postgres's English stopwords (tsearch_data/english.stop). Inside a phrase
# query they collapse it: to_tsquery('english', 'scanning <-> of') is 'scan'.
_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be been
being have has had having do does did doing a an the and but if or because as
until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don should now
""".split())

# A highlight term: one word, or two words that count only side by side.
HighlightTerm = Tuple[str, ...]


def _is_acronym(word: str) -> bool:
    """'MRI', 'fMRI', 'T1w', 'ADHD'; not 'Healthy' or 'Single'."""
    return any(c.isupper() for c in word[1:]) or (
        any(c.isdigit() for c in word) and any(c.isalpha() for c in word))


def _phrase_terms(phrase: str, acronyms_alone: bool = False) -> List[HighlightTerm]:
    """Highlight terms of one searched phrase.

    One word stands alone. Several words only count as adjacent pairs:
    "control participants" alone would mark every "participants", and
    stemming makes it worse ("subjective" marks "subject", "structural"
    marks the "Data Structure" of every BIDS README). *acronyms_alone*:
    acronyms in a coded label ("structural MRI") also stand alone.

    A digit stays as part of a pair ("7 tesla", "2 back"), never alone.
    """
    tokens = [w for w in _WORD_RE.findall(phrase)
              if (len(w) > 1 or w.isdigit()) and w.lower() not in _STOPWORDS]
    if len(tokens) == 1:
        return [(tokens[0],)] if len(tokens[0]) > 1 else []
    terms: List[HighlightTerm] = [(a.lower(), b.lower()) for a, b in zip(tokens, tokens[1:])]
    if acronyms_alone:
        terms.extend((w,) for w in tokens if _is_acronym(w))
    return terms


def highlight_terms(sql: Optional[str]) -> List[HighlightTerm]:
    """What to mark in a README: the chips' text words, and the labels of
    coded chips ("fMRI", "ADHD"), which READMEs often spell out too.

    Every chip that is not negated, whatever its kind: a dataset can match a
    chip in several ways (recorded ages, coded diagnoses, files, text), and
    the words of each way are marked where the text has them. An age chip
    carries its group's words ("infant", "newborn").

    Single words keep their case: a two-letter acronym ("MS", "PD") only
    counts in capitals, not as "ms" or "pd".
    """
    terms: List[HighlightTerm] = []
    for chip in describe(sql):
        if chip.negated:
            continue
        for phrase in chip.text_words:
            terms.extend(_phrase_terms(phrase))
        if chip.kind in (ChipKind.DIAGNOSIS, ChipKind.MODALITY, ChipKind.TASK):
            for name in chip.values[:chip.shown] or [chip.label]:
                terms.extend(_phrase_terms(name, acronyms_alone=True))
    seen, unique = set(), []
    for term in terms:
        key = tuple(w.lower() for w in term)
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def highlight_exact_words(sql: Optional[str]) -> FrozenSet[str]:
    """The words of highlight_terms(sql) that only count as written.

    A search limited to word forms ("ageing", "aging") must not mark what
    merely shares their stem ("age") in an excerpt.
    """
    return frozenset(form for clause in _asked_clauses(sql) for form in clause.guard_forms())


def _asked_clauses(sql: Optional[str]) -> List[Clause]:
    """The conditions of *sql* that are not negated, without their parentheses."""
    tree = sql_tree.parse_select(sql or "")
    if tree is None:
        return []
    asked = []
    for clause in read_clauses(tree):
        inner = clause.node
        negated = False
        while isinstance(inner, (exp.Paren, exp.Not)):
            negated ^= isinstance(inner, exp.Not)
            inner = inner.this
        if not negated:
            asked.append(clause.part(inner))
    return asked


# ── Field strength in the text ────────────────────────────────────────────────
#
# A field-strength condition is a number range on the sidecar key
# MagneticFieldStrength ("> 6.5 AND < 7.5"); it has no words to mark. READMEs write
# the strength as "7T", "7 T", "7-Tesla", "3.0T", "1.5 Tesla". These are the
# strengths scanners are sold at; a range marks the ones inside it.

_NUMBER = r"\d+(?:\.\d+)?"
# A searched phrase that is itself a strength: "7T", "7 tesla", "7tesla".
_STRENGTH_PHRASE_RE = re.compile(rf"^\s*({_NUMBER})\s*-?\s*(?:t|tesla)\s*$", re.IGNORECASE)


def highlight_field_strengths(sql: Optional[str]) -> List[float]:
    """The field strengths (tesla) to mark in a dataset's text: those a
    condition asks for, and searched phrases that name one ("7 tesla")."""
    strengths: List[float] = []
    for clause in _asked_clauses(sql):
        strengths.extend(asked.field_strengths(clause.node))
    for chip in describe(sql):
        if chip.negated:
            continue
        for phrase in chip.text_words:
            found = _STRENGTH_PHRASE_RE.match(phrase)
            if found:
                strengths.append(float(found.group(1)))
    return list(dict.fromkeys(strengths))


def field_strength_pattern(strengths: Iterable[float]) -> Optional["re.Pattern[str]"]:
    """Where a text names one of *strengths*: "7T", "7 T", "7-Tesla", "7.0T".

    Not inside another number ("17T", "0.7T"), a BIDS label ("ses-7T"), a
    word starting with T ("7 T1w images", "3 The") or a gradient spec
    ("200 T/m/s"). Spaces include the thin ones some READMEs use. Checked on
    the 1,720 OpenNeuro READMEs of 2026-11: no false matches found.
    """
    pattern = field_strength_regex(strengths)
    return re.compile(pattern) if pattern else None


def field_strength_tsquery(strengths: Iterable[float]) -> Optional[str]:
    """Postgres words for the strengths, to pick the README passage.

    The english parser keeps "7T" and "7Tesla" as one word; "T" is a
    stopword, so "1.5T" and "7 T" become the bare number. A decimal strength
    is therefore searched as its number; field_strength_pattern decides what
    is marked.
    """
    parts = []
    for value in dict.fromkeys(strengths):
        number = f"{value:g}"
        if "." in number:
            parts.append(number)
        else:
            parts.append(f"{number}t | {number}tesla | {number} <-> tesla")
    return " | ".join(parts) or None


def terms_tsquery(terms: Iterable[HighlightTerm]) -> Optional[str]:
    """An OR tsquery of *terms*; a pair matches side by side or with one
    word between ("Parkinson's disease", "scanning of one")."""
    parts = []
    for term in terms:
        words = [w.lower() for w in term]
        if len(words) == 1:
            parts.append(words[0])
        else:
            a, b = words
            parts.append(f"({a} <-> {b} | {a} <2> {b})")
    return " | ".join(parts) or None


def highlight_tsquery(sql: Optional[str]) -> Optional[str]:
    """The tsquery that marks *sql*'s highlight terms in a README."""
    return terms_tsquery(highlight_terms(sql))
