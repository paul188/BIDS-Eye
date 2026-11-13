"""
RAG/text_terms.py
-----------------
Terms for full-text search over dataset text (name, description, study
fields, README).

The SQL model never writes a text predicate itself. It copies a placeholder,

    text_match('epilepsy | epileptic | seizure', 'short')

and backend/services/text_search.py expands it into real SQL. This module owns
the placeholder: how a user term is split into words, which alternatives a
diagnosis gets, and how the placeholder is written and read back.

Placeholder grammar
    text_match('<alternatives>'[, '<mode>'])
    alternatives  one or more terms separated by " | "; any one may match
    term          words separated by spaces; every word must match
    word          letters and digits, hyphenated compounds stay one word
                  ("resting-state", "n-back") and must appear adjacent
    mode          all    every text field including the README (default)
                  short  name, description, study fields and the conditions
                         an LLM read as studied in the README
                  loose  every text field; only the rarest words of each term
                         are required (as many as still match a few
                         datasets), the others only raise the rank

Only [a-z0-9 |-] ever appears inside the quotes, so the placeholder survives
being copied by a language model and can be inlined into SQL safely.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

MODE_ALL = "all"
MODE_SHORT = "short"
MODE_LOOSE = "loose"
MODES = (MODE_ALL, MODE_SHORT, MODE_LOOSE)

# A word is a list of alphanumeric parts: "n-back" → ["n", "back"].
Word = List[str]
Term = List[Word]

_PART_RE = re.compile(r"[a-z0-9]+")
_POSSESSIVE_RE = re.compile(r"['’`]s\b")
_APOSTROPHE_RE = re.compile(r"['’`]")
_PARENTHESES_RE = re.compile(r"\([^)]*\)")

# Function words carry no search value; PostgreSQL drops them as well.
_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with",
})

# "Parkinson's disease" should find "Basal Ganglia in Parkinson".
_GENERIC_DIAGNOSIS_WORDS = frozenset({
    "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
    "spectrum",
})
# A synonym such as "epileptic patients" only repeats a shorter one; one such
# as "normal subjects" would shrink to a word that matches almost anything.
_POPULATION_WORDS = frozenset({
    "adult", "adults", "child", "children", "cohort", "group", "groups",
    "individual", "individuals", "participant", "participants", "patient",
    "patients", "people", "person", "subject", "subjects", "volunteer",
    "volunteers",
})

# The user's own words and the concept label are trusted down to three
# letters ("ASD", "OCD"). Two letters are not: "PD" is also proton density.
_MIN_TRUSTED_PART = 3
# Vocabulary synonyms were written to recognise user input, not to search
# documents: "load", "dat" and "fad" are listed for Alzheimer's disease.
_MIN_SYNONYM_PART = 6
MAX_DIAGNOSIS_ALTERNATIVES = 16


# ── Tokenising ────────────────────────────────────────────────────────────────

def _fold(text: str) -> str:
    """Lower-case and strip accents, possessives and apostrophes."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = _POSSESSIVE_RE.sub("", text)
    return _APOSTROPHE_RE.sub("", text)


def split_words(term: str) -> Term:
    """Split one search term into words, each a list of alphanumeric parts.

    "Resting-state fMRI of the brain" → [["resting", "state"], ["fmri"], ["brain"]]
    """
    words: Term = []
    for chunk in _fold(term).split():
        parts = _PART_RE.findall(chunk)
        if not parts or (len(parts) == 1 and parts[0] in _STOPWORDS):
            continue
        words.append(parts)
    return words


def _canonical(term: Term) -> str:
    return " ".join("-".join(word) for word in term)


def parse_alternatives(argument: str) -> List[Term]:
    """Read the first argument of a text_match() placeholder."""
    terms: List[Term] = []
    seen = set()
    for raw in (argument or "").split("|"):
        term = split_words(raw)
        key = _canonical(term)
        if term and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def format_text_match(alternatives: Iterable[str], mode: str = MODE_ALL) -> Optional[str]:
    """Write a text_match() placeholder, or None if nothing searchable is left."""
    if mode not in MODES:
        raise ValueError(f"unknown text_match mode: {mode!r}")
    terms = parse_alternatives(" | ".join(a for a in alternatives if a))
    if not terms:
        return None
    argument = " | ".join(_canonical(term) for term in terms)
    return f"text_match('{argument}')" if mode == MODE_ALL else f"text_match('{argument}', '{mode}')"


# ── Diagnosis alternatives ────────────────────────────────────────────────────

def _longest_part(term: Term) -> int:
    """Length of the longest part, not counting generic words.

    "oc disorder" scores 2: "disorder" says nothing about the condition.
    """
    return max(
        (len(part) for word in term for part in word
         if not (len(word) == 1 and word[0] in _GENERIC_DIAGNOSIS_WORDS)),
        default=0,
    )


def _flat(term: Term) -> List[str]:
    return [part for word in term for part in word]


def _core_words(text: str) -> Term:
    """Words of a diagnosis string without generic and population words."""
    words = split_words(_PARENTHESES_RE.sub(" ", text or ""))
    core = [
        word for word in words
        if not (len(word) == 1 and word[0] in _GENERIC_DIAGNOSIS_WORDS | _POPULATION_WORDS)
    ]
    return core or words


def _mentions_population(text: str) -> bool:
    return any(
        len(word) == 1 and word[0] in _POPULATION_WORDS
        for word in split_words(text)
    )


@lru_cache(maxsize=1)
def _vocabulary_strings() -> Dict[str, List[Tuple[str, float]]]:
    """code → [(label or synonym, weight)], the label first.

    Reuses the knowledge base the retriever has already parsed; parsing
    value_mappings.yaml a second time would cost several seconds.
    """
    from yaml_to_llamaindex import display_db, weight_db

    weights = weight_db.get("diagnosis", {})
    by_code: Dict[str, List[Tuple[str, float]]] = {}
    for string, code in display_db.get("diagnosis", {}).items():
        by_code.setdefault(code, []).append((string, float(weights.get(string, 1.0))))
    return by_code


def _lookup_strings(code: str) -> List[Tuple[str, float]]:
    try:
        return _vocabulary_strings().get(code, [])
    except Exception:
        # The vocabulary only widens the search; without it the user's own
        # words still apply.
        return []


def diagnosis_alternatives(
    term_to_codes: Dict[str, Sequence[str]],
    lookup: Optional[Callable[[str], List[Tuple[str, float]]]] = None,
    limit: int = MAX_DIAGNOSIS_ALTERNATIVES,
) -> List[str]:
    """Search terms that back up a structured diagnosis filter.

    *term_to_codes* maps each user term to the codes it resolved to, best
    match first. The result holds, in this order,
      1. the user's own terms,
      2. label and synonyms of each term's best code,
      3. labels of the other codes,
    without an alternative that a shorter one already covers, and capped at
    *limit*. Generic words ("disease", "disorder") are dropped from the user's
    terms and the best code's label, never from a synonym or another label.
    """
    lookup = lookup or _lookup_strings
    candidates: List[Tuple[Term, int]] = []   # (words, minimum longest part)

    for term in term_to_codes:
        candidates.append((_core_words(term), _MIN_TRUSTED_PART))

    other_labels: List[Tuple[Term, int]] = []
    for codes in term_to_codes.values():
        for position, code in enumerate(codes):
            strings = lookup(code)
            if not strings:
                continue
            label, synonyms = strings[0][0], strings[1:]
            if position > 0:
                # Kept whole: "Watanabe syndrome" without "syndrome" is a
                # common author name.
                other_labels.append(
                    (split_words(_PARENTHESES_RE.sub(" ", label)), _MIN_TRUSTED_PART)
                )
                continue
            candidates.append((_core_words(label), _MIN_TRUSTED_PART))
            for synonym, _weight in sorted(synonyms, key=lambda item: -item[1]):
                if not _mentions_population(synonym):
                    # Kept whole: without its "disorder", "reading disorder"
                    # would match every reading study.
                    words = split_words(_PARENTHESES_RE.sub(" ", synonym))
                    candidates.append((words, _MIN_SYNONYM_PART))
    candidates.extend(other_labels)

    kept: List[Term] = []
    kept_sets: List[frozenset] = []
    for term, min_part in candidates:
        if not term or _longest_part(term) < min_part:
            continue
        parts = frozenset(_flat(term))
        # "refractory epilepsy" adds nothing once "epilepsy" is searched.
        if any(existing <= parts for existing in kept_sets):
            continue
        # A shorter newcomer replaces the longer terms it covers.
        covered = [i for i, existing in enumerate(kept_sets) if parts < existing]
        for index in reversed(covered):
            del kept[index], kept_sets[index]
        kept.append(term)
        kept_sets.append(parts)
        if len(kept) >= limit:
            break
    return [_canonical(term) for term in kept]
