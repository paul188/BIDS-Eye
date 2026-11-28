"""What the planning model is told about a question besides its text.

The question goes to the model as typed (intent.py). Next to it go facts the
catalogue holds about its words, so the model does not have to guess them:

  shorthands       the abbreviations of the question with what the listings
                   say they stand for (listed_shorthands)
  stem collisions  for a word whose stem also matches unrelated words
                   ("ageing" and "age"), the catalogue's forms with their
                   dataset counts; the model names the forms that mean the
                   same (count_stem_forms, decided in
                   backend/vocabulary/stem_collisions.py)

planner_input() turns both into the extra input fields and instructions of
one prompt. The stem counts are looked up by the caller (pipeline.py), which
also needs them afterwards to decide the forms; the shorthands are looked up
here.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import catalogue as _catalogue

log = logging.getLogger("preprocess")


# Added to the intent prompt only for a question with a word whose stem
# collides (stem_collisions.prompt_block gives the lines above it). The examples
# are not benchmark words.
_SAME_MEANING_INSTRUCTION = """\
For these words, add a top-level key "same_meaning": a list of
{"word": "<the word>", "forms": ["<form>", ...]} with the forms a dataset must
use to match what the user means. Drop forms that are a different word with
the same stem ("organ" for "organization", "universe" for "university").
If every form fits, list them all."""


_SHORTHANDS_INSTRUCTION = """\
## Shorthands
`shorthands` lists what each abbreviation in the query stands for in the
catalogue (several entries = several possible meanings).
- Write each abbreviation exactly as typed in every field, including
  natural_language_summary. Do not expand it or pick one meaning.
- Use the list only to decide which field it belongs in and which other words
  describe it."""

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


@dataclass
class PlannerInput:
    """The extra input of one planning prompt, beside the question itself."""
    fields: Dict[str, Any] = field(default_factory=dict)   # added to the input JSON
    instructions: List[str] = field(default_factory=list)  # appended after it, in order
    asks_same_meaning: bool = False   # the answer carries a "same_meaning" key


def planner_input(query: str, stem_counts: Optional[Dict[str, Any]] = None) -> PlannerInput:
    """The facts about *query* for its planning prompt.

    *stem_counts*: count_stem_forms(query), or None. Without colliding words
    and shorthands the result is empty and the prompt is the question alone.
    """
    extra = PlannerInput()
    forms_block = stem_collisions_block(stem_counts)
    if forms_block:
        extra.fields["catalogue_stem_collisions"] = forms_block
    shorthands = listed_shorthands(query)
    if shorthands:
        extra.fields["shorthands"] = shorthands
        extra.instructions.append(_SHORTHANDS_INSTRUCTION)
    if forms_block:
        extra.instructions.append(_SAME_MEANING_INSTRUCTION)
        extra.asks_same_meaning = True
    return extra


def listed_shorthands(query: str) -> Dict[str, List[str]]:
    """The abbreviations of *query* with what the listings say they stand for.

    {"ERP": ["event-related potential"]}; {} when the query has none that a
    listing explains besides the plain scan abbreviations, or the vocabulary
    is not importable. The language model gets this
    instead of interpreting the letters itself: it read "APT" as "Arterial
    Perfusion Tagging" (2026-11-23).
    """
    try:
        from retriever import _PURE_MODALITY_TERMS, is_shorthand, shorthand_meanings
        from text_terms import POSTGRES_STOPWORDS
    except ImportError as exc:
        log.warning("Shorthands left out of the prompt (%s)", exc)
        return {}
    found: Dict[str, List[str]] = {}
    for token in _TOKEN_RE.findall(query or ""):
        if token in found or token.lower() in POSTGRES_STOPWORDS or not is_shorthand(token):
            continue
        if token.lower() in _PURE_MODALITY_TERMS:
            # "fMRI", "EEG", "MRI": the model knows them and 60 of the 70 gold
            # requests carry one. Listing them changed plans that were right
            # ("MRI" became "functional MRI", "M/EEG" two required scans):
            # recall 72% -> 65% (2026-11-23).
            continue
        meanings = shorthand_meanings(token)
        if meanings:
            # One no listing explains ("TR", "GB", "JSON", a dataset's name)
            # is left to the model as before; the other words it proposes
            # for such a term are dropped (topics._checked_alternatives).
            found[token] = meanings
    return found


def count_stem_forms(query: str) -> Dict[str, Any]:
    """The catalogue's forms of the question's words (backend/vocabulary/stem_collisions.py).

    {} when the exact-form search is off (TEXT_SEARCH_EXACT), outside the backend,
    or when the lookup fails: the search then works by stem, as before.
    """
    try:
        import stem_collisions
        vectors = _catalogue.vectors()  # backend/services
        word_form_counts = _catalogue.word_form_counts()
    except ImportError:  # outside the backend
        return {}
    if not vectors.exact_enabled():
        return {}
    # {} when the catalogue cannot answer (word_form_counts.lookup catches that).
    return word_form_counts.lookup(stem_collisions.question_words(query))


def stem_collisions_block(stem_counts: Optional[Dict[str, Any]]) -> str:
    """The lines of the prompt that show the colliding words' forms; "" for none."""
    if not stem_counts:
        return ""
    try:
        import stem_collisions
    except ImportError as exc:
        log.warning("Stem collisions left out of the prompt (%s)", exc)
        return ""
    return stem_collisions.prompt_block(stem_collisions.needs_decision(stem_counts))
