"""
RAG/word_forms.py
-----------------
Which forms of a searched word may match.

Full-text search matches a word by its stem (PostgreSQL's 'english'
configuration), so "depression" also finds "depressive". Usually that is
wanted. Sometimes the stem merges words that mean different things: "ageing"
shares the stem of "age" and "aged", "weight" that of "T1-weighted", "severe"
that of "several". A search for the first then returns every dataset with the
second.

For such a word the search can require the forms that were asked for
(backend/services/fulltext/text_search.py writes the condition). This module decides
which forms those are. It has no database access; the catalogue's forms come
in as WordEvidence (backend/services/catalogue/word_form_store.py).

A word's forms are decided by, in this order:
  1. word_forms.yaml, a short reviewed list. It applies to every searched
     word, also to synonyms a vocabulary or the language model added.
  2. the language model, for words of the question whose stem collides
     (is_collision). It sees the forms with their dataset counts
     (prompt_block) and names the ones to keep; decide() checks its answer.
  3. nobody: the word is searched by its stem, as before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

log = logging.getLogger(__name__)

_PATH = Path(__file__).with_name("word_forms.yaml")
_WORD_RE = re.compile(r"[a-z]+")
# A form goes into a tsquery literal as it is, so nothing else may pass.
_FORM_RE = re.compile(r"[a-z0-9]+")

# A stem collides when it matches at least MIN_RATIO times the datasets of the
# asked word's own forms, and at least MIN_EXTRA more. Tune these on
# development questions only (benchmarks/recall/held_out_questions.txt).
MIN_RATIO = 2.0
MIN_EXTRA = 10
# Forms shown to the language model per word, most frequent first.
PROMPT_FORMS = 8
# Forms a search requires at most (the most frequent are kept).
MAX_KEPT_FORMS = 12
# Words of a question looked up at most.
MAX_QUESTION_WORDS = 40

# word → the forms that may match (asked word first)
ExactForms = Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class WordEvidence:
    """What the catalogue says about one searched word."""
    word: str
    stem: str
    stem_docs: int                          # datasets the stemmed search matches
    forms: Tuple[Tuple[str, int], ...]      # (form, datasets with it), most frequent first

    def count(self, form: str) -> int:
        return next((n for f, n in self.forms if f == form), 0)


def question_words(question: str) -> List[str]:
    """The words of *question* worth looking up, in order, without repeats."""
    words = [w for w in _WORD_RE.findall(question.lower()) if len(w) >= 3]
    return list(dict.fromkeys(words))[:MAX_QUESTION_WORDS]


def inflections(word: str) -> Set[str]:
    """*word* with its regular plural and singular spellings.

    Deliberately crude: it is only ever intersected with forms that exist in
    the catalogue and share the word's stem.
    """
    found = {word, word + "s", word + "es"}
    if word.endswith("y"):
        found.add(word[:-1] + "ies")
    if word.endswith("ies"):
        found.add(word[:-3] + "y")
    if word.endswith("es"):
        found.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        found.add(word[:-1])
    return found


def family_docs(evidence: WordEvidence) -> int:
    """Datasets with the asked word itself (or its plural or singular).

    The largest single count, a lower bound: the forms mostly occur together.
    """
    own = inflections(evidence.word)
    return max((n for form, n in evidence.forms if form in own), default=0)


def is_collision(evidence: WordEvidence) -> bool:
    """True when the stem brings in far more datasets than the word's own forms."""
    own = family_docs(evidence)
    return (evidence.stem_docs >= MIN_RATIO * max(own, 1)
            and evidence.stem_docs - own >= MIN_EXTRA)


# ── Reviewed list ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _reviewed() -> Dict[str, Optional[Tuple[str, ...]]]:
    """word → its forms, or None where stemming is right. From word_forms.yaml."""
    try:
        data = yaml.safe_load(_PATH.read_text()) or {}
    except Exception as exc:  # a broken list must not stop the search
        log.warning("word_forms.yaml could not be read (%s) — ignoring it", exc)
        return {}
    found: Dict[str, Optional[Tuple[str, ...]]] = {}
    for entry in data.get("exact") or []:
        forms = tuple(dict.fromkeys(
            f for f in (str(f).lower() for f in entry.get("forms") or []) if _FORM_RE.fullmatch(f)))
        for form in forms:
            found[form] = forms
    for group in data.get("stemmed") or []:
        for word in group or []:
            found[str(word).lower()] = None
    return found


def reviewed(word: str) -> Tuple[bool, Optional[Tuple[str, ...]]]:
    """(listed, forms) for *word*; forms is None when the list says to stem it."""
    table = _reviewed()
    for candidate in (word, *sorted(inflections(word) - {word})):
        if candidate in table:
            forms = table[candidate]
            if forms is None:
                return True, None
            return True, _asked_first(word, forms)
    return False, None


def _asked_first(word: str, forms: Iterable[str]) -> Tuple[str, ...]:
    return tuple(f for f in dict.fromkeys([word, *forms]) if _FORM_RE.fullmatch(f))


# ── Deciding ──────────────────────────────────────────────────────────────────

def needs_decision(evidence: Mapping[str, WordEvidence]) -> List[WordEvidence]:
    """The words the language model is asked about: colliding and not reviewed."""
    return [e for e in evidence.values() if is_collision(e) and not reviewed(e.word)[0]]


def prompt_block(evidence: Sequence[WordEvidence]) -> str:
    """The catalogue's forms of each word in *evidence*, for the intent prompt.

    "" when there is nothing to ask.
    """
    lines = []
    for item in evidence:
        shown = list(item.forms[:PROMPT_FORMS])
        if item.word not in {form for form, _ in shown}:
            shown.append((item.word, item.count(item.word)))
        forms = ", ".join(f"{form} ({n})" for form, n in shown)
        lines.append(f"  {item.word}: {forms}")
    if not lines:
        return ""
    return ("Word forms in the catalogue. The search finds a word by its stem, so for each\n"
            "of these words of the query it also finds the other forms listed (in brackets:\n"
            "datasets that contain the form). Fill word_forms for them:\n" + "\n".join(lines))


def _complete(kept: Sequence[str], evidence: WordEvidence) -> List[str]:
    """*kept* plus the catalogue's regular inflections of each kept form."""
    wanted: Set[str] = set()
    for form in kept:
        wanted |= inflections(form)
    extra = [form for form, _ in evidence.forms if form in wanted and form not in kept]
    return list(kept) + extra


def decide(evidence: Mapping[str, WordEvidence],
           choices: Iterable[Tuple[str, Sequence[str]]]) -> ExactForms:
    """The forms each word of the question must match in; {} for none.

    *choices* are the language model's (word, forms to keep) answers. Anything
    that does not fit the evidence is dropped, never an error: a word that was
    not asked about, a form the catalogue does not have under that stem. A
    choice that keeps every form shown restricts nothing.
    """
    decided: ExactForms = {}
    asked = {e.word: e for e in needs_decision(evidence)}
    for word, keep in choices:
        word = str(word).strip().lower()
        item = asked.get(word)
        if item is None or word in decided:
            continue
        known = {form for form, _ in item.forms}
        kept = [f for f in dict.fromkeys(str(k).strip().lower() for k in keep) if f in known]
        kept = _complete(_asked_first(word, kept), item)
        shown = {form for form, _ in item.forms[:PROMPT_FORMS]}
        if not shown - set(kept):
            continue   # every form shown was kept: stemming is what was meant
        by_count = sorted(kept[1:], key=lambda form: -item.count(form))
        decided[word] = (word, *by_count[:MAX_KEPT_FORMS - 1])
    return decided


def forms_for(word: str, decided: Optional[Mapping[str, Sequence[str]]] = None,
              ) -> Optional[Tuple[str, ...]]:
    """The forms a search for *word* requires, or None to search by stem.

    The reviewed list wins over *decided* (the language model's choices).
    """
    listed, forms = reviewed(word)
    if listed:
        return forms
    chosen = (decided or {}).get(word)
    return _asked_first(word, chosen) if chosen else None
