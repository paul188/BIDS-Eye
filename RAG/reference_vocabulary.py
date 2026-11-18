"""Diagnoses and tasks from official sources (RAG/reference_vocabulary.yaml).

The curated vocabulary (value_mappings.yaml) lists what OpenNeuro datasets
contain. This one lists what exists: every MeSH concept under Nervous System
Diseases and Mental Disorders, and every Cognitive Atlas task. It lets the
search recognise a disease or paradigm no dataset has yet ("rasmussen" is
Rasmussen encephalitis, not the author Rasmus Sinn) instead of guessing.

Only exact matches count: the term, once case, apostrophes, hyphens and a
plural are normalised, is one of the source's own terms, or a surname that
names exactly one disease. There is no fuzzy matching here; a near miss is
left to the caller's other paths.

The curated vocabulary decides first. RAG/retriever.py asks this module only
for a diagnosis term the curated one could not resolve, and
RAG/text_terms.py only for the synonyms of a code the curated one lacks.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

PATH = Path(__file__).parent / "reference_vocabulary.yaml"

# A surname may be followed by one of these and still mean the same disease:
# "rasmussen", "rasmussen's", "rasmussen disease".
_EPONYM_TAILS = ("", " disease", " syndrome")
# Words around a task name that do not change which task it is.
_TASK_SCAFFOLD = re.compile(r"\b(?:task|tasks|paradigm|paradigms|test|tests|game)\b")


def normalize(text: str) -> str:
    """'Rasmussen’s  Syndrome' → 'rasmussen syndrome'; 'Guillain-Barré' → 'guillain barre'."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = text.lower().replace("’", "'")
    text = re.sub(r"(?<=\w)'s\b|(?<=s)'(?=\s|$)", "", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _singular(key: str) -> str:
    words = key.split()
    if words and len(words[-1]) > 4 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def _task_core(key: str) -> str:
    return " ".join(_TASK_SCAFFOLD.sub(" ", key).split())


class _Index:
    def __init__(self, data: dict):
        self.entries: Dict[str, Dict[str, dict]] = {
            field: data.get(field) or {} for field in ("diagnosis", "task")
        }
        self.keys: Dict[str, Dict[str, List[str]]] = {"diagnosis": {}, "task": {}}
        for code, entry in self.entries["diagnosis"].items():
            for string in [entry.get("label", "")] + list(entry.get("terms") or []):
                self._add("diagnosis", normalize(string), code)
            for surname in entry.get("eponyms") or []:
                for tail in _EPONYM_TAILS:
                    self._add("diagnosis", normalize(surname + tail), code)
        for code, entry in self.entries["task"].items():
            for string in [entry.get("label", "")] + list(entry.get("terms") or []):
                key = normalize(string)
                self._add("task", key, code)
                self._add("task", _task_core(key), code)

    def _add(self, field: str, key: str, code: str) -> None:
        if len(key.replace(" ", "")) < 3:
            return
        for form in {key, _singular(key)}:
            codes = self.keys[field].setdefault(form, [])
            if code not in codes:
                codes.append(code)

    def lookup(self, field: str, term: str) -> List[str]:
        keys = self.keys.get(field, {})
        key = normalize(term)
        forms = [key, _singular(key)]
        if field == "task":
            forms += [_task_core(key), _singular(_task_core(key))]
        for form in forms:
            if form and form in keys:
                return list(keys[form])
        return []


@lru_cache(maxsize=1)
def _index() -> Optional[_Index]:
    try:
        with open(PATH, encoding="utf-8") as fh:
            loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
            return _Index(yaml.load(fh, Loader=loader) or {})
    except FileNotFoundError:
        return None   # without the file the search works as before
    except (OSError, yaml.YAMLError) as exc:
        log.warning("%s not read (%s); searching without it", PATH.name, exc)
        return None


def match(field: str, term: str) -> List[Tuple[str, float]]:
    """(code, score) for *term* in *field* ('diagnosis' or 'task'); [] if not listed.

    A term listed under k concepts gives each 1/k, as in the curated resolver.
    """
    index = _index()
    if index is None:
        return []
    codes = index.lookup(field, term)
    return [(code, 1.0 / len(codes)) for code in codes]


def knows(field: str, term: str) -> bool:
    return bool(match(field, term))


def near_surname(term: str) -> Optional[str]:
    """The disease surname one letter away from *term* ("rasmussehn" → "rasmussen"), if exactly one.

    Only for a single word of six letters or more with the same first letter;
    used for a name that no author in the catalogue has.
    """
    index = _index()
    word = normalize(term)
    if index is None or " " in word or len(word) < 6:
        return None
    from rapidfuzz.distance import Levenshtein
    near = {
        surname for code, found in index.entries["diagnosis"].items()
        for surname in found.get("eponyms") or []
        if "-" not in surname and len(surname) >= 6 and surname[0] == word[0]
        and surname != word and Levenshtein.distance(surname, word, score_cutoff=1) <= 1
    }
    near = {s for s in near if s.rstrip("s") not in near or s.endswith("ss")} or near
    return next(iter(near)) if len(near) == 1 else None


def entry(field: str, code: str) -> Optional[dict]:
    index = _index()
    return index.entries.get(field, {}).get(code) if index else None


def strings(code: str) -> List[Tuple[str, float]]:
    """Label and terms of a reference diagnosis, label first (as text_terms expects)."""
    found = entry("diagnosis", code)
    if not found:
        return []
    out = [found.get("label", code).lower()]
    out += [t for t in found.get("terms") or [] if t.lower() not in out]
    return [(string, 1.0) for string in out]
