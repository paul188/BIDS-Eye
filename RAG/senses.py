"""
RAG/senses.py
-------------
Senses of ambiguous topic words, from senses.yaml.

"conflict" can mean conflict between people, response conflict in a Stroop
task or an approach-avoidance conflict. When a question qualifies the word
("social conflict"), pick_sense() returns the sense that qualifier selects,
with the phrases and task codes that stand for it and the words that mark the
other senses. The topic search (LLM_preprocessor/preprocess.py) adds the
former and excludes the latter.

The inventory is fixed and small on purpose: it makes the frequent cases
independent of how the language model phrases them on a given day.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

_PATH = Path(__file__).with_name("senses.yaml")
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


@dataclass(frozen=True)
class Sense:
    head: str
    name: str
    cues: Tuple[str, ...]
    alternatives: Tuple[str, ...]
    tasks: Tuple[str, ...]
    marks: Tuple[str, ...]


@dataclass
class SensePick:
    """What the inventory says about one topic."""
    head: str
    sense: Optional[Sense]                  # None: no qualifier picked one
    others: List[Sense] = field(default_factory=list)

    @property
    def alternatives(self) -> List[str]:
        return list(self.sense.alternatives) if self.sense else []

    @property
    def tasks(self) -> List[str]:
        return list(self.sense.tasks) if self.sense else []

    @property
    def not_meant(self) -> List[str]:
        """Marks of the other senses, minus anything the picked sense uses."""
        if self.sense is None:
            return []
        own = {m.lower() for m in self.sense.alternatives + self.sense.marks}
        marks = [m for other in self.others for m in other.marks if m.lower() not in own]
        return list(dict.fromkeys(marks))

    @property
    def excluded_tasks(self) -> List[str]:
        if self.sense is None:
            return []
        own = set(self.sense.tasks)
        return list(dict.fromkeys(t for other in self.others for t in other.tasks if t not in own))


def _words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@lru_cache(maxsize=1)
def inventory() -> Dict[str, List[Sense]]:
    """head → its senses, in file order. {} if the file is missing."""
    if not _PATH.exists():
        return {}
    data = yaml.safe_load(_PATH.read_text(encoding="utf-8")) or {}
    result: Dict[str, List[Sense]] = {}
    for head, senses in data.items():
        result[str(head).lower()] = [
            Sense(
                head=str(head).lower(),
                name=str(name),
                cues=tuple(str(c).lower() for c in entry.get("cues") or []),
                alternatives=tuple(str(a) for a in entry.get("alternatives") or []),
                tasks=tuple(str(t) for t in entry.get("tasks") or []),
                marks=tuple(str(m) for m in entry.get("marks") or []),
            )
            for name, entry in (senses or {}).items()
        ]
    return result


def senses_of(head: str) -> List[Sense]:
    words = _words(head)
    if not words:
        return []
    key = " ".join(words)
    senses = inventory()
    return senses.get(key) or senses.get(_singular(key)) or []


def _word_forms(texts: Iterable[str]) -> set:
    words = set()
    for text in texts:
        for word in _words(text):
            words.update({word, _singular(word)})
    return words


def pick_sense(head: str, qualifiers: Iterable[str], phrase: str = "") -> Optional[SensePick]:
    """The sense the qualifiers select; None when *head* is not in the inventory.

    A sense is picked when the qualifiers name more of its cues than of any
    other sense's. A cue in the *phrase* itself counts twice: in
    "psychosocial stress induction" the word next to "stress" decides, not
    "induction". A tie, or no cue at all, picks none: the question does not
    say which sense it means.
    """
    senses = senses_of(head)
    if not senses:
        return None
    in_phrase = _word_forms([phrase])
    words = _word_forms(qualifiers) | in_phrase

    def score(sense: Sense) -> int:
        return sum((2 if cue in in_phrase else 1) for cue in sense.cues if cue in words)

    hits = sorted(((score(s), i) for i, s in enumerate(senses)), reverse=True)
    best, index = hits[0]
    if best == 0 or (len(hits) > 1 and hits[1][0] == best):
        return SensePick(head=senses[0].head, sense=None, others=list(senses))
    chosen = senses[index]
    return SensePick(head=chosen.head, sense=chosen,
                     others=[s for s in senses if s is not chosen])
