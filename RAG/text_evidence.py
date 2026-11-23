"""
text_evidence.py
----------------
The words a dataset's own text (name, description, README) uses for something
a column may not record: "22 women" for participant sex, "adolescents" for
ages.

They are written in value_mappings.yaml and read here, so that no module
keeps its own list:

    text_words of an entry    the words for one stored code (sex: male, female)
    text_evidence section     words that are not one column's values: the
                              people a study is about, the age groups

They are not the request synonyms. A request may say "M" or "AMAB" for male;
a README that names its participants says "men" or "12 males".

Every function returns nothing (an empty tuple or dict) when the vocabulary
cannot be loaded; the caller then has no text evidence for the field.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _loaded() -> Tuple[Dict[str, Dict[str, list]], Dict[str, Any]]:
    """(text_words by category and code, the text_evidence section)."""
    import yaml
    try:
        from yaml_to_llamaindex import text_evidence_db, text_words_db
    except (ImportError, OSError, yaml.YAMLError) as exc:
        log.warning("curated vocabulary not available (%s); no text evidence words", exc)
        return {}, {}
    return text_words_db, text_evidence_db


def text_words(field: str) -> Dict[str, Tuple[str, ...]]:
    """stored code → the words a dataset's text uses for it, for one column
    ('sex' → {'male': ('male', 'males', 'man', …), 'female': (…)})."""
    return {code: tuple(words) for code, words in _loaded()[0].get(field, {}).items()}


def people_words() -> Tuple[str, ...]:
    """Words for the people a study is about: 'participants', 'subjects', …"""
    return tuple(str(word).lower() for word in _loaded()[1].get("people_words") or ())


def age_groups() -> Tuple[Tuple[float, float, Tuple[str, ...]], ...]:
    """(from, to, words) of each age group, ages in years, in file order."""
    groups = _loaded()[1].get("age_groups") or {}
    return tuple((float(group["from"]), float(group["to"]),
                  tuple(str(word).lower() for word in group["words"]))
                 for group in groups.values())
