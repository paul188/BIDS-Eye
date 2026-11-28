"""
participant_codes.py
--------------------
The stored code a word stands for, for the participant columns that hold
codes of the curated vocabulary: sex and handedness.

bids_participants.sex and .handedness hold one of STORED_CODES, or NULL; the
crawler's remote/participant_codes.py writes them from the sections of the
same name in value_mappings.yaml. Whoever reads such a value (from a request,
from a literal the SQL model wrote, for a chip) asks stored_code(), so that
no module keeps its own list of spellings.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, Optional

log = logging.getLogger(__name__)

# What each column holds. The vocabulary's entries for a value that is not
# known (unknown_missing_sex, unknown_handedness) are not among them: not
# known is stored as NULL.
STORED_CODES = {
    "sex": ("male", "female", "other_diverse"),
    "handedness": ("right_handed", "left_handed", "ambidextrous"),
}
# A number says nothing by itself: datasets use 1/2 and 0/1 both ways round.
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?")
_SEPARATORS = re.compile(r"[\s_-]+")


@lru_cache(maxsize=None)
def _spellings(field: str) -> Dict[str, str]:
    """label, synonym or code (lower case) → standard code, for one section.

    Empty when the vocabulary cannot be loaded; cached, so the import is not
    tried again on every call. The stored codes are then still read as
    themselves.
    """
    import yaml
    try:
        from yaml_to_llamaindex import leaf_db
    except (ImportError, OSError, yaml.YAMLError) as exc:
        log.warning("curated vocabulary not available (%s); only the stored codes are read as %s",
                    exc, field)
        return {}
    return dict(leaf_db.get(field, {}))


def stored_code(field: str, word: str) -> Optional[str]:
    """The stored code *word* stands for in *field* ('F', 'Women', 'left
    hander', 'LH'), else None. Words are matched whatever separates their
    parts: 'left_hander', 'left-hander' and 'left hander' are one."""
    key = str(word or "").strip().lower()
    if not key or _NUMBER.fullmatch(key):
        return None
    table = _spellings(field)
    stored = STORED_CODES[field]
    for form in dict.fromkeys((key, _SEPARATORS.sub(" ", key), _SEPARATORS.sub("_", key),
                               _SEPARATORS.sub("-", key))):
        code = table.get(form, form)
        if code in stored:
            return code
    return None


def sex_code(word: str) -> Optional[str]:
    return stored_code("sex", word)


def handedness_code(word: str) -> Optional[str]:
    return stored_code("handedness", word)
