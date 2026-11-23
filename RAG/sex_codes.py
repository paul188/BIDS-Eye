"""
sex_codes.py
------------
The participant sex a word stands for, from the curated vocabulary.

bids_participants.sex holds one of STORED_CODES, or NULL when the sex is not
known; the crawler's remote/participant_sex.py writes them from the `sex`
section of value_mappings.yaml. Whoever reads a sex (from a request, from a
literal the SQL model wrote, for a chip) asks sex_code(), so that no module
keeps its own list of spellings.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, Optional

log = logging.getLogger(__name__)

# What the sex column holds. The vocabulary's fourth entry, for a sex that is
# not known, is stored as NULL.
STORED_CODES = ("male", "female", "other_diverse")
# A number says nothing by itself: datasets use 1/2 and 0/1 both ways round.
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?")


@lru_cache(maxsize=1)
def _spellings() -> Dict[str, str]:
    """label, synonym or code (lower case) → standard code, for the sex entries.

    Empty when the vocabulary cannot be loaded; cached, so the import is not
    tried again on every call. The stored codes are then still read as
    themselves.
    """
    import yaml
    try:
        from yaml_to_llamaindex import leaf_db
    except (ImportError, OSError, yaml.YAMLError) as exc:
        log.warning("curated vocabulary not available (%s); only %s are read as a sex",
                    exc, ", ".join(STORED_CODES))
        return {}
    return dict(leaf_db.get("sex", {}))


def sex_code(word: str) -> Optional[str]:
    """The stored code *word* stands for ('F', 'Women', 'male'), else None."""
    key = str(word or "").strip().lower()
    if not key or _NUMBER.fullmatch(key):
        return None
    code = _spellings().get(key, key)
    return code if code in STORED_CODES else None
