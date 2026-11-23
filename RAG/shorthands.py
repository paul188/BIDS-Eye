"""
RAG/shorthands.py
-----------------
The written-out forms of a shorthand, from shorthands.yaml.

A shorthand ("ERP", "CEST") is searched as typed; what it stands for is
listed, not asked from the language model. The forms listed here are searched
as its other words (LLM_preprocessor/preprocess/topics.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

_PATH = Path(__file__).with_name("shorthands.yaml")


@lru_cache(maxsize=1)
def _listing() -> Dict[str, List[str]]:
    with open(_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return {str(key).strip().lower(): [str(form).strip() for form in forms or [] if str(form).strip()]
            for key, forms in raw.items()}


def written_out(term: str) -> List[str]:
    """Every listed written-out form of *term*; [] for a shorthand that has no entry."""
    return list(_listing().get((term or "").strip().lower(), []))


def listed() -> List[str]:
    """The shorthands of shorthands.yaml, lower-cased."""
    return sorted(_listing())
