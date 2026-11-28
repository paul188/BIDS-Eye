"""
code_names.py
-------------
The name a stored code is shown under ("fmri_bold" → "fMRI",
"parkinsons_disease" → "Parkinson's disease"), for chips, summaries and
anything else a person reads.

The names are the entries' short_label, else their label, in
value_mappings.yaml; no module keeps its own table of names. Labels are
written as they read inside a sentence (tools/label_case.py), so a
caller that starts a line with one capitalises it itself.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _names() -> Dict[str, Dict[str, str]]:
    """{field: {standard_code: name}}; empty when the vocabulary cannot be
    loaded, cached so the import is not tried again on every call."""
    import yaml
    try:
        from yaml_to_llamaindex import code_name_db
    except (ImportError, OSError, yaml.YAMLError) as exc:
        log.warning("curated vocabulary not available (%s); codes are shown as written", exc)
        return {}
    return code_name_db


def code_name(field: str, code: str) -> str:
    """The name of *code* in *field*; the code with spaces for underscores
    when the vocabulary has no entry for it."""
    code = str(code or "").strip()
    return _names().get(field, {}).get(code) or code.replace("_", " ")
