"""
backend/vocabulary/group_labels.py
----------------------------------
What a group column of participants.tsv calls a participant: the names of a
diagnosis, and the names of a control group.

A group column is a column of participants.tsv, kept in
bids_participants.extra, whose key is one of GROUP_KEYS ("Group": "PD").
Its value names a diagnosis when it is one of the names the vocabulary lists
for the code (value_mappings.yaml: the code, its label and its synonyms), and
a control group when it is one of the control names.

The words only; the SQL that reads them is in cohort_count.py.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

log = logging.getLogger(__name__)

_VOCABULARY = Path(__file__).resolve().parent / "value_mappings.yaml"

# Columns of participants.tsv that say which group a participant is in.
GROUP_KEYS: Tuple[str, ...] = (
    "group", "groups", "diagnosis", "dx", "diag", "participant_group", "patient_group",
    "subject_group", "clinical_group", "clinical_diagnosis", "disease", "cohort",
    "pathology", "population",
)

# What group columns call participants without the condition, besides the
# vocabulary's names for healthy_control.
_CONTROL_LABELS: Tuple[str, ...] = (
    "control", "controls", "ctl", "ctrl", "ctr", "con", "hc", "hcs", "cn", "nc", "td",
    "healthy", "normal", "healthy control", "healthy controls", "healthy_control",
    "control group", "neurotypical", "typically developing",
)
_CONTROL_CODE = "healthy_control"

_LABEL_OK_RE = re.compile(r"^[a-z0-9][a-z0-9 '/+._-]*$")


@lru_cache(maxsize=1)
def _vocabulary() -> Dict[str, dict]:
    """The diagnosis entries of value_mappings.yaml. Without them only coded
    rows count, so a file that cannot be read is logged, not silent."""
    import yaml
    try:
        data = yaml.safe_load(_VOCABULARY.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.warning("group labels: %s not read (%s); only coded diagnoses are counted",
                    _VOCABULARY, exc)
        return {}
    return (data or {}).get("diagnosis") or {}


def _clean(label: str) -> str:
    return " ".join(str(label).replace("’", "'").lower().split())


@lru_cache(maxsize=256)
def _labels_of(code: str) -> Tuple[str, ...]:
    entry = _vocabulary().get(code) or {}
    names = [code, code.replace("_", " "), entry.get("label") or ""]
    for synonym in entry.get("synonyms") or []:
        names.append(synonym.get("term") if isinstance(synonym, dict) else synonym)
    cleaned = []
    for name in names:
        name = _clean(name or "")
        if len(name) >= 2 and _LABEL_OK_RE.fullmatch(name) and name not in cleaned:
            cleaned.append(name)
    return tuple(cleaned)


def labels_for(codes: Sequence[str]) -> List[str]:
    """What a group column may call participants with one of these diagnoses."""
    found: List[str] = []
    for code in codes:
        for label in _labels_of(code):
            if label not in found:
                found.append(label)
    return found


def control_labels() -> List[str]:
    """What a group column may call participants without the condition."""
    found = list(_CONTROL_LABELS)
    for label in _labels_of(_CONTROL_CODE):
        if label not in found:
            found.append(label)
    return found
