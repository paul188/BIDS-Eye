"""
RAG/cohort_count.py
-------------------
"At least N patients with <diagnosis>": count the participants who have the
diagnosis, not the participants of the dataset.

A dataset's participants are known to have a diagnosis in two ways:

  coded   bids_participants.diagnosis holds a canonical code (about one
          dataset in six that has participant rows)
  label   a group column of participants.tsv kept in bids_participants.extra
          ("Group": "PD") holds one of the names the vocabulary lists for the
          code (RAG/value_mappings.yaml: label and synonyms)

The SQL model, or the plan compiler, writes a placeholder,

    patients_with('parkinsons_disease, de_novo_parkinsons') >= 40

and the backend expands it (expand) into a count over both. The uncertain tier
(backend/services/fulltext/uncertain_tier.py) gives the condition three outcomes:

  confirmed     at least N participants are counted
  uncertain     the count cannot be known: the dataset has no participant
                rows, or rows that say nothing recognisable about the group,
                and it has at least N subjects in all (unknown_sql)
  excluded      every participant row is either coded or labelled as the
                diagnosis or as a control, and fewer than N have it; or the
                dataset has fewer than N subjects

Only [a-z0-9_, ] appears inside the quotes of the placeholder, so it survives
being copied by a language model.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from services.schema.tables import objects as _objects, participants as _participants  # backend/ is on the path wherever RAG runs

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

_CODE_RE = re.compile(r"^[a-z0-9_]+$")
_LABEL_OK_RE = re.compile(r"^[a-z0-9][a-z0-9 '/+._-]*$")

PLACEHOLDER_RE = re.compile(
    r"patients_with\(\s*'(?P<codes>[a-z0-9_, ]+)'\s*\)\s*(?P<op>>=|>)\s*(?P<n>\d+)",
    re.IGNORECASE,
)
# The expanded count, as written by confirmed_sql and as sqlglot prints it back.
_EXPANDED_RE = re.compile(
    r"cohort_n\.diagnosis\s+IN\s*\((?P<codes>[^)]*)\).*>=\s*(?P<n>\d+)\s*\)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


@lru_cache(maxsize=1)
def _vocabulary() -> Dict[str, dict]:
    """The diagnosis entries of value_mappings.yaml. Without them only coded
    rows count, so a file that cannot be read is logged, not silent."""
    import yaml
    try:
        data = yaml.safe_load(_VOCABULARY.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.warning("cohort count: %s not read (%s); only coded diagnoses are counted",
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
    found = list(_CONTROL_LABELS)
    for label in _labels_of(_CONTROL_CODE):
        if label not in found:
            found.append(label)
    return found


def valid_codes(codes: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(c for c in (str(c).strip().lower() for c in codes) if _CODE_RE.fullmatch(c)))


def placeholder(codes: Sequence[str], n: int) -> Optional[str]:
    """patients_with('<codes>') >= n, or None without a usable code."""
    codes = valid_codes(codes)
    if not codes or int(n) < 1:
        return None
    return f"patients_with('{', '.join(codes)}') >= {int(n)}"


def _quoted(values: Sequence[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def _label_match(row: str, labels: Sequence[str]) -> str:
    """True when a group column of the participant row holds one of *labels*."""
    # extra is JSON null for rows without extra columns: only an object has keys.
    columns = f"CASE WHEN JSON_TYPEOF({row}.extra) = 'object' THEN {row}.extra END"
    return (f"EXISTS(SELECT 1 FROM JSON_EACH_TEXT({columns}) AS cohort_g "
            f"WHERE LOWER(cohort_g.key) IN ({_quoted(GROUP_KEYS)}) "
            f"AND LOWER(BTRIM(cohort_g.value)) IN ({_quoted(labels)}))")


def confirmed_sql(codes: Sequence[str], n: int, alias: str = "d") -> str:
    """At least *n* participants of the dataset are coded or labelled with the diagnosis."""
    codes = valid_codes(codes)
    has_it = f"cohort_n.diagnosis IN ({_quoted(codes)})"
    labels = labels_for(codes)
    if labels:
        has_it = f"({has_it} OR {_label_match('cohort_n', labels)})"
    return (f"(SELECT COUNT(DISTINCT cohort_n.participant_id) FROM {_participants.name} AS cohort_n "
            f"WHERE cohort_n.dataset_id = {alias}.id AND {has_it}) >= {int(n)}")


def unknown_sql(codes: Sequence[str], n: int, alias: str = "d",
                subject_count: Optional[str] = None) -> str:
    """The count cannot be known, and the dataset is large enough to hold *n* such participants.

    Known means: the dataset has participant rows, and every row is coded or
    carries a group label that names this diagnosis or a control group.
    """
    codes = valid_codes(codes)
    recognised = labels_for(codes) + [c for c in control_labels() if c not in labels_for(codes)]
    rows = f"SELECT 1 FROM {_participants.name} AS cohort_k WHERE cohort_k.dataset_id = {alias}.id"
    known = (f"(EXISTS({rows}) AND NOT EXISTS({rows} AND cohort_k.diagnosis IS NULL "
             f"AND NOT {_label_match('cohort_k', recognised)}))")
    # *subject_count*: a stored count of the dataset's subjects, when the database has one.
    subjects = subject_count or (
        f"(SELECT COUNT(DISTINCT cohort_s.subject) FROM {_objects.name} AS cohort_s "
        f"WHERE cohort_s.dataset_id = {alias}.id)")
    size = (f"GREATEST({subjects}, "
            f"(SELECT COUNT(*) FROM {_participants.name} AS cohort_k WHERE cohort_k.dataset_id = {alias}.id))")
    return f"(NOT {known} AND {size} >= {int(n)})"


def expand(sql: str, alias: str = "d") -> str:
    """Replace every patients_with(...) placeholder by its count."""
    def replace(match: re.Match) -> str:
        codes = [c.strip() for c in match.group("codes").split(",")]
        n = int(match.group("n")) + (1 if match.group("op") == ">" else 0)
        if not valid_codes(codes):
            return match.group(0)
        return confirmed_sql(codes, n, alias)
    return PLACEHOLDER_RE.sub(replace, sql)


def parse_expanded(sql: str) -> Optional[Tuple[List[str], int]]:
    """(codes, n) of an expanded count condition, else None."""
    if "cohort_n" not in sql:
        return None
    match = _EXPANDED_RE.search(sql)
    if not match:
        return None
    codes = valid_codes(re.findall(r"'([^']*)'", match.group("codes")))
    return (codes, int(match.group("n"))) if codes else None


REASON_UNKNOWN = "how many participants have the diagnosis is not recorded"
