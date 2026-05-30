"""
RAG/join_registry.py
--------------------
Lightweight loader for RAG/join_paths.yaml.

Provides a single source of truth for:
  - Which DB table and column each semantic field maps to
  - Which fields can be combined in one EXISTS clause vs. need separate ones
  - What field-value combinations indicate a user query contradiction

No external dependencies beyond PyYAML (already a project dependency).
Does NOT import from retriever.py or yaml_to_llamaindex.py — it is a
lower-level utility that those modules may safely import.

Public API
----------
  get_field_info(field)       -> dict | None
  get_fields_for_table(table) -> list[str]
  field_to_db_col(field)      -> tuple[str, str] | None
  check_contradictions(resolved) -> list[ContradictionWarning]
  build_join_context_block()  -> str
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent / "join_paths.yaml"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ContradictionWarning(NamedTuple):
    contradiction_id: str   # e.g. "eeg_mri_suffix"
    severity: str           # "hard" or "soft"
    message: str            # human-readable explanation


# ---------------------------------------------------------------------------
# Private loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load() -> dict:
    """Load and cache join_paths.yaml once per process."""
    path = _REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Join path registry not found: {path}\n"
            "Expected: RAG/join_paths.yaml"
        )
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"join_paths.yaml must be a YAML mapping, got {type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_field_info(field: str) -> Optional[Dict]:
    """Return the registry entry for *field*, or None if not found.

    Parameters
    ----------
    field : semantic field name, e.g. 'diagnosis', 'task', 'datatype', 'suffix',
            'sex', 'handedness', 'age', 'author', 'funding'

    Returns
    -------
    dict with keys: table, column, value_type, multi_value, exists_alias,
                    rag_backed, description, and optionally sql_pattern.
    None if the field is not registered.
    """
    return _load().get("fields", {}).get(field)


def get_fields_for_table(table: str) -> List[str]:
    """Return all field names that map to *table*.

    Parameters
    ----------
    table : 'bids_datasets', 'bids_objects', or 'bids_participants'
    """
    return [
        name
        for name, info in _load().get("fields", {}).items()
        if isinstance(info, dict) and info.get("table") == table
    ]


def field_to_db_col(field: str) -> Optional[Tuple[str, str]]:
    """Return (table, column) for *field*, or None.

    Drop-in replacement for the ``_FIELD_TO_DB_COL`` dict that used to live
    in ``RAG/retriever.py``.

    Returns None for array fields that live directly on bids_datasets
    (exists_alias == 'none') and for unknown fields.
    """
    info = get_field_info(field)
    if info is None:
        return None
    if info.get("exists_alias") == "none":
        return None
    return (info["table"], info["column"])


def check_contradictions(
    resolved: Dict[str, List[str]],
) -> List[ContradictionWarning]:
    """Detect contradictory field-value combinations in resolved RAG output.

    Parameters
    ----------
    resolved : {field_name: [canonical_codes]}
        e.g. {'datatype': ['eeg'], 'suffix': ['bold'], 'task': ['resting_state']}
        Values are standard_code strings already resolved by the RAG layer.

    Returns
    -------
    List of ContradictionWarning (may be empty). Hard contradictions sort first.

    Wildcard handling
    -----------------
    A trigger or incompatible value list of ``["*"]`` matches any non-empty
    code list (used for rules like "anat + any task = contradiction").
    """
    rules = _load().get("contradictions", [])
    warnings: List[ContradictionWarning] = []

    def _matches(field_codes: List[str], target_values: List[str]) -> bool:
        if target_values == ["*"]:
            return bool(field_codes)
        return bool(set(field_codes) & set(target_values))

    for rule in rules:
        trigger = rule.get("trigger", {})
        incompatible = rule.get("incompatible", {})

        trigger_matched = all(
            _matches(resolved.get(field, []), values)
            for field, values in trigger.items()
        )
        if not trigger_matched:
            continue

        incompatible_matched = any(
            _matches(resolved.get(field, []), values)
            for field, values in incompatible.items()
        )
        if not incompatible_matched:
            continue

        warnings.append(ContradictionWarning(
            contradiction_id=rule.get("id", "unknown"),
            severity=rule.get("severity", "soft"),
            message=rule.get("message", "Contradictory field combination.").strip(),
        ))

    warnings.sort(key=lambda w: 0 if w.severity == "hard" else 1)
    return warnings


def build_join_context_block() -> str:
    """Build a compact text block describing join paths for the SQL LLM.

    Returns a multi-line string formatted to match the context block style
    used in ``_build_augmented_question()`` (square-bracket header, indented
    lines). Inject this BEFORE the ``[Resolved DB filters]`` block so the
    SQL LLM has structural context before seeing the concrete EXISTS snippets.

    Output example::

        [Join paths — table routing for EXISTS subqueries]
          bids_datasets (direct, no EXISTS): author, funding
          bids_objects (alias o2): datatype, suffix, task
          bids_participants (alias p2): age, diagnosis, handedness, sex
          Rule: NEVER JOIN bids_participants with bids_objects directly — use separate
                correlated EXISTS subqueries, each referencing d.id.
          Rule: datatype+suffix from the same user term share one EXISTS (OR between
                them); task always gets its own EXISTS clause.
    """
    fields_cfg = _load().get("fields", {})

    table_fields: Dict[str, List[str]] = {}
    for fname, info in fields_cfg.items():
        if not isinstance(info, dict):
            continue
        alias = info.get("exists_alias", "none")
        table = info.get("table", "unknown")
        key = (
            f"{table} (direct, no EXISTS)" if alias == "none"
            else f"{table} (alias {alias})"
        )
        table_fields.setdefault(key, []).append(fname)

    lines = ["[Join paths — table routing for EXISTS subqueries]"]
    for table_label, fnames in sorted(table_fields.items()):
        lines.append(f"  {table_label}: {', '.join(sorted(fnames))}")

    lines.append(
        "  Rule: NEVER JOIN bids_participants with bids_objects directly — "
        "use separate correlated EXISTS subqueries, each referencing d.id."
    )
    lines.append(
        "  Rule: datatype+suffix from the same user term share one EXISTS (OR); "
        "task always gets its own EXISTS clause."
    )

    return "\n".join(lines)
