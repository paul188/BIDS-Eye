"""
backend/vocabulary/cohort_count.py
-------------------
"At least N patients with <diagnosis>": count the participants who have the
diagnosis, not the participants of the dataset.

A dataset's participants are known to have a diagnosis in two ways:

  coded   bids_participants.diagnosis holds a canonical code (about one
          dataset in six that has participant rows)
  label   a group column of participants.tsv kept in bids_participants.extra
          ("Group": "PD") holds one of the names the vocabulary lists for the
          code (group_labels.py, from value_mappings.yaml)

The SQL model, or the plan compiler, writes a placeholder,

    patients_with('parkinsons_disease, de_novo_parkinsons') >= 40

and the backend expands it (expand) into a count over both. The uncertain tier
(backend/services/uncertain/tier.py) gives the condition three outcomes:

  confirmed     at least N participants are counted
  uncertain     the count cannot be known: the dataset has no participant
                rows, or rows that say nothing recognisable about the group,
                and it has at least N subjects in all (unknown_sql)
  excluded      every participant row is either coded or labelled as the
                diagnosis or as a control, and fewer than N have it; or the
                dataset has fewer than N subjects

Only [a-z0-9_, ] appears inside the quotes of the placeholder, so it survives
being copied by a language model.

Whatever reads the query afterwards (the uncertain tier, the filter chips)
finds the expanded count with find(), on the syntax tree.
"""

from __future__ import annotations

import re
from typing import Any, List, NamedTuple, Optional, Sequence

from group_labels import GROUP_KEYS, control_labels, labels_for
from services.schema.tables import objects as _objects, participants as _participants  # backend/ is on the path wherever RAG runs

# The participant rows of the expanded count. find() knows the count by it:
# the model's own count of coded rows has the same shape under another alias.
ALIAS = "cohort_n"

_CODE_RE = re.compile(r"^[a-z0-9_]+$")

PLACEHOLDER_RE = re.compile(
    r"patients_with\(\s*'(?P<codes>[a-z0-9_, ]+)'\s*\)\s*(?P<op>>=|>)\s*(?P<n>\d+)",
    re.IGNORECASE,
)


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
    has_it = f"{ALIAS}.diagnosis IN ({_quoted(codes)})"
    labels = labels_for(codes)
    if labels:
        has_it = f"({has_it} OR {_label_match(ALIAS, labels)})"
    return (f"(SELECT COUNT(DISTINCT {ALIAS}.participant_id) FROM {_participants.name} AS {ALIAS} "
            f"WHERE {ALIAS}.dataset_id = {alias}.id AND {has_it}) >= {int(n)}")


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


class PatientCount(NamedTuple):
    codes: List[str]
    minimum: int


def find(node: Any) -> Optional[PatientCount]:
    """The count confirmed_sql() wrote, read from the syntax tree *node*, else None.

    Its shape: (SELECT COUNT(DISTINCT cohort_n.participant_id) FROM
    bids_participants AS cohort_n WHERE … cohort_n.diagnosis IN (<codes>) …) >= n.
    """
    from sqlglot import exp  # a backend dependency, not one of this package

    while isinstance(node, exp.Paren):
        node = node.this
    if not isinstance(node, exp.GTE):
        return None
    count, minimum = node.this, node.expression
    if not (isinstance(count, exp.Subquery) and isinstance(count.this, exp.Select)
            and isinstance(minimum, exp.Literal) and not minimum.is_string
            and str(minimum.this).isdigit()):
        return None
    select = count.this
    source = select.args.get("from") or select.args.get("from_")
    table = source.this if source is not None else None
    if not (isinstance(table, exp.Table) and table.name == _participants.name
            and table.alias == ALIAS):
        return None
    where = select.args.get("where")
    for found in where.find_all(exp.In) if where is not None else ():
        column = found.this
        if isinstance(column, exp.Column) and column.table == ALIAS and column.name == "diagnosis":
            codes = valid_codes([e.name for e in found.expressions
                                 if isinstance(e, exp.Literal) and e.is_string])
            return PatientCount(codes, int(minimum.this)) if codes else None
    return None
