"""
services/uncertain/conditions.py
------------------------------------
The pieces every part of the uncertain tier shares: the states one condition
can take (SoftCondition), the patterns that read a
generated text search, and the switch for the 'possible' level. What the
tier does: tier.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple


def possible_enabled() -> bool:
    """Whether a missing field without any hint (state 2) may keep a dataset."""
    # Off by default: on the benchmark this level added 2,211 rows for 0.4
    # points of recall, 11% of the judged ones relevant (2026-11-13).
    return os.getenv("UNCERTAIN_POSSIBLE", "false").strip().lower() in ("1", "true", "yes", "on")


# A stemmed text search, with the guard that may follow it: the word forms
# the unstemmed vector must contain (services/fulltext/text_search.py, exact word forms).
TSQUERY_RE = re.compile(
    r"(?P<negated>\bNOT\s+)?(?P<alias>\b[A-Za-z_]\w*)\.(?P<column>search_tsv(?:_papers)?)\s+@@\s+"
    r"TO_TSQUERY\(\s*'english'\s*,\s*'(?P<query>[^']*)'\s*\)"
    r"(?:\s+AND\s+(?P=alias)\.(?P<guard_column>search_tsv_exact(?:_papers)?)\s+@@\s+"
    r"TO_TSQUERY\(\s*'simple'\s*,\s*'(?P<guard>[^']*)'\s*\))?",
    re.IGNORECASE,
)
LABEL_RE = re.compile(r":[AB]+\b")


def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


@dataclass
class SoftCondition:
    sql: str                    # the condition as generated
    kind: str                   # age, diagnosis, sex or text
    probable: Optional[str]     # SQL for the weaker form, or None
    known: Optional[str]        # SQL: the field is recorded (None for text)
    # What the result card says for a dataset in that state; "" says nothing.
    # The wordings are in reason_texts.py.
    reason_probable: str        # e.g. "ages not recorded; the description names the age group"
    reason_possible: str        # e.g. "ages not recorded"
    paper: Optional[str] = None  # SQL: the terms match the dataset's paper text
    # (name, SELECT) sets the condition reads more than once: computed once
    # as a WITH query (a sidecar lookup scans file metadata; scanner.sidecar).
    ctes: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        # Paper evidence is one more weaker form, checked first.
        if self.paper:
            self.probable = f"({self.paper}) OR ({self.probable})" if self.probable else self.paper

    def state_sql(self) -> str:
        """0 confirmed, 1 probable, 2 possible, 9 contradicted."""
        probable = f"WHEN ({self.probable}) THEN 1 " if self.probable else ""
        possible = "2" if possible_enabled() else "9"
        if self.known is None:
            return f"CASE WHEN ({self.sql}) THEN 0 {probable}ELSE 9 END"
        if self.kind == "diagnosis":
            # Text evidence counts even when some diagnoses are recorded:
            # coded diagnoses are often incomplete.
            return (f"CASE WHEN ({self.sql}) THEN 0 {probable}"
                    f"WHEN NOT ({self.known}) THEN {possible} ELSE 9 END")
        # Recorded ages or sex that do not fit are a contradiction, whatever
        # the README says.
        unknown = (f"CASE WHEN ({self.probable}) THEN 1 ELSE {possible} END"
                   if self.probable else possible)
        return (f"CASE WHEN ({self.sql}) THEN 0 "
                f"WHEN NOT ({self.known}) THEN {unknown} ELSE 9 END")

    def keep_sql(self) -> str:
        """Pre-filter: the condition is not contradicted."""
        options = [f"({self.sql})"]
        if self.probable:
            options.append(f"({self.probable})")
        if self.known and possible_enabled():
            options.append(f"NOT ({self.known})")
        return "(" + " OR ".join(options) + ")"
