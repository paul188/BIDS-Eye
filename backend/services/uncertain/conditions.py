"""
services/uncertain/conditions.py
------------------------------------
The pieces every part of the uncertain tier shares: one condition with its
evidence in the order it counts (SoftCondition), the patterns that read a
generated text search, and the switch for the 'possible' level. What the
tier does: tier.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from services.evidence.base import Evidence, Strength


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


def own_text(alias: str) -> str:
    """The dataset's own name, description and README as one text."""
    return (f"(COALESCE({alias}.name, '') || ' ' || COALESCE({alias}.description_text, '') "
            f"|| ' ' || COALESCE({alias}.readme_text, ''))")


@dataclass
class SoftCondition:
    """One condition of the search and the evidence for it, in the order it
    counts: the first that holds for a dataset gives the condition's state
    (Strength) and the text on its result card.

    The list always starts with the condition as asked (weaker.AsAsked). What
    follows depends on the kind:

      a text search            [only in a paper,] a weaker form of the search
      a field that may be      other values recorded, [only in a paper,] [the
      missing (fields/,        text names the asked value or matches in part,]
      scanner.py)              not recorded
      the same, where text     [only in a paper,] [a weaker form,] other
      outweighs a record       values recorded, not recorded

    A dataset none of them holds for is contradicted.
    """
    kind: str                         # a field in fields/ (age, diagnosis, sex), text, sidecar or patient_count
    evidence: Tuple[Evidence, ...]
    # (name, SELECT) sets the condition reads more than once: computed once
    # as a WITH query (a sidecar lookup scans file metadata; scanner.sidecar).
    ctes: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        self.evidence = tuple(self.evidence)
        strengths = [e.strength for e in self.evidence]
        if strengths[:1] != [Strength.CONFIRMED] or strengths.count(Strength.CONFIRMED) != 1:
            raise ValueError("a condition's evidence starts with the condition as asked")
        if strengths.count(Strength.CONTRADICTED) > 1:
            raise ValueError("a condition has one record that contradicts it")
        if Strength.CONTRADICTED in strengths:
            at = strengths.index(Strength.CONTRADICTED)
            if Strength.PROBABLE in strengths[:at] and Strength.PROBABLE in strengths[at:]:
                # The tier keeps one column for all weaker evidence of a
                # condition (tier.apply); two groups would need two.
                raise ValueError("weaker evidence is listed all before or all after the record")

    # ── The parts of the list ─────────────────────────────────────────────────

    @property
    def sql(self) -> str:
        """The condition as generated."""
        return self.evidence[0].sql or ""

    @property
    def weaker(self) -> Tuple[Evidence, ...]:
        """The evidence for the probable state, in the order it counts."""
        return tuple(e for e in self.evidence if e.strength == Strength.PROBABLE)

    @property
    def probable(self) -> Optional[str]:
        """SQL: some weaker evidence holds. None without any."""
        parts = [e.sql for e in self.weaker]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else " OR ".join(f"({part})" for part in parts)

    @property
    def known(self) -> Optional[str]:
        """SQL: the field is recorded. None for a condition that reads no
        field a dataset may leave out (a text search)."""
        return next((e.sql for e in self.evidence if e.strength == Strength.CONTRADICTED), None)

    @property
    def weaker_beats_record(self) -> bool:
        """Weaker evidence counts although the field is recorded with other values."""
        strengths = [e.strength for e in self.evidence]
        return (Strength.CONTRADICTED in strengths
                and Strength.PROBABLE in strengths[:strengths.index(Strength.CONTRADICTED)])

    @property
    def reason_probable(self) -> str:
        """The card text of the weaker evidence that counts last."""
        return self.weaker[-1].reason if self.weaker else ""

    @property
    def reason_possible(self) -> str:
        return next((e.reason for e in self.evidence if e.strength == Strength.POSSIBLE), "")

    # ── SQL ───────────────────────────────────────────────────────────────────

    def state_sql(self, confirmed: Optional[str] = None, weak: Optional[str] = None,
                  recorded: Optional[str] = None) -> str:
        """The Strength the condition has for a dataset, as a number.

        *confirmed*, *weak*, *recorded*: columns that already hold whether the
        condition as asked, some weaker evidence, and the record are true
        (tier.apply computes each once per dataset). Without them the SQL of
        the evidence itself.
        """
        confirmed = confirmed or f"({self.sql})"
        weak = weak or (f"({self.probable})" if self.probable else "FALSE")
        yes, probably, no = (int(s) for s in (Strength.CONFIRMED, Strength.PROBABLE,
                                              Strength.CONTRADICTED))
        if self.known is None:
            return f"CASE WHEN {confirmed} THEN {yes} WHEN {weak} THEN {probably} ELSE {no} END"
        recorded = recorded or f"({self.known})"
        # Not recorded and no hint: possible, which keeps a dataset only while
        # that level is on.
        unrecorded = int(Strength.POSSIBLE) if possible_enabled() else no
        if self.weaker_beats_record:
            left = f"WHEN NOT {recorded} THEN {unrecorded} " if unrecorded != no else ""
            return (f"CASE WHEN {confirmed} THEN {yes} WHEN {weak} THEN {probably} "
                    f"{left}ELSE {no} END")
        # Recorded values that do not fit are a contradiction, whatever the
        # README says.
        return (f"CASE WHEN {confirmed} THEN {yes} WHEN NOT {recorded} THEN "
                f"CASE WHEN {weak} THEN {probably} ELSE {unrecorded} END ELSE {no} END")

    def keep_sql(self) -> str:
        """Pre-filter: the condition is not contradicted."""
        options = [f"({self.sql})"]
        if self.probable:
            options.append(f"({self.probable})")
        if self.known and possible_enabled():
            options.append(f"NOT ({self.known})")
        return "(" + " OR ".join(options) + ")"
