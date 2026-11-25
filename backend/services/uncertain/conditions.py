"""
services/uncertain/conditions.py
------------------------------------
The pieces every part of the uncertain tier shares: one condition with its
evidence in the order it counts (SoftCondition), the evidence of the whole
query, which combines its conditions (QueryEvidence), the patterns that read
a generated text search, and the switch for the 'possible' level. What the
tier does: tier.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def own_text_matches(alias: str, query: str) -> str:
    """SQL: the dataset's own text matches the tsquery *query*.

    search_tsv also holds the recorded labels and the LLM-written profile, so
    the index only finds the candidates. "/" as a space: as in search_tsv.
    """
    tsquery = f"TO_TSQUERY('english', {sql_literal(query)})"
    return (f"({alias}.search_tsv @@ {tsquery} "
            f"AND TO_TSVECTOR('english', TRANSLATE({own_text(alias)}, '/', ' ')) @@ {tsquery})")


@dataclass
class SoftCondition:
    """One condition (AND clause) of the generated query that a dataset can
    fail without contradicting it, and the evidence for a dataset it does not
    hold for, in the order it counts. The first that holds gives the
    condition's state (Strength); a condition that holds as generated is
    confirmed, and a dataset none of the evidence holds for is contradicted.

    What the list holds depends on the kind:

      a text search            [only in a paper,] a weaker form of the search
      a field that may be      other values recorded, [only in a paper,] [the
      missing (fields/,        text names the asked value or matches in part,]
      scanner.py)              not recorded
      the same, where text     [only in a paper,] [a weaker form,] other
      outweighs a record       values recorded, not recorded
    """
    kind: str                         # a field in fields/ (age, diagnosis, sex), text, sidecar or patient_count
    alias: str                        # the dataset row of the query
    sql: str                          # the condition as generated
    evidence: Tuple[Evidence, ...]
    # (name, SELECT) sets the condition reads more than once: computed once
    # as a WITH query (a sidecar lookup scans file metadata; scanner.sidecar).
    ctes: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        self.evidence = tuple(self.evidence)
        strengths = [e.strength for e in self.evidence]
        if Strength.CONFIRMED in strengths:
            raise ValueError("the condition itself is what confirms it")
        if strengths.count(Strength.CONTRADICTED) > 1 or strengths.count(Strength.POSSIBLE) > 1:
            raise ValueError("a condition has one record that contradicts it")
        if Strength.CONTRADICTED in strengths:
            at = strengths.index(Strength.CONTRADICTED)
            if Strength.PROBABLE in strengths[:at] and Strength.PROBABLE in strengths[at:]:
                # The tier keeps one column for all weaker evidence of a
                # condition (tier.apply); two groups would need two.
                raise ValueError("weaker evidence is listed all before or all after the record")

    # ── The parts of the list ─────────────────────────────────────────────────

    @property
    def weaker(self) -> Tuple[Evidence, ...]:
        """The evidence for the probable state, in the order it counts."""
        return tuple(e for e in self.evidence if e.strength == Strength.PROBABLE)

    @property
    def not_recorded(self) -> Optional[Evidence]:
        """The evidence for the possible state."""
        return next((e for e in self.evidence if e.strength == Strength.POSSIBLE), None)

    @property
    def probable(self) -> Optional[str]:
        """SQL: some weaker evidence holds. None without any."""
        parts = [e.sql(self.alias) for e in self.weaker]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else " OR ".join(f"({part})" for part in parts)

    @property
    def known(self) -> Optional[str]:
        """SQL: the field is recorded. None for a condition that reads no
        field a dataset may leave out (a text search)."""
        return next((e.sql(self.alias) for e in self.evidence
                     if e.strength == Strength.CONTRADICTED), None)

    @property
    def weaker_beats_record(self) -> bool:
        """Weaker evidence counts although the field is recorded with other values."""
        strengths = [e.strength for e in self.evidence]
        return (Strength.CONTRADICTED in strengths
                and Strength.PROBABLE in strengths[:strengths.index(Strength.CONTRADICTED)])

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


@dataclass
class QueryEvidence:
    """The evidence a whole query has for a dataset: that of its conditions,
    combined. The query is their AND, so a dataset is as strong as its weakest
    condition; whether it is kept, and what its result card says, is decided
    here and not by a single condition.

    Written as SQL over one row per dataset that holds, for condition i, its
    Strength (state_column) and which weaker evidence held (flag_columns);
    tier.apply builds that row.
    """
    conditions: Tuple[SoftCondition, ...]
    row: str = "evidence"   # SQL alias of that row

    def state_column(self, i: int) -> str:
        return f"u{i}"

    def flag_columns(self, i: int) -> List[str]:
        """Columns that tell the weaker evidence of condition *i* apart, one
        for each but the last: "paper_0" for evidence found in a paper."""
        names: List[str] = []
        for j, piece in enumerate(self.conditions[i].weaker[:-1]):
            name = f"{piece.source.value}_{i}"
            names.append(f"{name}_{j}" if name in names else name)
        return names

    def _states(self) -> List[str]:
        return [f"{self.row}.{self.state_column(i)}" for i in range(len(self.conditions))]

    @staticmethod
    def _weakest(states: Sequence[str]) -> str:
        """The weakest of several states: the largest Strength."""
        return f"GREATEST({', '.join(states)})" if len(states) > 1 else states[0]

    def strength_sql(self) -> str:
        """The Strength of the query for the dataset: its weakest condition."""
        return self._weakest(self._states())

    def strength_by_kind_sql(self) -> Dict[str, str]:
        """kind of condition → the weakest state among the conditions of
        that kind, for a ranker that tells them apart (results/rankers.py)."""
        of_kind: Dict[str, List[str]] = {}
        for state, condition in zip(self._states(), self.conditions):
            of_kind.setdefault(condition.kind, []).append(state)
        return {kind: self._weakest(states) for kind, states in of_kind.items()}

    def kept_sql(self, max_unsure: int) -> str:
        """SQL: the dataset stays in the result. No condition is
        contradicted, at most *max_unsure* are not confirmed, and at least
        one is confirmed or probable: a dataset that only matches the
        modality while everything else is unknown is not evidence of anything."""
        states = self._states()
        unsure = " + ".join(f"(CASE WHEN {state} > {int(Strength.CONFIRMED)} THEN 1 ELSE 0 END)"
                            for state in states)
        some_evidence = " OR ".join(f"{state} <= {int(Strength.PROBABLE)}" for state in states)
        return (f"{self.strength_sql()} < {int(Strength.CONTRADICTED)} "
                f"AND ({unsure}) <= {max_unsure} AND ({some_evidence})")

    def reason_sql(self) -> str:
        """SQL for what the result card says: for each condition that is not
        confirmed, the text of the first evidence that holds, joined with "; "."""
        cases = []
        for i, (state, condition) in enumerate(zip(self._states(), self.conditions)):
            flags = self.flag_columns(i)
            branches = []
            for j, piece in enumerate(condition.weaker):
                if reason := piece.reason():
                    # The last one needs no flag: nothing listed before it held.
                    flag = f" AND {self.row}.{flags[j]}" if j < len(flags) else ""
                    branches.append(f"WHEN {state} = {int(Strength.PROBABLE)}{flag} "
                                    f"THEN {sql_literal(reason)}")
            if condition.not_recorded is not None and (reason := condition.not_recorded.reason()):
                branches.append(f"WHEN {state} = {int(Strength.POSSIBLE)} THEN {sql_literal(reason)}")
            if branches:
                cases.append(f"CASE {' '.join(branches)} END")
        return f"NULLIF(CONCAT_WS('; ', {', '.join(cases)}), '')" if cases else "NULL"
