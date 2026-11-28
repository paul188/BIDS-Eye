"""
services/uncertain/conditions.py
------------------------------------
The pieces every part of the uncertain tier shares: one condition with its
evidence in the order it counts (ClauseEvidence), the evidence of the whole
query, which combines its conditions (QueryEvidence), and the switch for
the 'possible' level. What the tier does: tier.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from services.evidence.base import SourceGroup, Strength, UncertainEvidence
from services.evidence.confirmed import ConfirmedEvidence
from services.switches import Switch


# How the pieces of evidence of a clause, and the clauses of a query, add up
# to a score: services/evidence/SCORING.md. Both numbers are hand-set starting
# values, to be measured.
#
# What a second group of sources adds to the strongest one, from 0 (nothing:
# the groups say the same) to 1 (in full: they are independent).
SHARED_EVIDENCE_DISCOUNT = 0.5
# The power the clauses other than the weakest get in the query score, from
# 0 (only the weakest clause counts: the minimum) to 1 (the product).
CLAUSE_INDEPENDENCE = 1.0


def log_weight(score: float) -> float:
    """A score on the scale where independent evidence adds up: -ln(1 - score)."""
    return -math.log(1 - score)


def _greatest(terms: Sequence[str]) -> str:
    return terms[0] if len(terms) == 1 else f"GREATEST({', '.join(terms)})"


def possible_enabled() -> bool:
    """Whether a missing field without any hint (state 2) may keep a dataset."""
    # Off by default: on the benchmark this level added 2,211 rows for 0.4
    # points of recall, 11% of the judged ones relevant (2026-11-13).
    return Switch.UNCERTAIN_POSSIBLE.is_on()



def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def text_from_dataset(alias: str) -> str:
    """The dataset's own name, description and README as one text."""
    return (f"(COALESCE({alias}.name, '') || ' ' || COALESCE({alias}.description_text, '') "
            f"|| ' ' || COALESCE({alias}.readme_text, ''))")


def text_from_dataset_matches(alias: str, query: str) -> str:
    """SQL: the dataset's own text matches the tsquery *query*.

    search_tsv also holds the recorded labels and the LLM-written profile, so
    the index only finds the candidates. "/" as a space: as in search_tsv.
    """
    tsquery = f"TO_TSQUERY('english', {sql_literal(query)})"
    return (f"({alias}.search_tsv @@ {tsquery} "
            f"AND TO_TSVECTOR('english', TRANSLATE({text_from_dataset(alias)}, '/', ' ')) @@ {tsquery})")


@dataclass
class ClauseEvidence:
    """One condition (AND clause) of the generated query that a dataset can
    fail without contradicting it, and the evidence for a dataset it does not
    hold for, in the order it counts. The first that holds gives the
    condition's state (Strength); a condition that holds as generated is
    confirmed, and a dataset none of the evidence holds for is contradicted.

    What the list holds depends on the kind:

      a text search            [only in a paper,] a weaker form of the search
      a field that may be      records with other values, [only in a paper,] [the
      missing (fields/,        text names the asked value or matches in part,]
      scanner.py)              not recorded
      the same, where text     [only in a paper,] [a weaker form,] other
      outweighs a record       records with other values, not recorded
    """
    kind: str                         # a field in fields/ (age, diagnosis, sex), text, sidecar or patient_count
    sql: str                          # the condition as generated
    evidence: Tuple[UncertainEvidence, ...]
    # (name, SELECT) sets the condition reads more than once: computed once
    # as a WITH query (a sidecar lookup scans file metadata; scanner.sidecar).
    ctes: Tuple[Tuple[str, str], ...] = ()
    # The ways the condition holds as generated, one per OR branch
    # (evidence/branches.py). sql decides whether it holds: a branch that
    # cannot be evaluated alone is not listed.
    confirming: Tuple[ConfirmedEvidence, ...] = ()

    def __post_init__(self) -> None:
        self.evidence = tuple(self.evidence)
        self.confirming = tuple(self.confirming)
        if any(e.strength != Strength.CONFIRMED for e in self.confirming):
            raise ValueError("what confirms a condition has the strength CONFIRMED")
        strengths = [e.strength for e in self.evidence]
        if Strength.CONFIRMED in strengths:
            raise ValueError("the condition itself is what confirms it")
        if strengths.count(Strength.POSSIBLE) > 1:
            raise ValueError("a condition has one piece for 'not recorded'")
        weaker = [e for e in self.evidence if e.strength == Strength.PROBABLE]
        if len(weaker) > 1 and any(e.source.group is None for e in weaker):
            # The score combines weaker pieces by source group (score_sql).
            raise ValueError("weaker evidence without a source group is the only weaker piece")
        at = [i for i, strength in enumerate(strengths) if strength == Strength.CONTRADICTED]
        if at:
            if at != list(range(at[0], at[-1] + 1)):
                raise ValueError("the records that contradict a condition are listed together")
            if Strength.PROBABLE in strengths[:at[0]] and Strength.PROBABLE in strengths[at[-1]:]:
                # The tier keeps one column for all weaker evidence of a
                # condition (tier.apply); two groups would need two.
                raise ValueError("weaker evidence is listed all before or all after the record")

    # ── The parts of the list ─────────────────────────────────────────────────

    @property
    def weaker(self) -> Tuple[UncertainEvidence, ...]:
        """The evidence for the probable state, in the order it counts."""
        return tuple(e for e in self.evidence if e.strength == Strength.PROBABLE)

    @property
    def not_recorded(self) -> Optional[UncertainEvidence]:
        """The evidence for the possible state."""
        return next((e for e in self.evidence if e.strength == Strength.POSSIBLE), None)

    @property
    def probable(self) -> Optional[str]:
        """SQL: some weaker evidence holds. None without any."""
        parts = [e.sql() for e in self.weaker]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else " OR ".join(f"({part})" for part in parts)

    @property
    def known(self) -> Optional[str]:
        """SQL: the field is recorded, in any of its places. None for a
        condition that reads no field a dataset may leave out (a text search)."""
        parts = [e.sql() for e in self.evidence if e.strength == Strength.CONTRADICTED]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else " OR ".join(f"({part})" for part in parts)

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

    def score_sql(self, state: str, holds: Sequence[str], recorded: Optional[str] = None) -> str:
        """The score of the clause for a dataset, from 0 to 1
        (services/evidence/SCORING.md): 1 when it holds as generated, 0 when
        it is contradicted, else what its pieces of evidence add up to.

        Pieces of one source group count once, by their strongest; a second
        group adds SHARED_EVIDENCE_DISCOUNT of what it has above the field's
        "not recorded" score, which every piece already contains.

        *state*: the column with the clause's Strength. *holds*: for each
        weaker piece, SQL that says whether it holds. *recorded*: SQL that
        says the field is recorded, needed where weaker evidence counts
        although it is (weaker_beats_record).
        """
        pieces = self.weaker
        floor = self.not_recorded.score() if self.not_recorded is not None else 0.0
        branches = [f"WHEN {state} = {int(Strength.CONFIRMED)} THEN 1"]
        if pieces:
            branches.append(f"WHEN {state} = {int(Strength.PROBABLE)} "
                            f"THEN {self._probable_score_sql(holds, floor, recorded)}")
        if self.not_recorded is not None:
            branches.append(f"WHEN {state} = {int(Strength.POSSIBLE)} THEN {floor:g}")
        return f"CASE {' '.join(branches)} ELSE 0 END"

    def _probable_score_sql(self, holds: Sequence[str], floor: float, recorded: Optional[str]) -> str:
        pieces = self.weaker
        # The floor applies to a dataset that records nothing; where weaker
        # evidence counts beside a record, only when there is none.
        floor_always = floor > 0 and not self.weaker_beats_record
        if len(pieces) == 1 and (floor_always or floor == 0 or pieces[0].score() >= floor):
            # One piece, and it holds: nothing to combine.
            return f"{max(pieces[0].score(), floor if floor_always else 0.0):g}"
        floor_weight = None
        if floor > 0:
            floor_weight = (f"{log_weight(floor):.4f}" if floor_always
                            else f"CASE WHEN NOT {recorded} THEN {log_weight(floor):.4f} ELSE 0 END")
        of_group: Dict[Optional[SourceGroup], List[str]] = {}
        for piece, piece_holds in zip(pieces, holds):
            of_group.setdefault(piece.source.group, []).append(
                f"CASE WHEN {piece_holds} THEN {log_weight(piece.score()):.4f} ELSE 0 END")
        groups = [_greatest(terms) for terms in of_group.values()]
        strongest = _greatest(groups if floor_weight is None else [floor_weight, *groups])
        if len(groups) == 1:
            return f"1 - EXP(-({strongest}))"
        if floor_weight is None:
            others = f"{' + '.join(groups)} - {strongest}"
        else:
            above_floor = [f"GREATEST({group} - {floor_weight}, 0)" for group in groups]
            others = f"{' + '.join(above_floor)} - ({strongest} - {floor_weight})"
        return f"1 - EXP(-({strongest} + {SHARED_EVIDENCE_DISCOUNT:g} * ({others})))"

    @property
    def witnesses(self) -> Dict[SourceGroup, List[ConfirmedEvidence]]:
        """The confirming pieces by whose statement they are (Source.group).
        An assumed default states nothing, so it is none."""
        groups: Dict[SourceGroup, List[ConfirmedEvidence]] = {}
        for piece in self.confirming:
            if piece.source.group is not None:
                groups.setdefault(piece.source.group, []).append(piece)
        return groups

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
    conditions: Tuple[ClauseEvidence, ...]
    row: str = "evidence"   # SQL alias of that row

    def state_column(self, i: int) -> str:
        return f"u{i}"

    def flag_columns(self, i: int) -> List[str]:
        """Columns that say which weaker pieces of condition *i* hold, one
        per piece: "papers_0" for evidence found in a paper. None for a
        condition with a single weaker piece: its state says it."""
        pieces = self.conditions[i].weaker
        names: List[str] = []
        for j, piece in enumerate(pieces if len(pieces) > 1 else ()):
            name = f"{piece.source.value}_{i}"
            names.append(f"{name}_{j}" if name in names else name)
        return names

    def recorded_column(self, i: int) -> Optional[str]:
        """The column that says the field of condition *i* is recorded, where
        the score needs it (ClauseEvidence.score_sql)."""
        clause = self.conditions[i]
        if clause.weaker_beats_record and clause.not_recorded is not None:
            return f"known_{i}"
        return None

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

    def score_sql(self) -> str:
        """The score of the query for the dataset, from 0 to 1
        (services/evidence/SCORING.md): its weakest clause, times the other
        clauses to the power CLAUSE_INDEPENDENCE. With 1 that is the product
        of all clauses: the query is their AND, and taking the clauses as
        independent, the product is the share of relevant datasets among
        those with this evidence, relative to one that every clause confirms."""
        scores = []
        for i, (state, clause) in enumerate(zip(self._states(), self.conditions)):
            flags = [f"{self.row}.{flag}" for flag in self.flag_columns(i)] or ["TRUE"]
            recorded = self.recorded_column(i)
            scores.append("(" + clause.score_sql(state, flags, f"{self.row}.{recorded}" if recorded else None) + ")")
        product = " * ".join(scores)
        if CLAUSE_INDEPENDENCE == 1 or len(scores) == 1:
            return product
        weakest = f"LEAST({', '.join(scores)})"
        return (f"COALESCE({weakest} * POWER(({product}) / NULLIF({weakest}, 0), "
                f"{CLAUSE_INDEPENDENCE:g}), 0)")

    def witness_columns(self, i: int) -> Dict[str, List[ConfirmedEvidence]]:
        """Columns that say which source groups confirm condition *i*, with
        the pieces of each. None with a single group: it cannot tell datasets apart."""
        groups = self.conditions[i].witnesses
        if len(groups) < 2:
            return {}
        return {f"by_{group.value}_{i}": pieces for group, pieces in groups.items()}

    def corroboration_sql(self) -> Optional[str]:
        """How many source groups confirm the query's conditions beyond the
        first of each (services/evidence/SCORING.md). Counted only for a
        condition that holds as generated; None when no condition can have
        two."""
        counts = []
        for i, state in enumerate(self._states()):
            columns = self.witness_columns(i)
            if columns:
                held = " + ".join(f"{self.row}.{column}::int" for column in columns)
                counts.append(f"CASE WHEN {state} = {int(Strength.CONFIRMED)} "
                              f"THEN GREATEST({held} - 1, 0) ELSE 0 END")
        return " + ".join(counts) or None

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
                    flag = f" AND {self.row}.{flags[j]}" if j < len(condition.weaker) - 1 else ""
                    branches.append(f"WHEN {state} = {int(Strength.PROBABLE)}{flag} "
                                    f"THEN {sql_literal(reason)}")
            if condition.not_recorded is not None and (reason := condition.not_recorded.reason()):
                branches.append(f"WHEN {state} = {int(Strength.POSSIBLE)} THEN {sql_literal(reason)}")
            if branches:
                cases.append(f"CASE {' '.join(branches)} END")
        return f"NULLIF(CONCAT_WS('; ', {', '.join(cases)}), '')" if cases else "NULL"
