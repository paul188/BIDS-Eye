"""
services/uncertain_tier.py
--------------------------
Keep datasets that a filter cannot check, and show them after the confirmed
results.

Some filters fail only because a dataset does not record the field at all:
about half the datasets have no participant ages, most have no coded
diagnosis. Some text filters fail because a dataset uses only some of the
words ("MRI data of 3-12 year old children" for "young & children"). On the
167-question recall benchmark these exclusions are most of what the search
misses (benchmarks/recall/work/failure_analysis/).

Each top-level AND condition of the generated SQL gets one of four states for
a dataset:

    0  confirmed     the condition holds
    1  probable      a weaker form holds: some of the search words, the words
                     only in a paper the dataset names (with TEXT_SEARCH_PAPERS
                     on), or, for a missing field, the README names the asked
                     group ("children" for an age filter up to 12)
    2  possible      the field is not recorded and nothing hints either way
    9  contradicted  the dataset records the field and it does not fit

Only conditions on fields that can be missing (ages, diagnosis, sex) or on
text can be uncertain. Modality, file types, size and subject counts come
from the files themselves and stay hard filters, as does anything negated.

A dataset is kept when
  - no condition is contradicted,
  - at most MAX_UNSURE conditions are not confirmed, and
  - at least one uncertain-able condition is confirmed or probable: a dataset
    that only matches the modality while everything else is unknown is not
    evidence of anything.

The result gets two columns: uncertain_level (0, 1 or 2: the weakest state)
and uncertain_reasons (why, for the result card). The page query orders by
uncertain_level first (services/sql_rewriter.py). Datasets that confirm every
condition are exactly the rows the original query returned.

Switched on with UNCERTAIN_TIER=true; the benchmark compares both from one
translation (benchmarks/recall/paired_uncertain_run.py).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:  # sqlglot is a backend dependency; keep the module importable without it
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None
    exp = None

log = logging.getLogger(__name__)

ENV_FLAG = "UNCERTAIN_TIER"
# Conditions per dataset that may be unconfirmed.
MAX_UNSURE = int(os.getenv("UNCERTAIN_MAX_UNSURE", "1"))


def possible_enabled() -> bool:
    """Whether a missing field without any hint (state 2) may keep a dataset."""
    # Off by default: on the benchmark this level added 2,211 rows for 0.4
    # points of recall, 11% of the judged ones relevant (2026-11-13).
    return os.getenv("UNCERTAIN_POSSIBLE", "false").strip().lower() in ("1", "true", "yes", "on")

LEVEL_COLUMN = "uncertain_level"
REASONS_COLUMN = "uncertain_reasons"

_TSQUERY_RE = re.compile(
    r"(?P<negated>\bNOT\s+)?(?P<alias>\b[A-Za-z_]\w*)\.(?P<column>search_tsv(?:_papers)?)\s+@@\s+"
    r"TO_TSQUERY\(\s*'english'\s*,\s*'(?P<query>[^']*)'\s*\)",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r":[AB]+\b")
_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NUMBER_COMPARISON_RE = re.compile(
    r"\bage(?:_min|_max)?\s*(?P<op><=|>=|<|>|=)\s*(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE)
_BETWEEN_RE = re.compile(
    r"\bage(?:_min|_max)?\s+BETWEEN\s+(?P<low>\d+(?:\.\d+)?)\s+AND\s+(?P<high>\d+(?:\.\d+)?)",
    re.IGNORECASE)

# Age groups and the words a README uses for them, with the ages they cover.
_AGE_GROUPS: Tuple[Tuple[float, float, Tuple[str, ...]], ...] = (
    (0, 3, ("infant", "infants", "newborn", "newborns", "neonate", "neonates", "neonatal",
            "toddler", "toddlers", "baby", "babies")),
    (3, 12, ("child", "children", "kids", "pediatric", "paediatric", "preschool",
             "school-age", "schoolchildren")),
    (12, 19, ("adolescent", "adolescents", "adolescence", "teen", "teens", "teenager",
              "teenagers", "youth", "youths")),
    (18, 65, ("adult", "adults")),
    # Not "aging" / "ageing" / "aged": the stemmer turns them into "age", which
    # every README with an age range contains. Not "older" alone either.
    (60, 120, ("older-adults", "older-adult", "older-participants", "elderly", "geriatric",
               "seniors", "old-age")),
)

_REASON = {
    "age": "ages not recorded",
    "diagnosis": "diagnosis not recorded",
    "sex": "participant sex not recorded",
}


def enabled() -> bool:
    """True when UNCERTAIN_TIER is on. Read on every call."""
    return os.getenv(ENV_FLAG, "false").strip().lower() in ("1", "true", "yes", "on")


# ── Building blocks ───────────────────────────────────────────────────────────

def _top_level_and(query: str) -> List[str]:
    """Split a tsquery on the & signs outside parentheses."""
    parts, depth, current = [], 0, ""
    for char in query:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "&" and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    parts.append(current.strip())
    return [part for part in parts if part]


# Words that qualify the thing searched for rather than name it.
_MODIFIERS = frozenset({
    "young", "old", "older", "healthy", "many", "few", "single", "multiple", "different",
    "large", "small", "high", "low", "new", "long", "short", "early", "late", "normal",
    "typical", "various", "several", "repeated", "dense", "continuous", "natural",
})


def _rarest_parts(parts: List[str]) -> List[str]:
    """The rarer half of a term's parts (rounded up), in their original order.

    Rarity comes from how many datasets contain each part
    (text_search's frequency lookup); without it, the longest parts.
    """
    keep = (len(parts) + 1) // 2
    if keep >= len(parts):
        return parts
    counts = None
    try:
        from services import text_search
        counts = text_search._lookup_counts(parts)  # noqa: SLF001 — shared helper
    except Exception:  # pragma: no cover — no database available
        counts = None
    if counts and len(counts) == len(parts):
        # A modifier ("young", "healthy", "many") is never kept in place of a
        # noun, however rare it happens to be.
        order = sorted(range(len(parts)),
                       key=lambda i: (parts[i].strip("()") in _MODIFIERS, counts[i], -len(parts[i])))
    else:
        order = sorted(range(len(parts)), key=lambda i: -len(parts[i]))
    chosen = set(order[:keep])
    return [part for i, part in enumerate(parts) if i in chosen]


def relax_tsquery(query: str) -> Optional[str]:
    """The weaker form of a tsquery: its rarer half, anywhere in the text.

    "young & children" → "children" ("young" is a modifier, never kept alone);
    "evaluated & resective & surgery" → "resective & surgery";
    "language:AB" → "language" (the README counts too).
    None when relaxing changes nothing.
    """
    unscoped = _LABEL_RE.sub("", query)
    parts = [f"({p})" if ("|" in p and not p.startswith("(")) else p
             for p in _top_level_and(unscoped)]
    relaxed = " & ".join(_rarest_parts(parts))
    return relaxed if relaxed.replace(" ", "") != query.replace(" ", "") else None


def query_words(query: str) -> List[str]:
    words = _WORD_RE.findall(_LABEL_RE.sub("", query).replace("<->", " "))
    return list(dict.fromkeys(words))


def _negated_spans(sql: str) -> List[Tuple[int, int]]:
    from services import text_search  # local import: text_search imports heavy modules
    return text_search.negated_spans(sql)


_README_ONLY = "\x00readme"


def partial_reason(words: List[str]) -> str:
    """Card text for a probable text match."""
    missing = [w for w in words if not w.startswith(_README_ONLY)]
    if not missing:
        # _README_ONLY + the words whose scope was widened, e.g. "\x00readme harvard".
        terms = list(dict.fromkeys(
            t for w in words for t in w[len(_README_ONLY):].split() if t))
        if not terms:
            return "the search term appears only further down in the README"
        shown = " / ".join(f'"{t}"' for t in terms[:2]) + (" …" if len(terms) > 2 else "")
        return f"{shown} appears only further down in the README"
    shown = ", ".join(missing[:4]) + (" …" if len(missing) > 4 else "")
    return f"matches only part of the search; missing: {shown}"


def relax_text_condition(sql: str) -> Tuple[Optional[str], List[str]]:
    """The weaker text evidence for a condition, and the words involved.

    The weaker form is only the relaxed text searches of the condition, ORed:
    the rest of the condition (often an expensive check over the files) has
    already been evaluated for the confirmed state, and when it holds the
    dataset is confirmed anyway. (None, []) when no positive tsquery would
    change.
    """
    spans = _negated_spans(sql)
    words: List[str] = []
    relaxed_atoms: List[str] = []
    for match in _TSQUERY_RE.finditer(sql):
        inside_not = match.group("negated") or any(s < match.start() < e for s, e in spans)
        if inside_not:
            continue
        relaxed = relax_tsquery(match.group("query"))
        if relaxed is None:
            continue
        dropped = set(query_words(match.group("query"))) - set(query_words(relaxed))
        # No word dropped: only the scope was widened, to the whole README.
        words.extend(sorted(dropped) or [f"{_README_ONLY} {' '.join(query_words(relaxed))}"])
        relaxed_atoms.append(f"{match.group('alias')}.{match.group('column')} @@ "
                             f"TO_TSQUERY('english', '{relaxed}')")
    if not relaxed_atoms:
        return None, []
    return " OR ".join(dict.fromkeys(relaxed_atoms)), list(dict.fromkeys(words))


def asked_age_range(sql: str) -> Tuple[float, float]:
    """The age range an age condition asks for, read from its comparisons."""
    low, high = 0.0, 120.0
    for match in _BETWEEN_RE.finditer(sql):
        low = max(low, float(match.group("low")))
        high = min(high, float(match.group("high")))
    for match in _NUMBER_COMPARISON_RE.finditer(sql):
        value = float(match.group("value"))
        op = match.group("op")
        if op in (">", ">="):
            low = max(low, value)
        elif op in ("<", "<="):
            high = min(high, value)
    return low, high


def age_group_words(low: float, high: float) -> List[str]:
    """README words for the age groups that overlap [low, high]."""
    if low <= 0 and high >= 120:
        return []
    words: List[str] = []
    for group_low, group_high, group_words in _AGE_GROUPS:
        # The group must cover a good part of the asked range, or the asked
        # range a good part of the group: "60 and over" is not about adults
        # in general, "18 to 65" not about older adults.
        overlap = min(group_high, high) - max(group_low, low)
        needed = min((high - low) / 2, (group_high - group_low) / 2)
        if overlap > 0 and overlap >= needed:
            words.extend(group_words)
    return list(dict.fromkeys(words))


def _tsquery_for_words(words: Sequence[str]) -> str:
    parts = []
    for word in words:
        pieces = word.split("-")
        parts.append("(" + " <-> ".join(pieces) + ")" if len(pieces) > 1 else word)
    return " | ".join(parts)


def _known_sql(kind: str, alias: str) -> str:
    """True when the dataset records the field the condition reads."""
    if kind == "age":
        return (f"({alias}.age_min IS NOT NULL OR {alias}.age_max IS NOT NULL OR EXISTS("
                f"SELECT 1 FROM bids_participants AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND uk.age IS NOT NULL))")
    if kind == "diagnosis":
        return (f"EXISTS(SELECT 1 FROM bids_participants AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND uk.diagnosis IS NOT NULL)")
    if kind == "sex":
        return (f"EXISTS(SELECT 1 FROM bids_participants AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND uk.sex IS NOT NULL)")
    raise ValueError(kind)


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


# ── Classification ────────────────────────────────────────────────────────────

PAPER_REASON = "found only in a paper the dataset names"


@dataclass
class SoftCondition:
    sql: str                    # the condition as generated
    kind: str                   # age, diagnosis, sex or text
    probable: Optional[str]     # SQL for the weaker form, or None
    known: Optional[str]        # SQL: the field is recorded (None for text)
    reason_probable: str
    reason_possible: str
    paper: Optional[str] = None  # SQL: the terms match the dataset's paper text

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


def _columns_outside_subqueries(node) -> set:
    found = set()
    for column in node.find_all(exp.Column):
        if column.find_ancestor(exp.Subquery, exp.Exists) is None:
            found.add(column.table or "")
    return found


def paper_atoms(sql: str, alias: str) -> Optional[str]:
    """The condition's positive text searches run on the paper vector, ORed.

    Labels are dropped: paper text carries the lowest weight, so a search
    scoped to the name and short fields could never reach it.
    """
    from services.text_search import PAPER_FTS_COLUMN
    spans = _negated_spans(sql)
    atoms = []
    for match in _TSQUERY_RE.finditer(sql):
        if match.group("negated") or any(s < match.start() < e for s, e in spans):
            continue
        query = _LABEL_RE.sub("", match.group("query"))
        # The paper vector also holds the dataset's own text; "only in a
        # paper" means it matches there and not in the dataset's text.
        atoms.append(f"({alias}.{PAPER_FTS_COLUMN} @@ TO_TSQUERY('english', '{query}') "
                     f"AND NOT {alias}.search_tsv @@ TO_TSQUERY('english', '{query}'))")
    return " OR ".join(dict.fromkeys(atoms)) or None


def classify(node, alias: str, papers: bool = False) -> Optional[SoftCondition]:
    """A SoftCondition for a condition that may be uncertain, else None.

    *papers*: the database has paper text and it may serve as evidence.
    """
    if isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Not):
        return None
    sql = node.sql(dialect="postgres")
    columns = {column.name.lower() for column in node.find_all(exp.Column)}
    # Sidecar values (field strength, echo times …) are recorded with the
    # files, like the modality, and reading them is slow: evaluating such a
    # condition a second time for the tier took two minutes on one question.
    if "other_entities" in columns:
        return None
    # A resolved institution filter is evidence from the scanner headers. Its
    # weaker form would be the name anywhere in the README or a paper, and
    # those mentions are mostly suppliers, software and citations ('kindly
    # provided by … Harvard Medical School', surfer.nmr.mgh.harvard.edu).
    if "institutions" in columns:
        return None
    # A condition on the outer join's rows (o.datatype …) is about files.
    if _columns_outside_subqueries(node) - {alias, ""}:
        return None
    kinds = []
    if columns & {"age", "age_min", "age_max"}:
        kinds.append("age")
    if columns & {"diagnosis", "dx_status"}:
        kinds.append("diagnosis")
    if "sex" in columns:
        kinds.append("sex")
    if len(kinds) > 1:
        return None
    relaxed, words = relax_text_condition(sql)
    paper = paper_atoms(sql, alias) if papers else None
    if kinds:
        kind = kinds[0]
        probable = relaxed
        reason_probable = partial_reason(words) if relaxed else ""
        if kind == "age":
            group_words = age_group_words(*asked_age_range(sql))
            paper = None
            if group_words:
                vector = "search_tsv_papers" if papers else "search_tsv"
                probable = (f"{alias}.{vector} @@ TO_TSQUERY('english', "
                            f"'{_tsquery_for_words(group_words)}')")
                reason_probable = "ages not recorded; the description names the age group"
        return SoftCondition(
            sql=sql, kind=kind, probable=probable, known=_known_sql(kind, alias),
            reason_probable=reason_probable or _REASON[kind],
            reason_possible=_REASON[kind], paper=paper,
        )
    if relaxed or paper:
        return SoftCondition(
            sql=sql, kind="text", probable=relaxed, known=None,
            reason_probable=partial_reason(words) if relaxed else "",
            reason_possible="", paper=paper,
        )
    return None


# ── Rewrite ───────────────────────────────────────────────────────────────────

def _flatten_and(node) -> list:
    if isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
        return _flatten_and(node.this)
    return [node]


def _dataset_alias(select) -> Optional[str]:
    source = select.args.get("from") or select.args.get("from_")
    table = getattr(source, "this", None)
    if isinstance(table, exp.Table) and table.name == "bids_datasets":
        return table.alias_or_name
    return None


def apply(sql: str, papers: Optional[bool] = None) -> str:
    """The query with uncertain conditions softened and the two tier columns.

    *papers*: let paper text serve as evidence (default: TEXT_SEARCH_PAPERS
    and the column exist). Returns *sql* unchanged when it has nothing that
    can be uncertain, or when it cannot be parsed.
    """
    if papers is None:
        from services import text_search
        papers = text_search.papers_enabled()
    if sqlglot is None or LEVEL_COLUMN in sql:
        return sql
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return sql
    if not isinstance(tree, exp.Select):
        return sql
    # A top-N or ordered question ("the 10 largest …") keeps its own order and
    # limit; softening the filter inside them would let rows the tier later
    # drops fill the limit and push confirmed results out.
    if tree.args.get("limit") or tree.args.get("order") or tree.args.get("offset"):
        return sql
    alias = _dataset_alias(tree)
    where = tree.args.get("where")
    if alias is None or where is None:
        return sql

    hard: List[str] = []
    soft: List[SoftCondition] = []
    for node in _flatten_and(where.this):
        condition = None
        try:
            condition = classify(node, alias, papers=papers)
        except Exception as exc:  # one odd condition must not break the query
            log.debug("uncertain_tier: could not classify %s (%s)", node, exc)
        if condition is None:
            hard.append(node.sql(dialect="postgres"))
        else:
            soft.append(condition)
    if not soft:
        return sql

    inner = tree.copy()
    keep = " AND ".join([*(f"({h})" for h in hard), *(c.keep_sql() for c in soft)])
    inner.set("where", exp.Where(this=sqlglot.parse_one(keep, read="postgres")))
    inner_sql = inner.sql(dialect="postgres")

    states = ",\n      ".join(
        [f"{c.state_sql()} AS u{i}" for i, c in enumerate(soft)]
        + [f"({c.paper}) AS p{i}" for i, c in enumerate(soft) if c.paper])
    names = [f"us.u{i}" for i in range(len(soft))]
    unsure = " + ".join(f"(CASE WHEN {n} > 0 THEN 1 ELSE 0 END)" for n in names)
    evidence = " OR ".join(f"{n} <= 1" for n in names)
    level = f"GREATEST({', '.join(names)})" if len(names) > 1 else names[0]
    reasons = []
    for i, c in enumerate(soft):
        branches = []
        if c.paper:
            branches.append(f"WHEN us.u{i} = 1 AND us.p{i} THEN {_sql_literal(PAPER_REASON)}")
        if c.reason_probable:
            branches.append(f"WHEN us.u{i} = 1 THEN {_sql_literal(c.reason_probable)}")
        if c.reason_possible:
            branches.append(f"WHEN us.u{i} = 2 THEN {_sql_literal(c.reason_possible)}")
        if branches:
            reasons.append(f"CASE {' '.join(branches)} END")
    reasons_sql = (f"NULLIF(CONCAT_WS('; ', {', '.join(reasons)}), '')" if reasons else "NULL")

    return (
        f"SELECT tier_base.*, ut.{LEVEL_COLUMN}, ut.{REASONS_COLUMN}\n"
        f"FROM (\n{inner_sql}\n) AS tier_base\n"
        "CROSS JOIN LATERAL (\n"
        f"  SELECT {level} AS {LEVEL_COLUMN}, {reasons_sql} AS {REASONS_COLUMN},\n"
        f"         {unsure} AS n_unsure, ({evidence}) AS has_evidence\n"
        "  FROM (\n"
        f"    SELECT {states}\n"
        f"    FROM bids_datasets AS {alias} WHERE {alias}.id = tier_base.id\n"
        "  ) AS us\n"
        ") AS ut\n"
        f"WHERE ut.{LEVEL_COLUMN} < 9 AND ut.n_unsure <= {MAX_UNSURE} AND ut.has_evidence"
    )


def confirmed_only(rows: Sequence[dict]) -> list:
    """The rows every condition confirms (what the original query returns)."""
    return [row for row in rows if not row.get(LEVEL_COLUMN)]
