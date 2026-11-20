"""
services/fulltext/uncertain_tier.py
-----------------------------------
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

Only conditions on fields that can be missing (ages, diagnosis, sex, the
field strength and scanner make or model in the sidecars) or on text can be
uncertain. For each, the dataset's own text naming the asked value is the
probable form: "women", "all-female", "7T", "Siemens". Modality, file
types, size, subject counts and the other sidecar values come from the
files themselves and stay hard filters, as does anything negated.

"At least N participants with a diagnosis" is its own case (_patient_count):
confirmed when that many are coded or labelled with it, probable when the
dataset does not record who has the diagnosis, contradicted when its
participant rows show fewer.

A dataset is kept when
  - no condition is contradicted,
  - at most MAX_UNSURE conditions are not confirmed, and
  - at least one uncertain-able condition is confirmed or probable: a dataset
    that only matches the modality while everything else is unknown is not
    evidence of anything.

The result gets two columns: uncertain_level (0, 1 or 2: the weakest state)
and uncertain_reasons (why, for the result card). The page query orders by
uncertain_level first (services/translate/sql_rewriter.py). Datasets that confirm every
condition are exactly the rows the original query returned.

Switched on with UNCERTAIN_TIER=true; the benchmark compares both from one
translation (benchmarks/recall/paired_uncertain_run.py).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from text_terms import MODIFIER_WORDS, NEVER_ALONE_WORDS

try:  # sqlglot is a backend dependency; keep the module importable without it
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

ENV_FLAG = "UNCERTAIN_TIER"
# Conditions per dataset that may be unconfirmed.
MAX_UNSURE = int(os.getenv("UNCERTAIN_MAX_UNSURE", "1"))


def possible_enabled() -> bool:
    """Whether a missing field without any hint (state 2) may keep a dataset."""
    # Off by default: on the benchmark this level added 2,211 rows for 0.4
    # points of recall, 11% of the judged ones relevant (2026-11-13).
    return os.getenv("UNCERTAIN_POSSIBLE", "false").strip().lower() in ("1", "true", "yes", "on")

def hoist_terms_enabled() -> bool:
    """Name each tsquery once in a one-row CTE instead of repeating its literal.

    Off unless UNCERTAIN_HOIST_TERMS=true. It shortens cohort SQL by a third,
    but the planner no longer sees the tsquery as a constant: on the live
    database (2026-11-15, ten query shapes) two text-led shapes ran 4x faster
    and one age shape 2x slower; without it every shape ran as before.
    """
    return os.getenv("UNCERTAIN_HOIST_TERMS", "false").strip().lower() in ("1", "true", "yes", "on")


LEVEL_COLUMN = "uncertain_level"
REASONS_COLUMN = "uncertain_reasons"

# A stemmed text search, with the guard that may follow it: the word forms
# the unstemmed vector must contain (services/fulltext/text_search.py, exact word forms).
_TSQUERY_RE = re.compile(
    r"(?P<negated>\bNOT\s+)?(?P<alias>\b[A-Za-z_]\w*)\.(?P<column>search_tsv(?:_papers)?)\s+@@\s+"
    r"TO_TSQUERY\(\s*'english'\s*,\s*'(?P<query>[^']*)'\s*\)"
    r"(?:\s+AND\s+(?P=alias)\.(?P<guard_column>search_tsv_exact(?:_papers)?)\s+@@\s+"
    r"TO_TSQUERY\(\s*'simple'\s*,\s*'(?P<guard>[^']*)'\s*\))?",
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


# Words that qualify the thing searched for rather than name it, and nouns
# too general to stand for the term they were part of (RAG/text_terms.py).
_MODIFIERS = MODIFIER_WORDS
_NEVER_ALONE = NEVER_ALONE_WORDS

# A shortened term that dropped a content word is evidence only when what is
# left is rare: at most this many datasets (about 1% of the 1,721) contain it.
# "consolidation" from "memory consolidation" (10) or "resective & surgery"
# (7) still stand for the term; "imagery" from "motor imagery" (42),
# "recognition" from "emotion recognition" (52) or "cortex" from "visual
# cortex" (142) do not (live database, 2026-11-17).
RELAXED_MAX_MATCHES = 17


def _says_nothing(part: str) -> bool:
    """A tsquery part that is only a modifier or a too-general noun."""
    bare = part.strip("()")
    return bare in _MODIFIERS or bare in _NEVER_ALONE


def _rare_enough(kept: List[str], guard: Optional[str]) -> bool:
    """Whether the shortened term matches few enough datasets to stand for the whole.

    True when the catalogue cannot be asked (no database): the shortening
    then stays as it was.
    """
    query = " & ".join(kept)
    from services.fulltext import text_search
    # None when the catalogue cannot be asked (guarded_counts catches that).
    counts = text_search.guarded_counts([query], [guard_for(query, guard)])
    if not counts:
        return True
    return counts[0] <= RELAXED_MAX_MATCHES


def guard_for(query: str, guard: Optional[str]) -> Optional[str]:
    """The part of *guard* that belongs to the words of *query*, without labels.

    A guard has one group per exact word, the asked word first:
    "(ageing | aging) & weight". A relaxed query that dropped a word drops
    its group, and searches the whole text, so the labels go.
    """
    if not guard:
        return None
    words = set(query_words(query))
    kept = []
    for group in _top_level_and(_LABEL_RE.sub("", guard)):
        asked = _WORD_RE.search(group)
        if asked and asked.group(0) in words:
            kept.append(group)
    return " & ".join(kept) or None


def _rarest_parts(parts: List[str], guard: Optional[str] = None) -> List[str]:
    """The rarer half of a term's parts (rounded up), in their original order.

    Rarity comes from how many datasets contain each part
    (text_search's frequency lookup); without it, the longest parts. A part
    with exact word forms (*guard*) is counted in those forms.
    """
    keep = (len(parts) + 1) // 2
    if keep >= len(parts):
        return parts
    from services.fulltext import text_search
    # None when the catalogue cannot be asked (guarded_counts catches that).
    counts = text_search.guarded_counts(parts, [guard_for(part, guard) for part in parts])
    if counts and len(counts) == len(parts):
        # A modifier ("young", "healthy", "many") is never kept in place of a
        # noun, however rare it happens to be.
        order = sorted(range(len(parts)),
                       key=lambda i: (_says_nothing(parts[i]), counts[i], -len(parts[i])))
    else:
        order = sorted(range(len(parts)), key=lambda i: -len(parts[i]))
    chosen = set(order[:keep])
    return [part for i, part in enumerate(parts) if i in chosen]


# Exact word distances (`a <2> b`), written only by 'near' text searches;
# `<->` inside a hyphenated word is not one of them.
_DISTANCE_RE = re.compile(r"<\d+>")


def _top_level_or(query: str) -> List[str]:
    """Split a tsquery on the | signs outside parentheses."""
    parts, depth, current = [], 0, ""
    for char in query:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "|" and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    parts.append(current.strip())
    return [part for part in parts if part]


def _strip_parentheses(query: str) -> str:
    query = query.strip()
    while query.startswith("(") and query.endswith(")"):
        depth = 0
        for index, char in enumerate(query):
            depth += char == "("
            depth -= char == ")"
            if depth == 0 and index < len(query) - 1:
                return query
        query = query[1:-1].strip()
    return query


def relax_near(query: str) -> Optional[str]:
    """A 'near' search with its words anywhere instead of close together.

    "(social <1> conflict | … | conflict <2> social) | aggression"
    → "(social & conflict) | aggression". Never fewer words: the qualifier
    is what makes "social conflict" more than "conflict". None when the
    query has no word distances.
    """
    if not _DISTANCE_RE.search(query):
        return None
    alternatives = []
    for alternative in _top_level_or(query):
        if _DISTANCE_RE.search(alternative):
            first = _top_level_or(_strip_parentheses(alternative))[0]
            alternative = "(" + re.sub(r"\s*<\d+>\s*", " & ", first) + ")"
        alternatives.append(alternative)
    return " | ".join(dict.fromkeys(alternatives))


def relax_tsquery(query: str, guard: Optional[str] = None) -> Optional[str]:
    """The weaker form of a tsquery: its rarer half, anywhere in the text.

    "young & children" → "children" ("young" is a modifier, never kept alone);
    "evaluated & resective & surgery" → "resective & surgery";
    "language:AB" → "language" (the README counts too);
    a 'near' search → all its words anywhere (relax_near).
    None when relaxing changes nothing.
    """
    near = relax_near(query)
    if near is not None:
        return near
    unscoped = _LABEL_RE.sub("", query)
    parts = [f"({p})" if ("|" in p and not p.startswith("(")) else p
             for p in _top_level_and(unscoped)]
    kept = _rarest_parts(parts, guard)
    # A word that says nothing alone ("state" from "resting & state",
    # "syndrome", "healthy") is no evidence: no weaker form then.
    if len(kept) < len(parts) and all(_says_nothing(k) for k in kept):
        return None
    # Dropping a content word ("visual & cortex" → "cortex") needs a rare
    # remainder; dropping only general words ("emotional & state" →
    # "emotional") does not. Otherwise every word stays and only the scope
    # widens.
    dropped = [p for p in parts if p not in kept]
    if dropped and not all(_says_nothing(p) for p in dropped) and not _rare_enough(kept, guard):
        kept = parts
    relaxed = " & ".join(kept)
    return relaxed if relaxed.replace(" ", "") != query.replace(" ", "") else None


def query_words(query: str) -> List[str]:
    words = _WORD_RE.findall(re.sub(r"<(?:-|\d+)>", " ", _LABEL_RE.sub("", query)))
    return list(dict.fromkeys(words))


def _negated_spans(sql: str) -> List[Tuple[int, int]]:
    from services.fulltext import text_search  # local import: text_search imports heavy modules
    return text_search.negated_spans(sql)


_README_ONLY = "\x00readme"
# Markers for the two weaker forms of a topic search (preprocess.py, topics):
# its words anywhere instead of together, and a match on a meaning the
# question excluded. Each is followed by the words, e.g. "\x00apart social conflict".
_APART = "\x00apart"
_EXCLUDED = "\x00excluded"


def _marked_terms(words: List[str], marker: str) -> List[str]:
    return list(dict.fromkeys(
        t for w in words if w.startswith(marker) for t in w[len(marker):].split() if t))


def _shown(terms: List[str], limit: int = 3) -> str:
    return ", ".join(terms[:limit]) + (" …" if len(terms) > limit else "")


def partial_reason(words: List[str]) -> str:
    """Card text for a probable text match."""
    apart = _marked_terms(words, _APART)
    excluded = _marked_terms(words, _EXCLUDED)
    words = [w for w in words if not w.startswith((_APART, _EXCLUDED))]
    # One text serves every dataset at this level, so the topic's two weaker
    # forms are named as alternatives: either may be why a dataset is here.
    loose = []
    if apart:
        loose.append(f"the words {_shown(apart)} occur, but not together")
    if excluded:
        loose.append(f"a related term occurs next to a meaning the search excludes "
                     f"({_shown(excluded)})")
    reasons = [", or ".join(loose)] if loose else []
    if words or not reasons:
        reasons.append(_text_reason(words))
    return "; ".join(reasons)


def _text_reason(words: List[str]) -> str:
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
    matches = list(_TSQUERY_RE.finditer(sql))

    def negated(match: re.Match) -> bool:
        return bool(match.group("negated")) or any(s < match.start() < e for s, e in spans)

    # A topic search excludes the meanings it does not want:
    #   phrase OR (alternatives AND NOT unwanted meanings).
    # Dropping that NOT is a weaker form too, so a dataset that matches an
    # alternative and also an unwanted meaning moves down instead of out.
    excluded_words = [w for m in matches if negated(m) for w in query_words(m.group("query"))]
    # In a topic condition (a 'near' search or such an exclusion) only the
    # first text search, the user's own phrase, has a weaker form of its
    # words. The alternatives that follow are what the phrase was widened
    # with; loosening them too ("interpersonal & conflict" anywhere) brought
    # back what the phrase was meant to exclude.
    topic = bool(excluded_words) or any(_DISTANCE_RE.search(m.group("query")) for m in matches)
    first = True
    for match in matches:
        if negated(match):
            continue
        query = match.group("query")
        guard = match.group("guard")
        relaxed = relax_tsquery(query, guard) if (first or not topic) else None
        first = False
        if relaxed is None:
            if not excluded_words:
                continue
            relaxed = query
            words.append(f"{_EXCLUDED} {' '.join(excluded_words)}")
        elif relax_near(query) is not None:
            apart = [w for part in _top_level_or(relaxed) if "&" in part for w in query_words(part)]
            words.append(f"{_APART} {' '.join(apart)}")
        else:
            dropped = set(query_words(query)) - set(query_words(relaxed))
            # No word dropped: only the scope was widened, to the whole README.
            words.extend(sorted(dropped) or [f"{_README_ONLY} {' '.join(query_words(relaxed))}"])
        atom = (f"{match.group('alias')}.{match.group('column')} @@ "
                f"TO_TSQUERY('english', '{relaxed}')")
        # The weaker form still needs the word forms that were asked for:
        # "ageing" relaxed to its stem would be every README with an age.
        kept_guard = guard if relaxed == query else guard_for(relaxed, guard)
        if kept_guard:
            atom = (f"({atom} AND {match.group('alias')}.{match.group('guard_column')} @@ "
                    f"TO_TSQUERY('simple', '{kept_guard}'))")
        relaxed_atoms.append(atom)
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
    # (name, SELECT) sets the condition reads more than once: computed once
    # as a WITH query (a sidecar lookup scans file metadata; _sidecar).
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
    from services.catalogue import schema_capabilities
    from services.fulltext.text_search import EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN, PAPER_FTS_COLUMN
    spans = _negated_spans(sql)
    atoms = []
    for match in _TSQUERY_RE.finditer(sql):
        if match.group("negated") or any(s < match.start() < e for s, e in spans):
            continue
        query = _LABEL_RE.sub("", match.group("query"))
        paper = f"{alias}.{PAPER_FTS_COLUMN} @@ TO_TSQUERY('english', '{query}')"
        own = f"{alias}.search_tsv @@ TO_TSQUERY('english', '{query}')"
        if match.group("guard"):
            # Exact word forms count in the paper as in the dataset's text.
            # Without the unstemmed paper vector a paper is no evidence for them.
            if not schema_capabilities.has_exact_paper_fts():
                continue
            guard = _LABEL_RE.sub("", match.group("guard"))
            paper = f"({paper} AND {alias}.{EXACT_PAPER_FTS_COLUMN} @@ TO_TSQUERY('simple', '{guard}'))"
            own = f"({own} AND {alias}.{EXACT_FTS_COLUMN} @@ TO_TSQUERY('simple', '{guard}'))"
        # The paper vector also holds the dataset's own text; "only in a
        # paper" means it matches there and not in the dataset's text.
        atoms.append(f"({paper} AND NOT {own})")
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
    # files. Field strength and scanner make can be missing there and named
    # in the README instead (_sidecar); the rest stay hard filters.
    if "other_entities" in columns:
        return _sidecar(node, alias)
    # A resolved institution filter is evidence from the scanner headers. Its
    # weaker form would be the name anywhere in the README or a paper, and
    # those mentions are mostly suppliers, software and citations ('kindly
    # provided by … Harvard Medical School', surfer.nmr.mgh.harvard.edu).
    # Species likewise: the README of a human study says 'mouse click', and
    # its title/description fallback for unlabelled datasets is already
    # part of the condition.
    if columns & {"institutions", "species"}:
        return None
    # A condition on the outer join's rows (o.datatype …) is about files.
    if _columns_outside_subqueries(node) - {alias, ""}:
        return None
    counted = _patient_count(sql, alias)
    if counted is not None:
        return counted
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
                # The dataset's own text only. Of 63 datasets without ages whose
                # only child word was in a paper (live database, 2026-11-17),
                # about 9 studied children; the rest were adult studies citing
                # child work (Sherlock, BCIT) and two mouse datasets. Paper-only
                # matches fall to 'possible' instead.
                vector = "search_tsv"
                probable = (f"{alias}.{vector} @@ TO_TSQUERY('english', "
                            f"'{_tsquery_for_words(group_words)}')")
                reason_probable = "ages not recorded; the description names the age group"
        if kind == "sex":
            text = sex_text_condition(node, alias)
            if text is not None:
                probable, reason_probable = text
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


# ── Participant sex in the text ───────────────────────────────────────────────

# Words a README uses for each sex (stemmed by the english configuration:
# "females" is "femal", "women" stays). Not "man"/"woman" ("Isle of Man",
# "a woman's voice" in stimuli) nor "girls"/"boys" (stimuli and children's
# books as often as participants).
SEX_WORDS = {"female": ("female", "females", "women"), "male": ("male", "males", "men")}
_SEX_CODES = {"f": "female", "female": "female", "m": "male", "male": "male"}
# "Only female participants": "only" and "all" are english stopwords, so a
# tsquery cannot say it; a regex over the dataset's own text does.
_ONLY_SEX_RE = (r"(all[- ]?({w})|({w})[- ]only|only ({w})|exclusively ({w})"
                r"|all (participants|subjects|volunteers) (were|are) ({w}))")
_ONLY_SEX_WORDS = {"female": "female|females|women", "male": "male|males|men"}


def asked_sex(node) -> Optional[str]:
    """'female' or 'male' when a sex condition asks for one sex, else None."""
    values = set()
    for comparison in node.find_all(exp.EQ, exp.In):
        if not any(c.name.lower() == "sex" for c in comparison.find_all(exp.Column)):
            continue
        for literal in comparison.find_all(exp.Literal):
            if literal.is_string and literal.this.lower() in _SEX_CODES:
                values.add(_SEX_CODES[literal.this.lower()])
    return next(iter(values)) if len(values) == 1 else None


def _only_one_sex(node) -> bool:
    """The condition requires every participant row to have the sex
    (preprocess._participant_conditions: NOT EXISTS a row without it)."""
    for not_node in node.find_all(exp.Not):
        inner = not_node.this
        if isinstance(inner, exp.Exists) and any(
                c.name.lower() == "sex" for c in inner.find_all(exp.Column)):
            return True
    return False


def _own_text(alias: str) -> str:
    """The dataset's own name, description and README as one text."""
    return (f"(COALESCE({alias}.name, '') || ' ' || COALESCE({alias}.description_text, '') "
            f"|| ' ' || COALESCE({alias}.readme_text, ''))")


def _reads_participant_sex(node) -> bool:
    return any(t.name.lower() == "bids_participants" for t in node.find_all(exp.Table)) and any(
        c.name.lower() == "sex" for c in node.find_all(exp.Column))


def _keep_only_sex_pairs(nodes: list) -> list:
    """"Only female participants" as one condition again.

    preprocess writes it as EXISTS(a female row) AND NOT EXISTS(a row that is
    not female); flattening the ANDs made two conditions, so the tier never
    saw "only" (the README text for any female counted), and the NOT EXISTS
    alone held for a dataset with no participant rows at all.
    """
    rest = []
    pairs: dict[str, exp.Expression] = {}
    for node in nodes:
        bare = node.this if isinstance(node, exp.Paren) else node
        if isinstance(bare, exp.Not) and isinstance(bare.this, exp.Exists) and _reads_participant_sex(bare):
            pairs.setdefault("not", node)
        else:
            rest.append(node)
    if "not" not in pairs:
        return nodes
    partner = next((n for n in rest if not isinstance(n, exp.Not) and _reads_participant_sex(n)
                    and asked_sex(n)), None)
    if partner is None:
        return nodes
    joined = exp.Paren(this=exp.And(this=partner.copy(), expression=pairs["not"].copy()))
    return [joined if n is partner else n for n in rest]


def sex_text_condition(node, alias: str) -> Optional[Tuple[str, str]]:
    """(SQL, reason) for a dataset whose text names the asked sex; None if
    the condition does not ask for one sex."""
    sex = asked_sex(node)
    if sex is None:
        return None
    if _only_one_sex(node):
        pattern = _ONLY_SEX_RE.format(w=_ONLY_SEX_WORDS[sex])
        return (f"{_own_text(alias)} ~* {_sql_literal(r'(^|[^[:alnum:]_])' + pattern + r'($|[^[:alnum:]_])')}",
                f"participant sex not recorded; the description says all participants are {sex}")
    words = " | ".join(SEX_WORDS[sex])
    return (f"{alias}.search_tsv @@ TO_TSQUERY('english', '{words}')",
            f"participant sex not recorded; the description names {sex} participants")


# ── Scanner metadata in the text ──────────────────────────────────────────────

# Sidecar keys a README states as often as the sidecars do ("acquired on a
# 3T Siemens Prisma"). Of 1,210 datasets with MRI files on 2026-11-18, 299
# record no field strength; 52 of them name one in their text. 26 without a
# recorded manufacturer name it. Echo and repetition times are not here: a
# README states them in too many forms to read reliably.
FIELD_STRENGTH_KEY = "MagneticFieldStrength"
# The strengths scanners are sold at; a range asks for the ones inside it.
NOMINAL_TESLA = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.7, 7.0, 9.4, 10.5, 11.7, 14.0)


def _tesla_number(value: float) -> str:
    """The pattern of one strength's number: 7 also as "7.0", 1.5 also as "1,5"."""
    whole, _, fraction = f"{value:g}".partition(".")
    if not fraction:
        return rf"{whole}(?:\.0+)?"
    return rf"{whole}[.,]{fraction}0*"


def field_strength_regex(strengths: Iterable[float], postgres: bool = False) -> Optional[str]:
    """Where a text names one of *strengths*: "7T", "7 T", "7-Tesla", "7.0T".

    Not inside another number ("17T", "0.7T"), a BIDS label ("ses-7T"), a
    word starting with T ("7 T1w images", "3 The") or a gradient spec
    ("200 T/m/s"). Spaces include the thin ones some READMEs use.

    Python's re by default (query_filters.field_strength_pattern marks it in
    excerpts); *postgres*: for Postgres's ~ (POSIX classes, no backslashes),
    which this tier runs over the dataset's text. Both forms are checked
    against each other on a real database
    (tests/test_scanner_sex_venues_postgres.py).

    The Postgres form has plain groups, never "(?:": the query runs through
    SQLAlchemy's text(), which reads ":7" or ":T" after "?" as a bind
    parameter and fails the search.
    """
    numbers = [_tesla_number(v) for v in sorted(set(strengths), reverse=True)]
    if not numbers:
        return None
    space = "[ \u00a0\u2009\u202f]?"
    word = "[[:alnum:]_" if postgres else r"[\w"
    body = "|".join(numbers)
    group = "(?:"
    if postgres:
        body = body.replace(r"\.", "[.]").replace("(?:", "(")
        group = "("
    return (rf"(?<!{word}.,/-]){group}{body}){space}-?{space}"
            rf"{group}T|[Tt]esla|TESLA)(?!{word}/])")


def asked_field_strengths(node) -> List[float]:
    """The nominal strengths a field-strength condition asks for: "> 6.5 AND
    < 7.5" → [7.0], "= 1.5" → [1.5], "≥ 3" → [3.0, 4.0, …]."""
    low, high = 0.0, float("inf")
    low_open = high_open = False
    exact = None
    found = False
    ops = {exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=", exp.EQ: "="}
    for comparison in node.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.Between):
        if FIELD_STRENGTH_KEY.lower() not in [k.lower() for k in _json_keys(comparison.this)]:
            continue
        try:
            if isinstance(comparison, exp.Between):
                low = max(low, float(comparison.args["low"].this))
                high = min(high, float(comparison.args["high"].this))
                found = True
                continue
            right = comparison.expression
            if not (isinstance(right, exp.Literal) and not right.is_string):
                continue
            value = float(right.this)
        except (KeyError, ValueError, AttributeError):
            continue
        found = True
        op = ops[type(comparison)]
        if op == "=":
            exact = value
        elif op.startswith(">") and value >= low:
            low, low_open = value, op == ">"
        elif op.startswith("<") and value <= high:
            high, high_open = value, op == "<"
    if not found:
        return []
    if exact is not None:
        return [exact]
    return [t for t in NOMINAL_TESLA
            if (t > low if low_open else t >= low) and (t < high if high_open else t <= high)]


_TEXT_SIDECAR_KEYS = {"magneticfieldstrength": FIELD_STRENGTH_KEY,
                      "manufacturer": "Manufacturer",
                      "manufacturermodelname": "ManufacturerModelName"}
# Vendor names as a README writes them (tsquery, english configuration).
_VENDOR_WORDS = {
    "siemens": "siemens",
    "philips": "philips",
    "ge": "(ge <-> healthcare) | (ge <-> medical) | (general <-> electric) | signa | (ge <-> discovery)",
    "general electric": "(ge <-> healthcare) | (ge <-> medical) | (general <-> electric) | signa",
    "bruker": "bruker",
    "canon": "(canon <-> medical) | toshiba",
    "toshiba": "toshiba | (canon <-> medical)",
    "hitachi": "hitachi",
    "united imaging": "(united <-> imaging) | uih",
}
_SIDECAR_LABELS = {FIELD_STRENGTH_KEY: "field strength", "Manufacturer": "scanner manufacturer",
                   "ManufacturerModelName": "scanner model"}


def _json_keys(node) -> List[str]:
    keys = []
    for extract in node.find_all(exp.JSONExtractScalar, exp.JSONExtract):
        key = extract.expression
        name = key.this if isinstance(key, exp.Literal) else key.sql().strip("'$.")
        keys.append(name)
    return keys


def _key_values(node, key: str) -> List[str]:
    """String values compared with other_entities ->> *key* ('%GE%' → 'GE')."""
    values = []
    for comparison in node.find_all(exp.EQ, exp.In, exp.ILike, exp.Like):
        if key.lower() not in [k.lower() for k in _json_keys(comparison)]:
            continue
        for literal in comparison.find_all(exp.Literal):
            if literal.is_string and literal.this.lower() != key.lower():
                value = literal.this.strip("%").strip()
                if value:
                    values.append(value)
    return list(dict.fromkeys(values))


def sidecar_words(key: str, values: Sequence[str]) -> List[str]:
    """Words of a manufacturer or model a README would name ('GE' → 'ge healthcare' …)."""
    words: List[str] = []
    for value in values:
        query = _VENDOR_WORDS.get(value.lower()) if key == "Manufacturer" else None
        if query:
            words.extend(re.sub(r"[()]", "", part).replace(" <-> ", " ").strip()
                         for part in query.split("|"))
        else:
            words.append(re.sub(r"[_-]+", " ", value).strip())
    return [w for w in dict.fromkeys(words) if w]


def _value_tsquery(key: str, values: Sequence[str]) -> Optional[str]:
    parts = []
    for value in values:
        query = _VENDOR_WORDS.get(value.lower()) if key == "Manufacturer" else None
        if query is None:
            tokens = re.findall(r"[a-z0-9]+", value.lower())
            if not tokens or (len(tokens) == 1 and len(tokens[0]) < 3):
                return None   # "GE"-like fragments of unknown makes are no evidence
            query = " <-> ".join(tokens)
        parts.append(f"({query})")
    return " | ".join(parts) or None


def _exists_branches(node) -> Optional[list]:
    """The EXISTS subqueries an OR of them is made of; None for another shape."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Or):
        left, right = _exists_branches(node.left), _exists_branches(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, exp.Exists) and isinstance(node.this, exp.Select):
        return [node.this]
    return None


def _and_parts(node) -> list:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _and_parts(node.left) + _and_parts(node.right)
    return [node]


def _file_rows(select, alias: str) -> Optional[Tuple[str, List[str], List[str], set]]:
    """(inner alias, sidecar parts, file parts, file columns) of
    EXISTS(SELECT 1 FROM bids_objects x WHERE x.dataset_id = alias.id AND …);
    None for another shape (a join, a second correlation such as
    o_high.subject = o_low.subject)."""
    tables = list(select.find_all(exp.Table))
    if len(tables) != 1 or tables[0].name.lower() != "bids_objects":
        return None
    if select.args.get("joins") or select.args.get("group") or select.args.get("having"):
        return None
    inner = tables[0].alias_or_name
    where = select.args.get("where")
    if where is None:
        return None
    sidecar: List[str] = []
    files: List[str] = []
    file_columns: set[str] = set()
    correlated = False
    for part in _and_parts(where.this):
        columns = list(part.find_all(exp.Column))
        if (isinstance(part, exp.EQ) and {c.table for c in columns} == {inner, alias}
                and {c.name.lower() for c in columns} == {"dataset_id", "id"}):
            correlated = True
            continue
        if any(c.table not in (inner, "") for c in columns):
            return None
        if _json_keys(part):
            sidecar.append(part.sql(dialect="postgres"))
        else:
            files.append(part.sql(dialect="postgres"))
            file_columns.update(c.name.lower() for c in columns)
    if not correlated or not sidecar:
        return None
    return inner, sidecar, files, file_columns


def _cte_name(select_sql: str) -> str:
    """A stable name for a WITH query; the same lookup in two conditions is one."""
    return "_qf_sidecar_" + hashlib.sha256(select_sql.encode()).hexdigest()[:10]


def _file_set(inner: str, files: List[str], columns: set) -> Tuple[str, Optional[Tuple[str, str]]]:
    """The datasets with the file part's files: (IN subquery, WITH query or None).

    From bids_dataset_facets when the part reads only datatype, suffix and
    task (20,926 rows instead of the objects' ~1M; the same summary executed
    SQL uses, services/translate/sql_rewriter.py), else once from the objects.
    """
    where = " AND ".join(f"({p})" for p in files)
    try:
        from services.catalogue import schema_capabilities
        facets = schema_capabilities.has_dataset_facets()
    except ImportError:  # pragma: no cover
        facets = False
    if facets and columns <= {"datatype", "suffix", "task"}:
        return f"SELECT {inner}.dataset_id FROM bids_dataset_facets AS {inner} WHERE {where}", None
    select = f"SELECT DISTINCT {inner}.dataset_id FROM bids_objects AS {inner} WHERE {where}"
    name = _cte_name(select)
    return f"SELECT dataset_id FROM {name}", (name, select)


def _strength_tsquery(strengths: Sequence[float]) -> str:
    """Indexed words that every text field_strength_regex matches has: the
    english parser keeps "7T", "7Tesla", "7.0T" as one word and splits "7 T"
    and "7-Tesla" into the bare number. A prefilter only: the regex decides.
    Checked against the regex on Postgres (tests/test_scanner_sex_venues_postgres.py)."""
    words: List[str] = []
    for value in strengths:
        number = f"{value:g}"
        forms = [number, f"{value:.1f}"]
        if "." in number:
            forms.append(number.replace(".", ","))
        for form in dict.fromkeys(forms):
            words.extend((form, f"{form}t", f"{form}tesla"))
    return " | ".join(dict.fromkeys(words))


def _sidecar(node, alias: str) -> Optional[SoftCondition]:
    """A field strength, manufacturer or model condition with its text venue.

    Confirmed: the sidecars hold the value (as asked). Probably relevant: no
    file of the dataset records the key and its own text names the value, with
    the condition's file part still required ("7T fMRI": fMRI files). A
    recorded value that does not fit excludes the dataset, whatever the README
    says.
    """
    keys = {k.lower() for k in _json_keys(node)}
    if len(keys) != 1 or not keys <= set(_TEXT_SIDECAR_KEYS):
        return None
    key = _TEXT_SIDECAR_KEYS[next(iter(keys))]
    selects = _exists_branches(node)
    if not selects:
        return None
    rows = [r for s in selects if (r := _file_rows(s, alias)) is not None]
    if len(rows) < len(selects):
        return None
    if len({(tuple(r[2])) for r in rows}) > 1:
        return None   # branches on different files: no single file part

    # Each branch's matching datasets, computed once (the condition is read
    # in the candidate filter and again for its evidence).
    ctes = []
    confirmed = []
    for inner, sidecar, files, _ in rows:
        where = " AND ".join(f"({p})" for p in files + sidecar)
        select = f"SELECT DISTINCT {inner}.dataset_id FROM bids_objects AS {inner} WHERE {where}"
        name = _cte_name(select)
        ctes.append((name, select))
        confirmed.append(f"{alias}.id IN (SELECT dataset_id FROM {name})")
    inner, _, files, columns = rows[0]
    file_part = None
    if files:
        file_select, file_cte = _file_set(inner, files, columns)
        file_part = f"{alias}.id IN ({file_select})"
        if file_cte:
            ctes.append(file_cte)

    if key == FIELD_STRENGTH_KEY:
        strengths = asked_field_strengths(node)
        pattern = field_strength_regex(strengths, postgres=True)
        if not pattern:
            return None
        # The indexed words first: the regex then reads a few hundred texts.
        text = (f"{alias}.search_tsv @@ TO_TSQUERY('english', {_sql_literal(_strength_tsquery(strengths))}) "
                f"AND {_own_text(alias)} ~ {_sql_literal(pattern)}")
        named = " or ".join(f"{s:g} T" for s in strengths)
    else:
        values = _key_values(node, key)
        query = _value_tsquery(key, values) if values else None
        if not query:
            return None
        text = f"{alias}.search_tsv @@ TO_TSQUERY('english', {_sql_literal(query)})"
        named = " or ".join(values)
    label = _SIDECAR_LABELS[key]
    probable = f"({text}) AND ({file_part})" if file_part else text
    recorded = (f"EXISTS(SELECT 1 FROM bids_objects AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND NULLIF(TRIM(uk.other_entities ->> '{key}'), '') IS NOT NULL)")
    # Whether the key is recorded matters only where the text names the value:
    # checked for those few datasets, not for every candidate (6.6 ms per
    # dataset on the live objects table, 11 s for one question). The same
    # states while there is no 'possible' level; with it, every candidate.
    known = recorded if possible_enabled() else f"(({probable}) AND {recorded})"
    return SoftCondition(
        sql=" OR ".join(f"({c})" for c in confirmed) if len(confirmed) > 1 else confirmed[0],
        kind="sidecar", probable=probable, known=known,
        reason_probable=f"{label} not in the file metadata; the description names {named}",
        reason_possible=f"{label} not in the file metadata", ctes=tuple(ctes),
    )


def _patient_count(sql: str, alias: str) -> Optional[SoftCondition]:
    """"At least N participants with the diagnosis" (RAG/cohort_count.py).

    Confirmed when that many are coded or labelled with it. Uncertain when the
    dataset does not record who has the diagnosis and is large enough to hold
    them. Excluded when its participant rows show fewer.
    """
    try:
        import cohort_count
    except ImportError:  # the RAG package is not on the path
        return None
    parsed = cohort_count.parse_expanded(sql)
    if parsed is None:
        return None
    codes, minimum = parsed
    stored = None
    try:
        from services.catalogue import schema_capabilities
        if schema_capabilities.has_subject_counts():
            stored = f"COALESCE({alias}.{schema_capabilities.SUBJECT_COUNT_COLUMN}, 0)"
    except ImportError:
        pass
    return SoftCondition(
        sql=sql, kind="patient_count",
        probable=cohort_count.unknown_sql(codes, minimum, alias, subject_count=stored), known=None,
        reason_probable=cohort_count.REASON_UNKNOWN, reason_possible="",
    )


# ── Rewrite ───────────────────────────────────────────────────────────────────

def _flatten_and(node) -> list:
    from services.fulltext.text_search import is_guarded_pair
    if isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
        if is_guarded_pair(node):
            return [node]   # a text search and its word forms: one condition
        return _flatten_and(node.this)
    return [node]


def _dataset_alias(select) -> Optional[str]:
    source = select.args.get("from") or select.args.get("from_")
    table = getattr(source, "this", None)
    if isinstance(table, exp.Table) and table.name == "bids_datasets":
        return table.alias_or_name
    return None


def apply(sql: str, papers: Optional[bool] = None, hoist_terms: Optional[bool] = None) -> str:
    """The query with uncertain conditions softened and the two tier columns.

    *papers*: let paper text serve as evidence (default: TEXT_SEARCH_PAPERS
    and the column exist). Returns *sql* unchanged when it has nothing that
    can be uncertain, or when it cannot be parsed.
    """
    if papers is None:
        from services.fulltext import text_search
        papers = text_search.papers_enabled()
    if hoist_terms is None:
        hoist_terms = hoist_terms_enabled()
    if sqlglot is None or LEVEL_COLUMN in sql:
        return sql
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
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
    for node in _keep_only_sex_pairs(_flatten_and(where.this)):
        # classify has no expected failures (sqlglot and regex work on a tree
        # that parsed); it ran without one on the 1,972 conditions of the
        # benchmark runs' SQL (2026-11-19). A failure here is a bug: raised,
        # not turned silently into a hard filter as it used to be.
        condition = classify(node, alias, papers=papers)
        if condition is None:
            hard.append(node.sql(dialect="postgres"))
        else:
            soft.append(condition)
    if not soft:
        return sql

    # Keep the indexed candidate filter, then calculate each evidence channel
    # once for those candidates. The old wrapper copied the same full-text
    # expression into the candidate WHERE, the tier CASE, and the paper flag.
    inner = tree.copy()
    keep = " AND ".join([*(f"({h})" for h in hard), *(c.keep_sql() for c in soft)])
    inner.set("where", exp.Where(this=sqlglot.parse_one(keep, read="postgres")))
    inner_sql = inner.sql(dialect="postgres")

    reserved = sql.lower()
    def unique_cte_name(base: str) -> str:
        candidate = base
        suffix = 2
        while re.search(rf"\b{re.escape(candidate)}\b", reserved):
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate.lower()

    terms_name = unique_cte_name("_qf_tier_terms")
    terms_alias = unique_cte_name("_qf_q")

    evidence_columns = [f"{alias}.id"]
    state_cases = []
    paper_columns = []
    for i, condition in enumerate(soft):
        confirmed = f"confirmed_{i}"
        probable = f"probable_{i}"
        known = f"known_{i}"
        paper = f"paper_{i}"
        evidence_columns.extend((
            f"({condition.sql}) AS {confirmed}",
            f"({condition.probable}) AS {probable}" if condition.probable
            else f"FALSE AS {probable}",
            f"({condition.known}) AS {known}" if condition.known
            else f"TRUE AS {known}",
            f"({condition.paper}) AS {paper}" if condition.paper
            else f"FALSE AS {paper}",
        ))

        raw = f"ev.{confirmed}"
        weak = f"ev.{probable}"
        recorded = f"ev.{known}"
        possible = "2" if possible_enabled() else "9"
        if condition.known is None:
            state = f"CASE WHEN {raw} THEN 0 WHEN {weak} THEN 1 ELSE 9 END"
        elif condition.kind == "diagnosis":
            state = (f"CASE WHEN {raw} THEN 0 WHEN {weak} THEN 1 "
                     f"WHEN NOT {recorded} THEN {possible} ELSE 9 END"
                     if possible_enabled() else
                     f"CASE WHEN {raw} THEN 0 WHEN {weak} THEN 1 ELSE 9 END")
        else:
            unknown = f"CASE WHEN {weak} THEN 1 ELSE {possible} END"
            state = (f"CASE WHEN {raw} THEN 0 WHEN NOT {recorded} "
                     f"THEN {unknown} ELSE 9 END")
        state_cases.append(f"{state} AS u{i}")
        if condition.paper:
            paper_columns.append(f"ev.{paper} AS p{i}")

    raw_sql = (
        f"SELECT {', '.join(evidence_columns)}\n"
        f"FROM bids_datasets AS {alias}"
    )
    # Evidence is looked up per candidate row (LATERAL). Two CTEs joined on id
    # read the same, but the planner then sorts every file row of the
    # candidates instead of hashing them: 3 s became 5 s on the live database.
    raw_where = f"\nWHERE {alias}.id = tier_base.id"

    # Full-text predicates are used in the candidate filter and again to label
    # their evidence level. Hoist identical tsquery values into a one-row CTE;
    # both stages refer to q.qN instead of copying long synonym lists.
    tsquery_re = re.compile(
        r"TO_TSQUERY\(\s*'(?P<config>english|simple)'\s*,\s*'(?P<query>[^']*)'\s*\)",
        re.IGNORECASE,
    )
    query_values: dict[tuple[str, str], str] = {}
    for match in tsquery_re.finditer(inner_sql + "\n" + raw_sql):
        key = (match.group("config").lower(), match.group("query"))
        query_values.setdefault(key, f"q{len(query_values)}")

    terms_cte = None
    if query_values and hoist_terms:
        def replace_tsquery(match: re.Match) -> str:
            key = (match.group("config").lower(), match.group("query"))
            return f"{terms_alias}.{query_values[key]}"

        inner_sql = tsquery_re.sub(replace_tsquery, inner_sql)
        raw_sql = tsquery_re.sub(replace_tsquery, raw_sql)
        terms_select = ", ".join(
            f"TO_TSQUERY('{config}', '{query}') AS {column}"
            for (config, query), column in query_values.items()
        )
        terms_cte = f"{terms_name} AS (SELECT {terms_select})"

        # The query CTE is one row; joining it makes its named tsquery values
        # visible to the indexed candidate WHERE and the evidence expressions.
        inner_tree = sqlglot.parse_one(inner_sql, read="postgres")
        query_table = exp.Table(
            this=exp.to_identifier(terms_name),
            alias=exp.TableAlias(this=exp.to_identifier(terms_alias)),
        )
        joins = list(inner_tree.args.get("joins") or [])
        joins.append(exp.Join(this=query_table, kind="CROSS"))
        inner_tree.set("joins", joins)
        inner_sql = inner_tree.sql(dialect="postgres")
        raw_sql += f"\nCROSS JOIN {terms_name} AS {terms_alias}"
    raw_sql += raw_where

    state_sql = (
        f"SELECT ev.id, {', '.join(state_cases)}"
        + (f", {', '.join(paper_columns)}" if paper_columns else "")
        + f"\nFROM (\n{raw_sql}\n) AS ev"
    )

    names = [f"evidence.u{i}" for i in range(len(soft))]
    unsure = " + ".join(f"(CASE WHEN {name} > 0 THEN 1 ELSE 0 END)" for name in names)
    evidence = " OR ".join(f"{name} <= 1" for name in names)
    level = f"GREATEST({', '.join(names)})" if len(names) > 1 else names[0]
    reasons = []
    for i, c in enumerate(soft):
        branches = []
        if c.paper:
            branches.append(f"WHEN evidence.u{i} = 1 AND evidence.p{i} THEN {_sql_literal(PAPER_REASON)}")
        if c.reason_probable:
            branches.append(f"WHEN evidence.u{i} = 1 THEN {_sql_literal(c.reason_probable)}")
        if c.reason_possible:
            branches.append(f"WHEN evidence.u{i} = 2 THEN {_sql_literal(c.reason_possible)}")
        if branches:
            reasons.append(f"CASE {' '.join(branches)} END")
    reasons_sql = (f"NULLIF(CONCAT_WS('; ', {', '.join(reasons)}), '')" if reasons else "NULL")

    ctes = [f"{name} AS MATERIALIZED ({select})"
            for name, select in dict.fromkeys(cte for c in soft for cte in c.ctes)]
    if terms_cte:
        ctes.insert(0, terms_cte)
    with_sql = f"WITH {', '.join(ctes)}\n" if ctes else ""
    return (
        f"{with_sql}"
        f"SELECT tier_base.*, {level} AS {LEVEL_COLUMN}, "
        f"{reasons_sql} AS {REASONS_COLUMN}\n"
        f"FROM (\n{inner_sql}\n) AS tier_base\n"
        f"CROSS JOIN LATERAL (\n{state_sql}\n) AS evidence\n"
        f"WHERE {level} < 9 AND ({unsure}) <= {MAX_UNSURE} AND ({evidence})"
    )


def confirmed_only(rows: Sequence[dict]) -> list:
    """The rows every condition confirms (what the original query returns)."""
    return [row for row in rows if not row.get(LEVEL_COLUMN)]
