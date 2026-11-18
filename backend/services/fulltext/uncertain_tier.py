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

Only conditions on fields that can be missing (ages, diagnosis, sex) or on
text can be uncertain. Modality, file types, size and subject counts come
from the files themselves and stay hard filters, as does anything negated.

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

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_RAG_DIR = Path(__file__).resolve().parents[3] / "RAG"
if _RAG_DIR.exists() and str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from text_terms import MODIFIER_WORDS, NEVER_ALONE_WORDS  # noqa: E402

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
    try:
        from services.fulltext import text_search
        counts = text_search.guarded_counts([query], [guard_for(query, guard)])
    except Exception:  # pragma: no cover — no database available
        counts = None
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
    counts = None
    try:
        from services.fulltext import text_search
        counts = text_search.guarded_counts(parts, [guard_for(part, guard) for part in parts])
    except Exception:  # pragma: no cover — no database available
        counts = None
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
    # files, like the modality, and reading them is slow: evaluating such a
    # condition a second time for the tier took two minutes on one question.
    if "other_entities" in columns:
        return None
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
    query_values = {}
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

    with_sql = f"WITH {terms_cte}\n" if terms_cte else ""
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
