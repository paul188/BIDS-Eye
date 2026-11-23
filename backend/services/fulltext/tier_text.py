"""
services/fulltext/tier_text.py
------------------------------
Weaker forms of a text search, the 'probable' state of a text condition:
some of its words (relax_tsquery, relax_near), the words apart, the README
naming the asked age group (age_group_words), and the reasons a result card
shows for them.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from text_terms import MODIFIER_WORDS, NEVER_ALONE_WORDS

from services.fulltext.tier_conditions import LABEL_RE, TSQUERY_RE
from services.schema.tables import participants as _participants


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
    from services.fulltext import word_frequencies
    # None when the catalogue cannot be asked (guarded_counts catches that).
    counts = word_frequencies.guarded_counts([query], [guard_for(query, guard)])
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
    for group in _top_level_and(LABEL_RE.sub("", guard)):
        asked = _WORD_RE.search(group)
        if asked and asked.group(0) in words:
            kept.append(group)
    return " & ".join(kept) or None


def _rarest_parts(parts: List[str], guard: Optional[str] = None) -> List[str]:
    """The rarer half of a term's parts (rounded up), in their original order.

    Rarity comes from how many datasets contain each part
    (word_frequencies.py); without it, the longest parts. A part
    with exact word forms (*guard*) is counted in those forms.
    """
    keep = (len(parts) + 1) // 2
    if keep >= len(parts):
        return parts
    from services.fulltext import word_frequencies
    # None when the catalogue cannot be asked (guarded_counts catches that).
    counts = word_frequencies.guarded_counts(parts, [guard_for(part, guard) for part in parts])
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
    unscoped = LABEL_RE.sub("", query)
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
    words = _WORD_RE.findall(re.sub(r"<(?:-|\d+)>", " ", LABEL_RE.sub("", query)))
    return list(dict.fromkeys(words))


def negated_spans(sql: str) -> List[Tuple[int, int]]:
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
    spans = negated_spans(sql)
    words: List[str] = []
    relaxed_atoms: List[str] = []
    matches = list(TSQUERY_RE.finditer(sql))

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


def tsquery_for_words(words: Sequence[str]) -> str:
    parts = []
    for word in words:
        pieces = word.split("-")
        parts.append("(" + " <-> ".join(pieces) + ")" if len(pieces) > 1 else word)
    return " | ".join(parts)


def known_sql(kind: str, alias: str) -> str:
    """True when the dataset records the field the condition reads."""
    if kind == "age":
        return (f"({alias}.age_min IS NOT NULL OR {alias}.age_max IS NOT NULL OR EXISTS("
                f"SELECT 1 FROM {_participants.name} AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND uk.age IS NOT NULL))")
    if kind == "diagnosis":
        return (f"EXISTS(SELECT 1 FROM {_participants.name} AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND uk.diagnosis IS NOT NULL)")
    if kind == "sex":
        return (f"EXISTS(SELECT 1 FROM {_participants.name} AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND uk.sex IS NOT NULL)")
    raise ValueError(kind)
