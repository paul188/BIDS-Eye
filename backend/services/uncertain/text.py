"""
services/uncertain/text.py
------------------------------
Weaker forms of a text search, the 'probable' state of a text condition:
some of its words (relax_tsquery, relax_near), the words apart. The result
is evidence.probable.WeakerSearch, which writes the SQL and the card text.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from text_terms import MODIFIER_WORDS, NEVER_ALONE_WORDS

from services.evidence.probable import TextSearch, WeakenedSearch, WeakerSearch, Weakening
from services.uncertain.conditions import LABEL_RE, TSQUERY_RE


_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


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


def relax_text_condition(sql: str) -> Optional[WeakerSearch]:
    """The weaker text evidence for a condition; None when no positive text
    search of it would change.

    Only the relaxed text searches of the condition count, any one of them:
    the rest of the condition (often an expensive check over the files) has
    already been evaluated for the confirmed state, and when it holds the
    dataset is confirmed anyway.
    """
    spans = negated_spans(sql)
    searches: List[WeakenedSearch] = []
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
        words: Sequence[str]
        if relaxed is None:
            if not excluded_words:
                continue
            relaxed, weakening, words = query, Weakening.EXCLUDED_MEANING, excluded_words
        elif relax_near(query) is not None:
            weakening = Weakening.WORDS_APART
            words = [w for part in _top_level_or(relaxed) if "&" in part for w in query_words(part)]
        else:
            dropped = set(query_words(query)) - set(query_words(relaxed))
            if dropped:
                weakening, words = Weakening.WORDS_MISSING, sorted(dropped)
            else:
                # No word dropped: only the scope was widened, to the whole README.
                weakening, words = Weakening.README_ONLY, query_words(relaxed)
        # The word forms that were asked for stay with the words that are left.
        kept_guard = guard if relaxed == query else guard_for(relaxed, guard)
        searches.append(WeakenedSearch(
            TextSearch(relaxed, kept_guard), match.group("column"),
            match.group("guard_column") if kept_guard else None, weakening, tuple(words)))
    return WeakerSearch(tuple(searches)) if searches else None
