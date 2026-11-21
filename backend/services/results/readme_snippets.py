"""
services/results/readme_snippets.py
-----------------------------------
The README excerpt on each result card: the passages that match the search's
highlight words (services/results/query_filters.highlight_terms), or the README's
start. Matched words are wrapped in SNIPPET_START/SNIPPET_STOP, which READMEs
do not use, so the page can bold them without HTML.
"""

from __future__ import annotations

import logging
import re
from typing import Collection, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import DatasetSchema
from services.results import query_filters

log = logging.getLogger(__name__)

SNIPPET_START, SNIPPET_STOP = "⟦", "⟧"
_SNIPPET_README_CHARS = 20000   # bounds ts_headline's work on huge READMEs
_SNIPPET_PLAIN_CHARS = 320      # excerpt length without search words
_HEADLINE_OPTIONS = (
    f"StartSel={SNIPPET_START}, StopSel={SNIPPET_STOP}, MaxFragments=2, "
    "MaxWords=28, MinWords=12, ShortWord=2, FragmentDelimiter=\" … \""
)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_MD_SYNTAX_RE = re.compile(r"[#*`>|]+|(?<!\w)_{1,2}|_{1,2}(?!\w)|^-{3,}|={3,}", re.M)


def clean_snippet(text_: str) -> str:
    """README Markdown reduced to plain text: no images, link targets, URLs,
    heading marks, emphasis or table pipes; whitespace collapsed."""
    text_ = _MD_IMAGE_RE.sub(" ", text_)
    text_ = _MD_LINK_RE.sub(r"\1", text_)
    text_ = _URL_RE.sub(" ", text_)
    text_ = _MD_SYNTAX_RE.sub(" ", text_)
    return re.sub(r"\s+", " ", text_).strip()


def _cut_at_word(text_: str, limit: int) -> str:
    if len(text_) <= limit:
        return text_
    cut = text_[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + " …"


_MARK_RE = re.compile(f"({SNIPPET_START}[^{SNIPPET_STOP}]*{SNIPPET_STOP})")
_SNIPPET_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# The Postgres stems of a text, space-separated.
_LEXEMES_SQL = "array_to_string(tsvector_to_array(to_tsvector('english', {})), ' ')"


def keep_justified_marks(headline: str, mark_lexemes: Sequence[str],
                         terms: Sequence[tuple], term_lexemes: dict,
                         exact: Collection[str] = ()) -> str:
    """*headline* with only the marks that *terms* justify.

    ts_headline marks every query word in the passage, also a word of a
    pair whose partner is elsewhere ("structure" for "structural MRI").
    A pair's words keep their marks only next to each other, with at most
    one word between ("Parkinson's disease"). *mark_lexemes* holds the stems
    of each mark in order, *term_lexemes* those of each term word.

    *exact*: term words that only count as written. The search was limited
    to these word forms ("ageing"), so a mark on another word with the same
    stem ("age") goes.
    """
    parts = _MARK_RE.split(headline)
    marks = [i for i, part in enumerate(parts) if part.startswith(SNIPPET_START)]
    if len(marks) != len(mark_lexemes):
        return headline
    stems = [set(lexemes.split()) for lexemes in mark_lexemes]

    def matches(index: int, word: str) -> bool:
        if word.lower() in exact:
            written = re.split(r"['’]", parts[marks[index]][1:-1].lower())[0]
            return written == word.lower()
        if not stems[index] & term_lexemes.get(word.lower(), set()):
            return False
        # A two-letter acronym only in capitals: "MS", not "500 ms".
        return not (len(word) == 2 and word.isupper()) or \
            parts[marks[index]][1:-1].startswith(word)

    keep = set()
    for k in range(len(marks)):
        for term in terms:
            if len(term) == 1:
                if matches(k, term[0]):
                    keep.add(k)
            elif k + 1 < len(marks) and matches(k, term[0]) and matches(k + 1, term[1]):
                gap = "".join(parts[marks[k] + 1:marks[k + 1]])
                if len(_SNIPPET_WORD_RE.findall(gap)) <= 1:
                    keep.update((k, k + 1))
    for k, i in enumerate(marks):
        if k not in keep:
            parts[i] = parts[i][1:-1]
    return "".join(parts)


def mark_pattern(headline: str, pattern: Optional[re.Pattern]) -> str:
    """*headline* with every match of *pattern* outside a mark marked too."""
    if pattern is None:
        return headline
    parts = _MARK_RE.split(headline)
    for i, part in enumerate(parts):
        if not part.startswith(SNIPPET_START):
            parts[i] = pattern.sub(lambda m: f"{SNIPPET_START}{m.group(0)}{SNIPPET_STOP}", part)
    return "".join(parts)


async def attach_snippets(
    session: AsyncSession, datasets: list[DatasetSchema], terms: Optional[Sequence[tuple]],
    exact: Collection[str] = (), strengths: Sequence[float] = (),
) -> None:
    """Give each dataset a README excerpt: the passages matching the highlight
    *terms* (services/results/query_filters.highlight_terms), or the start.

    *exact*: the terms that only count as written
    (query_filters.highlight_exact_words). Where the README has one of them,
    the excerpt is the passage around it, found without stemming; a passage
    chosen by stem would show "age" for a search on "ageing".

    *strengths*: field strengths in tesla (query_filters.highlight_field_strengths).
    Postgres words only steer the passage towards them ("1.5T" is the word
    "1.5" there); the marks come from query_filters.field_strength_pattern.

    Read with its own query, like services/results/dataset_cards.py, so the generated SQL stays
    unchanged. A failure leaves
    the page without excerpts rather than failing it.
    """
    if not datasets:
        return
    ids = [d.id for d in datasets]
    strength_pattern = query_filters.field_strength_pattern(strengths)
    highlight = " | ".join(q for q in (query_filters.terms_tsquery(terms or ()),
                                       query_filters.field_strength_tsquery(strengths)) if q)
    words = list(dict.fromkeys(w.lower() for term in terms or () for w in term))
    exact = frozenset(w.lower() for w in exact if _SNIPPET_WORD_RE.fullmatch(w))
    if highlight:
        # The passage around a word that only counts as written, if there is one.
        exact_headline = (
            ", ts_headline('simple', left(readme_text, :chars), "
            "to_tsquery('simple', :exact_q), :opts) AS hl_exact" if exact else "")
        sql = (
            "SELECT h.id, h.hl, "
            "ARRAY(SELECT " + _LEXEMES_SQL.format("m.t[1]") + " FROM regexp_matches(h.hl, :mark, 'g') "
            "WITH ORDINALITY AS m(t, n) ORDER BY m.n) AS mark_lexemes, "
            "ARRAY(SELECT " + _LEXEMES_SQL.format("w.t") + " FROM unnest(CAST(:words AS text[])) "
            "WITH ORDINALITY AS w(t, n) ORDER BY w.n) AS term_lexemes"
            + (", h.hl_exact " if exact else " ") +
            "FROM (SELECT id, ts_headline('english', left(readme_text, :chars), "
            "to_tsquery('english', :q), :opts) AS hl" + exact_headline + " FROM bids_datasets "
            "WHERE id = ANY(:ids) AND readme_text IS NOT NULL AND readme_text <> '') AS h"
        )
        params = {"ids": ids, "q": highlight, "opts": _HEADLINE_OPTIONS,
                  "chars": _SNIPPET_README_CHARS, "words": words,
                  "mark": f"{SNIPPET_START}([^{SNIPPET_STOP}]*){SNIPPET_STOP}"}
        if exact:
            params["exact_q"] = " | ".join(sorted(exact))
    else:
        sql = (
            "SELECT id, left(readme_text, :chars) FROM bids_datasets "
            "WHERE id = ANY(:ids) AND readme_text IS NOT NULL AND readme_text <> ''"
        )
        params = {"ids": ids, "chars": _SNIPPET_PLAIN_CHARS * 4}
    try:
        async with session.begin_nested():
            rows = list(await session.execute(text(sql), params))
    except SQLAlchemyError:
        log.warning("README excerpts unavailable for this page", exc_info=True)
        return
    found = {}
    for row in rows:
        raw = row[1]
        if highlight and raw:
            by_form = row[4] if exact and len(row) > 4 else None
            if by_form and SNIPPET_START in by_form:
                raw = by_form   # every mark there is a word form asked for
            else:
                term_lexemes = {w: set(lexemes.split())
                                for w, lexemes in zip(words, row[3])}
                raw = keep_justified_marks(raw, row[2], terms or (), term_lexemes, exact)
            raw = mark_pattern(raw, strength_pattern)
        found[row[0]] = raw
    for dataset in datasets:
        raw = found.get(dataset.id)
        if not raw:
            continue
        snippet = clean_snippet(raw)
        if not highlight or SNIPPET_START not in snippet:
            snippet = _cut_at_word(snippet.replace(SNIPPET_START, "").replace(SNIPPET_STOP, ""),
                                   _SNIPPET_PLAIN_CHARS)
        else:
            # A matched passage usually starts and ends mid-sentence.
            if snippet.lstrip(SNIPPET_START)[:1].islower():
                snippet = "… " + snippet
            if snippet[-1:] not in ".!?…":
                snippet += " …"
        dataset.snippet = snippet or None
