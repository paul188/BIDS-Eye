"""
services/fulltext/text_search.py
--------------------------------
Full-text search over dataset text.

bids_datasets.search_tsv is a stored tsvector over
    weight A  name
    weight B  dx_status, study_domain, study_design, description_text,
              llm_conditions (studied conditions read from the README)
    weight C  llm_profile (LLM-written profile, services/indexing/readme_profile.py)
    weight D  readme_text
(see crawlers/migrate_add_search_tsv.sql).

The SQL model writes text_match('...') placeholders (grammar in
RAG/text_terms.py). This module turns them into

    d.search_tsv @@ to_tsquery('english', '<query>')

and builds the query that ranks the result. One column serves three scopes,
because a tsquery word can be restricted to weights: `word:A` matches the name
only, `word:AB` the short fields, plain `word` everything.

Queries are inlined as literals, not bound as parameters. They consist of
[a-z0-9] words and tsquery operators only, and a literal survives the sqlglot
round trips in services/translate/sql_rewriter.py unchanged, which a `:name` parameter
does not in every sqlglot version.

On a database without search_tsv the placeholders expand to ILIKE on the short
text columns, and there is no text rank.

Paper text (TEXT_SEARCH_PAPERS=true). search_tsv_papers is search_tsv plus,
at weight D, the title, abstract and keywords of the papers the dataset names
as its own (crawlers/paper_enrichment_plan.md). With the switch on and the
column present, a positive text_match() filters on it, so a dataset whose own
text lacks a word but whose paper has it is found. Everything else stays on
search_tsv:
  - the rank, so a dataset found only through its paper sorts after every
    dataset that matches on its own text (paper text adds results, it never
    promotes them);
  - NOT text_match(), so a condition a paper mentions only as background or
    as an exclusion never removes a dataset;
  - the word counts behind 'loose' terms.
Terms scoped to the name or the short fields (labels A, AB) cannot match the
paper text at all, because it carries weight D.

Exact word forms (TEXT_SEARCH_EXACT=true). The 'english' configuration
matches a word by its stem, and some stems merge words that mean different
things: "ageing" also finds "age", "weight" finds "T1-weighted". For a word
whose forms were decided (RAG/word_forms.py), a term also requires one of
those forms on search_tsv_exact, the same vector without stemming
(crawlers/migrate_add_exact_search.sql):

    (d.search_tsv @@ to_tsquery('english', 'ageing & mri')
     AND d.search_tsv_exact @@ to_tsquery('simple', '(ageing | aging)'))

Always this shape: the stemmed match, then its guard, in one pair of
parentheses. The stemmed match keeps the other words and the word distances;
the guard has one group per exact word, the asked word first, with the same
weight labels. Whatever reads the SQL afterwards treats the pair as one
condition (is_guarded_pair): the filter chips, the uncertain tier. A
placeholder with several alternatives becomes one pair per alternative that
has an exact word, ORed with the rest. The word counts count such words in
their forms, and they are ranked on search_tsv_exact: the rank query then
has two parts, "stemmed#exact" (join_rank, rank_expression), and travels as
one string like before. With paper text on, the guard reads
search_tsv_exact_papers.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from text_terms import (
    MODE_ALL,
    MODE_ALSO,
    MODE_LOOSE,
    MODE_NEAR,
    MODE_SHORT,
    MODES,
    MODIFIER_WORDS,
    NEAR_DISTANCE,
    NEVER_ALONE_WORDS,
    Term,
    Word,
    format_text_match,
    parse_alternatives,
    split_words,
)

from word_forms import forms_for  # noqa: E402

from services.catalogue import off_loop_query, schema_capabilities  # noqa: E402

log = logging.getLogger(__name__)

TS_CONFIG = "english"
FTS_COLUMN = "search_tsv"
PAPER_FTS_COLUMN = schema_capabilities.PAPER_FTS_COLUMN
PAPERS_ENV = "TEXT_SEARCH_PAPERS"
# The vectors without stemming, for words that must match in the form asked for.
EXACT_TS_CONFIG = "simple"
EXACT_FTS_COLUMN = schema_capabilities.EXACT_FTS_COLUMN
EXACT_PAPER_FTS_COLUMN = schema_capabilities.EXACT_PAPER_FTS_COLUMN
EXACT_ENV = "TEXT_SEARCH_EXACT"
# word → the forms it must match in, as decided for one question
# (RAG/word_forms.py). The reviewed list there applies without being passed.
ExactForms = Mapping[str, Sequence[str]]
# ts_rank_cd weights in the order {D, C, B, A}: README, LLM profile of the
# README, short fields (incl. conditions the README says were studied), name.
RANK_WEIGHTS = "{0.05, 0.4, 0.6, 1.0}"
# 32 scales the rank into 0..1 (rank / (rank + 1)), the range RAG scores use.
RANK_NORMALIZATION = 32

_SCOPE_LABEL = {MODE_ALL: "", MODE_LOOSE: "", MODE_SHORT: ":AB", MODE_NEAR: ""}
_NAME_LABEL = ":A"

_TEXT_MATCH_RE = re.compile(
    r"(?P<negated>\bNOT\s+)?\btext_match\s*\(\s*'(?P<terms>[^']*)'\s*(?:,\s*'(?P<mode>[^']*)'\s*)?\)",
    re.IGNORECASE,
)
_DATASET_ALIAS_RE = re.compile(r"\bbids_datasets\s+(?:AS\s+)?([A-Za-z_]\w*)", re.IGNORECASE)
_SQL_KEYWORDS = {"where", "left", "right", "inner", "join", "group", "order", "on", "limit", "having", "cross", "full"}
# Long free-text columns: a substring match there is what full-text search replaces.
_LONG_TEXT_ILIKE_RE = re.compile(
    r"\b[A-Za-z_]\w*\.(?P<column>description_text|readme_text)\s+(?P<negated>NOT\s+)?ILIKE\s+'%(?P<literal>[^'%]+)%'",
    re.IGNORECASE,
)
_NAME_ILIKE_RE = re.compile(
    r"(?P<negated>\bNOT\s+)?\b[A-Za-z_]\w*\.name\s+(?P<inner_negated>NOT\s+)?ILIKE\s+'%(?P<literal>[^'%]+)%'",
    re.IGNORECASE,
)
_URL_FRAGMENT_RE = re.compile(r"[./:@]")
_SAFE_TSQUERY_RE = re.compile(r"^[a-z0-9 &|()<>:AB-]+$")


# ── Word frequencies (for 'loose' terms) ──────────────────────────────────────

# Given tsquery strings, returns how many datasets match each one, or None
# when the answer is not available. A second argument, when given, holds a
# guard per query (or None): a tsquery the unstemmed vector must match too.
FrequencyLookup = Callable[..., Optional[List[int]]]

# Until something else is installed, the lookup opens its own short-lived
# connection (_own_connection_lookup), so every entry point gets frequencies:
# the API (main.py installs a pooled lookup at startup), one-off containers,
# the recall benchmark. set_frequency_lookup(None) switches lookups off.
_OWN_CONNECTION = object()
_frequency_lookup: Any = _OWN_CONNECTION
FREQUENCY_TIMEOUT_SECONDS = 3.0

_COUNT_SQL = (
    f"SELECT (SELECT count(*) FROM bids_datasets d "
    f"        WHERE d.{FTS_COLUMN} @@ to_tsquery('{TS_CONFIG}', q.tsq)) "
    "FROM unnest(CAST(:queries AS text[])) WITH ORDINALITY AS q(tsq, ord) "
    "ORDER BY q.ord"
)
_GUARDED_COUNT_SQL = (
    f"SELECT (SELECT count(*) FROM bids_datasets d "
    f"        WHERE d.{FTS_COLUMN} @@ to_tsquery('{TS_CONFIG}', q.tsq) "
    f"          AND (q.guard IS NULL OR "
    f"               d.{EXACT_FTS_COLUMN} @@ to_tsquery('{EXACT_TS_CONFIG}', q.guard))) "
    "FROM unnest(CAST(:queries AS text[]), CAST(:guards AS text[])) "
    "     WITH ORDINALITY AS q(tsq, guard, ord) "
    "ORDER BY q.ord"
)


def _count_statement(queries: List[str], guards: Optional[List[Optional[str]]]):
    """(SQL, parameters) for the counts; the guarded form only when needed."""
    if guards and any(guards):
        return _GUARDED_COUNT_SQL, {"queries": queries, "guards": list(guards)}
    return _COUNT_SQL, {"queries": queries}


def set_frequency_lookup(lookup: Optional[FrequencyLookup]) -> None:
    global _frequency_lookup
    _frequency_lookup = lookup


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _own_connection_lookup(queries: List[str],
                           guards: Optional[List[Optional[str]]] = None) -> Optional[List[int]]:
    """Count with a connection of its own; for callers that installed nothing.

    Runs only off the event loop (the translation pipeline runs in a worker
    thread) and only when the database has search_tsv.
    """
    if _on_event_loop() or not schema_capabilities.has_fts():
        return None
    from sqlalchemy import text

    async def _count(connection: Any) -> List[int]:
        sql, params = _count_statement(queries, guards)
        rows = await connection.execute(text(sql), params)
        return [int(row[0]) for row in rows]

    return off_loop_query.run_with_own_connection(_count, FREQUENCY_TIMEOUT_SECONDS)


def make_frequency_lookup(engine: Any, loop: asyncio.AbstractEventLoop,
                          timeout: float = FREQUENCY_TIMEOUT_SECONDS) -> FrequencyLookup:
    """Pooled frequency lookup for code running in a worker thread.

    The translation pipeline runs in asyncio.to_thread, while the database
    engine belongs to the event loop; the count is scheduled there.
    """
    from sqlalchemy import text

    async def _count(queries: List[str], guards: Optional[List[Optional[str]]]) -> List[int]:
        async with engine.connect() as connection:
            sql, params = _count_statement(queries, guards)
            rows = await connection.execute(text(sql), params)
            return [int(row[0]) for row in rows]

    def lookup(queries: List[str],
               guards: Optional[List[Optional[str]]] = None) -> Optional[List[int]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return None  # called on an event loop: blocking here would deadlock
        return off_loop_query.run_on_loop(_count(queries, guards), loop, timeout)

    return lookup


# ── tsquery building ──────────────────────────────────────────────────────────

def exact_enabled() -> bool:
    """True when words with decided forms must match in those forms.

    Read on every call, so a one-off container can set the variable.
    """
    flag = os.getenv(EXACT_ENV, "false").strip().lower() in ("1", "true", "yes", "on")
    return flag and schema_capabilities.has_exact_fts()


def _word_forms(word: Word, exact_forms: Optional[ExactForms]) -> Optional[Tuple[str, ...]]:
    """The forms *word* must match in, or None to match it by stem.

    Only plain words: the parts of a compound ("t1-weighted") stay together
    by adjacency, which already keeps them from matching anything else.
    """
    if len(word) != 1 or not exact_enabled():
        return None
    return forms_for(word[0], exact_forms)


def _form_group(forms: Sequence[str], label: str = "") -> str:
    labelled = [f"{form}{label}" for form in forms]
    return labelled[0] if len(labelled) == 1 else "(" + " | ".join(labelled) + ")"


def exact_guard(term: Term, label: str = "",
                exact_forms: Optional[ExactForms] = None) -> Optional[str]:
    """The query the unstemmed vector must match for *term*; None without exact words.

    One group per exact word, the asked word first: "(ageing | aging) & weight".
    """
    groups = [_form_group(forms, label)
              for forms in (_word_forms(word, exact_forms) for word in term) if forms]
    return " & ".join(groups) or None


def _word_query(word: Word, label: str = "") -> str:
    """One word; the parts of a hyphenated compound must be adjacent."""
    if len(word) == 1:
        return f"{word[0]}{label}"
    return "(" + " <-> ".join(f"{part}{label}" for part in word) + ")"


def _all_words_query(term: Term, label: str = "") -> str:
    return " & ".join(_word_query(word, label) for word in term)


# A 'near' term of more words than this needs only all of its words: the
# number of word orders and gaps to spell out grows too fast.
NEAR_MAX_WORDS = 4


def _near_query(term: Term, label: str = "") -> str:
    """The words of *term* close together.

    tsquery has only exact distances (`a <2> b`: b two positions after a), so
    every allowed gap is spelled out. Two words may come in either order
    ("social conflict", "conflict between social groups"); three or four keep
    their order with gaps up to 2. One word is just that word.
    """
    words = [_word_query(word, label) for word in term]
    if len(words) == 1:
        return words[0]
    if len(words) > NEAR_MAX_WORDS:
        return _all_words_query(term, label)
    variants: List[str] = []
    if len(words) == 2:
        first, second = words
        for left, right in ((first, second), (second, first)):
            variants.extend(f"{left} <{gap}> {right}" for gap in range(1, NEAR_DISTANCE + 1))
    else:
        for gaps in itertools.product(range(1, 3), repeat=len(words) - 1):
            phrase = words[0]
            for word, gap in zip(words[1:], gaps):
                phrase = f"{phrase} <{gap}> {word}"
            variants.append(phrase)
    return "(" + " | ".join(variants) + ")"


def _any(queries: Sequence[str]) -> str:
    queries = list(dict.fromkeys(q for q in queries if q))
    if len(queries) == 1:
        return queries[0]
    return " | ".join(f"({q})" if (" " in q and not _is_parenthesised(q)) else q for q in queries)


def _is_parenthesised(query: str) -> bool:
    if not (query.startswith("(") and query.endswith(")")):
        return False
    depth = 0
    for index, char in enumerate(query):
        depth += char == "("
        depth -= char == ")"
        if depth == 0 and index < len(query) - 1:
            return False
    return True


# A loose term keeps as many of its words as still leave this many datasets.
LOOSE_MIN_MATCHES = 3


def _lookup_counts(queries: List[str],
                   guards: Optional[Sequence[Optional[str]]] = None) -> Optional[List[int]]:
    """Datasets matching each query; with *guards*, also its unstemmed guard.

    The lookup gets the guards only when there is one, so a lookup that knows
    nothing of them (tests, older callers) keeps working. A lookup raises
    CatalogueLookupError when the database cannot answer; the search then
    goes on without the counts. Anything else it raises is a bug.
    """
    lookup = _own_connection_lookup if _frequency_lookup is _OWN_CONNECTION else _frequency_lookup
    if lookup is None:
        return None
    try:
        if guards and any(guards):
            counts = lookup(queries, list(guards))
        else:
            counts = lookup(queries)
    except off_loop_query.CatalogueLookupError as exc:
        log.warning("Word frequency lookup failed (%s) — using the longest word", exc)
        return None
    if not counts or len(counts) != len(queries):
        return None
    return counts


def guarded_counts(queries: List[str],
                   guards: Sequence[Optional[str]]) -> Optional[List[int]]:
    """_lookup_counts, passing the guards on only when there is one."""
    if any(guards):
        return _lookup_counts(queries, list(guards))
    return _lookup_counts(queries)


def _says_nothing(word: Word) -> bool:
    """A word that cannot stand for a term alone: a modifier or a general noun."""
    return len(word) == 1 and (word[0] in MODIFIER_WORDS or word[0] in NEVER_ALONE_WORDS)


def _loose_core(term: Term, exact_forms: Optional[ExactForms] = None) -> Term:
    """The words of an unresolved term that a dataset must contain.

    Words are taken rarest first, and as many as still leave
    LOOSE_MIN_MATCHES datasets are required: "eye tracking" keeps both words
    ("tracking" alone finds rat linear-track studies), "virtual reality
    navigation" drops "navigation" if the three together are too rare.

    A word that no dataset contains is usually the distinctive one ("sickle"
    in "sickle cell disease") or a typo. Searching the remaining words would
    answer a different question ("cell"), so then every word is required and
    the term honestly matches nothing.

    A general word ("syndrome", "task", "healthy") comes after every other
    word, however rare: "bipolar syndrome" keeps "bipolar", although this
    catalogue has "syndrome" in fewer datasets (11 against 43).

    With no frequencies available, the longest word, as the ILIKE search did.
    """
    if len(term) == 1:
        return term
    longest = [max(term, key=lambda word: (not _says_nothing(word), sum(len(part) for part in word)))]
    counts = guarded_counts([_word_query(word) for word in term],
                            [exact_guard([word], "", exact_forms) for word in term])
    if counts is None:
        return longest
    if any(count == 0 for count in counts):
        return term
    present = sorted((_says_nothing(term[index]), count, index) for index, count in enumerate(counts))
    by_rarity = [term[index] for _, _, index in present]
    if len(by_rarity) == 1:
        return by_rarity
    prefixes = [by_rarity[:k] for k in range(len(by_rarity), 1, -1)]
    combined = guarded_counts([_all_words_query(prefix) for prefix in prefixes],
                              [exact_guard(prefix, "", exact_forms) for prefix in prefixes])
    if combined is not None:
        for prefix, count in zip(prefixes, combined):
            if count >= LOOSE_MIN_MATCHES:
                return prefix
    return by_rarity[:1]


def _filter_term(term: Term, mode: str, exact_forms: Optional[ExactForms] = None) -> Term:
    """The words of *term* a dataset must contain: all, or a loose term's core."""
    return _loose_core(term, exact_forms) if mode == MODE_LOOSE else term


def build_filter_tsquery(terms: Sequence[Term], mode: str = MODE_ALL,
                         exact_forms: Optional[ExactForms] = None) -> Optional[str]:
    """tsquery a dataset must match; None when no term is searchable.

    *exact_forms* only tells a loose term's word counts which forms count.
    """
    label = _SCOPE_LABEL[mode]
    if mode == MODE_LOOSE:
        queries = [_all_words_query(_loose_core(term, exact_forms), label)
                   for term in terms if term]
    elif mode == MODE_NEAR:
        queries = [_near_query(term, label) for term in terms if term]
    else:
        queries = [_all_words_query(term, label) for term in terms if term]
    return _any(queries) or None


def count_matches(alternatives: Sequence[str], mode: str = MODE_NEAR,
                  exact_forms: Optional[ExactForms] = None) -> Optional[List[int]]:
    """How many datasets match each search term on its own; None when unknown.

    Used to check search terms a language model proposed before they are
    searched (LLM_preprocessor/preprocess/topics.py). A term with nothing
    searchable counts 0. A word with decided forms is counted in those forms,
    as the search will match it.
    """
    queries, guards = [], []
    for alternative in alternatives:
        terms = parse_alternatives(alternative)
        queries.append(build_filter_tsquery(terms, mode, exact_forms) if terms else None)
        # One guard per query: an alternative of several terms is counted by stem.
        guards.append(exact_guard(_filter_term(terms[0], mode, exact_forms),
                                  _SCOPE_LABEL[mode], exact_forms) if len(terms) == 1 else None)
    asked = [query for query in queries if query]
    if not asked:
        return [0] * len(queries)
    counts = guarded_counts(asked, [g for q, g in zip(queries, guards) if q])
    if counts is None:
        return None
    found = iter(counts)
    return [next(found) if query else 0 for query in queries]


def build_rank_tsquery(terms: Sequence[Term], mode: str = MODE_ALL,
                       exact_forms: Optional[ExactForms] = None) -> Optional[str]:
    """tsquery used for ranking: every field counts, whatever the filter scope.

    A diagnosis is filtered on the short fields only, but a README that
    mentions it still moves a matching dataset up. A word with decided forms
    is left out where it would rank alone: by stem, "ageing" counts every
    "age" (build_exact_rank_tsquery ranks it). Next to other words of its
    term it stays, since the term only ranks where all its words meet.
    """
    def exact(word: Word) -> bool:
        return bool(_word_forms(word, exact_forms))

    if mode == MODE_LOOSE:
        queries = [_word_query(word) for term in terms for word in term if not exact(word)]
    else:
        queries = [_all_words_query(term) for term in terms
                   if term and not all(exact(word) for word in term)]
    return _any(queries) or None


def build_exact_rank_tsquery(terms: Sequence[Term], mode: str = MODE_ALL,
                             exact_forms: Optional[ExactForms] = None) -> Optional[str]:
    """Ranking query for the unstemmed vector: the words with decided forms."""
    groups = [[_form_group(forms) for forms in (_word_forms(word, exact_forms) for word in term)
               if forms] for term in terms]
    if mode == MODE_LOOSE:
        queries = [group for term in groups for group in term]
    else:
        queries = [" & ".join(term) for term in groups if term]
    return _any(queries) or None


# ── Expansion ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ExpandedSearch:
    sql: str
    # What ranks the result: a tsquery for search_tsv, "stemmed#exact" when
    # words are ranked on search_tsv_exact, or "stemmed#exact#own" when other
    # words widen a search (join_rank).
    rank_tsquery: Optional[str]


def _dataset_alias(sql: str) -> str:
    for match in _DATASET_ALIAS_RE.finditer(sql):
        alias = match.group(1)
        if alias.lower() not in _SQL_KEYWORDS:
            return alias
    return "d"


def _ilike_fallback(terms: Sequence[Term], mode: str, alias: str) -> str:
    """Substring search on the short text columns, for databases without search_tsv."""
    columns = schema_capabilities.disease_text_columns()
    alternatives = []
    for term in terms:
        words = [max(term, key=lambda w: sum(len(p) for p in w))] if mode == MODE_LOOSE else term
        word_conditions = []
        for word in words:
            pattern = "%" + "%".join(word) + "%"
            word_conditions.append(
                "(" + " OR ".join(f"{alias}.{column} ILIKE '{pattern}'" for column in columns) + ")"
            )
        alternatives.append(" AND ".join(word_conditions))
    # Always one parenthesised unit, so a preceding NOT or a surrounding
    # AND / OR applies to the whole search.
    if len(alternatives) == 1:
        return alternatives[0] if len(alternatives[0].split(" AND ")) == 1 else f"({alternatives[0]})"
    return "(" + " OR ".join(f"({alt})" if " AND " in alt else alt for alt in alternatives) + ")"


def convert_long_text_ilike(sql: str) -> str:
    """Rewrite ILIKE on description_text / readme_text as text_match().

    The SQL model is told to use text_match(); this catches the cases where it
    falls back to the substring search it knows from its examples.
    """
    def replace(match: re.Match) -> str:
        literal = match.group("literal")
        # Full-text search indexes a URL as one token; a fragment of it
        # ('doi.org', 'ncbi.nlm') is only found by substring search.
        if _URL_FRAGMENT_RE.search(literal):
            return match.group(0)
        mode = MODE_ALL if match.group("column").lower() == "readme_text" else MODE_SHORT
        macro = format_text_match([literal], mode)
        if macro is None:
            return match.group(0)
        return f"NOT {macro}" if match.group("negated") else macro

    return _LONG_TEXT_ILIKE_RE.sub(replace, sql)


def papers_enabled() -> bool:
    """True when positive text matches should include paper text.

    Read on every call, so a one-off container can set the variable.
    """
    flag = os.getenv(PAPERS_ENV, "false").strip().lower() in ("1", "true", "yes", "on")
    return flag and schema_capabilities.has_paper_fts()


_VECTOR_MATCH_RE = re.compile(
    rf"(?P<negated>\bNOT\s+)?\b(?P<alias>[A-Za-z_]\w*)\."
    rf"(?P<column>{FTS_COLUMN}|{EXACT_FTS_COLUMN})\s+@@\s+to_tsquery\(",
    re.IGNORECASE)
_NOT_BEFORE_RE = re.compile(r"\bNOT\s*$", re.IGNORECASE)


def negated_spans(sql: str) -> List[Tuple[int, int]]:
    """Character ranges of the parenthesised groups that directly follow NOT.

    Quoted strings are skipped, so parentheses inside a literal do not count.
    """
    spans: List[Tuple[int, int]] = []
    stack: List[Tuple[int, bool]] = []
    position, length = 0, len(sql)
    while position < length:
        char = sql[position]
        if char == "'":
            end = position + 1
            while end < length:
                if sql[end] == "'" and (end + 1 >= length or sql[end + 1] != "'"):
                    break
                end += 2 if sql[end] == "'" else 1
            position = end + 1
            continue
        if char == "(":
            stack.append((position, bool(_NOT_BEFORE_RE.search(sql[max(0, position - 12):position]))))
        elif char == ")" and stack:
            start, negated = stack.pop()
            if negated:
                spans.append((start, position))
        position += 1
    return spans


def _is_negated(position: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in spans)


def with_paper_text(sql: str) -> str:
    """The same SQL with every positive full-text filter reading paper text too.

    Equivalent to expanding with papers on; used by the benchmark to compare
    both from one translation (benchmarks/recall/paired_papers_run.py).
    """
    spans = negated_spans(sql)
    exact_papers = schema_capabilities.has_exact_paper_fts()

    def replace(match: re.Match) -> str:
        if match.group("negated") or _is_negated(match.start(), spans):
            return match.group(0)
        column = match.group("column")
        if column == FTS_COLUMN:
            column = PAPER_FTS_COLUMN
        elif exact_papers:
            column = EXACT_PAPER_FTS_COLUMN   # the guard of a match that now reads papers
        return f"{match.group('alias')}.{column} @@ to_tsquery("

    return _VECTOR_MATCH_RE.sub(replace, sql)


def _exact_column(column: str) -> str:
    """The unstemmed twin of a full-text column."""
    if column == PAPER_FTS_COLUMN and schema_capabilities.has_exact_paper_fts():
        return EXACT_PAPER_FTS_COLUMN
    # Without the paper twin the guard reads the dataset's own text: a word
    # that must match exactly is then not found through a paper alone.
    return EXACT_FTS_COLUMN


_GUARD_RE = re.compile(
    rf"^\s*[A-Za-z_]\w*\.{EXACT_FTS_COLUMN}(?:_papers)?\s+@@\s+to_tsquery\(\s*'{EXACT_TS_CONFIG}'",
    re.IGNORECASE)


def is_guarded_pair(node: Any) -> bool:
    """True if the sqlglot *node* is "(stemmed match AND its guard)".

    Code that splits a WHERE clause into its AND conditions keeps such a pair
    whole: it is one text search, not two (module docstring).
    """
    if type(node).__name__ != "Paren" or type(node.this).__name__ != "And":
        return False
    # An And always has a right side, and .sql() only warns about what the
    # dialect cannot express (sqlglot's default unsupported_level).
    return _GUARD_RE.match(node.this.right.sql(dialect="postgres")) is not None


def expand_text_search(
    sql: str, fts: Optional[bool] = None, papers: Optional[bool] = None,
    exact_forms: Optional[ExactForms] = None,
) -> ExpandedSearch:
    """Replace every text_match() placeholder in *sql* and build the rank queries.

    *fts* and *papers* override the detected database capability and the
    TEXT_SEARCH_PAPERS switch (tests). *exact_forms*: the word forms decided
    for this question; with TEXT_SEARCH_EXACT on, those words and the ones on
    the reviewed list also need one of their forms (module docstring).
    """
    use_fts = schema_capabilities.has_fts() if fts is None else fts
    use_papers = use_fts and (papers_enabled() if papers is None else papers)
    if use_fts:
        sql = convert_long_text_ilike(sql)
    alias = _dataset_alias(sql)
    rank_parts: List[str] = []
    rank_exact_parts: List[str] = []
    # The searches that 'also' placeholders widen, and every search so far.
    own_parts: List[str] = []
    previous_filter: List[str] = []
    not_groups = negated_spans(sql) if use_papers else []

    def replace(match: re.Match) -> str:
        negated = match.group("negated") or ""
        inside_not = bool(negated) or _is_negated(match.start(), not_groups)
        mode = (match.group("mode") or MODE_ALL).lower()
        if mode not in MODES:
            mode = MODE_ALL
        # Other words for the search before it: searched like 'near'; a
        # dataset matching only them ranks below one matching that search.
        also = mode == MODE_ALSO
        if also:
            mode = MODE_NEAR
        terms = parse_alternatives(match.group("terms"))
        if not terms:
            # Nothing searchable ("???"): the term cannot match any dataset.
            return f"{negated}FALSE"
        if not use_fts:
            return negated + _ilike_fallback(terms, mode, alias)
        if also and not inside_not and previous_filter:
            own_parts.append(previous_filter[-1])
        elif not inside_not:
            own_filter = build_filter_tsquery(
                [_filter_term(term, mode, exact_forms) for term in terms],
                MODE_ALL if mode == MODE_LOOSE else mode)
            if own_filter:
                previous_filter.append(own_filter)
        if not negated:
            rank = build_rank_tsquery(terms, mode, exact_forms)
            if rank:
                rank_parts.append(rank)
            rank_exact = build_exact_rank_tsquery(terms, mode, exact_forms)
            if rank_exact:
                rank_exact_parts.append(rank_exact)
        column = PAPER_FTS_COLUMN if (use_papers and not inside_not) else FTS_COLUMN

        def atom(query: str) -> str:
            return f"{alias}.{column} @@ to_tsquery('{TS_CONFIG}', '{query}')"

        if not exact_enabled():
            return negated + atom(build_filter_tsquery(terms, mode))
        label = _SCOPE_LABEL[mode]
        # What each alternative requires (a loose term: its core, counted
        # once here), and its guard if it has exact words.
        required = [_filter_term(term, mode, exact_forms) for term in terms]
        guards = [exact_guard(term, label, exact_forms) for term in required]
        if not any(guards):
            return negated + atom(
                build_filter_tsquery(required, MODE_ALL if mode == MODE_LOOSE else mode))
        # One pair per alternative with exact words; the others share one
        # stemmed match, placed where the first of them stood.
        pieces: List[str] = []
        plain: List[Term] = []
        plain_at: Optional[int] = None
        for term, guard in zip(required, guards):
            if guard:
                pieces.append(
                    f"({atom(build_filter_tsquery([term], MODE_ALL if mode == MODE_LOOSE else mode))}"
                    f" AND {alias}.{_exact_column(column)} @@ "
                    f"to_tsquery('{EXACT_TS_CONFIG}', '{guard}'))")
            else:
                if plain_at is None:
                    plain_at = len(pieces)
                plain.append(term)
        if plain:
            pieces.insert(plain_at or 0, atom(
                build_filter_tsquery(plain, MODE_ALL if mode == MODE_LOOSE else mode)))
        if len(pieces) == 1:
            return negated + pieces[0]
        return negated + "(" + " OR ".join(pieces) + ")"

    expanded = _TEXT_MATCH_RE.sub(replace, sql)

    if use_fts:
        # A requested title word also ranks: an exact title hit comes first.
        for match in _NAME_ILIKE_RE.finditer(expanded):
            if match.group("negated") or match.group("inner_negated"):
                continue
            words = split_words(match.group("literal"))
            if words:
                rank_parts.append(_all_words_query(words, _NAME_LABEL))

    rank_tsquery = _any(rank_parts) if (use_fts and rank_parts) else None
    rank_exact = _any(rank_exact_parts) if (use_fts and rank_exact_parts) else None
    own = _any(list(dict.fromkeys(own_parts))) if (rank_tsquery or rank_exact) and own_parts else None
    return ExpandedSearch(sql=expanded, rank_tsquery=join_rank(rank_tsquery, rank_exact, own))


# Words of a question that say nothing about the datasets wanted.
_QUESTION_NOISE = frozenset({
    "dataset", "datasets", "data", "find", "show", "search", "looking", "look", "want",
    "need", "get", "list", "give", "me", "i", "we", "all", "any", "some", "which", "what",
    "that", "there", "have", "has", "had", "is", "are", "be", "was", "were", "do", "does",
    "can", "could", "would", "please", "available", "open", "public", "study", "studies",
    "recorded", "using", "use", "used", "where", "who", "about", "into", "their", "this",
})


def keyword_query(question: str) -> Optional[str]:
    """A tsquery that matches any content word of *question*, or None.

    Used when no language model is reachable (services/translate/text_to_sql.py): a
    plain keyword search ranked by full-text rank beats an arbitrary sample.
    """
    words = [word for word in split_words(question)
             if not (len(word) == 1 and (word[0] in _QUESTION_NOISE or len(word[0]) < 2))]
    queries = list(dict.fromkeys(_word_query(word) for word in words))
    query = " | ".join(queries[:12])
    return query if query and is_safe_tsquery(query) else None


def is_safe_tsquery(query: Optional[str]) -> bool:
    """True if *query* was built here and can be inlined into SQL."""
    return bool(query) and _SAFE_TSQUERY_RE.match(query) is not None


# Between the parts of a rank query: the tsquery for search_tsv, the one for
# search_tsv_exact, and the search that 'also' words widen. Not a tsquery
# character.
RANK_SEPARATOR = "#"
# Taken off the text rank of a dataset that matches only the other words
# ('also') of a search and not the search itself. Every ordering adds the
# text rank (0..1) to the score of the matched codes (a sum of at most a few
# dozen scores up to 1), so this puts such a dataset below every dataset
# that matches the user's own words, and keeps the order among them.
ALSO_ONLY_PENALTY = 100


def join_rank(stemmed: Optional[str], exact: Optional[str],
              own: Optional[str] = None) -> Optional[str]:
    """One rank string from its parts; the stemmed query alone as it is."""
    if own:
        return f"{stemmed or ''}{RANK_SEPARATOR}{exact or ''}{RANK_SEPARATOR}{own}"
    if not exact:
        return stemmed or None
    return f"{stemmed or ''}{RANK_SEPARATOR}{exact}"


def _rank_parts(rank: Optional[str]) -> List[Optional[str]]:
    parts = (rank or "").split(RANK_SEPARATOR, 2)
    parts += [""] * (3 - len(parts))
    return [part if is_safe_tsquery(part) else None for part in parts]


def split_rank(rank: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(stemmed, exact) of a rank string; a part is None when absent or unsafe."""
    stemmed, exact, _own = _rank_parts(rank)
    return stemmed, exact


def rank_own(rank: Optional[str]) -> Optional[str]:
    """The search that other words widen ('also'); None when there is none."""
    return _rank_parts(rank)[2]


def without_own(rank: Optional[str]) -> Optional[str]:
    """The rank string without the 'also' part: every match ranks alike."""
    return join_rank(*split_rank(rank))


def rank_expression(alias: str, rank: Optional[str]) -> Optional[str]:
    """SQL for the text rank of the dataset row *alias*; None without a rank.

    0..1; with both parts the raw ranks are added before scaling, as one
    query over one vector would have done. A dataset that matches only the
    other words of a search ('also') gets ALSO_ONLY_PENALTY taken off.
    """
    expression = _plain_rank_expression(alias, rank)
    own = rank_own(rank)
    if expression is None or not own:
        return expression
    return (f"({expression} - CASE WHEN {alias}.{FTS_COLUMN} @@ "
            f"to_tsquery('{TS_CONFIG}', '{own}') THEN 0 ELSE {ALSO_ONLY_PENALTY} END)")


def _plain_rank_expression(alias: str, rank: Optional[str]) -> Optional[str]:
    stemmed, exact = split_rank(rank)
    if exact and not schema_capabilities.has_exact_fts():
        exact = None
    if stemmed and not exact:
        return text_rank_expression(f"{alias}.{FTS_COLUMN}", stemmed)
    if exact and not stemmed:
        return text_rank_expression(f"{alias}.{EXACT_FTS_COLUMN}", exact, EXACT_TS_CONFIG)
    if not stemmed:
        return None
    raw = (f"COALESCE(ts_rank_cd('{RANK_WEIGHTS}', {alias}.{FTS_COLUMN}, "
           f"to_tsquery('{TS_CONFIG}', '{stemmed}'), 0), 0) + "
           f"COALESCE(ts_rank_cd('{RANK_WEIGHTS}', {alias}.{EXACT_FTS_COLUMN}, "
           f"to_tsquery('{EXACT_TS_CONFIG}', '{exact}'), 0), 0)")
    # Normalization 32 of text_rank_expression, applied to the sum.
    return f"(({raw}) / (({raw}) + 1))"


def text_rank_expression(vector_sql: str, rank_tsquery: str, config: str = TS_CONFIG) -> str:
    """SQL expression for the 0..1 text rank of one dataset.

    *config*: TS_CONFIG for search_tsv, EXACT_TS_CONFIG for search_tsv_exact.
    """
    if not is_safe_tsquery(rank_tsquery):
        raise ValueError("unsafe rank tsquery")
    if config not in (TS_CONFIG, EXACT_TS_CONFIG):
        raise ValueError("unknown text search configuration")
    return (
        f"COALESCE(ts_rank_cd('{RANK_WEIGHTS}', {vector_sql}, "
        f"to_tsquery('{config}', '{rank_tsquery}'), {RANK_NORMALIZATION}), 0)"
    )
