"""
services/text_search.py
-----------------------
Full-text search over dataset text.

bids_datasets.search_tsv is a stored tsvector over
    weight A  name
    weight B  dx_status, study_domain, study_design, description_text,
              llm_conditions (studied conditions read from the README)
    weight C  llm_profile (LLM-written profile, services/readme_profile.py)
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
round trips in services/sql_rewriter.py unchanged, which a `:name` parameter
does not in every sqlglot version.

On a database without search_tsv the placeholders expand to ILIKE on the short
text columns, and there is no text rank.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
_RAG_DIR = _ROOT / "RAG"
if _RAG_DIR.exists() and str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from text_terms import (  # noqa: E402
    MODE_ALL,
    MODE_LOOSE,
    MODE_SHORT,
    MODES,
    Term,
    Word,
    format_text_match,
    parse_alternatives,
    split_words,
)

from services import schema_capabilities  # noqa: E402

log = logging.getLogger(__name__)

TS_CONFIG = "english"
FTS_COLUMN = "search_tsv"
# ts_rank_cd weights in the order {D, C, B, A}: README, LLM profile of the
# README, short fields (incl. conditions the README says were studied), name.
RANK_WEIGHTS = "{0.05, 0.4, 0.6, 1.0}"
# 32 scales the rank into 0..1 (rank / (rank + 1)), the range RAG scores use.
RANK_NORMALIZATION = 32

_SCOPE_LABEL = {MODE_ALL: "", MODE_LOOSE: "", MODE_SHORT: ":AB"}
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
# when the answer is not available.
FrequencyLookup = Callable[[List[str]], Optional[List[int]]]

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


def set_frequency_lookup(lookup: Optional[FrequencyLookup]) -> None:
    global _frequency_lookup
    _frequency_lookup = lookup


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _own_connection_lookup(queries: List[str]) -> Optional[List[int]]:
    """Count with a connection of its own; for callers that installed nothing.

    Runs only off the event loop (the translation pipeline runs in a worker
    thread) and only when the database has search_tsv.
    """
    if _on_event_loop() or not schema_capabilities.has_fts():
        return None
    try:
        from db.db import engine as app_engine
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool
    except Exception as exc:  # noqa: BLE001
        log.warning("Word frequency lookup unavailable (%s) — using the longest word", exc)
        return None

    async def _count() -> List[int]:
        # NullPool: the connection belongs to this short-lived event loop only.
        engine = create_async_engine(app_engine.url, poolclass=NullPool)
        try:
            async with engine.connect() as connection:
                rows = await connection.execute(text(_COUNT_SQL), {"queries": queries})
                return [int(row[0]) for row in rows]
        finally:
            await engine.dispose()

    try:
        return asyncio.run(asyncio.wait_for(_count(), FREQUENCY_TIMEOUT_SECONDS))
    except Exception as exc:  # noqa: BLE001
        log.warning("Word frequency lookup failed (%s) — using the longest word", exc)
        return None


def make_frequency_lookup(engine: Any, loop: asyncio.AbstractEventLoop,
                          timeout: float = FREQUENCY_TIMEOUT_SECONDS) -> FrequencyLookup:
    """Pooled frequency lookup for code running in a worker thread.

    The translation pipeline runs in asyncio.to_thread, while the database
    engine belongs to the event loop; the count is scheduled there.
    """
    from sqlalchemy import text

    statement = text(_COUNT_SQL)

    async def _count(queries: List[str]) -> List[int]:
        async with engine.connect() as connection:
            rows = await connection.execute(statement, {"queries": queries})
            return [int(row[0]) for row in rows]

    def lookup(queries: List[str]) -> Optional[List[int]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return None  # called on an event loop: blocking here would deadlock
        future = asyncio.run_coroutine_threadsafe(_count(queries), loop)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            future.cancel()
            log.warning("Word frequency lookup failed (%s) — using the longest word", exc)
            return None

    return lookup


# ── tsquery building ──────────────────────────────────────────────────────────

def _word_query(word: Word, label: str = "") -> str:
    """One word; the parts of a hyphenated compound must be adjacent."""
    if len(word) == 1:
        return f"{word[0]}{label}"
    return "(" + " <-> ".join(f"{part}{label}" for part in word) + ")"


def _all_words_query(term: Term, label: str = "") -> str:
    return " & ".join(_word_query(word, label) for word in term)


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


def _lookup_counts(queries: List[str]) -> Optional[List[int]]:
    lookup = _own_connection_lookup if _frequency_lookup is _OWN_CONNECTION else _frequency_lookup
    if lookup is None:
        return None
    try:
        counts = lookup(queries)
    except Exception as exc:  # a lookup must never break a search
        log.warning("Word frequency lookup raised (%s) — using the longest word", exc)
        return None
    if not counts or len(counts) != len(queries):
        return None
    return counts


def _loose_core(term: Term) -> Term:
    """The words of an unresolved term that a dataset must contain.

    Words are taken rarest first, and as many as still leave
    LOOSE_MIN_MATCHES datasets are required: "eye tracking" keeps both words
    ("tracking" alone finds rat linear-track studies), "virtual reality
    navigation" drops "navigation" if the three together are too rare.

    A word that no dataset contains is usually the distinctive one ("sickle"
    in "sickle cell disease") or a typo. Searching the remaining words would
    answer a different question ("cell"), so then every word is required and
    the term honestly matches nothing.

    With no frequencies available, the longest word, as the ILIKE search did.
    """
    if len(term) == 1:
        return term
    longest = [max(term, key=lambda word: sum(len(part) for part in word))]
    counts = _lookup_counts([_word_query(word) for word in term])
    if counts is None:
        return longest
    if any(count == 0 for count in counts):
        return term
    present = sorted((count, index) for index, count in enumerate(counts))
    by_rarity = [term[index] for _, index in present]
    if len(by_rarity) == 1:
        return by_rarity
    prefixes = [by_rarity[:k] for k in range(len(by_rarity), 1, -1)]
    combined = _lookup_counts([_all_words_query(prefix) for prefix in prefixes])
    if combined is not None:
        for prefix, count in zip(prefixes, combined):
            if count >= LOOSE_MIN_MATCHES:
                return prefix
    return by_rarity[:1]


def build_filter_tsquery(terms: Sequence[Term], mode: str = MODE_ALL) -> Optional[str]:
    """tsquery a dataset must match; None when no term is searchable."""
    label = _SCOPE_LABEL[mode]
    if mode == MODE_LOOSE:
        queries = [_all_words_query(_loose_core(term), label) for term in terms if term]
    else:
        queries = [_all_words_query(term, label) for term in terms if term]
    return _any(queries) or None


def build_rank_tsquery(terms: Sequence[Term], mode: str = MODE_ALL) -> Optional[str]:
    """tsquery used for ranking: every field counts, whatever the filter scope.

    A diagnosis is filtered on the short fields only, but a README that
    mentions it still moves a matching dataset up.
    """
    if mode == MODE_LOOSE:
        queries = [_word_query(word) for term in terms for word in term]
    else:
        queries = [_all_words_query(term) for term in terms if term]
    return _any(queries) or None


# ── Expansion ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ExpandedSearch:
    sql: str
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


def expand_text_search(sql: str, fts: Optional[bool] = None) -> ExpandedSearch:
    """Replace every text_match() placeholder in *sql* and build the rank query.

    *fts* overrides the detected database capability (tests).
    """
    use_fts = schema_capabilities.has_fts() if fts is None else fts
    if use_fts:
        sql = convert_long_text_ilike(sql)
    alias = _dataset_alias(sql)
    rank_parts: List[str] = []

    def replace(match: re.Match) -> str:
        negated = match.group("negated") or ""
        mode = (match.group("mode") or MODE_ALL).lower()
        if mode not in MODES:
            mode = MODE_ALL
        terms = parse_alternatives(match.group("terms"))
        if not terms:
            # Nothing searchable ("???"): the term cannot match any dataset.
            return f"{negated}FALSE"
        if not use_fts:
            return negated + _ilike_fallback(terms, mode, alias)
        query = build_filter_tsquery(terms, mode)
        if not negated:
            rank = build_rank_tsquery(terms, mode)
            if rank:
                rank_parts.append(rank)
        return f"{negated}{alias}.{FTS_COLUMN} @@ to_tsquery('{TS_CONFIG}', '{query}')"

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
    return ExpandedSearch(sql=expanded, rank_tsquery=rank_tsquery or None)


def is_safe_tsquery(query: Optional[str]) -> bool:
    """True if *query* was built here and can be inlined into SQL."""
    return bool(query) and _SAFE_TSQUERY_RE.match(query) is not None


def text_rank_expression(vector_sql: str, rank_tsquery: str) -> str:
    """SQL expression for the 0..1 text rank of one dataset."""
    if not is_safe_tsquery(rank_tsquery):
        raise ValueError("unsafe rank tsquery")
    return (
        f"COALESCE(ts_rank_cd('{RANK_WEIGHTS}', {vector_sql}, "
        f"to_tsquery('{TS_CONFIG}', '{rank_tsquery}'), {RANK_NORMALIZATION}), 0)"
    )
