"""
services/fulltext/text_search.py
--------------------------------
Full-text search over dataset text: the text_match() placeholders in the
generated SQL become conditions on the stored vectors (vectors.py).

The SQL model writes text_match('...') placeholders (grammar in
RAG/text_terms.py). This module turns them into

    d.search_tsv @@ to_tsquery('english', '<query>')

and builds the query that ranks the result. The tsquery strings come from
tsquery.py, the rank from ranking.py, the word counts from
word_frequencies.py.

Queries are inlined as literals, not bound as parameters. They consist of
[a-z0-9] words and tsquery operators only, and a literal survives the sqlglot
round trips in services/translate/sql_rewriter.py unchanged, which a `:name` parameter
does not in every sqlglot version.

Paper text (TEXT_SEARCH_PAPERS=true). search_tsv_papers is search_tsv plus,
at weight D, the title, abstract and keywords of the papers the dataset names
as its own (crawlers/paper_enrichment_plan.md). With the switch on, a
positive text_match() filters on it, so a dataset whose own text lacks a word
but whose paper has it is found. Everything else stays on search_tsv:
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
those forms on search_tsv_exact, the same vector without stemming:

    (d.search_tsv @@ to_tsquery('english', 'ageing & mri')
     AND d.search_tsv_exact @@ to_tsquery('simple', '(ageing | aging)'))

Always this shape: the stemmed match, then its guard, in one pair of
parentheses. The stemmed match keeps the other words and the word distances;
the guard has one group per exact word, the asked word first, with the same
weight labels. Whatever reads the SQL afterwards finds searches and guards
with text_searches and treats the pair as one condition (is_guarded_pair):
the filter chips, the evidence, the uncertain tier. A
placeholder with several alternatives becomes one pair per alternative that
has an exact word, ORed with the rest. The word counts count such words in
their forms, and they are ranked on search_tsv_exact: the RankQuery then has
a stemmed and an exact part (ranking.rank_expression). With paper text on,
the guard reads search_tsv_exact_papers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sql_text import pattern_words
from sqlglot import exp
from text_terms import (
    MODE_ALL,
    MODE_ALSO,
    MODE_LOOSE,
    MODE_NEAR,
    MODE_SHORT,
    MODES,
    Term,
    format_text_match,
    parse_alternatives,
    split_words,
)

from services.fulltext import ranking, tsquery, vectors
from services.fulltext.vectors import (
    EXACT_FTS_COLUMN,
    EXACT_PAPER_FTS_COLUMN,
    EXACT_TS_CONFIG,
    FTS_COLUMN,
    PAPER_FTS_COLUMN,
    TS_CONFIG,
)

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
    r"(?P<negated>\bNOT\s+)?\b[A-Za-z_]\w*\.name\s+(?P<inner_negated>NOT\s+)?ILIKE\s+'%(?P<literal>(?:[^'%\\]|''|\\.)+)%'",
    re.IGNORECASE,
)
_URL_FRAGMENT_RE = re.compile(r"[./:@]")


# ── Expansion ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ExpandedSearch:
    sql: str
    # What ranks the result by its text; None without text terms.
    rank_query: Optional[ranking.RankQuery]


def _dataset_alias(sql: str) -> str:
    for match in _DATASET_ALIAS_RE.finditer(sql):
        alias = match.group(1)
        if alias.lower() not in _SQL_KEYWORDS:
            return alias
    return "d"


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
    exact_papers = vectors.exact_enabled()

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
    return EXACT_PAPER_FTS_COLUMN if column == PAPER_FTS_COLUMN else EXACT_FTS_COLUMN


# ── Reading the SQL back ──────────────────────────────────────────────────────

STEMMED_COLUMNS = frozenset({FTS_COLUMN, PAPER_FTS_COLUMN})
_EXACT_COLUMNS = frozenset({EXACT_FTS_COLUMN, EXACT_PAPER_FTS_COLUMN})


@dataclass(frozen=True)
class FoundSearch:
    """One to_tsquery() in parsed SQL."""
    query: str                # its query string, the last string argument
    call: Any                 # the sqlglot node, for where it stands (negation, branch)
    column: Optional[str]     # the vector it is matched against; None for an expression
    guard: Optional[FoundSearch] = None   # of a search: the guard written after it

    @property
    def is_guard(self) -> bool:
        """It reads the unstemmed vector: word forms of a search, not a search."""
        return self.column in _EXACT_COLUMNS


def _is_tsquery(node: Any) -> bool:
    if isinstance(node, exp.Anonymous):
        return node.name.lower() == "to_tsquery"
    return isinstance(node, exp.Func) and node.sql_name().lower() == "to_tsquery"


def _vector(call: Any) -> Optional[exp.Column]:
    """The column *call* is matched against (d.search_tsv @@ ...), if it is one."""
    match = call.parent
    if not isinstance(match, exp.MatchAgainst) or not match.expressions:
        return None
    vector = match.expressions[0]
    return vector if isinstance(vector, exp.Column) else None


def _is_guard(call: Any) -> bool:
    vector = _vector(call)
    return vector is not None and vector.name.lower() in _EXACT_COLUMNS


def _guarded_call(guard: Any) -> Optional[Any]:
    """The search a guard is written after: "d.search_tsv @@ s AND d.search_tsv_exact @@ guard"."""
    match = guard.parent
    pair = match.parent
    if not isinstance(pair, exp.And) or pair.right is not match:
        return None
    left = pair.left
    while isinstance(left, exp.And):
        left = left.right
    if not isinstance(left, exp.MatchAgainst) or not _is_tsquery(left.this) or _is_guard(left.this):
        return None
    vector, guard_vector = _vector(left.this), _vector(guard)
    same_row = vector is not None and guard_vector is not None and vector.table == guard_vector.table
    return left.this if same_row else None


def text_searches(node: Any) -> List[FoundSearch]:
    """Every to_tsquery() with a query string in the sqlglot *node*, subqueries
    included, in the order they are written; a search carries its guard."""
    calls = []
    for call in node.find_all(exp.Anonymous, exp.Func, bfs=False):
        if not _is_tsquery(call):
            continue
        literals = [e for e in call.expressions if isinstance(e, exp.Literal) and e.is_string]
        if literals:
            vector = _vector(call)
            calls.append((call, literals[-1].this, vector.name.lower() if vector else None))
    # The guards first, so that each search can carry its own.
    guards: Dict[int, FoundSearch] = {}
    guard_of: Dict[int, FoundSearch] = {}
    for call, query, column in calls:
        if column in _EXACT_COLUMNS:
            guards[id(call)] = FoundSearch(query, call, column)
            searched = _guarded_call(call)
            if searched is not None:
                guard_of[id(searched)] = guards[id(call)]
    return [guards.get(id(call)) or FoundSearch(query, call, column, guard_of.get(id(call)))
            for call, query, column in calls]


def is_guarded_pair(node: Any) -> bool:
    """True if the sqlglot *node* is "(stemmed match AND its guard)".

    Code that splits a WHERE clause into its AND conditions keeps such a pair
    whole: it is one text search, not two (module docstring).
    """
    if not isinstance(node, exp.Paren) or not isinstance(node.this, exp.And):
        return False
    right = node.this.right
    return (isinstance(right, exp.MatchAgainst) and _is_tsquery(right.this)
            and _is_guard(right.this))


def expand_text_search(
    sql: str, papers: Optional[bool] = None,
    exact_forms: Optional[tsquery.ExactForms] = None,
) -> ExpandedSearch:
    """Replace every text_match() placeholder in *sql* and build the rank queries.

    *papers* overrides the TEXT_SEARCH_PAPERS switch (tests). *exact_forms*: the word forms decided
    for this question; with TEXT_SEARCH_EXACT on, those words and the ones on
    the reviewed list also need one of their forms (module docstring).
    """
    use_papers = vectors.papers_enabled() if papers is None else papers
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
        # What each alternative requires: a loose term its core, counted once here.
        required = [tsquery.filter_term(term, mode, exact_forms) for term in terms]
        if also and not inside_not and previous_filter:
            own_parts.append(previous_filter[-1])
        elif not inside_not:
            own_filter = tsquery.build_filter_tsquery(
                required, MODE_ALL if mode == MODE_LOOSE else mode)
            if own_filter:
                previous_filter.append(own_filter)
        if not negated:
            rank = tsquery.build_rank_tsquery(terms, mode, exact_forms)
            if rank:
                rank_parts.append(rank)
            rank_exact = tsquery.build_exact_rank_tsquery(terms, mode, exact_forms)
            if rank_exact:
                rank_exact_parts.append(rank_exact)
        column = PAPER_FTS_COLUMN if (use_papers and not inside_not) else FTS_COLUMN

        def atom(query: Optional[str]) -> str:
            if query is None:
                return "FALSE"  # no term is searchable, as for "???" above
            return f"{alias}.{column} @@ to_tsquery('{TS_CONFIG}', '{query}')"

        filter_mode = MODE_ALL if mode == MODE_LOOSE else mode
        plain_core = atom(tsquery.build_filter_tsquery(required, filter_mode))

        def core_expression() -> str:
            if not vectors.exact_enabled():
                return plain_core
            label = tsquery.SCOPE_LABEL[mode]
            # Each alternative's guard if it has exact words.
            guards = [tsquery.exact_guard(term, label, exact_forms) for term in required]
            if not any(guards):
                return plain_core
            # One pair per alternative with exact words; the others share one
            # stemmed match, placed where the first of them stood.
            pieces: List[str] = []
            plain: List[Term] = []
            plain_at: Optional[int] = None
            for term, guard in zip(required, guards):
                if guard:
                    pieces.append(
                        f"({atom(tsquery.build_filter_tsquery([term], filter_mode))}"
                        f" AND {alias}.{_exact_column(column)} @@ "
                        f"to_tsquery('{EXACT_TS_CONFIG}', '{guard}'))")
                else:
                    if plain_at is None:
                        plain_at = len(pieces)
                    plain.append(term)
            if plain:
                pieces.insert(plain_at or 0, atom(tsquery.build_filter_tsquery(plain, filter_mode)))
            if len(pieces) == 1:
                return pieces[0]
            return "(" + " OR ".join(pieces) + ")"

        core = core_expression()
        lost_words = any(len(core_term) < len(term) for core_term, term in zip(required, terms))
        if mode == MODE_LOOSE and not inside_not and lost_words:
            # A loose term that lost words ("physical exercise" → "exercise"):
            # (all its words OR its core). The same rows as the core alone,
            # but the uncertain tier confirms only the datasets with every
            # word and calls the rest probable (services/uncertain/classify.py, loose_term):
            # breathing and visualization "exercises" were confirmed matches
            # for "physical exercise" (live, 2026-11-21).
            whole = tsquery.build_filter_tsquery(terms, MODE_ALL)
            if whole is not None:
                every_word = atom(whole)
                if core != plain_core:
                    # The core carries exact word forms: all words AND those.
                    every_word = f"({every_word} AND {core})"
                return f"({every_word} OR {core})"
        return negated + core

    expanded = _TEXT_MATCH_RE.sub(replace, sql)

    # A requested title word also ranks: an exact title hit comes first.
    for match in _NAME_ILIKE_RE.finditer(expanded):
        if match.group("negated") or match.group("inner_negated"):
            continue
        words = split_words(pattern_words(match.group("literal").replace("''", "'")))
        if words:
            rank_parts.append(tsquery.all_words_query(words, tsquery.NAME_LABEL))

    rank_stemmed = tsquery.any_of(rank_parts) if rank_parts else None
    rank_exact = tsquery.any_of(rank_exact_parts) if rank_exact_parts else None
    if not (rank_stemmed or rank_exact):
        return ExpandedSearch(sql=expanded, rank_query=None)
    own = tsquery.any_of(list(dict.fromkeys(own_parts))) if own_parts else None
    return ExpandedSearch(sql=expanded, rank_query=ranking.RankQuery(rank_stemmed, rank_exact, own))
