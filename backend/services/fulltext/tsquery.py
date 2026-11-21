"""
services/fulltext/tsquery.py
----------------------------
tsquery strings from the search terms of a text_match() placeholder
(RAG/text_terms.py parses them into words).

One vector serves three scopes, because a tsquery word can be restricted to
weights: `word:A` matches the name only, `word:AB` the short fields, plain
`word` everything (SCOPE_LABEL). A 'near' term spells out the allowed word
distances, a 'loose' term keeps its rarest words (word_frequencies.py), and a
word with decided forms gets a guard for the unstemmed vector (exact_guard).

The strings consist of [a-z0-9] words and tsquery operators only
(is_safe_tsquery), so text_search.py and ranking.py can inline them into SQL
as literals.
"""

from __future__ import annotations

import itertools
import re
from typing import List, Mapping, Optional, Sequence, Tuple

from text_terms import (
    MODE_ALL,
    MODE_LOOSE,
    MODE_NEAR,
    MODE_SHORT,
    MODIFIER_WORDS,
    NEAR_DISTANCE,
    NEVER_ALONE_WORDS,
    Term,
    Word,
    parse_alternatives,
    split_words,
)
from word_forms import WordEvidence, forms_for, search_variants

from services.fulltext import vectors, word_frequencies

# word → the forms it must match in, as decided for one question
# (RAG/word_forms.py). The reviewed list there applies without being passed;
# for a word not in the mapping it takes the catalogue's forms (_catalogue_forms).
ExactForms = Mapping[str, Sequence[str]]

SCOPE_LABEL = {MODE_ALL: "", MODE_LOOSE: "", MODE_SHORT: ":AB", MODE_NEAR: ""}
NAME_LABEL = ":A"
_SAFE_TSQUERY_RE = re.compile(r"^[a-z0-9 &|()<>:AB-]+$")


def _catalogue_forms(word: str) -> Optional[WordEvidence]:
    """The catalogue's forms of *word*, for a reviewed word the question's
    decisions do not cover (word_forms.forms_for)."""
    # Imported here: word_form_counts imports text_search, which imports this module.
    from services.fulltext import word_form_counts
    return word_form_counts.cached_evidence(word)


def _word_forms(word: Word, exact_forms: Optional[ExactForms]) -> Optional[Tuple[str, ...]]:
    """The forms *word* must match in, or None to match it by stem.

    Only plain words: the parts of a compound ("t1-weighted") stay together
    by adjacency, which already keeps them from matching anything else.
    """
    if len(word) != 1 or not vectors.exact_enabled():
        return None
    forms = forms_for(word[0], exact_forms, _catalogue_forms)
    if not forms:
        return forms
    return tuple(dict.fromkeys(f for form in forms for f in (form, *search_variants(form))))


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


def _spelled(part: str, label: str = "") -> str:
    """A word in all its spellings and numbers: "(behaviour | behavior)", "(stimulus | stimuli)"."""
    return _form_group([part, *search_variants(part)], label)


def _word_query(word: Word, label: str = "") -> str:
    """One word in every spelling and number; the parts of a hyphenated compound must be adjacent."""
    if len(word) == 1:
        return _spelled(word[0], label)
    return "(" + " <-> ".join(_spelled(part, label) for part in word) + ")"


def all_words_query(term: Term, label: str = "") -> str:
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
        return all_words_query(term, label)
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


# A word in all its forms: "(mouse:AB | mice:AB)", or a single word.
_WORD_GROUP_RE = re.compile(r"\([a-z0-9:AB]+(?: \| [a-z0-9:AB]+)+\)|[a-z0-9:AB]+")


def any_of(queries: Sequence[str]) -> str:
    queries = list(dict.fromkeys(q for q in queries if q))
    if len(queries) == 1:
        return queries[0]
    # Inside an OR, a word's forms join the OR: a list that names "mouse" and
    # "mice" gives "mouse | mice", not "(mouse | mice) | (mice | mouse)".
    flat: List[str] = []
    for query in queries:
        if _WORD_GROUP_RE.fullmatch(query):
            flat.extend(query.strip("()").split(" | "))
        else:
            flat.append(query)
    queries = list(dict.fromkeys(flat))
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
    counts = word_frequencies.guarded_counts([_word_query(word) for word in term],
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
    combined = word_frequencies.guarded_counts([all_words_query(prefix) for prefix in prefixes],
                              [exact_guard(prefix, "", exact_forms) for prefix in prefixes])
    if combined is not None:
        for prefix, count in zip(prefixes, combined):
            if count >= LOOSE_MIN_MATCHES:
                return prefix
    return by_rarity[:1]


def filter_term(term: Term, mode: str, exact_forms: Optional[ExactForms] = None) -> Term:
    """The words of *term* a dataset must contain: all, or a loose term's core."""
    return _loose_core(term, exact_forms) if mode == MODE_LOOSE else term


def build_filter_tsquery(terms: Sequence[Term], mode: str = MODE_ALL,
                         exact_forms: Optional[ExactForms] = None) -> Optional[str]:
    """tsquery a dataset must match; None when no term is searchable.

    *exact_forms* only tells a loose term's word counts which forms count.
    """
    label = SCOPE_LABEL[mode]
    if mode == MODE_LOOSE:
        queries = [all_words_query(_loose_core(term, exact_forms), label)
                   for term in terms if term]
    elif mode == MODE_NEAR:
        queries = [_near_query(term, label) for term in terms if term]
    else:
        queries = [all_words_query(term, label) for term in terms if term]
    return any_of(queries) or None


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
        guards.append(exact_guard(filter_term(terms[0], mode, exact_forms),
                                  SCOPE_LABEL[mode], exact_forms) if len(terms) == 1 else None)
    asked = [query for query in queries if query]
    if not asked:
        return [0] * len(queries)
    counts = word_frequencies.guarded_counts(asked, [g for q, g in zip(queries, guards) if q])
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
        queries = [all_words_query(term) for term in terms
                   if term and not all(exact(word) for word in term)]
    return any_of(queries) or None


def build_exact_rank_tsquery(terms: Sequence[Term], mode: str = MODE_ALL,
                             exact_forms: Optional[ExactForms] = None) -> Optional[str]:
    """Ranking query for the unstemmed vector: the words with decided forms."""
    groups = [[_form_group(forms) for forms in (_word_forms(word, exact_forms) for word in term)
               if forms] for term in terms]
    if mode == MODE_LOOSE:
        queries = [group for term in groups for group in term]
    else:
        queries = [" & ".join(term) for term in groups if term]
    return any_of(queries) or None


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
    query = any_of(queries[:12]) if queries else ""
    return query if query and is_safe_tsquery(query) else None


def is_safe_tsquery(query: Optional[str]) -> bool:
    """True if *query* was built here and can be inlined into SQL."""
    return bool(query) and _SAFE_TSQUERY_RE.match(query or "") is not None
