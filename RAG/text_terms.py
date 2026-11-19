"""
RAG/text_terms.py
-----------------
Terms for full-text search over dataset text (name, description, study
fields, README).

The SQL model never writes a text predicate itself. It copies a placeholder,

    text_match('epilepsy | epileptic | seizure', 'short')

and backend/services/fulltext/text_search.py expands it into real SQL. This module owns
the placeholder: how a user term is split into words, which alternatives a
diagnosis gets, and how the placeholder is written and read back.

Placeholder grammar
    text_match('<alternatives>'[, '<mode>'])
    alternatives  one or more terms separated by " | "; any one may match
    term          words separated by spaces; every word must match
    word          letters and digits, hyphenated compounds stay one word
                  ("resting-state", "n-back") and must appear adjacent
    mode          all    every text field including the README (default)
                  short  name, description, study fields and the conditions
                         an LLM read as studied in the README
                  loose  every text field; only the rarest words of each term
                         are required (as many as still match a few
                         datasets), the others only raise the rank
                  near   every text field; the words of each term must stand
                         close together (NEAR_DISTANCE positions, either order
                         for two words), so "social conflict" does not match a
                         README that says "social" in one place and "conflict"
                         in another
                  also   like near, for other words for the text_match it is
                         ORed with ("post-mortem" for "dead people"): a
                         dataset that matches only these is ranked below
                         every dataset that matches the first one

Only [a-z0-9 |-] ever appears inside the quotes, so the placeholder survives
being copied by a language model and can be inlined into SQL safely.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MODE_ALL = "all"
MODE_SHORT = "short"
MODE_LOOSE = "loose"
MODE_NEAR = "near"
MODE_ALSO = "also"
MODES = (MODE_ALL, MODE_SHORT, MODE_LOOSE, MODE_NEAR, MODE_ALSO)

# Largest gap between neighbouring words of a 'near' term, counted in word
# positions: "conflicts in social settings" puts "social" two positions after
# "conflict" ("in" keeps its position). 3 also matched "social cognition.
# Simon conflict task": a full stop does not separate positions.
NEAR_DISTANCE = 2

# A word is a list of alphanumeric parts: "n-back" → ["n", "back"].
Word = List[str]
Term = List[Word]

_PART_RE = re.compile(r"[a-z0-9]+")
_POSSESSIVE_RE = re.compile(r"['’`]s\b")
_APOSTROPHE_RE = re.compile(r"['’`]")
_PARENTHESES_RE = re.compile(r"\([^)]*\)")

# Function words carry no search value; PostgreSQL drops them as well.
_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with",
})

# Everyday words that name a condition only together with "disorder",
# "disease" or "syndrome": "eating disorders" searched as "eating" matched
# abstracts about meals (qp08: 1 of 12 results correct). A diagnosis string
# whose remaining words are all of these keeps its generic word;
# "Parkinson's disease" → "parkinson" still drops it.
_NEEDS_GENERIC_WORD = frozenset({
    "eating", "sleep", "sleeping", "movement", "movements", "motor", "personality",
    "language", "speech", "reading", "learning", "conduct", "mood", "panic",
    "use", "down", "williams", "heart", "kidney", "liver", "lung", "metabolic",
    "mental", "psychiatric", "neurological", "developmental", "neurodevelopmental",
    "neurodegenerative", "neuromuscular", "genetic", "rare", "functional",
    "attention", "communication", "adjustment", "stress", "pain",
})

# Words that qualify the thing searched for rather than name it. Never kept
# alone when a longer term is shortened.
MODIFIER_WORDS = frozenset({
    "young", "old", "older", "healthy", "many", "few", "single", "multiple", "different",
    "large", "small", "high", "low", "new", "long", "short", "early", "late", "normal",
    "typical", "various", "several", "repeated", "dense", "continuous", "natural",
})

# Nouns too general to stand for the term they were part of: "bipolar
# syndrome" shortened to "syndrome" matched Fragile X, Down syndrome mice and
# a phantom study (qf01: 38 of 50 results wrong).
NEVER_ALONE_WORDS = frozenset({
    "state", "resting", "rest", "syndrome", "syndromes", "disease", "diseases",
    "disorder", "disorders", "spectrum", "condition", "conditions", "task", "tasks",
    "study", "studies", "data", "dataset", "datasets", "brain", "control", "controls",
    "subjects", "participants", "patients", "people", "images", "image", "analysis",
    "results", "design", "level", "group", "groups", "type", "types", "related", "based",
    # General in a BIDS catalogue: "mriqc & derivatives" → "mriqc" keeps the
    # term (qx06 lost three correct datasets when "derivatives" counted).
    "derivatives", "derivative", "files", "file", "recordings", "recording",
    "scans", "scan", "sessions", "session", "version", "versions",
})

# "Parkinson's disease" should find "Basal Ganglia in Parkinson".
_GENERIC_DIAGNOSIS_WORDS = frozenset({
    "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
    "spectrum",
})
# A synonym such as "epileptic patients" only repeats a shorter one; one such
# as "normal subjects" would shrink to a word that matches almost anything.
_POPULATION_WORDS = frozenset({
    "adult", "adults", "child", "children", "cohort", "group", "groups",
    "individual", "individuals", "participant", "participants", "patient",
    "patients", "people", "person", "subject", "subjects", "volunteer",
    "volunteers",
})

# These words say that something is a task but do not identify its paradigm.
# Drop them before pairing structured task codes with a text fallback:
# "assessments related to memory" should search for "memory", while
# "decision making tasks" should retain both parts of the construct.
# "cognitive" is not one of them: "cognitive control" without it is "control",
# which every description with "healthy controls" matches.
_TASK_CONTEXT_WORDS = frozenset({
    "assessment", "assessments", "data", "dataset", "datasets",
    "paradigm", "paradigms", "related", "study", "studies", "task", "tasks",
    "test", "tests", "processing",
})
# A lone word shorter than this gets no task fallback. Without search_tsv the
# search is by substring, and "rest" is in "interest", "pain" in "Spain".
_MIN_LONE_TASK_WORD = 5

# The user's own words and the concept label are trusted down to three
# letters ("ASD", "OCD"). Two letters are not: "PD" is also proton density.
_MIN_TRUSTED_PART = 3
# Vocabulary synonyms were written to recognise user input, not to search
# documents: "load", "dat" and "fad" are listed for Alzheimer's disease.
_MIN_SYNONYM_PART = 6
MAX_DIAGNOSIS_ALTERNATIVES = 16


# ── Tokenising ────────────────────────────────────────────────────────────────

def _fold(text: str) -> str:
    """Lower-case and strip accents, possessives and apostrophes."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = _POSSESSIVE_RE.sub("", text)
    return _APOSTROPHE_RE.sub("", text)


def split_words(term: str) -> Term:
    """Split one search term into words, each a list of alphanumeric parts.

    "Resting-state fMRI of the brain" → [["resting", "state"], ["fmri"], ["brain"]]
    """
    words: Term = []
    for chunk in _fold(term).split():
        parts = _PART_RE.findall(chunk)
        if not parts or (len(parts) == 1 and parts[0] in _STOPWORDS):
            continue
        words.append(parts)
    return words


def _canonical(term: Term) -> str:
    return " ".join("-".join(word) for word in term)


def parse_alternatives(argument: str) -> List[Term]:
    """Read the first argument of a text_match() placeholder."""
    terms: List[Term] = []
    seen = set()
    for raw in (argument or "").split("|"):
        term = split_words(raw)
        key = _canonical(term)
        if term and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def format_text_match(alternatives: Iterable[str], mode: str = MODE_ALL) -> Optional[str]:
    """Write a text_match() placeholder, or None if nothing searchable is left."""
    if mode not in MODES:
        raise ValueError(f"unknown text_match mode: {mode!r}")
    terms = parse_alternatives(" | ".join(a for a in alternatives if a))
    if not terms:
        return None
    argument = " | ".join(_canonical(term) for term in terms)
    return f"text_match('{argument}')" if mode == MODE_ALL else f"text_match('{argument}', '{mode}')"


def format_task_text_match(terms: Iterable[str]) -> Optional[str]:
    """Text search that backs up the structured filter of a resolved task term.

    Task codes are incomplete: many datasets carry a task label the vocabulary
    does not map. A dataset whose name or description names the paradigm
    still belongs in the result.

    The search is deliberately narrow, because it is added to every resolved
    task term: every word of the phrase must match, and only in the short
    fields (mode 'short'), not anywhere in a README. The 'loose' mode this
    used before kept one word per phrase, so "working memory" searched for
    "working" and "face perception" for "perception".
    """
    cleaned: List[str] = []
    seen = set()
    for raw in terms:
        words = [
            word for word in split_words(raw)
            if not (len(word) == 1 and word[0] in _TASK_CONTEXT_WORDS)
        ]
        if not words:
            continue
        if len(words) == 1 and sum(len(part) for part in words[0]) < _MIN_LONE_TASK_WORD:
            continue
        phrase = " ".join("-".join(word) for word in words)
        if phrase not in seen:
            seen.add(phrase)
            cleaned.append(phrase)
    return format_text_match(cleaned, MODE_SHORT)


# ── Diagnosis alternatives ────────────────────────────────────────────────────

def _longest_part(term: Term) -> int:
    """Length of the longest part, not counting generic words.

    "oc disorder" scores 2: "disorder" says nothing about the condition.
    """
    return max(
        (len(part) for word in term for part in word
         if not (len(word) == 1 and word[0] in _GENERIC_DIAGNOSIS_WORDS)),
        default=0,
    )


def _flat(term: Term) -> List[str]:
    return [part for word in term for part in word]


def _core_words(text: str) -> Term:
    """Words of a diagnosis string without generic and population words.

    The generic word stays when the rest is only everyday words
    (_NEEDS_GENERIC_WORD): "eating disorders" stays whole.
    """
    words = split_words(_PARENTHESES_RE.sub(" ", text or ""))
    without_population = [
        word for word in words
        if not (len(word) == 1 and word[0] in _POPULATION_WORDS)
    ] or words
    core = [
        word for word in without_population
        if not (len(word) == 1 and word[0] in _GENERIC_DIAGNOSIS_WORDS)
    ]
    if core and len(core) < len(without_population) and all(
            len(word) == 1 and word[0] in _NEEDS_GENERIC_WORD for word in core):
        return without_population
    return core or words


def _mentions_population(text: str) -> bool:
    return any(
        len(word) == 1 and word[0] in _POPULATION_WORDS
        for word in split_words(text)
    )


@lru_cache(maxsize=1)
def _vocabulary_strings() -> Dict[str, List[Tuple[str, float]]]:
    """code → [(label or synonym, weight)], the label first.

    Reuses the knowledge base the retriever has already parsed; parsing
    value_mappings.yaml a second time would cost several seconds.
    """
    from yaml_to_llamaindex import display_db, weight_db

    weights = weight_db.get("diagnosis", {})
    by_code: Dict[str, List[Tuple[str, float]]] = {}
    for string, code in display_db.get("diagnosis", {}).items():
        by_code.setdefault(code, []).append((string, float(weights.get(string, 1.0))))
    return by_code


def _lookup_strings(code: str) -> List[Tuple[str, float]]:
    try:
        curated = _vocabulary_strings().get(code, [])
    except Exception:
        # The vocabulary only widens the search; without it the user's own
        # words still apply.
        curated = []
    if curated:
        return curated
    # A disease no dataset has yet (RAG/reference_vocabulary.py): its official
    # terms ("rasmussen" also searches "rasmussen encephalitis").
    try:
        import reference_vocabulary
        return reference_vocabulary.strings(code)
    except Exception:
        return []


# Words a diagnosis term may carry beyond the label it matched: which course,
# severity, age or certainty, or what kind of record, not which disease
# ("chronic stroke" is still stroke). Used by RAG/retriever.py and
# diagnosis_qualifier below.
DIAGNOSIS_COURSE_WORDS = frozenset({
    "chronic", "acute", "subacute", "early", "late", "onset", "mild", "moderate",
    "severe", "first", "episode", "recurrent", "remitted", "remission", "current",
    "past", "lifetime", "treatment", "resistant", "refractory", "drug", "naive",
    "medicated", "unmedicated", "untreated", "newly", "diagnosed", "diagnosis",
    "childhood", "adult", "adults", "adolescent", "adolescents", "pediatric",
    "paediatric", "juvenile", "young", "older", "elderly", "life", "idiopathic",
    "sporadic", "familial", "probable", "possible", "suspected", "clinical", "type",
    "disease", "disorder", "syndrome", "condition", "illness", "with", "without",
    "and", "of", "the", "in", "a", "people", "person", "group", "groups",
    "post", "pre", "high", "low", "functioning", "brain", "cerebral",
    "data", "dataset", "datasets", "recording", "recordings", "file", "files",
    "participants", "patients", "patient", "subjects", "individuals", "cohort",
})


def _diagnosis_word(word: str) -> str:
    """'parkinson's' → 'parkinson', 'dementias' → 'dementia' (for comparing words only)."""
    word = word.replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


_IGNORED_DIAGNOSIS_WORDS = frozenset(_diagnosis_word(w) for w in DIAGNOSIS_COURSE_WORDS)


def diagnosis_words(text: str) -> set:
    """The words of a diagnosis string that can name a disease."""
    words = re.sub(r"[-_/(),]+", " ", (text or "").lower()).split()
    return {_diagnosis_word(w) for w in words} - _IGNORED_DIAGNOSIS_WORDS


def diagnosis_qualifier(
    term: str,
    codes: Sequence[str],
    lookup: Optional[Callable[[str], List[Tuple[str, float]]]] = None,
) -> List[str]:
    """Words of *term* that name the disease but occur in no label or synonym of *codes*.

    "Lewy body dementia" resolves to the dementia group (Alzheimer's, MCI,
    Parkinson's MCI): its label sits inside the term. None of those codes says
    "lewy" or "body", so the dataset text must, or every Alzheimer's dataset
    answers the question. Empty when the codes cover every such word, or when
    the vocabulary is not available (then the codes are trusted as before).
    """
    lookup = lookup or _lookup_strings
    wanted = diagnosis_words(term)
    if not wanted or not codes:
        return []
    covered: set = set()
    found = False
    for code in codes:
        strings = lookup(code)
        found = found or bool(strings)
        for string, _weight in strings:
            covered |= diagnosis_words(string)
        covered |= diagnosis_words(code)
    if not found:
        return []
    order = [_diagnosis_word(w) for w in re.sub(r"[-_/(),]+", " ", term.lower()).split()]
    # A number alone ("type 2") matches nearly every text.
    return [w for w in dict.fromkeys(order)
            if w in wanted and w not in covered and not w.isdigit()]


def diagnosis_alternatives(
    term_to_codes: Mapping[str, Sequence[str]],
    lookup: Optional[Callable[[str], List[Tuple[str, float]]]] = None,
    limit: int = MAX_DIAGNOSIS_ALTERNATIVES,
) -> List[str]:
    """Search terms that back up a structured diagnosis filter.

    *term_to_codes* maps each user term to the codes it resolved to, best
    match first. The result holds, in this order,
      1. the user's own terms,
      2. label and synonyms of each term's best code,
      3. labels of the other codes,
    without an alternative that a shorter one already covers, and capped at
    *limit*. Generic words ("disease", "disorder") are dropped from the user's
    terms and the best code's label, never from a synonym or another label.
    """
    lookup = lookup or _lookup_strings
    candidates: List[Tuple[Term, int]] = []   # (words, minimum longest part)

    for term in term_to_codes:
        candidates.append((_core_words(term), _MIN_TRUSTED_PART))

    other_labels: List[Tuple[Term, int]] = []
    for codes in term_to_codes.values():
        for position, code in enumerate(codes):
            strings = lookup(code)
            if not strings:
                continue
            label, synonyms = strings[0][0], strings[1:]
            if position > 0:
                # Kept whole: "Watanabe syndrome" without "syndrome" is a
                # common author name.
                other_labels.append(
                    (split_words(_PARENTHESES_RE.sub(" ", label)), _MIN_TRUSTED_PART)
                )
                continue
            candidates.append((_core_words(label), _MIN_TRUSTED_PART))
            for synonym, _weight in sorted(synonyms, key=lambda item: -item[1]):
                if not _mentions_population(synonym):
                    # Kept whole: without its "disorder", "reading disorder"
                    # would match every reading study.
                    words = split_words(_PARENTHESES_RE.sub(" ", synonym))
                    candidates.append((words, _MIN_SYNONYM_PART))
    candidates.extend(other_labels)

    kept: List[Term] = []
    kept_sets: List[frozenset] = []
    for candidate, min_part in candidates:
        if not candidate or _longest_part(candidate) < min_part:
            continue
        parts = frozenset(_flat(candidate))
        # "refractory epilepsy" adds nothing once "epilepsy" is searched.
        if any(existing <= parts for existing in kept_sets):
            continue
        # A shorter newcomer replaces the longer terms it covers.
        covered = [i for i, existing in enumerate(kept_sets) if parts < existing]
        for index in reversed(covered):
            del kept[index], kept_sets[index]
        kept.append(candidate)
        kept_sets.append(parts)
        if len(kept) >= limit:
            break
    return [_canonical(term) for term in kept]
