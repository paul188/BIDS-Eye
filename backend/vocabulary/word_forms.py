"""
backend/vocabulary/word_forms.py
-----------------
Which forms of a searched word may match.

Full-text search matches a word by its stem (PostgreSQL's 'english'
configuration), so "depression" also finds "depressive". Usually that is
wanted. Sometimes the stem merges words that mean different things: "ageing"
shares the stem of "age" and "aged", "weight" that of "T1-weighted", "severe"
that of "several". A search for the first then returns every dataset with the
second.

For such a word the search can require the forms that may match
(backend/services/fulltext/text_search.py writes the condition). This module decides
which forms those are. It has no database access; the catalogue's forms come
in as FormCounts (backend/services/fulltext/word_form_counts.py).

Whether a form may match is a question about two words, never about one: a
search for "ageing" must not find "age", but a search for "aged" may well find
"ageing". So the decisions are pairs, with a direction: (asked word, other
form) → different. A pair nobody judged keeps matching by stem. The forms a
search requires are the catalogue's forms of the stem minus the ones judged
different (allowed_forms); a word without such a pair is searched by stem.

The pairs come from, in this order:
  1. word_forms.yaml, a short reviewed list: each form of a group against
     each of its excludes. It applies to every searched word, also to
     synonyms a vocabulary or the language model added.
  2. word_form_pairs.json, the catalogue's colliding pairs judged offline (a
     language model, its 'different' answers reviewed; how and when are in
     the file). Applies like the list; the list wins where both judge a pair.
  3. the language model, for colliding pairs of the question's words that
     neither file judged (open_forms). It sees those forms with their dataset
     counts (prompt_block) and names the ones to keep; a form shown and not
     kept is different (decide()).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

log = logging.getLogger(__name__)

_PATH = Path(__file__).with_name("word_forms.yaml")
_PAIRS_PATH = Path(__file__).with_name("word_form_pairs.json")
_WORD_RE = re.compile(r"[a-z]+")
# A form goes into a tsquery literal as it is, so nothing else may pass.
_FORM_RE = re.compile(r"[a-z0-9]+")

# Another form of the stem collides with the asked word when it alone brings
# in at least (MIN_RATIO - 1) times the datasets of the asked word's own forms,
# and at least MIN_EXTRA. Tune these on development questions only
# (benchmarks/recall/held_out_questions.txt).
MIN_RATIO = 2.0
MIN_EXTRA = 10
# Colliding forms shown to the language model per word, most frequent first.
PROMPT_FORMS = 8
# Words of a question looked up at most.
MAX_QUESTION_WORDS = 40

# word → the forms that may match (asked word first)
ExactForms = Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class FormCounts:
    """What the catalogue says about one searched word."""
    word: str
    stem: str
    stem_docs: int                          # datasets the stemmed search matches
    forms: Tuple[Tuple[str, int], ...]      # (form, datasets with it), most frequent first

    def count(self, form: str) -> int:
        return next((n for f, n in self.forms if f == form), 0)


def question_words(question: str) -> List[str]:
    """The words of *question* worth looking up, in order, without repeats."""
    words = [w for w in _WORD_RE.findall(question.lower()) if len(w) >= 3]
    return list(dict.fromkeys(words))[:MAX_QUESTION_WORDS]


def inflections(word: str) -> Set[str]:
    """*word* with its regular plural and singular spellings.

    Deliberately crude: it is only ever intersected with forms that exist in
    the catalogue and share the word's stem.
    """
    found = {word, word + "s", word + "es"}
    if word.endswith("y"):
        found.add(word[:-1] + "ies")
    if word.endswith("ies"):
        found.add(word[:-3] + "y")
    if word.endswith("es"):
        found.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        found.add(word[:-1])
    return found


# ── British and American spelling ─────────────────────────────────────────────
#
# The stemmer keeps "behaviour" and "behavior" apart, so a search in one
# spelling misses every dataset written in the other (272 datasets for
# behavioural/behavioral on 2026-11-21). Every searched word is therefore also
# searched in its other spelling. The rules run in both directions and may
# produce words nobody writes ("doctour"); those match nothing. What they must
# not produce is another real word, which backend/tests/test_spelling.py checks
# against words known to be different.

_OUR = ("behavi|col|hum|lab|neighb|fav|flav|hon|od|rum|tum|vap|vig|rig|harb|arm|savi|endeav|parl|val|"
        "cand|clam|ferv|sav|demean|ard|splend")
_OUR_END = "s|ed|ing|al|ally|ism|ist|ists|ite|ites|less|hood|ly|ful|able|ably|er|ers|y"
_RE_STEMS = ("(?:multi|epi)?cent|fib|(?:milli|centi|deci)?lit|(?:milli|centi|kilo|micro|nano)?met|"
             "theat|calib|sab|somb|spect|lust")

# (British pattern, its American form, American pattern, its British form);
# regexes over the whole word.
_UK_US: Tuple[Tuple[str, str, str, str], ...] = (
    # behaviour, colour, tumour, behavioural, neighbouring, unfavourable: listed stems, since
    # "motor", "factor" and "error" have no British -our
    (rf"({_OUR})our({_OUR_END})?$", r"\1or\2", rf"({_OUR})or({_OUR_END})?$", r"\1our\2"),
    # centre, fibre, litre, millimetre, theatre: listed stems from the word start, since "water",
    # "number", "parameter" and "cluster" have no -re
    (rf"^({_RE_STEMS})re(s|d)?$", r"\1er\2", rf"^({_RE_STEMS})er(s|ed)?$", r"\1re\2"),
    # randomise, normalisation, characterising; not size or prize (three letters before);
    # words that keep -ise in American English are skipped (_ISE_BOTH)
    (r"(?<=[a-z]{3})is(e|es|ed|ing|er|ers|ation|ations|ational)$", r"iz\1",
     r"(?<=[a-z]{3})iz(e|es|ed|ing|er|ers|ation|ations|ational)$", r"is\1"),
    # analyse, paralysed; not "analyses", the plural of analysis
    (r"ys(e|ed|ing|er|ers)$", r"yz\1", r"yz(e|ed|ing|er|ers)$", r"ys\1"),
    # labelled, modelling, signalling, travelled: two syllables at least, so "called" stays
    (r"(?:(?<=[aeiou][^aeiou])|(?<=[aeiou][^aeiou]{2}))([eao])ll(ed|ing|er|ers)$", r"\1l\2",
     r"(?:(?<=[aeiou][^aeiou])|(?<=[aeiou][^aeiou]{2}))([eao])l(ed|ing|er|ers)$", r"\1ll\2"),
    # catalogue, analogue, dialogue; not backlog or hedgehog
    (r"(catal|anal|dial|monol|prol|homol|pedag|demag|synag|epil)ogue(s)?$", r"\1og\2",
     r"(catal|anal|dial|monol|prol|homol|pedag|demag|synag|epil)og(s)?$", r"\1ogue\2"),
    (r"^grey", "gray", r"^gray", "grey"),
    (r"programme(s)?$", r"program\1", r"program(s)?$", r"programme\1"),
    (r"^(licen|defen|offen|preten)ce(s)?$", r"\1se\2", r"^(licen|defen|offen|preten)se(s)?$", r"\1ce\2"),
    (r"sulph", "sulf", r"sulf", "sulph"),
)
_ISE_RE = re.compile(r"is(e|es|ed|ing|er|ers|ation|ations|ational)$")
# Words that keep -ise in American English: no "exercize", "promize", "advize".
_ISE_BOTH = frozenset("""
exerc prom prec conc adv compr surpr expert rev superv telev disgu prem enterpr franch merchand comprom
otherw likew clockw anticlockw dev improv chast desp exc inc circumc surm treat parad advert appr dem repr
pra cru bru val porpo torto turquo
""".split())

# Medical stems with ae/oe in British spelling.
# These may stand anywhere in a word (hypoglycaemia, lymphoedema):
_STEMS_ANYWHERE: Tuple[Tuple[str, str], ...] = (
    ("paediatr", "pediatr"), ("aetiol", "etiol"), ("ischaem", "ischem"), ("leukaem", "leukem"),
    ("gynaec", "gynec"), ("orthopaed", "orthoped"), ("faeca", "feca"), ("faece", "fece"), ("oedem", "edem"),
    ("oesophag", "esophag"), ("diarrhoe", "diarrhe"), ("glycaem", "glycem"), ("septicaem", "septicem"),
    ("hyperaem", "hyperem"), ("anaemi", "anemi"), ("anaesth", "anesth"),
)
# These only where a word or a combining "o" starts them (neurohaemodynamic), so
# "chemotherapy", "hemisphere", "orchestra" and "fetch" stay as they are:
_STEMS_AT_START: Tuple[Tuple[str, str], ...] = (
    ("haemo", "hemo"), ("haema", "hema"), ("oestr", "estr"), ("foetal", "fetal"), ("foetus", "fetus"),
)


def _rule_variants(word: str) -> Set[str]:
    out: Set[str] = set()
    for uk_pat, to_us, us_pat, to_uk in _UK_US:
        out.add(re.sub(uk_pat, to_us, word))
        out.add(re.sub(us_pat, to_uk, word))
    ise = _ISE_RE.search(word)
    if ise and word[:ise.start()] in _ISE_BOTH:
        out.discard(_ISE_RE.sub(r"iz\1", word))
    for stems, start in ((_STEMS_ANYWHERE, ""), (_STEMS_AT_START, "(?:^|(?<=o))")):
        for uk, us in stems:
            if uk in word:
                out.add(re.sub(start + uk, us, word))
            else:       # "oedema" contains "edem" too
                out.add(re.sub(start + us, uk, word))
    out.discard(word)
    return {w for w in out if _FORM_RE.fullmatch(w)}


@lru_cache(maxsize=4096)
def spelling_variants(word: str) -> Tuple[str, ...]:
    """The other spellings of *word*, British and American ("behaviour" → "behavior").

    Two rounds, so a word needing two changes is covered ("anaesthetised" →
    "anesthetized"). Empty for most words.
    """
    word = word.lower()
    if len(word) < 4 or not _WORD_RE.fullmatch(word):
        return ()
    found = _rule_variants(word)
    for variant in list(found):
        found |= _rule_variants(variant)
    found.discard(word)
    return tuple(sorted(found))


# ── Latin and Greek plurals ───────────────────────────────────────────────────
#
# The stemmer keeps "stimulus" and "stimuli" apart (162 datasets write only
# "stimuli" on 2026-11-21), and "cortex"/"cortices", "criterion"/"criteria",
# "analysis"/"analyses" too. Every searched word is also searched in its other
# number; so are irregular English plurals (child/children, mouse/mice). Where an
# ending is reliably Latin (-ae, -mata, -sis) the rule is general; where names,
# abbreviations or English words share the ending ("status", "nifti", "complex",
# "dementia", "alpha") the stems are listed.

# -us / -i: listed stems. The catalogue's other -us and -i words are names, abbreviations
# and English words (status, bolus, rhesus, tinnitus, globus pallidus, nifti, afni, helsinki).
_US_I = frozenset("""
stimul gyr sulc nucle loc foc radi hippocamp parahippocamp thalam hypothalam collicul cune precune canth
menisc fung bacill alumn termin calcul embol homuncul uter
""".split())
# -um / -a
_UM_A = frozenset("""
spectr maxim minim optim strat sept striat cerebell cerebr claustr bacteri symposi curricul quant moment
continu referend memorand consorti addend errat corrigend millenni equilibri
""".split())
# Not media/medium ("social media" is no "culture medium") and not data/datum (nobody writes datum).
# -on / -a
_ON_A = frozenset("criteri phenomen gangli automat polyhedr tetrahedr".split())
# -ices → -ex / -ix (the stem, then its singular ending). Only from the plural: the stemmer cuts
# "indices" to "indic" (indicated), "vertices" to "vertic" (vertical) and "cortices" to "cortic"
# (cortical), so searching the plural for a singular would bring those in. "matrices" is safe.
_ICES = {"cort": "ex", "neocort": "ex", "subcort": "ex", "ind": "ex", "vert": "ex", "ap": "ex", "vort": "ex",
         "cod": "ex", "matr": "ix", "append": "ix", "hel": "ix", "cerv": "ix", "forn": "ix"}
_ICES_SAFE = frozenset({"matr"})
# -a / -ae, singular to plural (a plural in -ae always has a singular in -a)
_A_AE = frozenset("""
vertebr formul larv antenn nebul alg lamin foss ampull bull stri papill macul fistul scapul tibi fibul
patell corne amygdal insul uvul ven aort axill lacun plic tuberos
""".split())
# -ma / -mata, singular to plural
_MA_MATA = frozenset("sche sto stig lem dog".split())
# Irregular English plurals; whole words only ("human" is no "humen").
# Not person/people ("in-person", "first-person" are no people), life/lives
# ("lives" stems to "live", as live recordings) or leaf/leaves (leave).
_IRREGULAR = {"man": "men", "woman": "women", "child": "children", "grandchild": "grandchildren",
              "schoolchild": "schoolchildren", "stepchild": "stepchildren", "mouse": "mice", "foot": "feet",
              "tooth": "teeth", "goose": "geese", "calf": "calves", "half": "halves", "knife": "knives"}
_IRREGULAR_BACK = {plural: singular for singular, plural in _IRREGULAR.items()}
# words that end in -sis / -ses but are no such pair ("bases" is also the plural of "base")
_SIS_NOT = frozenset("basis bases oasis".split())


def _plural_variants(word: str) -> Set[str]:
    out: Set[str] = set()
    n = len(word)
    for table in (_IRREGULAR, _IRREGULAR_BACK):
        if word in table:
            out.add(table[word])
    if word.endswith("us") and word[:-2] in _US_I:
        out.add(word[:-2] + "i")                              # stimulus → stimuli
    if word.endswith("i") and word[:-1] in _US_I:
        out.add(word[:-1] + "us")                             # nuclei → nucleus, radii → radius
    if word.endswith("um") and word[:-2] in _UM_A:
        out.add(word[:-2] + "a")
    if word.endswith("a") and word[:-1] in _UM_A:
        out.add(word[:-1] + "um")
    if word.endswith("on") and word[:-2] in _ON_A:
        out.add(word[:-2] + "a")
    if word.endswith("a") and word[:-1] in _ON_A:
        out.add(word[:-1] + "on")
    for stem, ending in _ICES.items():
        if word == stem + ending and stem in _ICES_SAFE:
            out.add(stem + "ices")                            # matrix → matrices
        if word == stem + "ices":
            out.add(stem + ending)                            # matrices → matrix
    if word.endswith("ae") and n >= 5:
        out.add(word[:-1])                                    # vertebrae → vertebra
    if word.endswith("a") and word[:-1] in _A_AE:
        out.add(word + "e")                                   # vertebra → vertebrae
    if word.endswith("ma") and word[:-2] in _MA_MATA:
        out.add(word + "ta")                                  # schema → schemata
    if word.endswith("mata") and n >= 6:
        out.add(word[:-2])                                    # stomata → stoma
    if word not in _SIS_NOT:
        if word.endswith("sis") and n >= 6:
            out.add(word[:-2] + "es")                         # analysis → analyses, diagnosis → diagnoses
        if word.endswith(("yses", "oses", "eses")) and n >= 7:
            out.add(word[:-2] + "is")                         # hypotheses → hypothesis
    out.discard(word)
    return {w for w in out if _FORM_RE.fullmatch(w)}


@lru_cache(maxsize=4096)
def plural_variants(word: str) -> Tuple[str, ...]:
    """The Latin, Greek or irregular plural of *word*, or its singular
    ("stimulus" ↔ "stimuli", "child" ↔ "children")."""
    word = word.lower()
    if word in _IRREGULAR or word in _IRREGULAR_BACK:
        return tuple(sorted(_plural_variants(word)))
    if len(word) < 4 or not _WORD_RE.fullmatch(word):
        return ()
    return tuple(sorted(_plural_variants(word)))


@lru_cache(maxsize=4096)
def search_variants(word: str) -> Tuple[str, ...]:
    """Every other form a search for *word* should also match: the other
    spelling, the other number, and the other spelling of that
    ("haematoma"/"hematoma", "analysis"/"analyses"/"analyzes" stays out:
    no rule turns a plural noun into a verb)."""
    word = word.lower()
    found = set(spelling_variants(word)) | set(plural_variants(word))
    for variant in list(found):
        found |= set(spelling_variants(variant)) | set(plural_variants(variant))
    found.discard(word)
    return tuple(sorted(found))


def family_docs(counts: FormCounts) -> int:
    """Datasets with the asked word itself (or its plural or singular).

    The largest single count, a lower bound: the forms mostly occur together.
    """
    own = inflections(counts.word)
    return max((n for form, n in counts.forms if form in own), default=0)


def own_forms(word: str) -> Set[str]:
    """*word* in its plural and singular, and in its other spellings and numbers.

    Never judged against *word*: they are the word itself.
    """
    found = {word, *search_variants(word)}
    for form in list(found):
        found |= inflections(form)
    return found


def colliding_forms(counts: FormCounts) -> Tuple[str, ...]:
    """The other forms of the stem that bring in many datasets the word may not
    mean, most frequent first. Each is one pair to judge."""
    own = own_forms(counts.word)
    floor = max(MIN_EXTRA, (MIN_RATIO - 1) * max(family_docs(counts), 1))
    return tuple(form for form, n in counts.forms if form not in own and n >= floor)


def is_collision(counts: FormCounts) -> bool:
    """True when some other form of the stem collides with the word."""
    return bool(colliding_forms(counts))


# ── Reviewed list ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Group:
    """One `exact` group of word_forms.yaml."""
    forms: Tuple[str, ...]      # forms that mean the same
    excludes: FrozenSet[str]    # forms each of them is different from, in every spelling


def _safe_forms(values: object) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(f for f in (str(v).lower() for v in values) if _FORM_RE.fullmatch(f)))


@lru_cache(maxsize=1)
def _reviewed() -> Dict[str, Optional[_Group]]:
    """word → its group, or None where stemming is right. From word_forms.yaml."""
    # A broken list must not stop the search. UnicodeDecodeError is a ValueError.
    try:
        data = yaml.safe_load(_PATH.read_text()) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        log.warning("word_forms.yaml could not be read (%s) — ignoring it", exc)
        return {}
    if not isinstance(data, dict):
        log.warning("word_forms.yaml is not a mapping — ignoring it")
        return {}
    found: Dict[str, Optional[_Group]] = {}
    for entry in data.get("exact") or []:
        if not isinstance(entry, dict):
            continue
        forms = _safe_forms(entry.get("forms"))
        excludes = {f for x in _safe_forms(entry.get("excludes")) for f in (x, *search_variants(x))}
        group = _Group(forms, frozenset(excludes - set(forms)))
        for form in forms:
            found[form] = group
    for group in data.get("stemmed") or []:
        for word in group or []:
            found[str(word).lower()] = None
    return found


def _candidates(word: str) -> List[str]:
    """*word*, its other spellings and numbers, then their plurals and singulars."""
    spellings = [word, *search_variants(word)]
    return [*spellings, *sorted(set().union(*(inflections(w) for w in spellings)) - set(spellings))]


def _entry(word: str) -> Tuple[bool, Optional[_Group]]:
    """(listed, group) for *word*; group is None when the list says to stem it.

    A word in the other spelling or number counts as listed too: "localiser"
    gets the group of "localizer", or its stem would bring back "local" and
    "localized".
    """
    table = _reviewed()
    for candidate in _candidates(word):
        if candidate in table:
            return True, table[candidate]
    return False, None


# ── Pairs judged offline ──────────────────────────────────────────────────────

_NO_PAIRS: Tuple[FrozenSet[str], FrozenSet[str]] = (frozenset(), frozenset())


@lru_cache(maxsize=1)
def _stored_pairs() -> Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]:
    """asked word → (forms judged different, forms judged to keep matching by
    stem). From word_form_pairs.json; {} without it."""
    if not _PAIRS_PATH.exists():
        return {}
    # A broken file must not stop the search, as with word_forms.yaml.
    try:
        data = json.loads(_PAIRS_PATH.read_text())
    except (OSError, ValueError) as exc:
        log.warning("word_form_pairs.json could not be read (%s) — ignoring it", exc)
        return {}
    if not isinstance(data, dict):
        log.warning("word_form_pairs.json is not a mapping — ignoring it")
        return {}
    kinds = []
    for kind in ("different", "stem"):
        table = data.get(kind) or {}
        kinds.append({str(a).lower(): frozenset(_safe_forms(bs)) for a, bs in table.items()}
                     if isinstance(table, dict) else {})
    different, stem = kinds
    return {a: (different.get(a, frozenset()), stem.get(a, frozenset()))
            for a in set(different) | set(stem) if _FORM_RE.fullmatch(a)}


def _stored_for(word: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """(different, stem) stored for *word*, or for its other spelling or number."""
    table = _stored_pairs()
    for candidate in _candidates(word):
        if candidate in table:
            return table[candidate]
    return _NO_PAIRS


def reviewed(word: str) -> Tuple[bool, Optional[Tuple[str, ...]]]:
    """(listed, forms) for *word*: the forms its group names as the same, asked
    word first; forms is None when the list says to stem it."""
    listed, group = _entry(word)
    if group is None:
        return listed, None
    return True, _asked_first(word, group.forms)


def excluded_forms(word: str) -> FrozenSet[str]:
    """The forms judged different from *word*: by the reviewed list, and by
    the stored pairs where the list does not name the form as the same."""
    listed, group = _entry(word)
    if listed and group is None:
        return frozenset()      # the list says the stem is right for every form
    different = set(_stored_for(word)[0])
    if group is not None:
        different = (different - set(group.forms)) | group.excludes
    return frozenset(different - own_forms(word))


def _asked_first(word: str, forms: Iterable[str]) -> Tuple[str, ...]:
    return tuple(f for f in dict.fromkeys([word, *forms]) if _FORM_RE.fullmatch(f))


def allowed_forms(counts: FormCounts, different: Iterable[str]) -> Optional[Tuple[str, ...]]:
    """The forms a search for the word requires: the catalogue's forms of its
    stem except the ones judged *different*, asked word first, then by count.

    None when none of *different* is a form of the stem: the search then
    matches by stem, as before.
    """
    drop = set(different) - own_forms(counts.word)
    if not drop & {form for form, _ in counts.forms}:
        return None
    return _asked_first(counts.word, [form for form, _ in counts.forms if form not in drop])


# ── Deciding ──────────────────────────────────────────────────────────────────

def open_forms(counts: FormCounts) -> Tuple[str, ...]:
    """The colliding forms of the word that neither the reviewed list nor the
    stored pairs have judged against it, most frequent first."""
    listed, group = _entry(counts.word)
    if listed and group is None:
        return ()       # the list says the stem is right for every form
    judged = (set(group.forms) | group.excludes) if group else set()
    judged |= set().union(*_stored_for(counts.word))
    return tuple(form for form in colliding_forms(counts) if form not in judged)


def needs_decision(form_counts: Mapping[str, FormCounts]) -> List[FormCounts]:
    """The words the language model is asked about: those with open pairs."""
    return [e for e in form_counts.values() if open_forms(e)]


def prompt_block(form_counts: Sequence[FormCounts]) -> str:
    """The open forms of each word in *form_counts*, for the intent prompt.

    "" when there is nothing to ask.
    """
    lines = []
    for item in form_counts:
        shown = [(form, item.count(form)) for form in open_forms(item)[:PROMPT_FORMS]]
        if not shown:
            continue
        shown.append((item.word, item.count(item.word)))
        forms = ", ".join(f"{form} ({n})" for form, n in shown)
        lines.append(f"  {item.word}: {forms}")
    if not lines:
        return ""
    return ("Word forms in the catalogue. The search finds a word by its stem, so for each\n"
            "of these words of the query it also finds the other forms listed (in brackets:\n"
            "datasets that contain the form). Fill word_forms for them:\n" + "\n".join(lines))


def _complete(kept: Sequence[str], counts: FormCounts) -> List[str]:
    """*kept* plus the catalogue's regular inflections of each kept form."""
    wanted: Set[str] = set()
    for form in kept:
        wanted |= inflections(form)
    extra = [form for form, _ in counts.forms if form in wanted and form not in kept]
    return list(kept) + extra


def decide(form_counts: Mapping[str, FormCounts],
           choices: Iterable[Tuple[str, Sequence[str]]]) -> ExactForms:
    """The forms each word of the question must match in; {} for none.

    *choices* are the language model's (word, forms to keep) answers about the
    forms prompt_block showed. A form shown and not kept is different from the
    word; a form not shown was not judged and stays. The reviewed list's pairs
    count too, so a listed word of the question gets its forms here from the
    catalogue's forms of its stem.

    Anything in an answer that does not fit the counts is dropped, never an
    error: a word that was not asked about, a form the catalogue does not have
    under that stem. A choice that keeps every form shown restricts nothing.
    """
    different: Dict[str, Set[str]] = {}
    asked = {e.word: e for e in needs_decision(form_counts)}
    for word, keep in choices:
        word = str(word).strip().lower()
        item = asked.get(word)
        if item is None or word in different:
            continue
        known = {form for form, _ in item.forms}
        kept = [f for f in dict.fromkeys(str(k).strip().lower() for k in keep) if f in known]
        kept_all = set(_complete(_asked_first(word, kept), item))
        different[word] = {form for form in open_forms(item)[:PROMPT_FORMS] if form not in kept_all}
    decided: ExactForms = {}
    for item in form_counts.values():
        forms = allowed_forms(item, different.get(item.word, set()) | excluded_forms(item.word))
        if forms:
            decided[item.word] = forms
    return decided


# The catalogue's forms of one word, or None when it cannot say.
Catalogue = Callable[[str], Optional[FormCounts]]


def forms_for(word: str, decided: Optional[Mapping[str, Sequence[str]]] = None,
              catalogue: Optional[Catalogue] = None) -> Optional[Tuple[str, ...]]:
    """The forms a search for *word* requires, or None to search by stem.

    *decided*: the forms decide() gave for this question's words. A word it
    does not cover that has forms judged different (a synonym, a word the
    model wrote) gets the forms of its stem minus those, from *catalogue*.
    Without the catalogue the pairs cannot say what else the stem holds; a
    word on the reviewed list then matches the forms its group names, any
    other word its stem.
    """
    listed, group = _entry(word)
    if listed and group is None:
        return None
    different = excluded_forms(word)
    chosen = (decided or {}).get(word)
    if chosen:
        return _asked_first(word, [form for form in chosen if form not in different])
    if not different:
        return None
    counts = catalogue(word) if catalogue else None
    if counts is not None:
        return allowed_forms(counts, different)
    return _asked_first(word, group.forms) if group is not None else None
