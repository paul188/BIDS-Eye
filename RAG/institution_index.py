"""
RAG/institution_index.py
------------------------
Resolve an institution named in a user question ("Stanford", "MIT",
"Beijing Normal University") to the exact strings stored in
bids_datasets.institutions.

The stored names come from the InstitutionName field of DICOM headers, so the
same site appears in many spellings ('NIH_FMRIF', 'NIH FMRIF',
'Beijing Normal Universtiy/2773C4/'), and names are short codes as often as
full names. This is a string-matching problem, not a semantic one: an
embedding would place 'University of Washington' next to 'Washington
University' (two different institutions) and 'MIT' nowhere near 'McGovern
Institute'. Matching works on normalised word tokens instead:

  1. Normalise both sides: fold accents and German umlaut spellings
     ('Universitaet' = 'Universität' = 'Universitat'), lower-case, split on
     anything that is not a letter or digit, rejoin possessives ('Queen s').
  2. Expand the user term with a small curated alias table
     (MIT ↔ Massachusetts Institute of Technology ↔ McGovern Institute).
  3. Phrase match: the term's tokens must occur as consecutive whole tokens
     of a stored name, in order. Whole tokens keep 'MIT' from matching
     'Brains Unlimited'; order keeps 'Washington University' apart from
     'University of Washington'. Tokens of six or more letters may differ by
     one edit, which absorbs typos in the stored data ('Universtiy',
     'Nortwestern').
  4. Only if no phrase matches: every distinctive token of the term
     (generic words such as 'university' or 'institute' removed) must occur
     in the name, in any order. 'Oxford University' then finds
     'University of Oxford'.

The resolved strings become an exact array filter
(d.institutions && ARRAY[...]) that the GIN index on the column serves; see
LLM_preprocessor/preprocess/. Terms that resolve to nothing fall back to
full-text search via text_alternatives().

The names are loaded from the database at backend startup
(refresh_from_db), with RAG/name_index.json as the fallback source.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rapidfuzz.distance import OSA

log = logging.getLogger(__name__)

Tokens = Tuple[str, ...]

# A term that matches more stored names than this is too vague to be useful
# as a filter (it would mostly match by accident).
MAX_NAMES_PER_TERM = 60

# Tokens this long may differ by one edit (insertion, deletion, substitution
# or transposition of adjacent letters).
_MIN_FUZZY_TOKEN = 6

# Text fallback: phrases this short are left out, because full-text search
# cannot tell the acronym from an ordinary word ('MIT' is German for 'with').
_MIN_TEXT_PHRASE = 4

SCORE_PHRASE = 1.0
SCORE_PHRASE_FUZZY = 0.9
ALIAS_FACTOR = 0.95
SCORE_TOKENS = 0.8

_UMLAUT_SPELLINGS = (("ae", "a"), ("oe", "o"), ("ue", "u"))

# Dropped on both sides, so 'Max Planck Institute in Leipzig' can match
# 'Max Planck Institute, Leipzig' and 'University of Texas at Austin' still
# matches 'Univ of Texas Austin'.
_PLACE_CONNECTORS = frozenset({"in", "at"})

# Words that say what kind of place it is, not which one.
_GENERIC_TOKENS = frozenset("""
    a an and at de del della der des di du el en et for fur in la le of on the und von zu
    univ uni university universitat universite universidad universita universidade
    universitetet universiteit universitatsklinikum universitatsspital
    institute institut instituto istituto institution institutes
    hospital hospitals hosp clinic clinics clinical klinik klinikum clinica
    center centre centrum centro ctr college school faculty campus department dept
    medical medicine med research lab labs laboratory laboratories unit facility group
    imaging brain neuroimaging neuroscience neurosciences science sciences health
    national state city foundation fundacion technology technical
    mri fmri meg eeg pet scanner
""".split())

# Each line is one institution under several names. A user term equal to one
# entry is searched under all of them. Entries are normalised like the data.
# Keep entries unambiguous: 'penn' (Penn State?), 'uva' (Virginia or
# Amsterdam?) and 'ntu' (Nanyang or Taiwan?) are left out on purpose.
_ALIAS_GROUPS: Sequence[Sequence[str]] = (
    ("mit", "massachusetts institute of technology", "mcgovern institute"),
    ("nih", "national institutes of health"),
    ("nimh", "national institute of mental health"),
    ("ninds", "national institute of neurological disorders and stroke"),
    ("niaaa", "national institute on alcohol abuse and alcoholism"),
    ("nyu", "new york university"),
    ("ucl", "university college london", "wellcome centre for human neuroimaging",
     "wellcome trust centre for neuroimaging", "wtcn"),
    ("ucla", "university of california los angeles"),
    ("ucsd", "uc san diego", "university of california san diego"),
    ("ucsb", "uc santa barbara", "university of california santa barbara"),
    ("uc berkeley", "university of california berkeley"),
    ("ucsf", "uc san francisco", "university of california san francisco"),
    ("uci", "uc irvine", "university of california irvine"),
    ("usc", "university of southern california"),
    ("caltech", "california institute of technology"),
    ("mgh", "massachusetts general hospital", "mass general hospital", "martinos center"),
    ("washu", "wustl", "wash u", "washington university in st louis", "washington university"),
    ("upenn", "university of pennsylvania", "hospital univ of pennsylvania", "hup"),
    ("unc", "unc chapel hill", "university of north carolina", "univ of north carolina"),
    ("ut austin", "university of texas at austin", "univ of texas at austin"),
    ("utsw", "ut southwestern"),
    ("kcl", "kings college london"),
    ("mpi", "max planck institute", "max planck institut"),
    ("mpi cbs", "mpi leipzig", "max planck institute leipzig", "max planck leipzig",
     "max planck institute for human cognitive and brain sciences",
     "max planck institut fuer kognitions und neurowissenschaften"),
    ("mrc cbu", "mrc cognition and brain sciences unit", "mrc cognition brain sciences unit"),
    ("ohsu", "o h s u", "oregon health and science university"),
    ("ku leuven", "kuleuven", "kul", "leuven"),
    ("ghent", "gent", "ghent university", "ugent", "uz gent"),
    ("tum", "technical university of munich", "technische universitat munchen",
     "klinikum rechts der isar"),
    ("epfl", "ecole polytechnique federale de lausanne"),
    ("eth", "eth zurich"),
    ("uzh", "university of zurich", "universitat zurich"),
    ("bnu", "beijing normal university"),
    ("pku", "peking university"),
    ("skku", "sungkyunkwan university"),
    ("cmu", "carnegie mellon university"),
    ("donders", "donders institute", "radboud university", "radboudumc"),
    ("chuv", "lausanne university hospital", "centre hospitalier universitaire vaudois"),
    ("cubric", "cardiff university"),
    ("uiowa", "university of iowa", "uihc", "ui hospitals and clinics", "iowa mrrf"),
    ("uw madison", "university of wisconsin madison", "waisman"),
    ("umich", "university of michigan", "u of m"),
    ("cimcyc", "university of granada"),
    ("cmrr", "center for magnetic resonance research", "university of minnesota"),
    ("mcgill", "mc gill", "mcgill university", "mcconnell brain imaging centre",
     "mc connell brain imaging centre", "montreal neurological institute"),
    ("ubc", "university of british columbia"),
    ("ucdenver", "cu denver", "university of colorado denver"),
    ("cuny", "city college of new york", "city university of new york"),
    ("jhu", "johns hopkins", "johns hopkins university"),
    ("polyu", "hong kong polytechnic university"),
    ("uq", "university of queensland", "queensland brain institute"),
    ("oxford", "oxford university", "university of oxford", "john radcliffe hospital",
     "warneford hospital"),
    ("neurospin", "cea neurospin"),
    ("mount sinai", "mt sinai", "icahn school of medicine at mount sinai"),
    ("musc", "medical university of south carolina"),
    ("unm", "university of new mexico"),
    ("byu", "brigham young university"),
    ("usf", "university of south florida"),
    ("osu", "ohio state university", "ohio state univ"),
    ("psu", "penn state", "pennsylvania state university"),
    ("fiu", "florida international university"),
    ("umd", "university of maryland"),
    ("uoft", "university of toronto"),
    ("tau", "tel aviv university"),
    ("huji", "hebrew university"),
    ("weizmann", "weizmann institute", "weizmann institute of science"),
    ("snu", "seoul national university", "snubic"),
    ("bcbl", "basque center on cognition brain and language"),
    ("umc utrecht", "university medical center utrecht"),
    ("umcg", "university medical center groningen"),
    ("fu berlin", "freie universitat berlin"),
    ("tu berlin", "technische universitat berlin"),
)

# One-way aliases: a parent institution that scans under the names of its
# hospitals and centres. Scanner headers almost never say "Harvard"; they say
# 'MGH 7T', 'Martinos Center Bay 4 Prisma' or 'McLean Hospital BIC'. A user
# term equal to a key is searched under all its members, but a member does
# not pull in the others: "MGH" must not match McLean Hospital.
_UMBRELLAS: Dict[str, Sequence[str]] = {
    "harvard": (
        "harvard medical school", "harvard university",
        "massachusetts general hospital", "mass general hospital", "mgh",
        "martinos center", "mclean hospital",
        "mass general brigham", "mgb", "brigham and womens hospital", "bwh",
        "boston childrens hospital", "beth israel deaconess medical center", "bidmc",
        "dana farber cancer institute", "spaulding rehabilitation hospital",
        "massachusetts eye and ear", "joslin diabetes center",
    ),
}
for _alias in ("harvard university", "harvard medical school", "hms"):
    _UMBRELLAS[_alias] = _UMBRELLAS["harvard"]


# ── Normalisation ─────────────────────────────────────────────────────────────

def _fold_token(token: str) -> str:
    for spelled, plain in _UMLAUT_SPELLINGS:
        token = token.replace(spelled, plain)
    return token


def tokens(text: str) -> Tokens:
    """Normalised word tokens of an institution name or a user term.

    'Max-Planck-Institut für Kognitions_ und Neurowissenschaften'
        → ('max', 'planck', 'institut', 'fur', 'kognitions', 'und', 'neurowissenschaften')
    """
    text = unicodedata.normalize("NFKD", (text or "").replace("ß", "ss"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"['’`]", "", text)   # Queen's → queens
    raw = re.findall(r"[a-z0-9]+", text)
    merged: List[str] = []
    for token in raw:
        if token == "s" and merged:      # 'Queen s University' (apostrophe lost upstream)
            merged[-1] += "s"
        elif token not in _PLACE_CONNECTORS:
            merged.append(token)
    return tuple(_fold_token(token) for token in merged)


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def _distinctive(toks: Tokens) -> Tokens:
    return tuple(t for t in toks if t not in _GENERIC_TOKENS and len(t) >= 3 and not t.isdigit())


def _same_token(query: str, stored: str) -> Optional[bool]:
    """True: identical. False: one edit apart (long tokens only). None: different."""
    if query == stored:
        return True
    if (len(query) >= _MIN_FUZZY_TOKEN and len(stored) >= _MIN_FUZZY_TOKEN
            and OSA.distance(query, stored, score_cutoff=1) <= 1):
        return False
    return None


def _phrase_score(query: Tokens, stored: Tokens) -> float:
    """Score of *query* occurring as consecutive tokens of *stored*; 0 if not."""
    n = len(query)
    best = 0.0
    for start in range(len(stored) - n + 1):
        exact = True
        for offset, token in enumerate(query):
            same = _same_token(token, stored[start + offset])
            if same is None:
                break
            exact = exact and same
        else:
            best = max(best, SCORE_PHRASE if exact else SCORE_PHRASE_FUZZY)
            if best == SCORE_PHRASE:
                break
    return best


def _contains_all(query: Tokens, stored: Tokens) -> bool:
    return all(any(_same_token(t, s) is not None for s in stored) for t in query)


# ── Alias table ───────────────────────────────────────────────────────────────

# Alias entry → the other entries of its group, as (tokens, written form).
def _build_aliases() -> Dict[Tokens, List[Tuple[Tokens, str]]]:
    aliases: Dict[Tokens, List[Tuple[Tokens, str]]] = {}
    for group in _ALIAS_GROUPS:
        members = list({tokens(entry): entry for entry in group}.items())
        for member, _ in members:
            others = aliases.setdefault(member, [])
            for other in members:
                if other[0] != member and other not in others:
                    others.append(other)
    return aliases


_ALIASES = _build_aliases()
_UMBRELLA_MEMBERS: Dict[Tokens, List[Tuple[Tokens, str]]] = {
    tokens(parent): [(tokens(member), member) for member in members]
    for parent, members in _UMBRELLAS.items()
}


def _variants(term: str) -> List[Tuple[Tokens, float, str]]:
    """The term itself, then its aliases: (tokens, score factor, written form)."""
    own = tokens(term)
    written = (term or "").strip()
    if own and own[0] == "the":
        own = own[1:]
        written = re.sub(r"(?i)^the\s+", "", written)
    if not own:
        return []
    found = [(own, 1.0, written)]
    for alias, alias_written in _ALIASES.get(own, []) + _UMBRELLA_MEMBERS.get(own, []):
        if all(alias != f[0] for f in found):
            found.append((alias, ALIAS_FACTOR, alias_written))
    return found


def text_alternatives(term: str) -> List[str]:
    """Phrases for a full-text fallback search on the dataset text.

    The user's own term plus the long forms of its aliases, as written (the
    umlaut folding used for matching is not applied). Phrases shorter than
    four characters are dropped: as free text they collide with ordinary
    words, while the structured filter matches them as whole name tokens.
    """
    phrases: List[str] = []
    seen = set()
    for variant, _, written in _variants(term):
        if len("".join(variant)) >= _MIN_TEXT_PHRASE and variant not in seen:
            seen.add(variant)
            phrases.append(written)
    return phrases


# ── Index ─────────────────────────────────────────────────────────────────────

class InstitutionIndex:
    """Exact institution strings from the database, searchable by user term."""

    def __init__(self, names: Iterable[str]):
        self._entries: List[Tuple[str, Tokens]] = []
        seen = set()
        for name in names:
            if not isinstance(name, str) or not name.strip() or name in seen:
                continue
            seen.add(name)
            toks = tokens(name)
            if toks:
                self._entries.append((name, toks))

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, term: str, limit: int = MAX_NAMES_PER_TERM) -> List[Tuple[str, float]]:
        """Stored names for *term*, best first, as (exact DB string, score 0..1)."""
        variants = _variants(term)
        if not variants:
            return []

        scores: Dict[str, float] = {}
        for variant, factor, _ in variants:
            for name, stored in self._entries:
                score = _phrase_score(variant, stored) * factor
                if score > scores.get(name, 0.0):
                    scores[name] = score

        if not scores:
            # No phrase anywhere: fall back to the distinctive words, any order.
            for variant, factor, _ in variants:
                key = _distinctive(variant)
                if not key:
                    continue
                for name, stored in self._entries:
                    if _contains_all(key, stored):
                        score = SCORE_TOKENS * factor
                        if score > scores.get(name, 0.0):
                            scores[name] = score

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) > limit:
            log.info("Institution term %r matches %d names — too vague, ignored", term, len(ranked))
            return []
        return ranked


# ── Shared instance ───────────────────────────────────────────────────────────

_index: Optional[InstitutionIndex] = None


def set_names(names: Iterable[str]) -> InstitutionIndex:
    """Replace the shared index (startup, tests)."""
    global _index
    _index = InstitutionIndex(names)
    return _index


def get_index(fallback_names: Iterable[str] = ()) -> InstitutionIndex:
    """The shared index; built from *fallback_names* if the database load did not run."""
    global _index
    if _index is None:
        _index = InstitutionIndex(fallback_names)
    return _index


async def refresh_from_db(engine: Any) -> int:
    """Load the distinct institution names from bids_datasets. Returns the count.

    On any error (column missing, database down) the index is left unset, and
    the retriever falls back to RAG/name_index.json.
    """
    import asyncio

    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError
    from services.schema.tables import datasets

    try:
        async with engine.connect() as connection:
            rows = await connection.execute(text(
                f"SELECT DISTINCT inst FROM {datasets.name} d, unnest(d.institutions) AS inst "
                "WHERE btrim(inst) <> ''"
            ))
            names = [row[0] for row in rows]
    # A refused connection can reach us as a bare OSError (asyncpg), a
    # connect timeout as asyncio.TimeoutError.
    except (SQLAlchemyError, OSError, asyncio.TimeoutError) as exc:
        log.warning("Could not load institution names (%s) — using name_index.json", exc)
        return 0
    set_names(names)
    log.info("Loaded %d distinct institution names for query matching", len(names))
    return len(names)
