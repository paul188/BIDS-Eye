"""
benchmarks/phrases/expander.py
------------------------------
Spell out what a sentence says in shortened form, so that a term the text
means is also written as the plain phrase.

    "the temporal and frontal cortices"   -> temporal cortices; frontal cortices
    "superior and middle temporal gyri"   -> superior temporal gyri; middle temporal gyri
    "gray/white matter"                   -> gray matter; white matter
    "the temporal lobe and pole"          -> temporal pole            (attested pairs only)
    "healthy young adults"                -> healthy adults           (attested pairs only)
    "Alzheimer's disease"                 -> Alzheimer disease
    "ventral temporal (VT) cortex"        -> ventral temporal cortex
    "eyes open or closed"                 -> eyes closed
    "participants watched short movies"   -> movies watched           (the stem of "movie watching")
    "corrected for head motion"           -> head motion corrected
    "fronto-parietal" / "frontoparietal"  -> the other spelling
    "white matter (WM) ... WM volume"     -> white matter volume      (definitions in the same text)

The input is a parsed sentence: a list of tokens [text, lemma, pos, dep,
head, space] as benchmarks/phrases/analysis/grammar/parse.py writes them
(spaCy en_core_web_sm; head is the index of the head token in the sentence).
The output is a list of Expansion(kind, text, start, end). The text is meant
to be indexed next to the original, so a phrase search finds the term; it is
never shown to anyone.

Two constructions guess more than they read (a shared noun after a list of
nouns, a modifier separated from its noun). They are emitted only when the
resulting two-word pair is written adjacent somewhere in the corpus: pass
`attested`, a set of (word, word) lemma pairs. Without it they are skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

TEXT, LEMMA, POS, DEP, HEAD, SPACE = range(6)

NOUNISH = {"NOUN", "PROPN"}
MODIFIER_POS = {"ADJ", "NOUN", "PROPN", "VERB", "NUM", "ADV"}
MODIFIER_DEPS = {"amod", "compound", "nmod", "npadvmod", "nummod", "advmod", "conj", "cc", "punct", "poss", "case"}
SEPARATORS = {",", "and", "or", "/", "&", "and/or", "+"}

# Words that name a position. A list item made only of them borrows the rest of
# the next item: "superior and middle temporal gyri" -> superior temporal gyri.
POSITIONAL = set("""
left right bilateral ipsilateral contralateral superior inferior middle mid anterior posterior medial lateral
dorsal ventral rostral caudal dorsolateral ventrolateral dorsomedial ventromedial anteromedial posteromedial
anterolateral posterolateral mesial basal apical upper lower inner outer deep superficial primary secondary
early late proximal distal
""".split())

# Verbs whose object says nothing a search would ask for.
LIGHT_VERBS = set("""
be have do use include contain show find provide see report allow describe give get need want let become seem
appear consider indicate suggest note mean represent require follow refer
""".split())
VERB_PREPS = {"to", "for", "at", "on", "with"}
UNITS = set("hz khz ms millisecond second sec minute min hour day week month year mm cm m t tesla mhz db %".split())

# Region words a fused compound can end in: "frontoparietal" = fronto + parietal.
FUSABLE = set("""
temporal frontal parietal occipital cortical spinal striatal thalamic cerebellar limbic insular cingulate
hippocampal central spatial motor sensory medial lateral
""".split())


_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'’]*$")

# List items that only count or point ("other trials", "both groups").
QUANTIFIERS = set("""
other such same more fewer less many several both each all any first second third last few some various
different respective additional further following previous next own certain
""".split())


def _clean(text: str, min_words: int = 2) -> bool:
    """Plain words only: no paths, URLs, numbers, code or markup."""
    parts = [w for w in re.split(r"[ \-‐]+", text) if w]
    return (text[:1].isalpha() and min_words <= len(parts) <= 7
            and all(_WORD_RE.match(w) and 2 <= len(w) <= 30 for w in parts))


def _stem(w: str) -> str:
    """Crude suffix stripping, enough to see that 'watched' and 'watching' are one verb."""
    w = w.lower()
    for suf in ("ations", "ation", "ions", "ion", "ings", "ing", "ments", "ment", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


@dataclass(frozen=True)
class Expansion:
    kind: str
    text: str
    start: int      # first token the expansion is read from
    end: int        # last token


class Sentence:
    """Tokens of one parsed sentence with the tree around them."""

    def __init__(self, toks: Sequence[Sequence]):
        self.t = toks
        self.n = len(toks)
        self.kids: Dict[int, List[int]] = {}
        for i, tok in enumerate(toks):
            h = tok[HEAD]
            if h != i and 0 <= h < self.n:
                self.kids.setdefault(h, []).append(i)

    def text(self, i: int) -> str:
        return self.t[i][TEXT]

    def low(self, i: int) -> str:
        return self.t[i][TEXT].lower()

    def lemma(self, i: int) -> str:
        return self.t[i][LEMMA].lower()

    def pos(self, i: int) -> str:
        return self.t[i][POS]

    def dep(self, i: int) -> str:
        return self.t[i][DEP]

    def head(self, i: int) -> int:
        return self.t[i][HEAD]

    def ancestors(self, i: int) -> List[int]:
        out, seen = [], {i}
        while True:
            h = self.head(i)
            if h == i or h in seen or not (0 <= h < self.n):
                return out
            out.append(h)
            seen.add(h)
            i = h

    def noun_owner(self, i: int) -> Optional[int]:
        """The first noun at or above i, climbing only through modifier links."""
        while True:
            if self.pos(i) in NOUNISH and self.dep(i) not in ("compound", "nmod", "poss"):
                return i
            if self.dep(i) not in MODIFIER_DEPS:
                return i if self.pos(i) in NOUNISH else None
            h = self.head(i)
            if h == i:
                return i if self.pos(i) in NOUNISH else None
            i = h

    def subtree(self, i: int) -> List[int]:
        out, stack = [], [i]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(self.kids.get(x, ()))
        return sorted(out)

    def join(self, idx: Iterable[int]) -> str:
        """Tokens with their original spacing where they are contiguous, a space elsewhere."""
        idx = list(idx)
        out = []
        for k, i in enumerate(idx):
            out.append(self.text(i))
            if k + 1 < len(idx):
                out.append(" " if idx[k + 1] != i + 1 or self.t[i][SPACE] else "")
        return "".join(out)


def _is_sep(s: Sentence, i: int) -> bool:
    return s.low(i) in SEPARATORS or s.pos(i) == "CCONJ"


def _left_region(s: Sentence, h: int) -> List[int]:
    """Tokens just left of noun h that belong to its premodifiers (lists included)."""
    region: List[int] = []
    i = h - 1
    while i >= 0:
        tok_ok = (s.pos(i) in MODIFIER_POS or _is_sep(s, i) or s.text(i) in ("-", "‐", "'s", "’s"))
        if not tok_ok or s.pos(i) == "DET":
            break
        # a token belongs to h only if its chain of heads reaches h through modifier links
        j, ok = i, False
        for _ in range(8):
            if j == h:
                ok = True
                break
            if s.dep(j) not in MODIFIER_DEPS:
                break
            nj = s.head(j)
            if nj == j:
                break
            j = nj
        if not ok and not _is_sep(s, i):
            break
        region.insert(0, i)
        i -= 1
    while region and _is_sep(s, region[0]):
        region.pop(0)
    return region


def _segments(s: Sentence, region: List[int]) -> List[List[int]]:
    segs, cur = [], []
    for i in region:
        if _is_sep(s, i):
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        segs.append(cur)
    return segs


def _coordinated_modifiers(s: Sentence, h: int, attested=None) -> List[Expansion]:
    if h > 0 and s.text(h - 1) in ("-", "‐"):
        return []             # "six or seven time-points": the list belongs to "time"
    region = _left_region(s, h)
    seps = [i for i in region if _is_sep(s, i)]
    if not seps:
        return []
    # the list must belong to h and not to a noun inside the region ("gray/white matter boundary":
    # the list belongs to "matter")
    for i in seps:
        left, right = i - 1, i + 1
        if left < 0 or right > h:
            return []
        common = set([left] + s.ancestors(left)) & set([right] + s.ancestors(right))
        owner = None
        for a in [right] + s.ancestors(right):
            if a in common:
                owner = a
                break
        if owner is None:
            return []
        nown = owner if s.pos(owner) in NOUNISH and owner <= h else s.noun_owner(owner)
        if nown is not None and nown < h and s.pos(nown) in NOUNISH and s.dep(nown) != "conj":
            return []         # a list item the tagger called a noun ("intraparietal") is not an owner
    segs = _segments(s, region)
    # counts and quantifiers in front of the first item belong to the whole list ("five healthy, young adults")
    while segs and segs[0] and (s.pos(segs[0][0]) == "NUM" or s.low(segs[0][0]) in QUANTIFIERS):
        segs[0] = segs[0][1:]
    segs = [g for g in segs if g]
    if len(segs) < 2:
        return []
    out = []
    for k, seg in enumerate(segs):
        words = list(seg)
        if all(s.low(i) in POSITIONAL for i in words):
            # borrow the content words of the next item that has some
            for later in segs[k + 1:]:
                tail = [i for i in later if s.low(i) not in POSITIONAL]
                if tail:
                    first_content = later.index(tail[0])
                    words = words + later[first_content:]
                    break
        if k == len(segs) - 1:
            continue          # the last item already stands next to the noun
        if all(s.low(i) in QUANTIFIERS for i in seg) or any(s.pos(i) in ("ADV", "NUM") for i in seg):
            continue
        last = seg[-1]
        if s.pos(last) in NOUNISH and (attested is None or (s.lemma(last), s.lemma(h)) not in attested):
            continue          # "age, gender, body mass index" is a list of nouns, not of modifiers
        out.append(Expansion("list", s.join(words + [h]), region[0], h))
    return out


def _coordinated_heads(s: Sentence, h: int, attested) -> List[Expansion]:
    """'the temporal lobe and pole' -> temporal pole, if 'temporal pole' is written somewhere."""
    if attested is None:
        return []
    mods = [m for m in s.kids.get(h, ()) if m < h and s.dep(m) in ("amod", "compound", "nmod")]
    if not mods:
        return []
    out = []
    for c in s.kids.get(h, ()):
        if c <= h or s.dep(c) != "conj" or s.pos(c) not in NOUNISH:
            continue
        if any(m < c and s.dep(m) in ("amod", "compound", "nmod", "det", "poss") for m in s.kids.get(c, ())):
            continue
        between = range(h + 1, c)
        if not all(_is_sep(s, i) for i in between) or len(between) > 2:
            continue
        near = max(mods)
        if "PROPN" in (s.pos(near), s.pos(h), s.pos(c)):
            continue          # names: "NIMH Data Science & Sharing Team"
        if (s.lemma(near), s.lemma(c)) not in attested:
            continue
        region = _left_region(s, h)
        if not region or any(_is_sep(s, i) for i in region):
            continue
        out.append(Expansion("shared_modifier", s.join(region + [c]), region[0], c))
    return out


def _noun_list(s: Sentence, x: int, attested) -> List[Expansion]:
    """'word and picture naming' parsed as two nouns -> word naming, if 'word naming' is written somewhere."""
    if attested is None or any(m < x and s.dep(m) in ("amod", "compound", "nmod") for m in s.kids.get(x, ())):
        return []
    out = []
    for c in s.kids.get(x, ()):
        if c <= x or s.dep(c) != "conj" or s.pos(c) not in NOUNISH:
            continue
        mods = [m for m in s.kids.get(c, ()) if x < m < c and s.dep(m) in ("amod", "compound", "nmod")]
        if s.pos(x) == "ADJ" and len(mods) != 1:
            continue
        if not mods or not all(_is_sep(s, i) for i in range(x + 1, min(mods))):
            continue
        if (s.lemma(x), s.lemma(c)) in attested:
            out.append(Expansion("list", f"{s.text(x)} {s.text(c)}", x, c))
    return out


def _skipped_modifiers(s: Sentence, h: int, attested, fixed=None) -> List[Expansion]:
    """'healthy young adults' -> healthy adults, if 'healthy adults' is written somewhere."""
    if attested is None:
        return []
    region = _left_region(s, h)
    while region and (s.pos(region[0]) == "NUM" or s.low(region[0]) in QUANTIFIERS):
        region = region[1:]      # "40 healthy young adults"
    if not region or any(_is_sep(s, i) or s.pos(i) in ("NUM", "PUNCT") for i in region):
        return []
    direct = [m for m in s.kids.get(h, ()) if m < h and m in region and s.dep(m) in ("amod", "compound", "nmod")]
    out = []
    for m in direct:
        if m == h - 1 or s.low(m) in POSITIONAL:
            continue          # "superior temporal gyrus" is not a "superior gyrus"
        inner = [i for i in region if i > max(s.subtree(m)) or (i > m and i not in s.subtree(m))]
        inner = [i for i in inner if i < h]
        if any(s.pos(i) == "VERB" for i in inner) or (fixed and any((s.lemma(i), s.lemma(h)) in fixed for i in inner)):
            continue          # "verbal working memory" is not "verbal memory"
        if (s.lemma(m), s.lemma(h)) not in attested or (s.pos(m) == "PROPN" and s.pos(h) == "PROPN"):
            continue
        sub = [i for i in s.subtree(m) if i < h and i in region]
        out.append(Expansion("skip", s.join(sub + [h]), sub[0], h))
    return out


def _possessive(s: Sentence, h: int) -> List[Expansion]:
    region = _left_region(s, h)
    case = [i for i in region if s.dep(i) == "case" and s.low(i) in ("'s", "’s", "'")]
    if not case:
        return []
    keep = [i for i in region if i not in case and not _is_sep(s, i)]
    if not keep or not any(s.pos(i) == "PROPN" or s.text(i)[:1].isupper() for i in keep):
        return []
    return [Expansion("possessive", s.join(keep + [h]), region[0], h)]


def _parenthetical(s: Sentence) -> List[Expansion]:
    """'ventral temporal (VT) cortex' -> ventral temporal cortex."""
    out = []
    for i in range(s.n):
        if s.text(i) != "(":
            continue
        close = next((j for j in range(i + 1, min(s.n, i + 8)) if s.text(j) == ")"), None)
        if close is None or i == 0 or close + 1 >= s.n:
            continue
        left, right = i - 1, close + 1
        if s.pos(left) not in ("ADJ", "NOUN", "PROPN") or s.pos(right) not in ("ADJ", "NOUN", "PROPN"):
            continue
        h = s.noun_owner(right)
        if h is None or h < right or h - right > 3:
            continue
        if h not in [left] + s.ancestors(left) and s.noun_owner(left) != h:
            continue
        start = left
        while start > 0 and s.pos(start - 1) in ("ADJ", "NOUN", "PROPN") and h in s.ancestors(start - 1):
            start -= 1
        words = list(range(start, i)) + list(range(right, h + 1))
        out.append(Expansion("parenthesis", s.join(words), start, h))
    return out


def _predicative(s: Sentence, i: int) -> bool:
    return s.pos(i) == "ADJ" and s.dep(i) in ("advcl", "acomp", "xcomp", "oprd", "amod", "conj", "ccomp")


def _postposed(s: Sentence) -> List[Expansion]:
    """'eyes open or closed' -> eyes closed."""
    out = []
    for i in range(1, s.n - 2):
        n = i - 1
        if s.pos(n) not in NOUNISH or not _predicative(s, i):
            continue
        if s.head(n) == i or s.head(i) >= i:
            continue          # "participants memorized and repeated", "visual and auditory oddball"
        for c in s.kids.get(i, ()):
            if c > i and s.dep(c) == "conj" and _predicative(s, c) and c - i <= 3 \
                    and all(_is_sep(s, k) for k in range(i + 1, c)):
                out.append(Expansion("postposed", f"{s.text(i - 1)} {s.text(c)}", i - 1, c))
    return out


def _noun_core(s: Sentence, n: int) -> List[int]:
    """The noun with its compound modifiers: 'head motion'."""
    core = [n]
    for m in sorted(s.kids.get(n, ()), reverse=True):
        if m < n and s.dep(m) == "compound" and m == core[0] - 1:
            core.insert(0, m)
    return core


def _verb_object(s: Sentence, compounds) -> List[Expansion]:
    """'watched short movies' -> movies watched; 'stimuli were presented' -> stimuli presented."""
    out = []
    for v in range(s.n):
        if s.pos(v) not in ("VERB", "ADJ") or s.lemma(v) in LIGHT_VERBS:
            continue
        if s.pos(v) == "ADJ" and not s.low(v).endswith("ed"):
            continue
        if s.dep(v) in ("amod", "compound", "npadvmod", "nmod", "acl", "advcl", "relcl"):
            continue                # "Accompanying data" is an adjective use
        objs = []
        for c in s.kids.get(v, ()):
            if s.dep(c) in ("dobj", "nsubjpass") and s.pos(c) in NOUNISH:
                objs.append(c)
            elif s.dep(c) == "prep" and s.low(c) in VERB_PREPS and c > v:
                objs.extend(p for p in s.kids.get(c, ()) if s.dep(p) == "pobj" and s.pos(p) in NOUNISH)
        for o in objs:
            if s.lemma(o) in UNITS:
                continue
            core = _noun_core(s, o)
            if core[-1] == v - 1:
                continue            # already "object verb" in the text
            if o == v + 1 or core[0] == v + 1:
                continue            # "encoding phase" is a noun compound, not "encoding the phase"
            if compounds is None or (s.lemma(o), _stem(s.text(v))) not in compounds:
                continue            # only where "movie watching" is written somewhere
            out.append(Expansion("verb_object", s.join(core) + " " + s.text(v), min(core[0], v), max(o, v)))
    return out


_FUSED_RE = re.compile(r"^([a-z]+o)(" + "|".join(sorted(FUSABLE, key=len, reverse=True)) + r")$")


def _fused(s: Sentence) -> List[Expansion]:
    out = []
    for i in range(s.n):
        w = s.low(i)
        if i + 2 < s.n and s.text(i + 1) in ("-", "‐") and w.endswith("o") and s.low(i + 2) in FUSABLE:
            out.append(Expansion("fused", w + s.low(i + 2), i, i + 2))
        m = _FUSED_RE.match(w)
        if m and len(m.group(1)) >= 4:
            out.append(Expansion("fused", f"{m.group(1)}-{m.group(2)}", i, i))
    return out


def _slash_split(s: Sentence) -> List[Expansion]:
    """'gray/white matter' when the tokenizer kept 'gray/white' as one token."""
    out = []
    for i in range(s.n - 1):
        w = s.text(i)
        if "/" in w and w.count("/") == 1 and s.pos(i + 1) in NOUNISH:
            a, b = w.split("/")
            if a.isalpha() and b.isalpha() and len(a) > 2 and len(b) > 2:
                out.append(Expansion("list", f"{a} {s.text(i + 1)}", i, i + 1))
    return out


def expand_sentence(toks: Sequence[Sequence], attested: Optional[Set[Tuple[str, str]]] = None,
                    abbreviations: Optional[Dict[str, str]] = None,
                    compounds: Optional[Set[Tuple[str, str]]] = None,
                    fixed: Optional[Set[Tuple[str, str]]] = None) -> List[Expansion]:
    """Expansions of one sentence.

    attested   (modifier, noun) lemma pairs the corpus writes adjacent; gates the guessing constructions
    abbreviations  {short: long} defined in the same text
    compounds  (noun, verb stem) pairs written as a noun compound; gates the verb-object rewording
    fixed      (word, noun) lemma pairs that are one term ("working memory"); never split by a skip
    """
    s = Sentence(toks)
    out: List[Expansion] = []
    for h in range(s.n):
        if s.pos(h) == "ADJ" and s.dep(h) in ("pobj", "dobj", "nsubj", "nsubjpass", "conj"):
            out += _noun_list(s, h, attested)      # "grey + white matter" parsed with "grey" as the noun
        if s.pos(h) not in NOUNISH:
            continue
        out += _coordinated_modifiers(s, h, attested)
        out += _coordinated_heads(s, h, attested)
        out += _noun_list(s, h, attested)
        out += _skipped_modifiers(s, h, attested, fixed)
        out += _possessive(s, h)
    out += _parenthetical(s)
    out += _postposed(s)
    out += _verb_object(s, compounds)
    out += _fused(s)
    out += _slash_split(s)
    if abbreviations:
        out += _abbreviations(s, abbreviations)
    seen, uniq = set(), []
    for e in out:
        key = (e.kind, e.text.lower())
        if key not in seen and _clean(e.text, 1 if e.kind == "fused" else 2):
            seen.add(key)
            uniq.append(e)
    return uniq


# ---------------------------------------------------------------- abbreviations

_DEF_RE = re.compile(r"((?:[A-Za-z0-9][\w'’./-]*[ ‐-]){1,10})\(\s*([A-Za-z][A-Za-z0-9-]{1,9}?)s?\s*[);,]")


def _long_form(before: str, short: str) -> Optional[str]:
    """Schwartz & Hearst (2003): the shortest preceding words whose letters cover the abbreviation."""
    si, li = len(short) - 1, len(before) - 1
    while si >= 0:
        c = short[si].lower()
        if not c.isalnum():
            si -= 1
            continue
        while li >= 0 and (before[li].lower() != c or (si == 0 and li > 0 and before[li - 1].isalnum())):
            li -= 1
        if li < 0:
            return None
        si -= 1
        li -= 1
    lf = before[li + 1:].strip()
    words = re.split(r"[ ‐-]+", lf)
    if not (2 <= len(words) <= min(len(short) + 5, len(short) * 2)):
        return None
    if short.lower() in (w.lower() for w in words):
        return None
    return lf


def abbreviations_in(text: str) -> Dict[str, str]:
    """Abbreviations a text defines: {'WM': 'white matter', ...}. First definition wins."""
    defs: Dict[str, str] = {}
    for m in _DEF_RE.finditer(text):
        short = m.group(2)
        if sum(ch.isupper() for ch in short) < 2:
            continue
        lf = _long_form(m.group(1), short)
        if lf and short not in defs:
            defs[short] = re.sub(r"\s+", " ", lf)
    return defs


def _abbreviations(s: Sentence, abbreviations: Dict[str, str]) -> List[Expansion]:
    out = []
    for i in range(s.n):
        w = s.text(i)
        key = w if w in abbreviations else (w[:-1] if w.endswith("s") and w[:-1] in abbreviations else None)
        if key is None:
            continue
        if i > 0 and s.text(i - 1) == "(":
            continue                # the definition itself
        long = abbreviations[key]
        h = s.head(i)
        if s.dep(i) in ("compound", "nmod", "amod") and h > i and s.pos(h) in NOUNISH:
            out.append(Expansion("abbreviation", f"{long} {s.join(range(i + 1, h + 1))}", i, h))
        else:
            out.append(Expansion("abbreviation", long, i, i))
    return out


# ---------------------------------------------------------------- whole texts

def expand_document(sentences: Iterable[Sequence[Sequence]], text: str = "",
                    attested: Optional[Set[Tuple[str, str]]] = None,
                    compounds: Optional[Set[Tuple[str, str]]] = None) -> List[Expansion]:
    abbr = abbreviations_in(text) if text else None
    out: List[Expansion] = []
    for toks in sentences:
        out += expand_sentence(toks, attested, abbr, compounds)
    return out


def index_text(expansions: Iterable[Expansion], gap: int = 4) -> str:
    """The expansions as one text to index, each separated by `gap` stopwords.

    PostgreSQL drops stopwords but keeps their positions, so the padding keeps
    the last word of one expansion from standing next to the first word of the
    next ("temporal cortices | frontal cortices" must not give "cortices frontal").
    """
    pad = " " + " ".join(["the"] * gap) + " "
    return pad.join(e.text for e in expansions)


def attested_pairs(sentences: Iterable[Sequence[Sequence]], min_count: int = 3
                   ) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """What the corpus writes adjacent, at least min_count times.

    Returns (attested, compounds): attested holds (modifier lemma, noun lemma)
    pairs ("temporal", "pole"); compounds holds (noun lemma, stem of the next
    word) pairs ("movie", "watch" from "movie watching"), joined by a space or a
    hyphen, for the verb-object rewording.
    """
    from collections import Counter
    c: Counter = Counter()
    comp: Counter = Counter()
    for toks in sentences:
        n = len(toks)
        for i in range(n - 1):
            a, b = toks[i], toks[i + 1]
            if b[POS] in NOUNISH and a[POS] in ("ADJ", "NOUN", "PROPN", "VERB"):
                c[(a[LEMMA].lower(), b[LEMMA].lower())] += 1
            if a[POS] in NOUNISH:
                j = i + 2 if b[TEXT] in ("-", "‐") and i + 2 < n else i + 1
                if toks[j][POS] == "NOUN" and toks[j][TEXT].lower().endswith(("ing", "ion", "ions", "ment")):
                    comp[(a[LEMMA].lower(), _stem(toks[j][TEXT]))] += 1
    return ({k for k, v in c.items() if v >= min_count}, {k for k, v in comp.items() if v >= min_count})
