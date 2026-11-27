#!/usr/bin/env python3
"""
benchmarks/phrases/build_phrase_candidates.py
---------------------------------------------
Candidate multi-word terms for review: phrases a search should match with
their words close together, not anywhere in the text ("cognitive control" is
in 73 datasets as two words, in 22 as the term).

Candidates come from two sources:
  vocabulary  multi-word labels and synonyms of backend/vocabulary/value_mappings.yaml,
              backend/vocabulary/reference_vocabulary.yaml (MeSH diagnoses, Cognitive Atlas
              tasks) and backend/vocabulary/cognitive_atlas_constructs.yaml, kept when the
              catalogue writes them as a phrase in >= MIN_VOCAB_DOCS datasets
  catalogue   two- and three-word sequences the catalogue writes in
              >= MIN_MINED_DOCS datasets, whose words stand together in at
              least MIN_SHARE of the datasets that contain them all, and that
              co-occur more than chance (PMI > MIN_PMI). A mined phrase in
              fewer than MIN_DOCS_ANY_CONTEXT datasets also needs
              MIN_CONTEXTS different neighbouring words, so a sentence copied
              across one lab's READMEs drops out; one found (almost) only
              inside a longer candidate is left to that one

A phrase counts only where its words are separated by spaces (one line break
allowed), a hyphen or a slash ("go/no-go"), never across a full stop, a comma
or a blank line. Spellings are counted apart: "resting state",
"resting-state", "restingstate".
A final plural is merged with its singular ("eye movement(s)").

The script decides nothing. It mines both research terms ("fixation cross")
and README boilerplate ("exclusion criteria", "public domain"); the review
page sorts them. Counts and sources are shown, the script's own guess is not.

    DATABASE_URL=postgresql://... python benchmarks/phrases/build_phrase_candidates.py

Writes, next to this file:
    phrase_candidates.json  every candidate with its counts and examples
    phrase_review.html      the page to review in; export the labels as JSON

Read-only transaction; nothing is written to the database.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

SEED = 20261120
MIN_VOCAB_DOCS = 2
MIN_MINED_DOCS = 5
MIN_SHARE = 0.5
MIN_PMI = 1.0
MAX_WORDS = 5
MIN_CONTEXTS = 3           # distinct (word before, word after) pairs of a rare mined phrase
MIN_DOCS_ANY_CONTEXT = 15  # from here on a mined phrase may be boilerplate in one wording
SUBSUMED_SHARE = 0.9
EXAMPLES = 2
SNIPPET_CHARS = 80

# Words a phrase must not start or end with. Inside a phrase they may stand
# ("theory of mind"), as PostgreSQL keeps their positions.
_EDGE_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have he her his i if in into is it its
may me more most my no not of on or our she so some such than that the their them then there these they this those
to was we were what when where which while who will with would you your also each other all any both only same very
using used use per via about after before between during over under up out than just well one two three
""".split())

# Pieces of URLs, file names and e-mail addresses.
_NOISE_TOKENS = frozenset("""
http https www com org io gov edu uk de nl fr ca au ac net html htm pdf doi github osf en txt csv tsv json
nii gz mat py zip md readme sub ses acq run gmail
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# What may stand between two words of a phrase: spaces with at most one line
# break, or a hyphen.
_SPACE_RE = re.compile(r"[ \t]*\n?[ \t]*")
_FULL_SQL = ("SELECT accession_id, concat_ws(' • ', name, description_text, readme_text) "
             "FROM bids_datasets ORDER BY accession_id")
_SHORT_SQL = ("SELECT accession_id, concat_ws(' • ', name, description_text, dx_status::text, "
              "study_domain::text, study_design::text, llm_conditions::text) "
              "FROM bids_datasets ORDER BY accession_id")

Key = Tuple[str, ...]


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("Set DATABASE_URL.")
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


async def _load() -> Tuple[Dict[str, str], Dict[str, str]]:
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            full = {str(d): t or "" for d, t in await conn.execute(text(_FULL_SQL))}
            short = {str(d): t or "" for d, t in await conn.execute(text(_SHORT_SQL))}
            await conn.rollback()
    finally:
        await engine.dispose()
    return full, short


# ── Tokens with their separators ──────────────────────────────────────────────

class Doc:
    """One dataset's text as tokens, and how each token joins the previous one."""

    __slots__ = ("text", "tokens", "starts", "joins", "words")

    def __init__(self, body: str) -> None:
        self.text = body
        low = body.lower()
        self.tokens: List[str] = []
        self.starts: List[int] = []
        self.joins: List[str] = []   # " ", "-" or "/" when the token continues a phrase, "" otherwise
        end = None
        for m in _TOKEN_RE.finditer(low):
            sep = low[end:m.start()] if end is not None else None
            if sep is None:
                join = ""
            elif sep in ("-", "/"):
                join = sep
            elif sep and _SPACE_RE.fullmatch(sep):
                join = " "
            else:
                join = ""
            self.tokens.append(m.group())
            self.starts.append(m.start())
            self.joins.append(join)
            end = m.end()
        self.words: Set[str] = set(self.tokens)

    def ngrams(self, n: int) -> Iterable[Tuple[Key, int, str]]:
        """(words, start index, spelling) of every n-word phrase."""
        toks, joins = self.tokens, self.joins
        for i in range(len(toks) - n + 1):
            seps = joins[i + 1:i + n]
            if all(seps):
                kinds = set(seps)
                spelling = {"-": "hyphen", " ": "space", "/": "slash"}[seps[0]] if len(kinds) == 1 else "mixed"
                yield tuple(toks[i:i + n]), i, spelling


def _singular(word: str) -> Optional[str]:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return None


# ── Vocabulary terms ──────────────────────────────────────────────────────────

def _vocabulary() -> Dict[Key, Set[str]]:
    """Multi-word terms of the repo's vocabularies → their sources."""
    found: Dict[Key, Set[str]] = defaultdict(set)

    def add(term: object, source: str) -> None:
        if not isinstance(term, str):
            return
        words = tuple(_TOKEN_RE.findall(term.lower()))
        while words and words[0] in _EDGE_STOPWORDS:
            words = words[1:]
        while words and words[-1] in _EDGE_STOPWORDS:
            words = words[:-1]
        if 2 <= len(words) <= MAX_WORDS:
            found[words].add(source)

    vm = yaml.safe_load((_REPO / "backend" / "vocabulary" / "value_mappings.yaml").read_text())
    for kind in ("diagnosis", "task"):
        for entry in (vm.get(kind) or {}).values():
            add(entry.get("label"), f"curated {kind}")
            for syn in entry.get("synonyms") or []:
                add(syn.get("term") if isinstance(syn, dict) else syn, f"curated {kind}")
    ref = yaml.safe_load((_REPO / "backend" / "vocabulary" / "reference_vocabulary.yaml").read_text())
    names = {"diagnosis": "MeSH", "task": "Cognitive Atlas task"}
    for kind, name in names.items():
        for entry in (ref.get(kind) or {}).values():
            add(entry.get("label"), name)
            for term in entry.get("terms") or []:
                add(term, name)
    ca = yaml.safe_load((_REPO / "backend" / "vocabulary" / "cognitive_atlas_constructs.yaml").read_text())
    for entry in (ca.get("cognitive_construct") or {}).values():
        add(entry.get("label"), "Cognitive Atlas construct")
        for syn in entry.get("synonyms") or []:
            add(syn, "Cognitive Atlas construct")
    return found


# ── Counting ──────────────────────────────────────────────────────────────────

def _mined(docs: Dict[str, Doc]) -> Set[Key]:
    n_docs = len(docs)
    word_df: Counter = Counter()
    for d in docs.values():
        word_df.update(d.words)
    out: Set[Key] = set()
    for n in (2, 3):
        df: Counter = Counter()
        for d in docs.values():
            df.update({k for k, _, _ in d.ngrams(n)})
        for key, c in df.items():
            if c < MIN_MINED_DOCS:
                continue
            if key[0] in _EDGE_STOPWORDS or key[-1] in _EDGE_STOPWORDS:
                continue
            if any(w.isdigit() or w in _NOISE_TOKENS for w in key):
                continue
            if n == 3 and key[1] in _EDGE_STOPWORDS and key[1] not in {"of", "to", "on", "in"}:
                continue
            content = [w for w in key if w not in _EDGE_STOPWORDS]
            pmi = math.log(c * n_docs ** (len(content) - 1) / math.prod(word_df[w] for w in content))
            if pmi <= MIN_PMI:
                continue
            together = sum(1 for d in docs.values() if all(w in d.words for w in content))
            if together and c / together >= MIN_SHARE:
                out.add(key)
    return out


def _group(keys: Iterable[Key]) -> Dict[Key, List[Key]]:
    """Merge a final plural into its singular: head key → its forms."""
    keys = set(keys)
    groups: Dict[Key, List[Key]] = {}
    for key in sorted(keys, key=len):
        sing = _singular(key[-1])
        head = key[:-1] + (sing,) if sing else key
        if head != key and head in keys:
            groups.setdefault(head, [head]).append(key)
        elif key not in groups:
            groups.setdefault(key, [key])
    # A plural whose singular is also a plural head is already merged above.
    merged = {k for forms in groups.values() for k in forms[1:]}
    return {h: f for h, f in groups.items() if h not in merged}


# dataset → (char start, char end, spelling, (word before, word after)) per occurrence
Hits = Dict[str, List[Tuple[int, int, str, Tuple[str, str]]]]


def _index(docs: Dict[str, Doc], keys: Set[Key]) -> Dict[Key, Hits]:
    """phrase → dataset → its occurrences (Hits).

    One pass over the text per phrase length; a phrase written as one word
    ("restingstate") is found under its key with spelling "joined".
    """
    out: Dict[Key, Hits] = defaultdict(lambda: defaultdict(list))
    joined = {"".join(k): k for k in keys}
    for n in sorted({len(k) for k in keys}):
        for ds, d in docs.items():
            for key, i, spelling in d.ngrams(n):
                if key in keys:
                    last = i + n - 1
                    out[key][ds].append((d.starts[i], d.starts[last] + len(d.tokens[last]), spelling,
                                         _context(d, i, last)))
    for ds, d in docs.items():
        for i, tok in enumerate(d.tokens):
            key = joined.get(tok)
            if key is not None:
                out[key][ds].append((d.starts[i], d.starts[i] + len(tok), "joined", _context(d, i, i)))
    return out


def _context(d: Doc, first: int, last: int) -> Tuple[str, str]:
    return (d.tokens[first - 1] if first else "", d.tokens[last + 1] if last + 1 < len(d.tokens) else "")


def _occurrences(index: Dict[Key, Hits], forms: Sequence[Key]) -> Hits:
    merged: Hits = {}
    for f in forms:
        for ds, hits in index.get(f, {}).items():
            merged.setdefault(ds, []).extend(hits)
    return merged


def _snippet(body: str, start: int, end: int) -> Dict[str, str]:
    a, b = max(0, start - SNIPPET_CHARS), min(len(body), end + SNIPPET_CHARS)
    clip = lambda s: re.sub(r"\s+", " ", s)  # noqa: E731
    return {"before": ("…" if a else "") + clip(body[a:start]), "match": body[start:end],
            "after": clip(body[end:b]) + ("…" if b < len(body) else "")}


def build() -> Dict:
    full_text, short_text = asyncio.run(_load())
    docs = {ds: Doc(t) for ds, t in full_text.items()}
    short = {ds: Doc(t) for ds, t in short_text.items()}
    rng = random.Random(SEED)

    vocab = _vocabulary()
    mined = _mined(docs)
    groups = _group(set(vocab) | mined)
    all_forms = {f for forms in groups.values() for f in forms}
    full_index = _index(docs, all_forms)
    short_index = _index(short, all_forms)

    candidates = []
    for head, forms in groups.items():
        occ = _occurrences(full_index, forms)
        sources = sorted({s for f in forms for s in vocab.get(f, ())})
        in_vocab = bool(sources)
        if len(occ) < (MIN_VOCAB_DOCS if in_vocab else MIN_MINED_DOCS):
            continue
        if any(f in mined for f in forms):
            sources.append("catalogue")
        spellings = Counter()
        for hits in occ.values():
            spellings.update({s for _, _, s, _ in hits})
        contexts = {ctx for hits in occ.values() for *_, ctx in hits}
        # A rare mined phrase in one wording only is a sentence copied across a
        # series of READMEs ("series of laser"), not a term.
        if not in_vocab and len(occ) < MIN_DOCS_ANY_CONTEXT and len(contexts) < MIN_CONTEXTS:
            continue
        content = [w for w in head if w not in _EDGE_STOPWORDS]
        if not content:
            continue
        last_forms = {f[-1] for f in forms}
        bag = [ds for ds, d in docs.items()
               if ds in occ or (all(w in d.words for w in content[:-1]) and (last_forms & d.words))]
        short_occ = _occurrences(short_index, forms)

        examples = []
        # One example from a dataset whose short fields have the phrase too, when there is one.
        picks = sorted(set(short_occ) & set(occ))[:1] + rng.sample(sorted(occ), min(len(occ), EXAMPLES + 1))
        for ds in list(dict.fromkeys(picks))[:EXAMPLES]:
            s, e, _, _ = occ[ds][0]
            examples.append({"dataset": ds, **_snippet(full_text[ds], s, e)})
        apart = []
        rest = [ds for ds in bag if ds not in occ]
        rarest = min(content, key=lambda w: sum(1 for ds in rest if w in docs[ds].words) or 10 ** 6)
        for ds in rng.sample(rest, min(len(rest), EXAMPLES)):
            d = docs[ds]
            i = next((k for k, t in enumerate(d.tokens) if t == rarest or t in last_forms), 0)
            apart.append({"dataset": ds, **_snippet(full_text[ds], d.starts[i], d.starts[i] + len(d.tokens[i]))})

        candidates.append({
            "phrase": " ".join(head),
            "forms": [" ".join(f) for f in forms],
            "sources": sources,
            "datasets_phrase": len(occ),
            "datasets_words_anywhere": len(bag),
            "datasets_phrase_short_fields": len(short_occ),
            "spellings": dict(spellings),
            "examples": examples,
            "words_apart": apart,
            "_head": head,
            "_docs": set(occ),
            "_mined_only": not in_vocab,
        })

    # A mined phrase that (almost) only occurs inside a longer candidate is a
    # piece of it: "organizing electrophysiological" of "... data".
    longer = defaultdict(list)
    for c in candidates:
        h = c["_head"]
        for n in range(2, len(h)):
            for k in range(len(h) - n + 1):
                longer[h[k:k + n]].append(c)
    kept = []
    for c in candidates:
        covering = [o for o in longer.get(c["_head"], ())]
        if c["_mined_only"] and any(len(c["_docs"] & o["_docs"]) >= SUBSUMED_SHARE * len(c["_docs"])
                                    for o in covering):
            continue
        kept.append(c)
    candidates = kept
    for c in candidates:
        for k in ("_head", "_docs", "_mined_only"):
            del c[k]

    candidates.sort(key=lambda c: (-c["datasets_words_anywhere"], -c["datasets_phrase"], c["phrase"]))
    for k, c in enumerate(candidates):
        c["id"] = f"c{k:04d}"
    digest = hashlib.sha1(json.dumps([c["phrase"] for c in candidates]).encode()).hexdigest()[:12]
    return {"set_id": f"phrases-{digest}", "datasets": len(docs),
            "settings": {"min_vocab_docs": MIN_VOCAB_DOCS, "min_mined_docs": MIN_MINED_DOCS,
                         "min_share": MIN_SHARE, "min_pmi": MIN_PMI},
            "candidates": candidates}


def main() -> None:
    data = build()
    (_HERE / "phrase_candidates.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    page = (_HERE / "review_template.html").read_text()
    page = page.replace("__SET_ID__", data["set_id"]).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (_HERE / "phrase_review.html").write_text(page)
    by_source = Counter(s for c in data["candidates"] for s in c["sources"])
    print(f"{len(data['candidates'])} candidates from {data['datasets']} datasets; set {data['set_id']}")
    print("by source:", dict(by_source.most_common()))


if __name__ == "__main__":
    main()
