#!/usr/bin/env python3
"""
benchmarks/word_forms/build_label_set.py
----------------------------------------
Build the blind labelling set for word forms: pairs of forms under one stem,
each to be labelled "should a search for A find datasets that only say B?".

The set scores the methods that could decide stem collisions offline
(a language model, word vectors, the catalogue's own contexts). It is a test
set, not a list the search reads.

Population: the stems where a form passes the search's own collision test
(RAG/word_forms.py is_collision, 2x and +10 datasets), forms in >= 2 datasets.
Sample: SAMPLE_PER_STRATUM stems from each quartile of the stem's dataset
count, without the stems word_forms.yaml already covers, plus every group of
word_forms.yaml. Forms are grouped into families by plural and singular
(word_forms.inflections); a pair is two families of one stem.

Each pair shows README snippets of both families, so a form's use in this
catalogue can be judged (functional = fMRI). Snippets come from the dataset
name, description and README only: llm_profile is written by a language
model, one of the methods under test. Counts and the pair's origin are kept
out of the page.

    DATABASE_URL=postgresql://... python benchmarks/word_forms/build_label_set.py

Writes, next to this file:
    label_set.json   every pair with its hidden facts (stem, origin, counts)
    label.html       the page to label in; labels are exported from it as JSON

Read-only transaction; nothing is written to the database.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path[:0] = [str(_REPO / "RAG"), str(_REPO / "crawlers" / "openneuro-crawler")]

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from remote.openneuro_metadata import EXACT_SEARCH_VECTOR_EXPRESSION  # noqa: E402
from word_forms import WordEvidence, inflections, is_collision  # noqa: E402

SEED = 20261119
SAMPLE_PER_STRATUM = 25
PAIRS_PER_STEM = 3
SNIPPETS_PER_FAMILY = 3
SNIPPET_CHARS = 90
MIN_FORM_DOCS = 2

_FORMS_SQL = (
    "SELECT s.word, array_to_string(tsvector_to_array(to_tsvector('english', s.word)), ' '), s.ndoc "
    f"FROM ts_stat($${'SELECT ' + EXACT_SEARCH_VECTOR_EXPRESSION + ' FROM bids_datasets'}$$) AS s "
    "WHERE s.word ~ '^[a-z]{2,40}$'"
)
_STEMS_SQL = "SELECT word, ndoc FROM ts_stat('SELECT search_tsv FROM bids_datasets')"
_TEXTS_SQL = ("SELECT accession_id, concat_ws(' • ', name, description_text, readme_text) "
              "FROM bids_datasets ORDER BY accession_id")

Family = Tuple[str, ...]


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("Set DATABASE_URL.")
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


async def _load() -> Tuple[List[Sequence], Dict[str, int], List[Tuple[str, str]]]:
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            forms = list(await conn.execute(text(_FORMS_SQL)))
            stems = {w: int(n) for w, n in await conn.execute(text(_STEMS_SQL))}
            texts = [(str(d), t or "") for d, t in await conn.execute(text(_TEXTS_SQL))]
            await conn.rollback()
    finally:
        await engine.dispose()
    return forms, stems, texts


def _families(forms: Dict[str, int]) -> List[Family]:
    """Forms grouped by plural and singular, commonest family first."""
    parent = {f: f for f in forms}

    def root(f: str) -> str:
        while parent[f] != f:
            f = parent[f]
        return f

    for f in forms:
        for g in inflections(f) & forms.keys():
            parent[root(g)] = root(f)
    groups: Dict[str, List[str]] = defaultdict(list)
    for f in forms:
        groups[root(f)].append(f)
    fams = [tuple(sorted(g, key=lambda f: (-forms[f], f))) for g in groups.values()]
    return sorted(fams, key=lambda fam: -max(forms[f] for f in fam))


def _reviewed_groups() -> List[Tuple[str, List[List[str]], List[List[str]]]]:
    """word_forms.yaml as (kind, wanted form lists, excluded form lists)."""
    import yaml
    data = yaml.safe_load((_REPO / "RAG" / "word_forms.yaml").read_text())
    groups = [("yaml_exact", [e["forms"]], [e.get("excludes") or []]) for e in data.get("exact") or []]
    groups += [("yaml_stemmed", [g], []) for g in data.get("stemmed") or []]
    return groups


def _snippets(family: Family, texts: List[Tuple[str, str]], rng: random.Random) -> List[Dict[str, str]]:
    pattern = re.compile(r"(?<![a-z])(" + "|".join(map(re.escape, family)) + r")(?![a-z])", re.I)
    hits = []
    for dataset, body in texts:
        m = pattern.search(body)
        if m:
            hits.append((dataset, body, m))
    rng.shuffle(hits)
    out = []
    for dataset, body, m in hits[:SNIPPETS_PER_FAMILY]:
        start, end = max(0, m.start() - SNIPPET_CHARS), min(len(body), m.end() + SNIPPET_CHARS)
        clip = lambda s: re.sub(r"\s+", " ", s)  # noqa: E731
        out.append({"dataset": dataset,
                    "before": ("…" if start else "") + clip(body[start:m.start()]),
                    "match": body[m.start():m.end()],
                    "after": clip(body[m.end():end]) + ("…" if end < len(body) else "")})
    return out


def build() -> Dict:
    rows, stem_docs, texts = asyncio.run(_load())
    by_stem: Dict[str, Dict[str, int]] = defaultdict(dict)
    for word, stem, n in rows:
        if stem and re.fullmatch(r"[a-z0-9]+", stem) and int(n) >= MIN_FORM_DOCS:
            by_stem[stem][word] = int(n)

    colliding: Dict[str, List[str]] = {}
    for stem, forms in by_stem.items():
        if len(forms) < 2:
            continue
        evidence = tuple(sorted(forms.items(), key=lambda x: -x[1]))
        hits = [w for w in forms if is_collision(WordEvidence(w, stem, stem_docs.get(stem, 0), evidence))]
        if hits:
            colliding[stem] = hits

    reviewed = _reviewed_groups()
    reviewed_forms = {f for _, wanted, excl in reviewed for g in wanted + excl for f in g}
    rng = random.Random(SEED)

    pool = sorted((s for s in colliding if not reviewed_forms & by_stem[s].keys()),
                  key=lambda s: (stem_docs.get(s, 0), s))
    quarter = len(pool) / 4
    sampled = []
    for i in range(4):
        stratum = pool[round(i * quarter):round((i + 1) * quarter)]
        sampled += [(s, i) for s in rng.sample(stratum, min(SAMPLE_PER_STRATUM, len(stratum)))]

    pairs = []

    def add(stem: str, a: Family, b: Family, origin: str, extra: Dict) -> None:
        forms = by_stem.get(stem, {})
        pairs.append({"stem": stem, "a": list(a), "b": list(b), "origin": origin,
                      "a_docs": max(forms.get(f, 0) for f in a), "b_docs": max(forms.get(f, 0) for f in b),
                      "stem_docs": stem_docs.get(stem, 0), **extra})

    for stem, stratum in sampled:
        fams = _families(by_stem[stem])
        hit = [f for f in fams if set(f) & set(colliding[stem])]
        chosen = []
        for c in hit:
            for other in fams:
                if other != c and {c, other} not in chosen:
                    chosen.append({c, other})
        for pair in chosen[:PAIRS_PER_STEM]:
            a, b = sorted(pair, key=lambda fam: fam[0])
            add(stem, a, b, "sample", {"stratum": stratum})

    for kind, wanted, excluded in reviewed:
        present = [f for g in wanted + excluded for f in g]
        stem_of = {w: s for s, forms in by_stem.items() for w in forms if w in present}
        if not stem_of:
            continue
        stem = max(set(stem_of.values()), key=list(stem_of.values()).count)
        forms = {f: by_stem[stem][f] for f in present if stem_of.get(f) == stem}
        fams = _families(forms)
        want = {f for g in wanted for f in g}
        chosen = []
        for x in fams:
            for y in fams:
                if x < y and (kind == "yaml_stemmed" or (bool(set(x) & want) != bool(set(y) & want))):
                    chosen.append((x, y))
        for a, b in chosen[:PAIRS_PER_STEM]:
            add(stem, a, b, kind, {})

    for p in pairs:
        if rng.random() < 0.5:
            p["a"], p["b"], p["a_docs"], p["b_docs"] = p["b"], p["a"], p["b_docs"], p["a_docs"]
    rng.shuffle(pairs)
    for i, p in enumerate(pairs):
        p["id"] = f"p{i:03d}"
        p["a_snippets"] = _snippets(tuple(p["a"]), texts, rng)
        p["b_snippets"] = _snippets(tuple(p["b"]), texts, rng)

    digest = hashlib.sha256(json.dumps([(p["id"], p["a"], p["b"]) for p in pairs]).encode()).hexdigest()[:12]
    return {"set_id": f"word-forms-{digest}", "seed": SEED, "population_stems": len(colliding),
            "pool_stems": len(pool), "datasets": len(texts), "pairs": pairs}


def _page(label_set: Dict) -> str:
    public = [{k: p[k] for k in ("id", "a", "b", "a_snippets", "b_snippets")} for p in label_set["pairs"]]
    data = json.dumps({"set_id": label_set["set_id"], "pairs": public}).replace("</", "<\\/")
    return (_HERE / "label_template.html").read_text().replace("__DATA__", data).replace(
        "__SET_ID__", html.escape(label_set["set_id"]))


def main() -> None:
    label_set = build()
    (_HERE / "label_set.json").write_text(json.dumps(label_set, indent=1, ensure_ascii=False))
    (_HERE / "label.html").write_text(_page(label_set))
    origins = defaultdict(int)
    for p in label_set["pairs"]:
        origins[p["origin"]] += 1
    print(f"{label_set['set_id']}: {len(label_set['pairs'])} pairs {dict(origins)}; "
          f"{label_set['population_stems']} colliding stems, {label_set['pool_stems']} not in word_forms.yaml")


if __name__ == "__main__":
    main()
