"""
benchmarks/phrases/evaluate_expander.py
---------------------------------------
Run expander.py over a parsed corpus and report what it adds.

    python3 benchmarks/phrases/evaluate_expander.py PARSED.jsonl.gz PARAS.jsonl OUT.json \
        [--attested-from OTHER.parsed.jsonl.gz ...] [--samples 40]

PARSED is parse.py output (one sentence per line, grouped by document), PARAS
the paragraphs it was parsed from (for abbreviation definitions per document).

Reports
- expansions per kind, and documents with at least one;
- random samples per kind with the sentence they come from, for reading;
- per term (analysis/grammar/termlist.py) and per phrase candidate
  (phrase_candidates.json): documents that write the term adjacent, and
  documents that only an expansion makes write it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis" / "grammar"))
from expander import abbreviations_in, attested_pairs, expand_sentence  # noqa: E402
from termlist import terms  # noqa: E402

LATIN = {"cortices": "cortex", "gyri": "gyrus", "sulci": "sulcus", "stimuli": "stimulus", "nuclei": "nucleus",
         "hippocampi": "hippocampus", "foci": "focus", "data": "data", "criteria": "criterion"}


def norm(w: str) -> str:
    w = w.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    if w in LATIN:
        return LATIN[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def words(text: str):
    return [norm(w) for w in re.findall(r"[A-Za-z0-9]+(?:'s)?", text)]


def has_phrase(seq, phrase):
    n = len(phrase)
    return any(seq[i:i + n] == phrase for i in range(len(seq) - n + 1))


def documents(path):
    cur, sents = None, []
    for line in gzip.open(path, "rt"):
        doc, part, toks = json.loads(line)
        if doc != cur and sents:
            yield cur, sents
            sents = []
        cur = doc
        sents.append(toks)
    if sents:
        yield cur, sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed")
    ap.add_argument("paras")
    ap.add_argument("out")
    ap.add_argument("--attested-from", nargs="*", default=[])
    ap.add_argument("--min-attested", type=int, default=3)
    ap.add_argument("--samples", type=int, default=40)
    a = ap.parse_args()

    def all_sents():
        for p in [a.parsed] + a.attested_from:
            for _, sents in documents(p):
                yield from sents
    attested, compounds = attested_pairs(all_sents(), a.min_attested)
    print("attested pairs", len(attested), "noun compounds", len(compounds))

    texts = defaultdict(list)
    for line in open(a.paras):
        d = json.loads(line)
        texts[d["paper"]].append(d["text"])

    targets = [(f"term:{name}", [norm(w[0]) for w in ws]) for _, name, ws in terms()]
    cands = json.load(open(HERE / "phrase_candidates.json"))["candidates"]
    targets += [(f"cand:{c['phrase']}", words(c["phrase"])) for c in cands if len(words(c["phrase"])) >= 2]
    by_first = defaultdict(list)
    for name, ph in targets:
        by_first[ph[0]].append((name, ph))

    kinds, kind_docs = Counter(), defaultdict(set)
    pool = defaultdict(list)
    seen = Counter()
    rnd = random.Random(1)
    plain_docs, gained_docs = defaultdict(set), defaultdict(set)
    gained_by_kind = defaultdict(Counter)
    gain_examples = defaultdict(list)
    ndocs = 0
    for doc, sents in documents(a.parsed):
        ndocs += 1
        abbr = abbreviations_in("\n".join(texts.get(doc, [])))
        orig_seq, exp_list = [], []
        for toks in sents:
            seq = [norm(t[0]) for t in toks if t[2] not in ("PUNCT", "SYM", "SPACE")]
            orig_seq.append(seq)
            for e in expand_sentence(toks, attested, abbr, compounds):
                kinds[e.kind] += 1
                kind_docs[e.kind].add(doc)
                exp_list.append((e, toks))
                seen[e.kind] += 1
                src = "".join(t[0] + (" " if t[5] else "") for t in toks[max(0, e.start - 8):e.end + 8]).strip()
                item = {"doc": doc, "text": e.text, "source": src}
                if len(pool[e.kind]) < a.samples:
                    pool[e.kind].append(item)
                elif rnd.random() < a.samples / seen[e.kind]:
                    pool[e.kind][rnd.randrange(a.samples)] = item
        present = set()
        for seq in orig_seq:
            for w in set(seq):
                for name, ph in by_first.get(w, ()):
                    if name not in present and has_phrase(seq, ph):
                        present.add(name)
        for name in present:
            plain_docs[name].add(doc)
        for e, toks in exp_list:
            seq = words(e.text)
            for w in set(seq):
                for name, ph in by_first.get(w, ()):
                    if name in present or doc in gained_docs[name]:
                        continue
                    if has_phrase(seq, ph):
                        gained_docs[name].add(doc)
                        gained_by_kind[e.kind][name] += 1
                        if len(gain_examples[name]) < 4:
                            src = "".join(t[0] + (" " if t[5] else "") for t in toks[max(0, e.start - 8):e.end + 8]).strip()
                            gain_examples[name].append({"doc": doc, "kind": e.kind, "text": e.text, "source": src})

    report = {
        "documents": ndocs,
        "attested_pairs": len(attested),
        "compounds": len(compounds),
        "kinds": {k: {"expansions": v, "documents": len(kind_docs[k])} for k, v in kinds.most_common()},
        "samples": pool,
        "targets": sorted(
            [{"target": name, "plain": len(plain_docs[name]), "gained": len(gained_docs[name]),
              "examples": gain_examples[name]} for name, _ in targets if gained_docs[name] or plain_docs[name]],
            key=lambda r: -r["gained"]),
        "gained_by_kind": {k: sum(v.values()) for k, v in gained_by_kind.items()},
    }
    json.dump(report, open(a.out, "w"), indent=1)
    print("documents", ndocs)
    for k, v in report["kinds"].items():
        print(f"  {k:16} expansions {v['expansions']:7}  documents {v['documents']:6}")
    for prefix in ("term:", "cand:"):
        rows = [r for r in report["targets"] if r["target"].startswith(prefix)]
        p = sum(r["plain"] for r in rows)
        g = sum(r["gained"] for r in rows)
        print(f"{prefix} target-document pairs: written adjacent {p}, gained only by expansion {g} (+{100 * g / max(p, 1):.1f}%)")
    print("gained pairs by kind:", report["gained_by_kind"])


if __name__ == "__main__":
    main()
