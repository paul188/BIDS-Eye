"""
benchmarks/phrases/recall_by_class.py
-------------------------------------
For the 138 two-word terms of analysis/grammar/termlist.py: in sentences that
contain both words but not as the adjacent phrase, how often does an expansion
write the phrase? Broken down by the arrangement terms.py assigns (list,
inserted word, head first, unlinked near/far), so the gain on real mentions
(list, inserted word) can be told from the gain on unrelated pairs (far).

    python3 benchmarks/phrases/recall_by_class.py PARSED.jsonl.gz OUT.json --attested-from A.gz B.gz
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis" / "grammar"))
from evaluate_expander import documents, norm, words  # noqa: E402
from expander import abbreviations_in, attested_pairs, expand_sentence  # noqa: E402
from termlist import terms  # noqa: E402
from terms import classify2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed")
    ap.add_argument("paras")
    ap.add_argument("out")
    ap.add_argument("--attested-from", nargs="+", required=True)
    ap.add_argument("--min-attested", type=int, default=3)
    a = ap.parse_args()

    def all_sents():
        for p in a.attested_from:
            for _, sents in documents(p):
                yield from sents
    attested, compounds = attested_pairs(all_sents(), a.min_attested)
    print("attested", len(attested), "compounds", len(compounds))

    texts = defaultdict(list)
    for line in open(a.paras):
        d = json.loads(line)
        texts[d["paper"]].append(d["text"])

    two = [(name, ws) for _, name, ws in terms() if len(ws) == 2]
    allforms = {f for _, ws in two for w in ws for f in w}
    total, gained = Counter(), Counter()
    samples = defaultdict(list)
    rnd = random.Random(2)
    for doc, sents in documents(a.parsed):
        abbr = None
        for toks in sents:
            present = defaultdict(list)
            for i, t in enumerate(toks):
                low = t[0].lower()
                for f in (t[1], low, low[:-1] if low.endswith("s") else low):
                    if f in allforms:
                        present[f].append(i)
            if len(present) < 2:
                continue
            wpos, c = [], 0
            for t in toks:
                wpos.append(c)
                if t[2] not in ("PUNCT", "SYM", "SPACE"):
                    c += 1
            hits = []
            for name, ws in two:
                idx = [sorted({i for f in w for i in present.get(f, ())}) for w in ws]
                if all(idx):
                    cls, _, m, h = classify2(toks, wpos, idx[0], idx[1])
                    if cls != "adjacent":
                        hits.append((name, cls, m, h))
            if not hits:
                continue
            if abbr is None:
                abbr = abbreviations_in("\n".join(texts.get(doc, [])))
            exp = [words(e.text) for e in expand_sentence(toks, attested, abbr, compounds)]
            for name, cls, m, h in hits:
                ph = [norm(w) for w in name.split()]
                total[cls] += 1
                ok = any(seq[i:i + 2] == ph for seq in exp for i in range(len(seq) - 1))
                if ok:
                    gained[cls] += 1
                    lo, hi = min(m, h), max(m, h)
                    src = "".join(t[0] + (" " if t[5] else "") for t in toks[max(0, lo - 8):hi + 6]).strip()
                    s = samples[cls]
                    if len(s) < 25:
                        s.append(f"[{name}] {src}")
                    elif rnd.random() < 0.05:
                        s[rnd.randrange(25)] = f"[{name}] {src}"
    rep = {c: {"instances": total[c], "now_phrase": gained[c], "share": round(gained[c] / total[c], 3)}
           for c in total}
    json.dump({"classes": rep, "samples": samples}, open(a.out, "w"), indent=1)
    for c, v in sorted(rep.items(), key=lambda kv: -kv[1]["instances"]):
        print(f"  {c:14} instances {v['instances']:7}  now written as the phrase {v['now_phrase']:6} ({100 * v['share']:.1f}%)")


if __name__ == "__main__":
    main()
