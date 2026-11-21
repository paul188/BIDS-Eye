"""Fused and hyphenated compounds of anatomical/technical adjectives in running text (paras.jsonl or snapshot)."""
import json, re, sys
from collections import Counter, defaultdict
BASES = ("temporal frontal parietal occipital cortical spinal striatal thalamic cerebellar limbic insular cingulate hippocampal "
         "amygdala motor sensory visual spatial lateral medial dorsal ventral central").split()
TOK = re.compile(r"[A-Za-z][A-Za-z‐-]*[A-Za-z]")
def main(src, fields):
    alone = Counter(); fused = defaultdict(Counter); hyph = defaultdict(Counter); docs = defaultdict(lambda: defaultdict(set))
    for n, line in enumerate(open(src)):
        try: d = json.loads(line)
        except Exception: continue
        text = " ".join(str(d.get(f) or "") for f in fields); doc = d.get("paper") or d.get("accession_id") or n
        for w in TOK.findall(text):
            w = w.lower().replace("‐", "-")
            for b in BASES:
                if w == b: alone[b] += 1; docs[b]["alone"].add(doc)
                elif w.endswith(b) and len(w) > len(b) + 2:
                    pre = w[:-len(b)]
                    if pre.endswith("-"): hyph[b][w] += 1; docs[b]["compound"].add(doc)
                    elif pre.endswith("o") or pre in ("pre", "sub", "extra", "inter", "intra", "sensori", "peri", "para", "supra", "infra", "bi", "multi", "non", "meso"):
                        fused[b][w] += 1; docs[b]["compound"].add(doc)
    print(f"{'word':12} {'alone':>7} {'fused':>7} {'hyphen':>7} {'docs alone':>10} {'docs compound only':>18}  top compounds")
    for b in BASES:
        only = len(docs[b]["compound"] - docs[b]["alone"])
        top = (fused[b] + hyph[b]).most_common(6)
        print(f"{b:12} {alone[b]:7} {sum(fused[b].values()):7} {sum(hyph[b].values()):7} {len(docs[b]['alone']):10} {only:18}  " + ", ".join(f"{w} {c}" for w, c in top))
if __name__ == "__main__": main(sys.argv[1], sys.argv[2].split(","))
