"""Classify each labelled pair: inflection (same word, -s/-ed/-ing) or derivation
(-al, -ion, -ive, -ly, -er, -ity, -ance ...), and compare with the labels."""
import json, sys, os
from collections import Counter, defaultdict

S = sys.argv[1]
lab = json.load(open(f"{S}/raw_labels.json"))
pairs = {p["id"]: p for p in json.load(open(sys.argv[2]))["pairs"]}
gem = json.load(open(sys.argv[3]))["labels"]
neigh = json.load(open(f"{S}/neigh.json"))

INFL = {"", "s", "es", "ed", "d", "ing", "e", "ies", "ied", "y", "ying", "ie"}

def suffixes(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]: i += 1
    sa, sb = a[i:], b[i:]
    # doubled consonant: transfer/transferred, scan/scanned
    for s in (sa, sb):
        pass
    if sa[:1] and sa[:1] == a[i-1:i] and sa[1:] in INFL: sa = sa[1:]
    if sb[:1] and sb[:1] == b[i-1:i] and sb[1:] in INFL: sb = sb[1:]
    return sa, sb

def kind(p):
    a, b = p["a"][0], p["b"][0]
    sa, sb = suffixes(a, b)
    if sa in INFL and sb in INFL:
        # -ed/-ing vs bare noun-ish forms are still inflection
        return "inflection", sa, sb
    return "derivation", sa, sb

def dirs(l):
    return {"both": (1, 1), "a_to_b": (1, 0), "b_to_a": (0, 1), "neither": (0, 0)}.get(l)

by = defaultdict(Counter); rows = []
for pid, v in lab.items():
    p = pairs[pid]; k, sa, sb = kind(p); L = v["label"]
    by[k][L] += 1
    rows.append((k, pid, p["a"][0], p["b"][0], sa, sb, L, gem.get(pid)))
for k, c in by.items():
    n = sum(c.values()); d = n - c["context"]
    print(f"{k:11} pairs {n:3}  {dict(c)}  merge-both share of decided {100*c['both']/d:.0f}%")

# inflection pairs Paul split: is one form lexicalised in a fixed phrase?
print("\ninflection pairs labelled neither / one-way / context:")
for k, pid, a, b, sa, sb, L, G in sorted(rows):
    if k == "inflection" and L != "both":
        na = neigh.get(a, {}); nb = neigh.get(b, {})
        def top(n):
            if not n or not n.get("top"): return ""
            ph, c = n["top"][0]; return f"{ph} {c}/{n['df']}"
        print(f"  {pid} {a:14} {b:14} {L:8} G={G:8} | {top(na):30} | {top(nb)}")
print("\nderivation pairs labelled both:")
for k, pid, a, b, sa, sb, L, G in sorted(rows):
    if k == "derivation" and L == "both":
        print(f"  {pid} {a:16} {b:16} -{sa or '0'} / -{sb or '0'}  G={G}")
json.dump(rows, open(f"{S}/morph.json", "w"))
