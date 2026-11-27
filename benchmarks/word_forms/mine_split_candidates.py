#!/usr/bin/env python3
r"""
benchmarks/word_forms/mine_split_candidates.py
----------------------------------------------
Word-form directions nobody has judged, in both directions (2026-11-27).

word_forms.colliding_forms() only flags a form with at least MIN_RATIO times
the datasets of the asked word, so a common word that shares its stem with a
rarer, different word (functional / functioning) was never asked about, and
word_form_pairs.json was mined with the same test. This lists every direction
A -> B under one stem where B has at least MIN_EXTRA datasets and A at least
MIN_ASKED, minus the pairs word_forms.yaml or word_form_pairs.json decide.

Input: bids_word_forms as CSV (form,stem,ndoc), exported read-only:
    \copy (SELECT form, stem, ndoc FROM bids_word_forms) TO STDOUT WITH (FORMAT csv, HEADER)

    python benchmarks/word_forms/mine_split_candidates.py word_forms_table.csv out.json

Judged candidates of the first run: results/stem_split_candidates_2026-11-27.json
("judge": Claude, from words and counts only, blind to the labels; "paul": the
blind label of label_set.json where one exists).
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "backend" / "vocabulary")]

import word_forms as wf  # noqa: E402

MIN_OTHER = wf.MIN_EXTRA
MIN_ASKED = 5


def families(forms):
    """Forms of one stem grouped into words: plural, singular and spellings together."""
    names = [f for f, _ in forms]
    parent = {f: f for f in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in names:
        own = wf.own_forms(f)
        for g in names:
            if g in own:
                parent[find(g)] = find(f)
    groups = collections.defaultdict(list)
    for f, n in forms:
        groups[find(f)].append((f, n))
    return [sorted(v, key=lambda x: -x[1]) for v in groups.values()]


def judged(a_forms, b_forms):
    """Which list already decides A -> B, or None."""
    for a in a_forms:
        listed, group = wf._entry(a)
        if listed and group is None:
            return "yaml:stemmed"
        if group and set(b_forms) & (set(group.forms) | group.excludes):
            return "yaml"
        different, same = wf._stored_for(a)
        if set(b_forms) & (different | same):
            return "pairs"
    return None


def blind_labels():
    """(form A, form B) -> (pair id, label, side of A) from the blind label set."""
    label_set = json.loads((HERE / "label_set.json").read_text())
    labels = json.loads((HERE / "labels_p000-p189_chrome_2026-11-19.json").read_text())["labels"]
    found = {}
    for p in label_set["pairs"]:
        if p["id"] not in labels:
            continue
        label = labels[p["id"]]["label"]
        for a in p["a"]:
            for b in p["b"]:
                found[(a, b)] = (p["id"], label, "a")
                found[(b, a)] = (p["id"], label, "b")
    return found, {p["stem"] for p in label_set["pairs"]}


def main() -> None:
    table, out = sys.argv[1], sys.argv[2]
    stems = collections.defaultdict(list)
    with open(table) as fh:
        for row in csv.DictReader(fh):
            stems[row["stem"]].append((row["form"], int(row["ndoc"])))
    paul, label_stems = blind_labels()

    candidates, stats = [], collections.Counter()
    for stem, forms in stems.items():
        fams = families(forms)
        if len(fams) < 2:
            continue
        for fam_a in fams:
            a_docs = fam_a[0][1]
            if a_docs < MIN_ASKED:
                continue
            for fam_b in fams:
                b_docs = fam_b[0][1]
                if fam_b is fam_a or b_docs < MIN_OTHER:
                    continue
                stats["directions"] += 1
                old = b_docs >= max(MIN_OTHER, (wf.MIN_RATIO - 1) * a_docs)
                a_f, b_f = [f for f, _ in fam_a], [f for f, _ in fam_b]
                done = judged(a_f, b_f)
                if done:
                    stats["judged:" + done] += 1
                    continue
                stats["old_test" if old else "new_direction"] += 1
                label = next((paul[(a, b)] for a in a_f for b in b_f if (a, b) in paul), None)
                candidate = {"stem": stem, "a": a_f, "b": b_f, "a_docs": a_docs, "b_docs": b_docs,
                             "old_test": old, "label_set_stem": stem in label_stems}
                if label:
                    candidate["paul"] = {"pair": label[0], "label": label[1], "a_side": label[2]}
                candidates.append(candidate)
    candidates.sort(key=lambda c: -c["b_docs"])
    for i, c in enumerate(candidates):
        c["id"] = f"c{i:04d}"
    Path(out).write_text(json.dumps(candidates, indent=0))
    print(dict(stats))
    print(len(candidates), "candidates ->", out)


if __name__ == "__main__":
    main()
