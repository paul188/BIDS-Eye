#!/usr/bin/env python3
"""
benchmarks/word_forms/score.py
------------------------------
Score methods against the person's word form labels (label.html export).

Each label says, per direction, whether a search for one form should find
the other:

    both     A->B yes, B->A yes        a_to_b   A->B yes, B->A no
    neither  A->B no,  B->A no         b_to_a   A->B no,  B->A yes
    context  undecidable from the words alone; scored apart

Per direction a method can go wrong two ways:
    wrong split   the person says match, the method says no   -> lost recall
    wrong merge   the person says no, the method says match   -> false hits

Two reference methods are always scored, so a method that splits (or merges)
everything cannot look good: "stem" (today's search: every pair matches both
ways) and "split" (no pair matches).

    python benchmarks/word_forms/score.py --labels ~/Downloads/word_form_labels.json \\
        results/llm_gemini-2.5-flash_<set_id>.json [more prediction files]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

_HERE = Path(__file__).resolve().parent
DIRECTIONS = {"both": (True, True), "a_to_b": (True, False), "b_to_a": (False, True),
              "neither": (False, False), "context": None}


def _load_labels(path: Path, set_id: str) -> Dict[str, str]:
    data = json.loads(path.read_text())
    if data.get("set_id") != set_id:
        raise SystemExit(f"{path} is for set {data.get('set_id')}, not {set_id}")
    return {pid: v["label"] for pid, v in data["labels"].items() if v.get("label") in DIRECTIONS}


def _directions(label: str) -> Optional[Tuple[bool, bool]]:
    return DIRECTIONS[label]


def score(truth: Dict[str, str], pred: Dict[str, str], pairs: Dict[str, dict]) -> Dict[str, dict]:
    """Counts per group: all, and by the pair's origin."""
    groups: Dict[str, Counter] = defaultdict(Counter)
    for pid, label in truth.items():
        guess = pred.get(pid)
        keys = ["all", pairs[pid]["origin"]]
        for key in keys:
            c = groups[key]
            c["pairs"] += 1
            if guess is None:
                c["unanswered"] += 1
                continue
            c["exact"] += guess == label
            want, got = _directions(label), _directions(guess)
            if want is None:
                c["truth_context"] += 1
                c[f"context_as_{guess}"] += 1
                continue
            if got is None:
                c["said_context"] += 2       # two directions left undecided
                c["directions"] += 2
                continue
            for w, g in zip(want, got):
                c["directions"] += 1
                c["right"] += w == g
                c["should_match"] += w
                c["should_split"] += not w
                c["wrong_split"] += w and not g
                c["wrong_merge"] += g and not w
    return groups


def _fmt(name: str, groups: Dict[str, Counter]) -> str:
    lines = [f"== {name}"]
    for key in ["all", "sample", "yaml_exact", "yaml_stemmed"]:
        c = groups.get(key)
        if not c:
            continue
        decided = c["pairs"] - c["truth_context"] - c["unanswered"]
        pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "   - "  # noqa: E731
        lines.append(
            f"  {key:12} pairs {c['pairs']:3}  exact label {pct(c['exact'], c['pairs'] - c['unanswered'])}"
            f"  directions right {pct(c['right'], c['directions'])}"
            f"  wrong split {c['wrong_split']:3}/{c['should_match']:<3} ({pct(c['wrong_split'], c['should_match'])})"
            f"  wrong merge {c['wrong_merge']:3}/{c['should_split']:<3} ({pct(c['wrong_merge'], c['should_split'])})"
            f"  said context {c['said_context'] // 2:2}  decided {decided}")
        ctx = {k[len('context_as_'):]: v for k, v in c.items() if k.startswith("context_as_")}
        if ctx:
            lines.append(f"  {'':12} person said context ({c['truth_context']}): method said {dict(ctx)}")
        if c["unanswered"]:
            lines.append(f"  {'':12} unanswered {c['unanswered']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="export of label.html")
    parser.add_argument("predictions", type=Path, nargs="*", help="files with {set_id, labels: {id: label}}")
    parser.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args()

    label_set = json.loads((_HERE / "label_set.json").read_text())
    pairs = {p["id"]: p for p in label_set["pairs"]}
    truth = _load_labels(args.labels, label_set["set_id"])
    counts = Counter(truth.values())
    report = [f"set {label_set['set_id']}: {len(truth)} of {len(pairs)} pairs labelled; "
              f"labels {dict(sorted(counts.items()))}"]

    methods = {"stem (today)": {pid: "both" for pid in pairs}, "split all": {pid: "neither" for pid in pairs}}
    for path in args.predictions:
        data = json.loads(path.read_text())
        if data.get("set_id") != label_set["set_id"]:
            raise SystemExit(f"{path} is for set {data.get('set_id')}")
        methods[data.get("model") or path.stem] = data["labels"]
    for name, pred in methods.items():
        report.append(_fmt(name, score(truth, pred, pairs)))
    text = "\n\n".join(report)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
