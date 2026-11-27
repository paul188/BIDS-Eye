#!/usr/bin/env python3
"""
benchmarks/word_forms/llm_baseline.py
-------------------------------------
The language-model baseline for the word form labels (build_label_set.py):
the model gets each stem's forms with their dataset counts, as the search's
intent prompt does (backend/vocabulary/word_forms.py prompt_block), and labels every pair of
the set the way a person does on label.html.

Only words and counts are sent, never dataset text. One call per stem, at
temperature 0, the production model without the Claude fallback, so the
answers are one model's. Every prompt and answer is logged.

    GEMINI_API_KEY=... python benchmarks/word_forms/llm_baseline.py
    python benchmarks/word_forms/llm_baseline.py --dry-run     # print one prompt

Writes results/llm_<model>_<set_id>.jsonl (one line per stem: prompt, answer,
model, seconds) and results/llm_<model>_<set_id>.json (pair id -> label).
Prints nothing about the answers: the labels are made blind.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "backend" / "llm"))

from llm_client import llm_generate  # noqa: E402

MODEL = "gemini-2.5-flash"
LABELS = ["both", "a_to_b", "b_to_a", "neither", "context"]
# Output limit, timeouts and retries: profile word_forms_benchmark in
# backend/llm/llm_config.yaml (Gemini 2.5's thinking counts against the
# output limit; 2048 cut answers off).

SYSTEM = (
    "You help a full-text search over a catalogue of neuroimaging datasets "
    "(OpenNeuro, BIDS: MRI, EEG, MEG and similar studies). The search reduces "
    "every word to its stem, so all forms in one request share a stem and "
    "currently match each other. You decide, for pairs of these forms, whether "
    "that is right in this catalogue."
)

INSTRUCTIONS = """Forms under one stem, with the number of datasets in the catalogue that contain each:
{forms}

Forms joined by "/" are plural and singular of one word; treat them as one.
For each pair below, answer: should a search for A find datasets that only say B?
  both      the two mean the same here; each search should find the other
  a_to_b    a search for A should find B, but a search for B should not find A
  b_to_a    the reverse
  neither   different meanings here; neither search should find the other
  context   sometimes the same meaning, sometimes not; the word alone cannot decide

Pairs:
{pairs}

Answer every pair id once."""

SCHEMA = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "label": {"type": "string", "enum": LABELS}},
        "required": ["id", "label"]}}},
    "required": ["pairs"],
}


def _parsed(answer: str):
    """The answer as a dict, or None when it is not complete JSON."""
    try:
        value = json.loads(answer)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def prompts(label_set: dict):
    by_stem = defaultdict(list)
    for p in label_set["pairs"]:
        by_stem[p["stem"]].append(p)
    for stem, pairs in sorted(by_stem.items()):
        counts = {}
        for p in pairs:
            counts[" / ".join(p["a"])] = p["a_docs"]
            counts[" / ".join(p["b"])] = p["b_docs"]
        forms = "\n".join(f"  {f}: {n}" for f, n in sorted(counts.items(), key=lambda x: -x[1]))
        lines = "\n".join(f"  {p['id']}: A = {' / '.join(p['a'])}; B = {' / '.join(p['b'])}" for p in pairs)
        yield stem, [p["id"] for p in pairs], INSTRUCTIONS.format(forms=forms, pairs=lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt, call nothing")
    args = parser.parse_args()

    label_set = json.loads((_HERE / "label_set.json").read_text())
    if args.dry_run:
        stem, ids, prompt = next(prompts(label_set))
        print(f"[system]\n{SYSTEM}\n\n[prompt for stem {stem!r}]\n{prompt}")
        return

    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)
    name = f"llm_{args.model}_{label_set['set_id']}"
    log_path, pred_path = out_dir / f"{name}.jsonl", out_dir / f"{name}.json"
    done = {}
    if log_path.exists():   # resume: stems already answered are kept
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            if _parsed(row["answer"]) is not None:   # a cut-off answer is asked again
                done[row["stem"]] = row
    predictions, missing = {}, 0
    with log_path.open("a") as log:
        for stem, ids, prompt in prompts(label_set):
            row = done.get(stem)
            if row is None:
                t0 = time.monotonic()
                answer = llm_generate(prompt, profile="word_forms_benchmark", system=SYSTEM,
                                      models=[args.model], json_schema=SCHEMA)
                row = {"stem": stem, "model": args.model, "prompt": prompt, "system": SYSTEM,
                       "answer": answer, "seconds": round(time.monotonic() - t0, 2)}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            parsed = _parsed(row["answer"])
            if parsed is None:
                missing += len(ids)
                continue
            got = {a.get("id"): a.get("label") for a in parsed.get("pairs", [])}
            for pid in ids:
                if got.get(pid) in LABELS:
                    predictions[pid] = got[pid]
                else:
                    missing += 1
    pred_path.write_text(json.dumps({"set_id": label_set["set_id"], "model": args.model,
                                     "labels": predictions}, indent=1))
    print(f"{len(predictions)} pairs answered, {missing} missing; log {log_path.name}")


if __name__ == "__main__":
    main()
