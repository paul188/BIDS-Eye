#!/usr/bin/env python3
"""How reliably does the intent step mark "or" lists of scan terms as alternatives?

Runs ONLY the intent step (one model call each, no database) on phrasings
with an "or" list, several times each, and counts how often the plan comes
back right:

  or-phrasing   one ScanRequirement with any_of=true holds the listed
                alternatives, and the modality every result must have (when
                the question names one) is not inside that list
  control       an "and" / "both" phrasing: no requirement has any_of=true

    python check_any_of.py --backend-dir /app/backend [--repeats 3] [--out any_of.json]

Exit code 1 when any phrasing is wrong in more than one run in three.
The phrasings are in no benchmark, and none is an example of the intent prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# (question, alternatives that must be listed together, modality that must stay outside)
OR_PHRASINGS = [
    ("EEG datasets with a Stroop or flanker task", ["stroop", "flanker"], "eeg"),
    ("fMRI datasets with n-back, Sternberg or digit span tasks", ["n-back", "sternberg", "digit span"], "fmri"),
    ("datasets with either EEG or MEG", ["eeg", "meg"], None),
    ("MEG or iEEG recordings of speech", ["meg", "ieeg"], None),
    ("fMRI with a gambling or reward task", ["gambling", "reward"], "fmri"),
    ("diffusion or perfusion MRI datasets", ["diffusion", "perfusion"], None),
    ("EEG during sleep or anesthesia", ["sleep", "anesthesia"], "eeg"),
    ("datasets with T2-weighted or FLAIR images", ["t2", "flair"], None),
    ("fMRI studies using music, speech or environmental sounds", ["music", "speech", "environmental sounds"], "fmri"),
    ("task fMRI such as emotion regulation, fear conditioning or reappraisal",
     ["emotion regulation", "fear conditioning", "reappraisal"], "fmri"),
    ("iEEG datasets with a memory task or a language task", ["memory", "language"], "ieeg"),
    ("are there PET or MRS datasets?", ["pet", "mrs"], None),
    ("EEG with oddball / go-no-go / stop-signal paradigms (any of them)",
     ["oddball", "go-no-go", "stop-signal"], "eeg"),
    ("fNIRS or EEG during a verbal fluency task", ["fnirs", "eeg"], None),
]
CONTROLS = [
    "EEG datasets with both a Stroop and a flanker task",
    "datasets with EEG and fMRI",
    "fMRI datasets with n-back and resting state",
    "resting-state fMRI datasets",
    "MEG datasets with a verbal fluency task",
]


def _norm(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def judge_or(plan, alternatives, modality) -> str:
    """'' when the plan is right, else what is wrong."""
    wanted = [_norm(a) for a in alternatives]
    for requirement in plan.scan_requirements:
        terms = [t for t in (_norm(t) for t in requirement.scan_terms) if t]
        held = [w for w in wanted if any(w in t or t in w for t in terms)]
        if len(held) < len(wanted):
            continue
        if not requirement.any_of:
            return "listed together without any_of"
        if modality and any(_norm(modality) in t for t in terms):
            return "the required modality is inside the list"
        return ""
    return "alternatives not in one requirement"


def judge_control(plan) -> str:
    return "any_of set" if any(r.any_of for r in plan.scan_requirements) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, args.backend_dir)
    import main as _app  # noqa: F401 — sets up the import paths as the app does
    import preprocess
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    cases = [(q, lambda plan, a=a, m=m: judge_or(plan, a, m), "or") for q, a, m in OR_PHRASINGS]
    cases += [(q, judge_control, "control") for q in CONTROLS]
    results, failed = [], 0
    for question, judge, kind in cases:
        wrong = []
        for _ in range(args.repeats):
            try:
                problem = judge(preprocess.preprocess_query(question, api_key=api_key))
            except Exception as exc:  # noqa: BLE001
                problem = f"error: {str(exc)[:80]}"
            if problem:
                wrong.append(problem)
        right = args.repeats - len(wrong)
        failed += len(wrong) * 3 > args.repeats
        print(f"{right}/{args.repeats}  {kind:<8} {question}" + (f"   [{'; '.join(sorted(set(wrong)))}]" if wrong else ""),
              flush=True)
        results.append({"question": question, "kind": kind, "right": right, "runs": args.repeats, "wrong": wrong})
    total = sum(r["right"] for r in results)
    print(f"\n{total} of {len(results) * args.repeats} runs right; "
          f"{sum(1 for r in results if r['right'] == r['runs'])} of {len(results)} phrasings right every time")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
