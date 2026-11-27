#!/usr/bin/env python3
"""How reliably does the intent step mark an excluded scan as required: false?

Runs ONLY the intent step (one model call each, no database), several times
per phrasing. A plan is right when

  exclusion  every excluded scan is in a requirement with required=false, in
             no required one and not in description_search, and what the
             dataset must have is required and not excluded
  together   the same, with the words of the excluded scan in one requirement
             ("resting-state fMRI" excludes neither alone)
  control    no requirement has required=false

    python check_exclusions.py --backend-dir /app/backend [--repeats 3] [--out exclusions.json]

Exit code 1 when any phrasing is wrong in more than one run in three.
The phrasings are in no benchmark, and none is an example of the intent prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# (question, excluded scans, required scans); the scans are regular expressions.
EXCLUSIONS = [
    ("MEG datasets that have no T1w images", [r"\bT1"], [r"\bMEG\b"]),
    ("EEG datasets without any fMRI", [r"\bfMRI\b|functional MRI"], [r"\bEEG\b"]),
    ("task fMRI data, no resting state", [r"\brest"], [r"fMRI|functional MRI"]),
    ("diffusion MRI, but not with ASL", [r"\bASL\b|arterial spin"], [r"diffusion"]),
    ("iEEG recordings without a sleep session", [r"sleep"], [r"iEEG|intracranial"]),
    ("T2-weighted images but no FLAIR", [r"FLAIR"], [r"\bT2"]),
    ("fMRI datasets that include neither EEG nor eye tracking", [r"\bEEG\b", r"eye.?track"], [r"fMRI"]),
    ("PET scans; exclude anything with CT", [r"\bCT\b"], [r"\bPET\b"]),
    ("fNIRS datasets apart from those with a finger tapping task", [r"finger.?tapping"], [r"fNIRS"]),
    ("structural MRI datasets lacking field maps", [r"field.?map"], [r"structural|anatomical|\bMRI\b"]),
    ("movie-watching fMRI, minus any resting-state runs", [r"\brest"], [r"movie"]),
    ("hi all, i need meg data w/ a picture naming task; no eeg pls, thanks!!", [r"\bEEG\b"], [r"\bMEG\b", r"naming"]),
    ("Looking for anatomical scans - SWI excluded - of adults", [r"\bSWI\b|susceptib"], [r"anatomical"]),
    ("auditory oddball EEG without passive listening", [r"passive"], [r"oddball", r"\bEEG\b"]),
    ("EEG recordings free of simultaneous fMRI", [r"fMRI"], [r"\bEEG\b"]),
    # Long requests, where the exclusion is one clause of several.
    ("resting-state EEG and a T1w scan acquired with MPRAGE instead of FLASH", [r"FLASH"], [r"MP.?RAGE", r"\bEEG\b"]),
    ("healthy adults with task fMRI and anatomical images acquired as T2w, not as FLAIR", [r"FLAIR"], [r"\bT2"]),
    ("BOLD data recorded with spin echo, not gradient echo, at 3 Tesla", [r"gradient"], [r"spin"]),
    ("EEG dataset with at least 20 subjects, a working memory task, two sessions per subject, "
     "and no resting-state recording", [r"\brest"], [r"\bEEG\b", r"working memory"]),
    ("Siemens fMRI, several runs per task (no localizer runs), more than 30 people", [r"locali[sz]er"], [r"fMRI"]),
    ("MEG of language comprehension with human subjects (listening, not reading)", [r"reading"], [r"\bMEG\b"]),
    ("children with ADHD; resting-state fMRI; structural MRI; no task fMRI", [r"task"], [r"\brest"]),
    ("7T diffusion images, acquired with DSI and not with HARDI", [r"HARDI"], [r"\bDSI\b|diffusion"]),
    ("MPRAGE but not MEMPRAGE", [r"\bME.?MPRAGE|multi.?echo"], [r"\bMP.?RAGE"]),
]
TOGETHER = [
    ("MEG datasets without resting-state fMRI", [r"\brest", r"fMRI"], [r"\bMEG\b"]),
    ("diffusion datasets that do not include n-back fMRI", [r"n.?back", r"fMRI"], [r"diffusion"]),
]
CONTROLS = [
    "non-invasive brain stimulation with EEG",
    "go/no-go task fMRI",
    "fMRI datasets that contain not only resting state but also a motor task",
    "MEG datasets, the type of task does not matter",
    "iEEG datasets without epilepsy patients",
    "EEG datasets not collected at Harvard",
    "fMRI with and without caffeine",
    "fMRI datasets no larger than 20 GB",
    "EEG besides fMRI in the same dataset",
    "unmedicated depression patients with fMRI",
    "drug-naive schizophrenia, resting-state fMRI",
    "fMRI without simultaneous multi-slice acceleration",
    "EEG data that were not filtered",
    "MEG without Maxwell filtering such as tSSS",
]


def _found(pattern: str, terms) -> bool:
    return any(re.search(pattern, t, re.I) for t in terms)


def judge_exclusion(plan, excluded, required, together: bool = False) -> str:
    """'' when the plan is right, else what is wrong."""
    out = [r.scan_terms for r in plan.scan_requirements if not r.required]
    out_terms = [t for terms in out for t in terms]
    in_terms = [t for r in plan.scan_requirements if r.required for t in r.scan_terms]
    for pattern in excluded:
        if not _found(pattern, out_terms):
            where = ("required" if _found(pattern, in_terms)
                     else "description_search" if _found(pattern, [plan.description_search or ""])
                     else "nowhere")
            return f"exclusion is {where}"
        if _found(pattern, in_terms):
            return "excluded and required"
        if _found(pattern, [plan.description_search or ""]):
            return "also in description_search"
    if together and not any(all(_found(p, terms) for p in excluded) for terms in out):
        return "not in one requirement"
    for pattern in required:
        if not _found(pattern, in_terms):
            return "required scan missing"
    return ""


def judge_control(plan) -> str:
    terms = [t for r in plan.scan_requirements if not r.required for t in r.scan_terms]
    return f"excludes {terms}" if terms else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = Path(args.backend_dir).resolve().parent
    sys.path[:0] = [str(root / name) for name in ("backend", "backend/llm", "backend/vocabulary")]
    import preprocess
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    cases = [(q, lambda p, e=e, r=r: judge_exclusion(p, e, r), "exclusion") for q, e, r in EXCLUSIONS]
    cases += [(q, lambda p, e=e, r=r: judge_exclusion(p, e, r, True), "together") for q, e, r in TOGETHER]
    cases += [(q, judge_control, "control") for q in CONTROLS]
    results, failed = [], 0
    for question, judge, kind in cases:
        wrong, errors = [], []
        for _ in range(args.repeats):
            try:
                problem = judge(preprocess.preprocess_query(question))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {str(exc)[:80]}")
                continue
            if problem:
                wrong.append(problem)
        runs = args.repeats - len(errors)  # a failed call says nothing about the plan
        right = runs - len(wrong)
        failed += runs == 0 or len(wrong) * 3 > runs
        notes = sorted(set(wrong)) + sorted(set(errors))
        print(f"{right}/{runs}  {kind:<9} {question}" + (f"   [{'; '.join(notes)}]" if notes else ""), flush=True)
        results.append({"question": question, "kind": kind, "right": right, "runs": runs,
                        "wrong": wrong, "errors": errors})
    print()
    for kind in ("exclusion", "together", "control"):
        rows = [r for r in results if r["kind"] == kind]
        print(f"{kind:<9} {sum(r['right'] for r in rows)} of {sum(r['runs'] for r in rows)} runs right; "
              f"{sum(1 for r in rows if r['runs'] and r['right'] == r['runs'])} of {len(rows)} phrasings every time")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
