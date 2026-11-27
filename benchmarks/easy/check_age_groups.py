#!/usr/bin/env python3
"""How reliably does the intent step turn age-group words into age_groups?

Runs ONLY the intent step (one model call each, no database) on questions
that name an age group in words, several times each, and on controls that
name none (or give ages as numbers). A plan is right when the age groups the
search will use (plan.age_groups, plus description_search when it holds only
an age-group word) are one of the accepted sets, and no age-group word is left
in a longer description_search.

    python check_age_groups.py --backend-dir /app/backend [--repeats 3] [--out age_groups.json]

Exit code 1 when any question is wrong in more than one run in three.
None of the questions is an example of the intent prompt; the first is the
development question of 2026-11-15 that showed the gap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# (question, accepted sets of codes); an empty set means no age group.
CASES = [
    ("show me datasets related to impulse control in children", [{"children"}]),
    ("MEG recordings from school-age kids", [{"children"}]),
    ("pediatric epilepsy iEEG", [{"children"}]),
    ("youth with depression, resting state", [{"children"}]),
    ("autism versus typically developing children, EEG", [{"children"}]),
    ("teenagers playing video games", [{"adolescents"}, {"children"}]),
    ("language development in toddlers", [{"infants"}, {"children"}]),
    ("neonatal brain MRI", [{"infants"}]),
    ("memory decline in elderly people", [{"older_adults"}]),
    ("fMRI of seniors listening to music", [{"older_adults"}]),
    # controls
    ("healthy adults doing a memory task", [set()]),
    ("participants aged 8 to 12 with dyslexia", [set()]),
    ("EEG of professional musicians", [set()]),
    ("resting state fMRI in bilingual speakers", [set()]),
]

_AGE_WORD_RE = re.compile(
    r"\b(child|children|kids?|pediatric|paediatric|youth|adolescen\w*|teen\w*|infants?|"
    r"toddlers?|newborns?|neonat\w*|bab(y|ies)|elderly|seniors?|older adults?)\b", re.I)


def judge(plan, accepted, preprocess) -> str:
    """'' when the plan is right, else what is wrong."""
    from criteria import normalise_age_groups

    only_age = preprocess._age_groups_only(plan.description_search) or []
    codes = set(normalise_age_groups(list(plan.age_groups) + only_age))
    if not only_age and plan.description_search and _AGE_WORD_RE.search(plan.description_search):
        return f"age word left in description_search {plan.description_search!r}"
    if codes not in accepted:
        return f"age_groups {sorted(codes) or '[]'}"
    return ""


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

    results, failed = [], 0
    for question, accepted in CASES:
        wrong = []
        for _ in range(args.repeats):
            try:
                problem = judge(preprocess.preprocess_query(question), accepted, preprocess)
            except Exception as exc:  # noqa: BLE001
                problem = f"error: {str(exc)[:80]}"
            if problem:
                wrong.append(problem)
        right = args.repeats - len(wrong)
        failed += len(wrong) * 3 > args.repeats
        kind = "control" if accepted == [set()] else "age"
        print(f"{right}/{args.repeats}  {kind:<8} {question}"
              + (f"   [{'; '.join(sorted(set(wrong)))}]" if wrong else ""), flush=True)
        results.append({"question": question, "kind": kind, "right": right,
                        "runs": args.repeats, "wrong": wrong})
    total = sum(r["right"] for r in results)
    print(f"\n{total} of {len(results) * args.repeats} runs right; "
          f"{sum(1 for r in results if r['right'] == r['runs'])} of {len(results)} questions right every time")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
