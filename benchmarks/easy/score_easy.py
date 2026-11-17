#!/usr/bin/env python3
"""Score a run of the easy searches against the datasets' own labels.

    python score_easy.py --run results/<run>.json \
        --facts ../openneuro_facets/results/2026-11-14_db_facts.jsonl [--json out.json]

The run comes from ../demand/record_run.py --queries easy_searches.yaml --prefix "".
For each search with a `check`: every dataset whose labels pass the check must
be among the results ("missed" lists those that are not). With `strict: true`
every result must pass it too ("wrong"). The facts file must be dumped from
the database the run used (../openneuro_facets/dump_facts.sql).

Verdicts: ok / MISSES / WRONG / EMPTY (no result at all) / ERROR / FALLBACK,
and "unchecked" for searches without a check.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "openneuro_facets"))
import compare  # noqa: E402

# Values the OpenNeuro filter benchmark did not need.
compare.DATATYPES.update({
    "anat": {"anatomical_mri"},
    "mrs": {"magnetic_resonance_spectroscopy"},
    "motion": {"motion_capture"},
    "fmap": {"field_maps"},
})
compare.DIAGNOSES.update({
    "depression": (r"depress|\bmdd\b", {"major_depressive_disorder", "depression", "unipolar_depression"}),
    "autism": (r"autis|\basd\b", {"autism_spectrum_disorder"}),
    "parkinson": (r"parkinson", {"parkinsons_disease"}),
    "epilepsy": (r"epilep", {"epilepsy", "non_lesional_epilepsy", "genetic_epilepsy", "epilepsy_unclear_etiology"}),
    "stroke": (r"stroke", {"stroke"}),
    "bipolar": (r"bipolar", {"bipolar_disorder"}),
    "ms": (r"multiple sclerosis", {"multiple_sclerosis"}),
    "tbi": (r"traumatic brain|\btbi\b", {"traumatic_brain_injury"}),
    "dyslexia": (r"dyslexi", {"dyslexia"}),
    "anxiety": (r"anxiety", {"anxiety_disorder"}),
    "chronic_pain": (r"chronic.*pain", {"chronic_pain"}),
    "fibromyalgia": (r"fibromyalgia", {"fibromyalgia"}),
})
compare.TASKS.update({
    "eyes_closed": (r"eyes?_?closed|\bec\b", r"eyes_closed"),
    "eyes_open": (r"eyes?_?open|\beo\b", r"eyes_open"),
    "working_memory": (r"working_?mem|n_?back|\bwm\b", r"working_memory|n_back"),
    "stroop": (r"stroop", r"stroop"),
    "simon": (r"simon", r"simon"),
    "switching": (r"task_?switch|switching", r"task_switching"),
    "tapping": (r"tapping", r"finger_tapping"),
    "story": (r"story|narrative", r"story|narrative"),
    "speech": (r"speech", r"speech"),
    "music": (r"music", r"music"),
    "emotion": (r"emotion", r"emotion"),
    "reward": (r"reward", r"reward"),
    "gambling": (r"gambl", r"gambling"),
    "decision": (r"decision", r"decision"),
    "reading": (r"reading", r"reading"),
    "naming": (r"naming", r"naming"),
    "retinotopy": (r"retino|\bprf\b", r"retinotopy|population_receptive_field"),
    "localizer": (r"locali[sz]er", r"localizer"),
    "encoding": (r"encod", r"memory_encoding"),
    "tom": (r"theory_?of_?mind|false_?belief|\btom\b", r"theory_of_mind"),
    "pain": (r"pain", r"pain"),
    "imagery": (r"imagery", r"mental_imagery"),
    "arithmetic": (r"arithmetic", r"arithmetic"),
    "breath_hold": (r"breath_?hold", r"breath_hold"),
})
_MONKEY = r"monkey|macaque|macaca|primate|marmoset"
compare.SPECIES.update({
    "monkey": lambda s: bool(s) and re.search(_MONKEY, s, re.I) is not None and "human" not in s.lower().replace("non-human", ""),
    "macaque": lambda s: bool(s) and re.search(r"macaque|macaca", s, re.I) is not None,
    "animal": lambda s: bool(s) and (compare.SPECIES["rodent"](s) or compare.SPECIES["other_animal"](s)),
})


def make_check(spec: dict):
    """compare.make_check plus two keys: modality_all and published_year."""
    spec = dict(spec)
    tests = []
    for kind in spec.pop("modality_all", []):
        wanted = compare.DATATYPES[kind]
        tests.append(lambda f, w=wanted: bool(set(f.get("dts") or []) & w))
    if "published_year" in spec:
        lo, hi = spec.pop("published_year")
        tests.append(lambda f, lo=lo, hi=hi: bool(f.get("publish"))
                     and compare.in_range(int(f["publish"][:4]), lo, hi))
    rest = compare.make_check(spec)
    return lambda f: rest(f) and all(t(f) for t in tests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", type=Path, default=HERE / "easy_searches.yaml")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    queries = yaml.safe_load(args.queries.read_text())["queries"]
    run = {r["id"]: r for r in json.loads(args.run.read_text())["results"]}
    facts = {}
    for line in args.facts.read_text().splitlines():
        if line.strip():
            fact = json.loads(line)
            facts[fact["acc"]] = fact

    detail, counts = {}, {}
    print(f"{'search':20} {'found':>5} {'must':>5} {'missed':>6} {'wrong':>5}  verdict  question")
    for q in queries:
        result = run.get(q["id"])
        if result is None:
            continue
        returned = [a for a in result.get("returned") or [] if a in facts]
        got = set(returned)
        must, missed, wrong = set(), [], []
        if result.get("error"):
            verdict = "ERROR"
        elif result.get("fallback"):
            verdict = "FALLBACK"
        elif "check" not in q:
            verdict = "unchecked" if got else "EMPTY"
        else:
            check = make_check(q["check"])
            must = {a for a, fact in facts.items() if check(fact)}
            missed = sorted(must - got)
            wrong = sorted(a for a in got if not check(facts[a])) if q.get("strict") else []
            verdict = "MISSES" if missed else "WRONG" if wrong else "EMPTY" if not got and must else "ok"
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"{q['id']:20} {len(got):5d} {len(must):5d} {len(missed):6d} {len(wrong):5d}  {verdict:9} {q['question']}")
        detail[q["id"]] = {"question": q["question"], "verdict": verdict, "found": len(got),
                           "must_find": len(must), "missed": missed, "wrong": wrong,
                           "sql": result.get("sql"), "error": result.get("error")}
    print("\n" + json.dumps(counts))
    if args.show:
        for qid, d in detail.items():
            for key in ("missed", "wrong"):
                if d[key]:
                    more = f" … +{len(d[key]) - args.show}" if len(d[key]) > args.show else ""
                    print(f"{qid}: {key} {len(d[key])}: {', '.join(d[key][:args.show])}{more}")
    if args.json:
        args.json.write_text(json.dumps(detail, indent=1))


if __name__ == "__main__":
    main()
