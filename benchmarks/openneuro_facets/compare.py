"""
benchmarks/openneuro_facets/compare.py
--------------------------------------
Compare BIDS-Eye's answers with OpenNeuro's filter results.

    python benchmarks/openneuro_facets/compare.py \\
        --openneuro openneuro_results.json --run run_live.json --facts db_facts.jsonl

db_facts.jsonl has one line per dataset in the BIDS-Eye database (see
dump_facts.sql). Each dataset that either side returns is checked against
its own data with the `check` of the question (facets.yaml), so a
disagreement can be settled: a dataset is "correct" when the data says it
meets every condition of the question.

Only datasets that are in the BIDS-Eye database count; OpenNeuro datasets
published after the last crawl are reported separately.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional

import yaml

HERE = Path(__file__).resolve().parent

DATATYPES = {
    "mri": {"anatomical_mri", "functional_mri", "diffusion_mri", "perfusion_asl", "field_maps"},
    "func": {"functional_mri"},
    "dwi": {"diffusion_mri"},
    "perf": {"perfusion_asl"},
    "pet": {"positron_emission_tomography"},
    "eeg": {"electroencephalography"},
    "ieeg": {"intracranial_eeg"},
    "meg": {"magnetoencephalography"},
    "nirs": {"fnirs"},
}

NOT_ANIMAL = re.compile(r"human|homo sapiens|phantom|synthetic|printed|fibre|fiber", re.I)
SPECIES = {
    "rat": lambda s: bool(s) and re.search(r"\brats?\b|rattus", s, re.I) is not None,
    "mouse": lambda s: bool(s) and re.search(r"\b(mouse|mice|mus)\b", s, re.I) is not None,
    "rodent": lambda s: bool(s) and re.search(r"\b(rats?|rattus|mouse|mice|mus)\b", s, re.I) is not None,
    "other_animal": lambda s: bool(s) and s.strip().lower() not in ("", "other") and not NOT_ANIMAL.search(s)
    and not re.search(r"\brats?\b|rattus|\b(mouse|mice|mus)\b", s, re.I),
}

# diagnosis: (pattern on the uploader's dx_status, participant diagnosis codes)
DIAGNOSES = {
    "healthy": (r"healthy|control", {"healthy_control"}),
    "schizophrenia": (r"schizo", {"schizophrenia", "schizoaffective_disorder"}),
    "adhd": (r"adhd|add/|attention.deficit", {"adhd"}),
    "alzheimer": (r"alzheimer", {"alzheimers_disease", "alzheimer_disease", "alzheimers"}),
}

# task: pattern on raw task labels (from file names) and on normalised codes
TASKS = {
    "rest": (r"rest", r"^resting_state"),
    "nback": (r"n?back", r"n_back"),
    "stopsignal": (r"stop", r"stop_signal"),
    "flanker": (r"flank", r"flanker"),
    "faces": (r"face", r"face"),
    "movie": (r"movie|film", r"movie"),
    "sleep": (r"sleep|nap", r"sleep"),
    "oddball": (r"oddball", r"oddball"),
    "gonogo": (r"go_?no_?go|nogo", r"go_no_go|gonogo"),
    "motor": (r"motor", r"motor"),
    "mid": (r"^mid$|monetary|incentive", r"monetary_incentive"),
}

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


def in_range(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> bool:
    return value is not None and (lo is None or value >= lo) and (hi is None or value <= hi)


def make_check(spec: dict) -> Callable[[dict], bool]:
    tests: List[Callable[[dict], bool]] = []
    for key, value in spec.items():
        if key == "modality":
            wanted = set().union(*(DATATYPES[v] for v in value))
            tests.append(lambda f, w=wanted: bool(set(f.get("dts") or []) & w))
        elif key == "species":
            tests.append(lambda f, t=SPECIES[value]: t(f.get("species") or ""))
        elif key == "diagnosis":
            pattern, codes = DIAGNOSES[value]
            tests.append(lambda f, p=pattern, c=codes: bool(re.search(p, f.get("dx_status") or "", re.I))
                         or bool(set(f.get("dx") or []) & c))
        elif key == "section":
            pattern = r"longitudinal" if value == "longitudinal" else r"cross"
            tests.append(lambda f, p=pattern: bool(re.search(p, f.get("long") or "", re.I)))
        elif key == "sex":
            tests.append(lambda f: any((s or "").lower() in ("f", "female") for s in f.get("sexes") or []))
        elif key == "age":
            lo, hi = value
            def age_test(f, lo=lo, hi=hi):
                if any(in_range(float(a), lo, hi) for a in f.get("ages") or []):
                    return True
                amin, amax = f.get("age_min"), f.get("age_max")
                if amin is None and amax is None:
                    return False
                amin = amin if amin is not None else amax
                amax = amax if amax is not None else amin
                return (hi is None or amin <= hi) and (lo is None or amax >= lo)
            tests.append(age_test)
        elif key == "subjects":
            lo, hi = value
            tests.append(lambda f, lo=lo, hi=hi: in_range(f.get("n_sub") or None, lo, hi)
                         or in_range(f.get("n_part") or None, lo, hi))
        elif key == "task":
            raw, code = TASKS[value]
            tests.append(lambda f, r=raw, c=code: any(re.search(r, t, re.I) for t in f.get("raw_tasks") or [])
                         or any(re.search(c, t) for t in f.get("tasks") or []))
        elif key == "domain":
            tests.append(lambda f, v=value: v.lower().split("'")[0] in (f.get("domain") or "").lower())
        elif key == "tracer":
            pattern = {"fdg": r"fdg|fluorodeoxy", "pib": r"pib|pittsburgh", "c11": r"11c|c11|c-11|carbon"}[value]
            tests.append(lambda f, p=pattern: any(re.search(p, t, re.I) for t in (f.get("tracer") or []) + (f.get("nuclide") or [])))
        elif key == "pet_manufacturer":
            tests.append(lambda f, v=value: any(re.search(rf"\b{v}", m, re.I) for m in f.get("pet_manu") or []))
        elif key == "published_within_days":
            cutoff = (NOW - timedelta(days=value)).strftime("%Y-%m-%d")
            tests.append(lambda f, c=cutoff: bool(f.get("publish")) and f["publish"][:10] >= c)
        elif key == "author":
            tests.append(lambda f, v=value: any(v in (a or "").lower() for a in (f.get("authors") or []) + [f.get("senior") or ""]))
        else:
            raise KeyError(key)
    return lambda f: all(t(f) for t in tests)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=HERE / "facets.yaml")
    parser.add_argument("--openneuro", type=Path, default=HERE / "openneuro_results.json")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="write per-query detail here")
    parser.add_argument("--show", type=int, default=6, help="ids to list per disagreement")
    args = parser.parse_args()

    queries = yaml.safe_load(args.queries.read_text())["queries"]
    on = json.loads(args.openneuro.read_text())["results"]
    run = {r["id"]: r for r in json.loads(args.run.read_text())["results"]}
    facts = {}
    for line in args.facts.read_text().splitlines():
        if line.strip():
            f = json.loads(line)
            facts[f["acc"]] = f

    header = (f"{'query':22} {'ON':>5} {'ON∩db':>6} {'ONok':>5} {'ours':>5} {'ok':>5} "
              f"{'ONok∖ours':>9} {'ours ok∖ON':>10}  verdict")
    print(header)
    detail = {}
    totals = {"on_ok": 0, "ours_ok_of_on_ok": 0}
    for q in queries:
        qid = q["id"]
        if qid not in on or qid not in run:
            continue
        check = make_check(q["check"])
        on_ids = set(on[qid]["ids"])
        on_db = {a for a in on_ids if a in facts}
        ours = [a for a in run[qid].get("returned") or [] if a in facts]
        ours_set = set(ours)
        on_ok = {a for a in on_db if check(facts[a])}
        ours_ok = {a for a in ours_set if check(facts[a])}
        missed_ok = sorted(on_ok - ours_set)            # OpenNeuro right, we miss it
        missed_bad = sorted((on_db - on_ok) - ours_set)  # OpenNeuro returns, data says no
        extra_ok = sorted(ours_ok - on_ids)              # we find more correct ones
        extra_bad = sorted((ours_set - ours_ok) - on_ids)
        if run[qid].get("error"):
            verdict = "ERROR " + run[qid]["error"][:40]
        elif run[qid].get("fallback"):
            verdict = "FALLBACK"
        elif len(ours_ok) >= len(on_ok) and not missed_ok:
            verdict = "ok"
        elif len(ours_ok) >= len(on_ok):
            verdict = f"more, but misses {len(missed_ok)}"
        else:
            verdict = "FEWER"
        totals["on_ok"] += len(on_ok)
        totals["ours_ok_of_on_ok"] += len(on_ok & ours_set)
        print(f"{qid:22} {len(on_ids):5d} {len(on_db):6d} {len(on_ok):5d} {len(ours_set):5d} {len(ours_ok):5d} "
              f"{len(missed_ok):9d} {len(extra_ok):10d}  {verdict}")
        detail[qid] = {
            "question": q["question"], "check": q["check"], "openneuro": q["openneuro"],
            "on_total": len(on_ids), "on_not_in_db": sorted(on_ids - set(facts)),
            "on_ok": sorted(on_ok), "ours": ours, "ours_ok": sorted(ours_ok),
            "missed_ok": missed_ok, "missed_bad": missed_bad, "extra_ok": extra_ok, "extra_bad": extra_bad,
            "sql": run[qid].get("sql"), "explanation": run[qid].get("explanation"),
            "verdict": verdict,
        }
    if totals["on_ok"]:
        print(f"\nOf {totals['on_ok']} correct OpenNeuro hits (summed over questions) BIDS-Eye returns "
              f"{totals['ours_ok_of_on_ok']} ({totals['ours_ok_of_on_ok'] / totals['on_ok']:.0%}).")
    print("\nColumns: ON = OpenNeuro total; ON∩db = of those, in our database; ONok = of those, correct by the data;\n"
          "ours = BIDS-Eye results; ok = correct by the data; ONok∖ours = correct OpenNeuro hits we miss;\n"
          "ours ok∖ON = correct results OpenNeuro does not return.")
    if args.show:
        for qid, d in detail.items():
            if d["missed_ok"] or d["missed_bad"] or d["extra_bad"]:
                print(f"\n{qid}: {d['question']}")
                for key in ("missed_ok", "missed_bad", "extra_bad"):
                    if d[key]:
                        more = f" … +{len(d[key]) - args.show}" if len(d[key]) > args.show else ""
                        print(f"  {key:10} {len(d[key]):4d}  {', '.join(d[key][:args.show])}{more}")
    if args.json:
        args.json.write_text(json.dumps(detail, indent=1))


if __name__ == "__main__":
    main()
