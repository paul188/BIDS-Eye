"""Compare what people ask for with what the recall benchmark tests.

For every criterion: its share of the real requests (requests.yaml), how many
benchmark questions carry it, and the recall the search reaches on those
questions in a given run of the development set.

    python3 benchmarks/demand/coverage.py [run.json]

Benchmark questions are tagged from their text with the patterns below; the
tagging is rough on purpose (it only has to show which criteria are missing).
Held-out questions are counted, never listed, and never scored here.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RECALL = HERE.parent / "recall"

# criterion -> pattern over the question text (benchmark side)
PATTERNS = {
    "modality": r"\b(f?mri|eeg|meg|pet|ieeg|ecog|intracranial|fnirs|dwi|dti|diffusion|asl|arterial spin|spectroscopy|mrs|"
                r"structural|t1w?|optical|qsm|tms|imaging|scans?|recordings?)\b",
    "task": r"\b(task|tasks|paradigm|oddball|n-?back|stop-signal|flanker|learning|recall|naming|imagery|decision|"
            r"discounting|gambl|conditioning|reactivity|perception|processing|regulation|search|listening|watch|viewing|"
            r"tapping|ssvep|memory|language|reward|attention|interoception|music|speech|conversation|dialogue)\b",
    "rest": r"\brest(ing)?([- ]state)?\b",
    "stimulus": r"\b(movie|movies|stories|story|podcast|narrative|videos?|images|faces?|chord|words|speech|music)\b",
    "design": r"\b(block[- ]design|event[- ]related|randomi[sz]ed|rest-task-rest|cross-sectional)\b",
    "population": r"(alzheimer|epilep|parkinson|schizo|bipolar|autis|adhd|depress|stroke|aphasia|tumou?r|sclerosis|"
                  r"dementia|anxiety|pain|migraine|tbi|concussion|injury|dyslexia|neuralgia|aneurysm|dysplasia|psychosis|"
                  r"fragile|tremor|sickle|patients?|disorder|blind|twins?|musicians|bilingual|experts)",
    "healthy": r"\b(healthy|controls?|normal|cognitively healthy)\b",
    "controls": r"(vs\.? controls?|versus controls?|plus a control group|and controls|and healthy controls|patients and)",
    "age": r"\b(child|children|adolescen|teen|infant|toddler|newborn|fetal|older|elderly|aged?|ageing|aging|adults?|"
           r"young|youth|paediatric|pediatric|school-age|life-?span|years? old)\b",
    "sex": r"\b(male|female|men|women|sex|gender)\b",
    "species": r"\b(mouse|mice|rats?|rodent|monkey|macaque|primate|animal|non-human|nhp|human subjects)\b",
    "sample_size": r"(at least \d+ (subjects|participants)|more than \d+ (subjects|participants|healthy)|large cohort|"
                   r"thousand participants|\d+ subjects|sample size)",
    "subgroup_count": r"(at least \w+ (adult|healthy|patients)|more than \d+ healthy)",
    "sessions": r"\b(sessions?|longitudinal|test-retest|repeated|follow-up|over time|multiple days|several days)\b",
    "derivatives": r"\b(preprocessed|derivatives?|connectivity matri|connectomes?|segmentation|brain extracted)\b",
    "pipeline": r"\b(fmriprep|mriqc|freesurfer|qsiprep|xcp)\b",
    "acquisition": r"(\btesla\b|\b\d(\.\d)?\s?t\b|\btr\b|repetition time|multi-?echo|multi-?shell|mp2?-?rage|voxel|"
                   r"sampling|siemens|philips|\bge\b|scanner|ultra-?high|field strength|high-density|sequence\b)",
    "cooccurrence": r"(both .* and |same (subjects?|session|people|time)|simultaneous|recorded together|paired|"
                    r"that also have|combined|together with)",
    "auxiliary": r"\b(physio|physiological|eye[- ]?tracking|digiti[sz]ation|electrode positions|stimulus files)\b",
    "amount": r"(\bgb\b|no larger than|hours of|duration|trials per|many directions)",
    "phenotype": r"\b(questionnaire|assessments?|scores?|audit|bdi|cognitive test)\b",
    "labels": r"\b(label+ed|ground truth|classification|classify)\b",
    "licence": r"\b(licen[cs]e|cc0|redistribut|copyright)\b",
    "format": r"\b(png|jpg|graphml|nifti|\.nii|edf|brainvision)\b",
    "bids_valid": r"\bbids[- ]valid",
    "quality": r"\b(quality|qc|artifact|complete (task )?events)\b",
    "linked_paper": r"\b(paper|publication|data descriptor|published in|cited)\b",
    "like_dataset": r"\b(like|similar to) (abide|hcp|deap|adni|[A-Z]{3,})",
    "region": r"\b(cortex|hippocamp|cerebell|spinal|striatum|amygdala|brainstem|subthalamic)\b",
    "site": r"\b(university|hospital|institute|collected (at|in)|scanned at|harvard|stanford|lab\b|country|countries)\b",
    "author": r"\b(authored|coauthored|by [A-Z][a-z]+ (et al|lab)|'s lab)\b",
    "date": r"\b(published (in|since|after|before)|last \d+ (months|years)|recent|newest|since 20\d\d)\b",
    "exclusion": r"\b(no|not|without|excluding|only|but not)\b",
    "numeric": r"\d",
}
COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def tag(question: str) -> set:
    return {name for name, rx in COMPILED.items() if rx.search(question)}


def per_question_recall(run_path: Path) -> dict:
    truth = json.loads((RECALL / "ground_truth.json").read_text())
    truth = truth.get("queries", truth)
    run = {r["id"]: r for r in json.loads(run_path.read_text())["results"]}
    out = {}
    for qid, t in truth.items():
        if qid not in run or not t.get("relevant"):
            continue
        relevant = set(t["relevant"])
        returned = run[qid].get("returned") or []
        out[qid] = len(relevant & set(returned)) / len(relevant)
    return out


def main() -> None:
    requests = yaml.safe_load((HERE / "requests.yaml").read_text())["requests"]
    demand = Counter(c for r in requests for c in r["criteria"])
    by_kind = defaultdict(Counter)
    for r in requests:
        for c in r["criteria"]:
            by_kind[r["id"][:2]][c] += 1
    n_kind = Counter(r["id"][:2] for r in requests)

    queries = yaml.safe_load((RECALL / "queries.yaml").read_text())["queries"]
    held = {l.strip() for l in (RECALL / "held_out_questions.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")}
    dev_tags = {q["id"]: tag(q["question"]) for q in queries if q["id"] not in held}
    held_count = Counter(c for q in queries if q["id"] in held for c in tag(q["question"]))
    dev_count = Counter(c for tags in dev_tags.values() for c in tags)

    recall = per_question_recall(Path(sys.argv[1])) if len(sys.argv) > 1 else {}

    names = sorted(set(demand) | set(PATTERNS), key=lambda c: -demand[c])
    print(f"{len(requests)} real requests ({n_kind['qf']} forum, {n_kind['qx']} capability, {n_kind['qp']} paper); "
          f"{len(dev_tags)} development and {len(held)} held-out benchmark questions\n")
    print(f"{'criterion':16} {'requests':>8} {'share':>6} {'forum':>6} {'capab.':>6} {'paper':>6} | "
          f"{'dev q':>6} {'held q':>6} {'bench share':>11} | {'dev recall':>10} {'(n scored)':>10}")
    total_bench = len(queries)
    for c in names:
        scored = [recall[q] for q, tags in dev_tags.items() if c in tags and q in recall]
        r = f"{sum(scored) / len(scored):9.0%}" if scored else "        —"
        print(f"{c:16} {demand[c]:8d} {demand[c] / len(requests):6.0%} {by_kind['qf'][c]:6d} {by_kind['qx'][c]:6d} "
              f"{by_kind['qp'][c]:6d} | {dev_count[c]:6d} {held_count[c]:6d} "
              f"{(dev_count[c] + held_count[c]) / total_bench:11.0%} | {r:>10} {len(scored):10d}")
    if recall:
        print(f"\nall scored development questions: {sum(recall.values()) / len(recall):.0%} (n={len(recall)})")


if __name__ == "__main__":
    main()
