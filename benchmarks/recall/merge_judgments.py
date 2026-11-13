"""
benchmarks/recall/merge_judgments.py
------------------------------------
Combine the judge output (one JSON line per query × candidate) and the
structural queries into ground_truth.json, the file eval_recall.py scores
against. Also writes judgments.jsonl, the human-readable record with the
reason behind every label, so labels can be reviewed and corrected.

    python benchmarks/recall/merge_judgments.py POOLS.json JUDGED_DIR --snapshot-date 2026-11-13
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pools", type=Path)
    parser.add_argument("judged_dir", type=Path)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--judged-by", default="LLM judge (Claude Sonnet), criteria in queries.yaml; not yet reviewed by a human")
    args = parser.parse_args()

    queries = yaml.safe_load((HERE / "queries.yaml").read_text())["queries"]
    pools = json.loads(args.pools.read_text())
    labels = {}
    for path in sorted(args.judged_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[(row["query"], row["acc"])] = row

    truth, missing = {}, []
    for q in queries:
        qid = q["id"]
        meta = {"source": q.get("source"), "type": q.get("type")}
        if q.get("kind") == "structural":
            truth[qid] = {"method": "structural", **meta, "relevant": pools[qid]["structural_relevant"]}
            continue
        # partially_relevant: the dataset meets the topic but misses one
        # constraint of the question (graded relevance as in bioCADDIE 2016).
        entry = {"method": "judged", **meta, "relevant": [], "partially_relevant": [],
                 "not_relevant": [], "unsure": []}
        # Every pooled candidate must have a label; labels from earlier rounds
        # whose candidate is no longer in the (shallower) pool are kept too.
        earlier = sorted(a for (q, a) in labels if q == qid)
        for acc in dict.fromkeys(list(pools[qid]["candidates"]) + earlier):
            row = labels.get((qid, acc))
            if row is None:
                missing.append((qid, acc))
                continue
            entry[row["label"]].append(acc)
        truth[qid] = entry
    if missing:
        raise SystemExit(f"{len(missing)} candidates have no judgment, e.g. {missing[:3]}")

    (HERE / "ground_truth.json").write_text(json.dumps({
        "snapshot_date": args.snapshot_date,
        "judged_by": args.judged_by,
        "note": "Relevance of datasets outside each query's pool is assumed false. "
                "Structural truth is computed from recorded modalities and sizes on the snapshot date.",
        "queries": truth,
    }, indent=1))
    with (HERE / "judgments.jsonl").open("w") as out:
        for key in sorted(labels):
            out.write(json.dumps(labels[key], ensure_ascii=False) + "\n")

    counts = Counter()
    for t in truth.values():
        for label in ("relevant", "partially_relevant", "not_relevant", "unsure"):
            counts[label] += len(t.get(label, []))
    print(f"{len(truth)} queries, {len(labels)} judgments: {dict(counts)}")
    print("relevant per query:", {k: len(v["relevant"]) for k, v in truth.items()})


if __name__ == "__main__":
    main()
