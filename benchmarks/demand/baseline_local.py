"""Two plain searches over the catalogue snapshot, as a reference point.

    python3 baseline_local.py bm25      --out work/run_bm25.json
    python3 baseline_local.py embedding --out work/run_embedding.json

Neither is the BIDS-Eye pipeline. They answer "how far does ordinary keyword
or embedding search get on these questions", need no server and no LLM, and
write the same result format as ../recall/eval_recall.py, so
score_known_item.py scores them the same way. Each question gets the 100
best-ranked datasets.

bm25       Okapi BM25 over name, uploader fields, task labels and README
embedding  cosine similarity with BAAI/bge-base-en-v1.5 against the dataset
           vectors of ../recall/work/emb (name, fields, first 1,800 README
           characters; see ../recall/embed_datasets.py)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RECALL = HERE.parent / "recall"
DEPTH = 100
STOP = set("a an and are as at be but by for from has have i in is it of on or that the their there these this to want "
           "was were which with would need data dataset datasets find looking any some my me can so plan test".split())


def tokens(text: str) -> list:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP or len(t) < 2:
            continue
        for suffix in ("ing", "es", "s"):       # crude stemming: patients/patient, imaging/imag
            if len(t) > 4 and t.endswith(suffix):
                t = t[:-len(suffix)]
                break
        out.append(t)
    return out


def load_snapshot() -> list:
    rows = [json.loads(l) for l in (RECALL / "work" / "snapshot.jsonl").open() if l.startswith("{")]
    return [r for r in rows if r.get("acc")]


def bm25_runs(questions: list) -> dict:
    rows = load_snapshot()
    docs = []
    for r in rows:
        text = " ".join([
            (r.get("name") or "") * 3, r.get("dx_status") or "", r.get("study_domain") or "",
            r.get("study_design") or "", " ".join(r.get("datatypes") or []), " ".join(r.get("tasks") or []),
            " ".join(r.get("diagnoses") or []), r.get("description") or "", r.get("readme") or ""])
        docs.append(Counter(tokens(text)))
    n = len(docs)
    avg = sum(sum(d.values()) for d in docs) / n
    df = Counter(t for d in docs for t in d)
    k1, b = 1.2, 0.75
    out = {}
    for q in questions:
        scores = []
        terms = set(tokens(q["question"]))
        for r, d in zip(rows, docs):
            length = sum(d.values())
            s = 0.0
            for t in terms:
                f = d.get(t, 0)
                if f:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg))
            if s > 0:
                scores.append((s, r["acc"]))
        scores.sort(reverse=True)
        out[q["id"]] = [acc for _, acc in scores[:DEPTH]]
    return out


def embedding_runs(questions: list) -> dict:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    emb_dir = RECALL / "work" / "emb"
    doc = np.load(emb_dir / "doc_emb.npy")
    ids = json.loads((emb_dir / "doc_ids.json").read_text())
    model = SentenceTransformer(json.loads((emb_dir / "model.json").read_text())["model"])
    # bge-v1.5 wants this instruction in front of a retrieval query
    prefix = "Represent this sentence for searching relevant passages: "
    q_emb = model.encode([prefix + q["question"] for q in questions], batch_size=32,
                         normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ doc.T
    out = {}
    for q, row in zip(questions, sims):
        best = np.argsort(-row)[:DEPTH]
        out[q["id"]] = [ids[i] for i in best]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("bm25", "embedding"))
    parser.add_argument("--queries", type=Path, default=HERE / "demand_queries.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    questions = yaml.safe_load(args.queries.read_text())["queries"]
    runs = bm25_runs(questions) if args.method == "bm25" else embedding_runs(questions)
    meta = {"label": f"local baseline: {args.method} over the 2026-11-13 snapshot, top {DEPTH}",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    results = [{"id": q["id"], "question": q["question"], "returned": runs[q["id"]], "total": len(runs[q["id"]])}
               for q in questions]
    args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"wrote {args.out}: {len(results)} questions")


if __name__ == "__main__":
    main()
