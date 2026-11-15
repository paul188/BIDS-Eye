"""Cut the reuse candidates into batches for the question writers.

    python3 -I make_writer_batches.py [--size 40]  -> work/writer_batches/batch_NN.json

A writer sees the reusing paper only (title, abstract, the paragraphs that
name the dataset or the repository) plus the surnames of the dataset's
authors, so it can tell a reuse from the authors' own paper. It never sees
the dataset's title or README: the question must use the searcher's words,
not the uploader's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
PAPERS = HERE.parent / "recall" / "work" / "papers"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    args = parser.parse_args()
    ds_authors = {}
    for line in (PAPERS / "authors.jsonl").open():
        if line.startswith("{"):
            r = json.loads(line)
            ds_authors[r["acc"]] = r.get("authors") or []
    papers = []
    for line in (WORK / "reuse_candidates.jsonl").open():
        p = json.loads(line)
        papers.append({
            "pmcid": p["pmcid"], "title": p["title"], "year": p["year"], "journal": p["journal"],
            "paper_authors": p["authors"], "abstract": p["abstract"],
            "datasets": [{"acc": r["acc"], "dataset_authors": ds_authors.get(r["acc"], [])[:12]} for r in p["reused"]],
            "paragraphs_naming_the_dataset": [{"section": c["section"], "text": c["text"]} for c in p["contexts"]],
            "paragraphs_on_selection": [{"section": c["section"], "text": c["text"]} for c in p["selection"]],
            "other_paragraphs_naming_the_repository": [{"section": c["section"], "text": c["text"]} for c in p["repository"]]
            if len(p["contexts"]) < 2 else [],
        })
    out = WORK / "writer_batches"
    out.mkdir(exist_ok=True)
    # Papers already in a batch stay there: batches that were handed to a writer
    # are never rewritten, new papers go into new batches.
    batched, n = set(), 0
    for old in sorted(out.glob("batch_*.json")):
        batched.update(p["pmcid"] for p in json.loads(old.read_text()))
        n = max(n, int(old.stem.split("_")[1]))
    papers = sorted((p for p in papers if p["pmcid"] not in batched), key=lambda p: p["pmcid"])
    first_new = n + 1
    for start in range(0, len(papers), args.size):
        n += 1
        (out / f"batch_{n:02d}.json").write_text(json.dumps(papers[start:start + args.size], indent=1))
    print(f"{len(papers)} new papers -> batches {first_new}..{n} in {out} ({len(batched)} papers were already batched)")


if __name__ == "__main__":
    main()
