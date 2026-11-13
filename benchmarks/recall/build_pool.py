"""
benchmarks/recall/build_pool.py
-------------------------------
Build the candidate pool of every judged query in queries.yaml, and the
evidence card a judge reads for each candidate.

Input: a dataset snapshot (JSON lines, one object per dataset) exported from
the database with snapshot.sql. Output: pools.json

    {query_id: {"candidates": [accession, ...], "cards": {accession: card}}}

A card holds what a careful reader needs to decide relevance: name, uploader
fields, recorded modalities, tasks and diagnoses, ages, and the README — the
first part plus the passages around every pool keyword.

    python benchmarks/recall/build_pool.py SNAPSHOT.jsonl --out pools.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

HERE = Path(__file__).resolve().parent
README_HEAD = 600
SNIPPET_RADIUS = 140
MAX_SNIPPETS = 3


def load_snapshot(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.startswith("{")]


def _text_fields(d: dict) -> Dict[str, str]:
    return {
        "name": d.get("name") or "",
        "description": d.get("description") or "",
        "dx_status": d.get("dx_status") or "",
        "study_domain": d.get("study_domain") or "",
        "readme": d.get("readme") or "",
    }


def _min_age(d: dict):
    ages = [a for a in (d.get("age_min"), d.get("p_age_min")) if isinstance(a, (int, float))]
    return min(ages) if ages else None


def _max_age(d: dict):
    ages = [a for a in (d.get("age_max"), d.get("p_age_max")) if isinstance(a, (int, float))]
    return max(ages) if ages else None


def satisfies_datatypes(q: dict, d: dict) -> bool:
    have = set(d.get("datatypes") or [])
    any_of = q.get("require_datatypes_any")
    all_of = q.get("require_datatypes_all")
    if any_of and not have & set(any_of):
        return False
    if all_of and not set(all_of) <= have:
        return False
    return True


def structural_relevant(rule: dict, d: dict) -> bool:
    have = set(d.get("datatypes") or [])
    if rule.get("datatypes_all") and not set(rule["datatypes_all"]) <= have:
        return False
    if rule.get("datatypes_any") and not have & set(rule["datatypes_any"]):
        return False
    if rule.get("max_size_gb") is not None:
        size = d.get("size_bytes")
        if size is None or size > rule["max_size_gb"] * 1024 ** 3:
            return False
    return True


def pool_hits(q: dict, d: dict) -> List[str]:
    """Names of the pool signals this dataset triggers (empty: not a candidate)."""
    pool = q.get("pool") or {}
    hits: List[str] = []
    fields = _text_fields(d)
    short = {k: v for k, v in fields.items() if k != "readme"}
    # `text` searches every field; `text_short` skips the README, for words
    # that READMEs use in passing ("memory", "language") and that would
    # otherwise put most of the collection into the pool.
    for key, scope in (("text", fields), ("text_short", short)):
        for pattern in pool.get(key, []):
            rx = re.compile(pattern, re.IGNORECASE)
            for field, value in scope.items():
                if rx.search(value):
                    hits.append(f"text:{pattern}@{field}")
    for pattern in pool.get("diagnoses", []):
        if any(re.search(pattern, code or "", re.IGNORECASE) for code in d.get("diagnoses") or []):
            hits.append(f"diagnosis:{pattern}")
    for pattern in pool.get("tasks", []):
        if any(re.search(pattern, code or "", re.IGNORECASE) for code in d.get("tasks") or []):
            hits.append(f"task:{pattern}")
    for pattern in pool.get("datatypes", []):
        if any(re.search(pattern, code or "", re.IGNORECASE) for code in d.get("datatypes") or []):
            hits.append(f"datatype:{pattern}")
    if "age_below" in pool and (_min_age(d) is not None and _min_age(d) < pool["age_below"]):
        hits.append(f"age<{pool['age_below']}")
    if "age_above" in pool and (_max_age(d) is not None and _max_age(d) > pool["age_above"]):
        hits.append(f"age>{pool['age_above']}")
    return hits


def card(q: dict, d: dict, hits: List[str]) -> Dict[str, Any]:
    readme = (d.get("readme") or "").replace("\r", "")
    snippets: List[str] = []
    pool = q.get("pool") or {}
    for pattern in pool.get("text", []) + pool.get("text_short", []):
        for m in re.finditer(pattern, readme, re.IGNORECASE):
            if m.start() < README_HEAD:
                continue  # already inside the README head
            start, end = max(0, m.start() - SNIPPET_RADIUS), min(len(readme), m.end() + SNIPPET_RADIUS)
            snippet = " ".join(readme[start:end].split())
            if not any(snippet[:60] in s for s in snippets):
                snippets.append("…" + snippet + "…")
            if len(snippets) >= MAX_SNIPPETS:
                break
        if len(snippets) >= MAX_SNIPPETS:
            break
    tasks = d.get("tasks") or []
    return {
        "acc": d["acc"],
        "name": d.get("name"),
        "dx_status": d.get("dx_status"),
        "study_domain": d.get("study_domain"),
        "study_design": d.get("study_design"),
        "datatypes": d.get("datatypes"),
        "tasks": tasks[:25] + ([f"(+{len(tasks) - 25} more)"] if len(tasks) > 25 else []),
        "diagnoses": d.get("diagnoses"),
        "ages": [_min_age(d), _max_age(d)],
        "n_subjects": d.get("n_subjects"),
        "description": (d.get("description") or "")[:500] or None,
        "readme_head": " ".join(readme[:README_HEAD].split()) or None,
        "readme_snippets": snippets or None,
        "pool_signals": sorted(set(h.split("@")[0] for h in hits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--queries", type=Path, default=HERE / "queries.yaml")
    parser.add_argument("--out", type=Path, default=HERE / "pools.json")
    args = parser.parse_args()

    datasets = [d for d in load_snapshot(args.snapshot) if d.get("acc")]
    queries = yaml.safe_load(args.queries.read_text())["queries"]
    out: Dict[str, Any] = {}
    print(f"{'query':28} {'pool':>5} {'after modality':>15}")
    for q in queries:
        if q.get("kind") == "structural":
            relevant = [d["acc"] for d in datasets if structural_relevant(q["rule"], d)]
            out[q["id"]] = {"structural_relevant": relevant}
            print(f"{q['id']:28} {'—':>5} {len(relevant):>15}  (structural)")
            continue
        raw, kept = 0, {}
        for d in datasets:
            if not d.get("acc"):
                continue  # 3 local test datasets without an accession id
            hits = pool_hits(q, d)
            if not hits:
                continue
            raw += 1
            if satisfies_datatypes(q, d):
                kept[d["acc"]] = card(q, d, hits)
        out[q["id"]] = {"candidates": sorted(kept), "cards": kept}
        print(f"{q['id']:28} {raw:>5} {len(kept):>15}")
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    total = sum(len(v.get("candidates", [])) for v in out.values())
    print(f"\n{total} candidates to judge → {args.out}")


if __name__ == "__main__":
    main()
