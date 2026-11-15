"""What the reuse papers tell us about demand and about the catalogue.

    python3 benchmarks/demand/analyse_demand.py [run.json ...]

1. how often each criterion appears in the needs recovered from reuse papers
2. which datasets are reused most
3. how well the catalogue describes the datasets people reuse (title, README),
   and, for every run given, how often the search finds the reused dataset in
   each group (development questions only)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parent / "recall" / "work" / "snapshot.jsonl"


def text_group(d: dict) -> str:
    readme = len(d.get("readme") or "")
    bare_title = bool(re.fullmatch(r"\s*ds\d{6}\S*\s*", d.get("name") or "")) or not (d.get("name") or "").strip()
    if bare_title and readme < 1500:
        return "1 no real title, README under 1,500 characters"
    if readme == 0:
        return "2 title only, no README"
    if readme < 1500:
        return "3 README under 1,500 characters"
    return "4 README of 1,500 characters or more"


def main() -> None:
    queries = yaml.safe_load((HERE / "demand_queries.yaml").read_text())["queries"]
    reuse = [q for q in queries if q["source"] == "reuse_paper"]
    needs = [q for q in reuse if q["form"] == "need"]           # one per paper x purpose
    snapshot = {}
    for line in SNAPSHOT.open():
        if line.startswith("{"):
            d = json.loads(line)
            if d.get("acc"):
                snapshot[d["acc"]] = d

    papers = {q["paper"]["pmcid"] for q in needs}
    print(f"{len(needs)} needs from {len(papers)} reuse papers "
          f"({sum(1 for q in needs if q['id'].startswith('h'))} held out), "
          f"{len({a for q in needs for a in q['known_items']})} distinct datasets reused\n")

    print("criterion        needs  share")
    crit = Counter(c for q in needs for c in set(q["criteria"]))
    for c, n in crit.most_common():
        print(f"{c:15} {n:6d} {n / len(needs):6.0%}")

    print("\nmost reused datasets (papers in this sample)")
    per_ds = Counter(a for q in needs for a in q["known_items"])
    for acc, n in per_ds.most_common(20):
        d = snapshot.get(acc, {})
        print(f"{n:4d}  {acc}  {(d.get('name') or '')[:60]:60}  README {len(d.get('readme') or ''):6d} chars")
    reused_once = sum(1 for n in per_ds.values() if n == 1)
    print(f"{reused_once} of {len(per_ds)} datasets were reused by exactly one paper; "
          f"the top 10 account for {sum(n for _, n in per_ds.most_common(10)) / sum(per_ds.values()):.0%} of all reuses")

    print("\nhow the catalogue describes the reused datasets")
    groups = defaultdict(set)
    for acc in per_ds:
        groups[text_group(snapshot.get(acc, {}))].add(acc)
    all_groups = Counter(text_group(d) for d in snapshot.values())
    print(f"{'group':48} {'reused':>7} {'whole catalogue':>16}")
    for g in sorted(all_groups):
        print(f"{g[2:]:48} {len(groups[g]):7d} {all_groups[g]:16d}")

    for run_path in sys.argv[1:]:
        data = json.loads(Path(run_path).read_text())
        returned = {r["id"]: r.get("returned") or [] for r in data["results"]}
        print(f"\n{(data.get('meta') or {}).get('label')}: known dataset on the first page (top 20), development questions")
        by_group = defaultdict(list)
        for q in reuse:
            if q["id"].startswith("h") or q["id"] not in returned or len(q["known_items"]) != 1:
                continue
            acc = q["known_items"][0]
            by_group[text_group(snapshot.get(acc, {}))].append(acc in returned[q["id"]][:20])
        for g in sorted(by_group):
            v = by_group[g]
            print(f"  {g[2:]:48} {sum(v) / len(v):5.0%}  (n={len(v)})")


if __name__ == "__main__":
    main()
