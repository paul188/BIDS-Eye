#!/usr/bin/env python3
"""Expected answers of the ordering gold requests (requests.yaml) → truth.json.

For each request: the set being ordered (every dataset, or the relevant
answers of a structural gold request), each member's value of the sort key
from facts.jsonl, and which datasets may fill the first N places. Ties count:
every dataset whose value equals the N-th one's may take that place.
Datasets without a value come last, as NULLS LAST does in the search.

    python3 build_truth.py      # in this folder; reads ../requests_truth.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PAGE = 20  # without a limit, the first page is what counts


def _values(facts: list[dict], key: str) -> dict[str, object]:
    """accession → value; dates as seconds, so that they compare as dates."""
    def value(raw):
        if raw is None or key != "published":
            return raw
        return datetime.fromisoformat(raw).timestamp()
    return {row["acc"]: value(row[key]) for row in facts}


def ranked(members: list[str], values: dict, descending: bool) -> list[str]:
    """Members by value, those without a value last; the accession breaks ties."""
    with_value = sorted((m for m in members if values.get(m) is not None),
                        key=lambda m: m)
    with_value.sort(key=lambda m: values[m], reverse=descending)
    return with_value + sorted(m for m in members if values.get(m) is None)


def acceptable_top(order: list[str], values: dict, n: int) -> list[str]:
    """The datasets that may stand in the first n places: the first n, and any
    tied with the n-th one."""
    if len(order) <= n:
        return list(order)
    cutoff = values.get(order[n - 1])
    top = order[:n]
    if cutoff is None:
        return top + [m for m in order[n:] if values.get(m) is None]
    return top + [m for m in order[n:] if values.get(m) == cutoff]


def main() -> None:
    spec = yaml.safe_load((HERE / "requests.yaml").read_text())["queries"]
    facts = [json.loads(line) for line in (HERE / "facts.jsonl").read_text().splitlines() if line]
    # A dataset without an accession id cannot be named in a run's results.
    facts = [row for row in facts if row["acc"]]
    gold = json.loads((HERE.parent / "requests_truth.json").read_text())["queries"]
    catalogue = [row["acc"] for row in facts]

    out: dict[str, dict] = {}
    for request in spec:
        base = request["base"]
        if base == "all":
            members, judged = catalogue, set(catalogue)
        else:
            truth = gold[base]
            if truth["method"] != "structural":
                raise SystemExit(f"{request['id']}: base {base} is not a structural request")
            members = list(truth["relevant"])
            judged = set(members) | set(truth.get("not_relevant", [])) | \
                set(truth.get("partially_relevant", [])) | set(truth.get("unsure", []))
        limit = request.get("limit")
        entry = {
            "question": request["question"], "base": base, "limit": limit,
            "sort_key": request.get("sort_key"), "direction": request.get("direction", "DESC"),
            "members": sorted(members),
            # Datasets the base request decided either way; any other returned
            # dataset is newer than the gold snapshot and is reported, not scored.
            "judged": sorted(judged),
        }
        key = request.get("sort_key")
        if key:
            values = _values(facts, key)
            order = ranked(members, values, request.get("direction", "DESC") == "DESC")
            n = limit or PAGE
            entry["n"] = n
            entry["order"] = order[:max(n, PAGE)]
            entry["acceptable_top"] = acceptable_top(order, values, n)
            entry["values"] = {m: values.get(m) for m in members}
        else:
            entry["n"] = limit or PAGE
        out[request["id"]] = entry

    note = ("Ordering gold requests (requests.yaml). Values from facts.jsonl (production, read-only, "
            "see facts.sql); base answer lists from ../requests_truth.json (snapshot 2026-11-14).")
    (HERE / "truth.json").write_text(json.dumps({"note": note, "queries": out}, indent=1) + "\n")
    for qid, entry in out.items():
        print(f"{qid:36} members={len(entry['members']):5} n={entry['n']:3} "
              f"top={len(entry.get('acceptable_top', [])):3}")


if __name__ == "__main__":
    main()
