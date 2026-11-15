"""Assemble the demand question set from the real requests and the reuse papers.

    python3 build_queries.py   -> demand_queries.yaml    (runnable with
                                  ../recall/eval_recall.py run --queries … --split all)
                                  demand_first_run.yaml (a subset for a first run)
                                  held_out_ids.txt

Reuse questions come in two forms with the same answer: `short` (keywords, as
typed into a search box) and `need` (one or two sentences). The known answer
is the dataset the paper's authors actually used (`known_items`); it is a
lower bound, other datasets may fit as well.

Three papers in ten are held out (picked by a hash of the paper id, so the
split is stable when papers are added). Their ids start with "h". The rules of
../recall/held_out_questions.txt apply to them.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
FORBIDDEN = re.compile(r"(ds\s?00\d{4}|openneuro|openfmri)", re.I)
HELD_OUT_SHARE = 3  # of 10
FIRST_RUN_PAPERS = 150
MAX_KNOWN = 40
SNAPSHOT = HERE.parent / "recall" / "work" / "snapshot.jsonl"

# Pairings found wrong on review (label work, done from the paper's text, not
# from any search result). The paper is dropped with the reason given.
EXCLUDED_PAPERS = {
    "PMC9467511": "cites ds000221 for resting EEG, but that EEG is not part of the OpenNeuro copy",
}

# A recording type named in the keyword form -> the catalogue's datatype.
MODALITY_WORDS = {
    "electroencephalography": r"\b(EEG|electroencephalogra\w*)\b",
    "magnetoencephalography": r"\b(MEG|magnetoencephalogra\w*)\b",
    "functional_mri": r"\b(fMRI|functional MRI)\b",
    "diffusion_mri": r"\b(DWI|DTI|diffusion[- ](MRI|weighted|tensor|imaging)|tractograph\w*)\b",
    "intracranial_eeg": r"\b(iEEG|ECoG|sEEG|stereo-?EEG|intracranial (EEG|electro\w+|recordings?)|electrocorticogra\w*)\b",
    "positron_emission_tomography": r"\b(PET|positron emission)\b",
    "fnirs": r"\b(fNIRS|near-infrared)\b",
}


def catalogue_datatypes() -> dict:
    """accession -> datatypes the catalogue records (empty when there is no snapshot)."""
    out = {}
    if SNAPSHOT.exists():
        for line in SNAPSHOT.open():
            if line.startswith("{"):
                d = json.loads(line)
                if d.get("acc"):
                    out[d["acc"]] = set(d.get("datatypes") or [])
    return out


def modality_not_in_catalogue(short: str, known: list, datatypes: dict) -> list:
    """Recording types the keyword form asks for that the catalogue records for
    none of the known datasets. Mostly a gap in the catalogue (EEG stored
    outside the BIDS folders), sometimes a wrong pairing; scored separately."""
    if not datatypes or any(a not in datatypes for a in known):
        return []
    have = set().union(*(datatypes[a] for a in known))
    asked = {m for m, pattern in MODALITY_WORDS.items() if re.search(pattern, short, re.I)}
    if "intracranial_eeg" in asked:
        asked.discard("electroencephalography")  # "intracranial EEG" is not a request for scalp EEG
    return sorted(asked - have)


def held_out(pmcid: str) -> bool:
    return int(hashlib.sha1(pmcid.encode()).hexdigest(), 16) % 10 < HELD_OUT_SHARE


def main() -> None:
    queries, stats = [], Counter()
    for r in yaml.safe_load((HERE / "requests.yaml").read_text())["requests"]:
        kind = {"qf": "forum_request", "qx": "capability_request", "qp": "paper_criteria"}[r["id"][:2]]
        queries.append({"id": r["id"], "source": kind, "form": "request", "question": r["question"],
                        "criteria": r["criteria"], "in_scope": r.get("in_scope", True),
                        "needs": r["needs"], "ref": r["ref"], "url": r["url"]})
        stats[kind] += 1

    candidates = {}
    cand_file = WORK / "reuse_candidates.jsonl"
    if cand_file.exists():
        for line in cand_file.open():
            p = json.loads(line)
            candidates[p["pmcid"]] = p
    verdicts = Counter()
    datatypes = catalogue_datatypes()
    for path in sorted((WORK / "written").glob("batch_*.jsonl")):
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            w = json.loads(line)
            verdicts[w.get("verdict")] += 1
            paper = candidates.get(w["pmcid"])
            if w.get("verdict") != "reuse" or paper is None:
                continue
            if w["pmcid"] in EXCLUDED_PAPERS:
                stats["dropped: pairing found wrong on review"] += len(w.get("questions") or [])
                continue
            allowed = {r["acc"]: r for r in paper["reused"]}
            for n, q in enumerate(w.get("questions") or [], 1):
                known = [a for a in q.get("accessions", []) if a in allowed]
                if not known:
                    stats["dropped: accession not in the paper's list"] += 1
                    continue
                if len(known) > MAX_KNOWN:
                    # "every MRI dataset on the site": not a need a search can be scored on
                    stats[f"dropped: more than {MAX_KNOWN} datasets pooled"] += 1
                    continue
                if FORBIDDEN.search(q.get("short", "") + " " + q.get("need", "")):
                    stats["dropped: names the dataset or repository"] += 1
                    continue
                prefix = "h" if held_out(w["pmcid"]) else ""
                base = f"{prefix}qr_{w['pmcid'].lower()}_{n}"
                shared = {
                    "source": "reuse_paper", "known_items": known,
                    "criteria": q.get("criteria", []), "requirements": q.get("requirements", []),
                    "confidence": q.get("confidence", "low"), "quote": q.get("quote", ""),
                    # the search could match the dataset through this very paper's text
                    "paper_linked_in_catalogue": any(allowed[a]["linked_role"] for a in known),
                    "paper": {"pmcid": w["pmcid"], "doi": paper.get("doi"), "year": paper.get("year"),
                              "title": paper.get("title")},
                }
                gap = modality_not_in_catalogue(q.get("short") or "", known, datatypes)
                if gap:
                    shared["modality_not_in_catalogue"] = gap
                    stats["flagged: known dataset lacks the requested modality in the catalogue"] += 1
                for form in ("short", "need"):
                    text = (q.get(form) or "").strip()
                    if text:
                        queries.append({"id": f"{base}_{form[0]}", "form": form, "question": text, **shared})
                        stats[f"reuse_paper {form}{' (held out)' if prefix else ''}"] += 1

    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids)), "duplicate ids"
    header = ("# Demand question set: real requests and needs recovered from papers that reused\n"
              "# OpenNeuro datasets. Built by build_queries.py; do not edit by hand.\n"
              "# Ids starting with 'h' are held out (held_out_ids.txt): do not tune on them.\n")
    (HERE / "demand_queries.yaml").write_text(
        header + yaml.safe_dump({"queries": queries}, sort_keys=False, allow_unicode=True, width=1000))
    (HERE / "held_out_ids.txt").write_text(
        "# Held-out demand questions (picked by hash of the paper id). Same rules as\n"
        "# ../recall/held_out_questions.txt: averages only, no tuning on them.\n"
        + "".join(i + "\n" for i in ids if i.startswith("h")))
    # A first run that fits in an afternoon: every real request, plus one
    # development paper per reused dataset (so no dataset dominates), both forms.
    seen, first = set(), [q for q in queries if q["source"] != "reuse_paper"]
    by_paper = {}
    for q in queries:
        if q["source"] == "reuse_paper" and not q["id"].startswith("h"):
            by_paper.setdefault(q["id"].rsplit("_", 1)[0], []).append(q)
    order = sorted(by_paper, key=lambda k: hashlib.sha1(k.encode()).hexdigest())
    picked = 0
    for key in order:
        items = set(by_paper[key][0]["known_items"])
        if items & seen or picked >= FIRST_RUN_PAPERS:
            continue
        seen |= items
        first.extend(by_paper[key])
        picked += 1
    (HERE / "demand_first_run.yaml").write_text(
        "# Subset of demand_queries.yaml for a first run: all real requests plus one\n"
        "# development paper per reused dataset. Built by build_queries.py.\n"
        + yaml.safe_dump({"queries": first}, sort_keys=False, allow_unicode=True, width=1000))
    print(f"{len(first)} questions -> demand_first_run.yaml ({picked} reuse papers)", file=sys.stderr)
    print(f"{len(queries)} questions -> demand_queries.yaml", file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"{v:6d}  {k}", file=sys.stderr)
    if verdicts:
        print("writer verdicts:", dict(verdicts), file=sys.stderr)


if __name__ == "__main__":
    main()
