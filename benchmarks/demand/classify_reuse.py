"""Split accession mentions into the dataset's own paper and reuse by others.

A paper that names an OpenNeuro accession is either the paper of the people
who shared the dataset, or a paper by others who went looking for data and
picked this dataset. Only the second kind tells us what searchers need.

    python3 -I classify_reuse.py   -> work/reuse_candidates.jsonl
                                      (one line per paper with >= 1 reused dataset)

Evidence used, in this order:
  linked   the paper is already linked to the dataset in the catalogue
           (benchmarks/recall/work/papers/roles.json): role 'primary' = own
           paper; any other role is kept but flagged, because the search can
           match the dataset through that paper's text
  authors  the paper shares authors with the dataset: own paper
  else     reuse candidate (a writer still reads the text and can reject it)
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
PAPERS = HERE.parent / "recall" / "work" / "papers"
SNAPSHOT = HERE.parent / "recall" / "work" / "snapshot.jsonl"

BODY_KINDS = {"p", "table-wrap"}
# sections where a mention says "our data is here", not "we used their data"
AVAILABILITY = re.compile(r"(data|code|resource)s? (and \w+ )?(availability|accessibility|sharing)|availability|"
                          r"open practices|key resources", re.I)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def surname_set(names) -> set:
    """Surnames of the dataset authors, whatever the format they were typed in."""
    out = set()
    for name in names:
        name = re.sub(r"\([^)]*\)", " ", name)             # "(data curation)"
        name = re.sub(r"[^A-Za-z ,\-']", " ", _ascii(name)).strip()
        if "," in name:                                   # "Poldrack, R.A."
            parts = name.split(",")[0].split()
            last = parts[-1] if parts else ""
        else:
            parts = name.split()
            if not parts:
                continue
            # "Poldrack RA": the last token is initials; "Russell A. Poldrack": it is the surname
            initials = len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3
            last = parts[0] if initials else parts[-1]
        if len(last) >= 2:                                # Li, Ma, Gu are surnames
            out.add(last.lower())
    return out


def paper_surnames(author_string: str) -> set:
    # Europe PMC: "Markiewicz CJ, Gorgolewski KJ, Feingold F, ..."
    out = set()
    for a in (author_string or "").rstrip(".").split(","):
        parts = a.strip().split()
        if len(parts) >= 2:
            s = re.sub(r"[^a-z \-']", "", _ascii(" ".join(parts[:-1])).lower()).split()
            if s and len(s[-1]) >= 2:
                out.add(s[-1])
    return out


def main() -> None:
    catalogue = set()
    for line in SNAPSHOT.open():
        if line.startswith("{"):
            catalogue.add(json.loads(line)["acc"])
    roles = json.loads((PAPERS / "roles.json").read_text())
    ds_authors = {}
    for line in (PAPERS / "authors.jsonl").open():
        if line.startswith("{"):
            r = json.loads(line)
            ds_authors[r["acc"]] = surname_set(r.get("authors") or [])

    stats = Counter()
    out = (WORK / "reuse_candidates.jsonl").open("w")
    for line in (WORK / "mentions.jsonl").open():
        paper = json.loads(line)
        if not paper["accession_counts"]:
            continue
        stats["papers naming an accession"] += 1
        keys = {f"doi:{(paper.get('doi') or '').lower()}", f"pmid:{paper.get('pmid')}", f"pmcid:{paper['pmcid'].lower()}"}
        authors = paper_surnames(paper.get("authors"))
        reused, own, outside = [], [], []
        for acc in sorted(paper["accession_counts"]):
            if acc not in catalogue:
                outside.append(acc)
                continue
            linked = [r for r in roles.get(acc, []) if r["key"].lower() in keys]
            shared = authors & ds_authors.get(acc, set())
            n_ds = len(ds_authors.get(acc, set()))
            body = [m for m in paper["mentions"] if acc in m["accessions"] and m["kind"] in BODY_KINDS]
            only_availability = bool(body) and all(AVAILABILITY.search(m["section"]) for m in body)
            if any(r["role"] == "primary" for r in linked) or len(shared) >= 2 or (shared and n_ds <= 3):
                own.append(acc)
            else:
                reused.append({"acc": acc, "linked_role": linked[0]["role"] if linked else None,
                               "shared_authors": sorted(shared), "body_mentions": len(body),
                               "only_in_availability": only_availability})
        stats["accessions: own paper"] += len(own)
        stats["accessions: outside the catalogue"] += len(outside)
        stats["accessions: reuse candidate"] += len(reused)
        if not reused:
            stats["papers without a reused dataset"] += 1
            continue
        stats["papers with >= 1 reused dataset"] += 1
        stats[f"  reusing {min(len(reused), 5)}{'+' if len(reused) >= 5 else ''} dataset(s)"] += 1
        reused_ids = {r["acc"] for r in reused}
        out.write(json.dumps({
            "pmcid": paper["pmcid"], "doi": paper.get("doi"), "title": paper.get("title"), "year": paper.get("year"),
            "journal": paper.get("journal"), "authors": paper.get("authors"), "abstract": paper.get("abstract"),
            "reused": reused, "own": own, "outside_catalogue": outside,
            "contexts": [m for m in paper["mentions"] if reused_ids & set(m["accessions"]) and m["kind"] in BODY_KINDS][:8],
            "selection": paper.get("selection", [])[:4], "repository": paper.get("repository", [])[:4],
        }) + "\n")
    for k, v in stats.items():
        print(f"{v:6d}  {k}", file=sys.stderr)


if __name__ == "__main__":
    main()
