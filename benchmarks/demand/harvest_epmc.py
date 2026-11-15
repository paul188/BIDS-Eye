"""Harvest open-access papers that mention OpenNeuro / OpenfMRI from Europe PMC.

Step 1 (`search`): list every open-access paper whose full text mentions the
repository, with its metadata             -> work/epmc_search.jsonl
Step 2 (`fulltext`): download each paper's full-text XML (resumable)
                                           -> work/epmc_xml/<PMCID>.xml

The XML is third-party data: it is only ever parsed (extract_mentions.py),
never executed, and it lives in its own folder.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
HEADERS = {"User-Agent": "dataset-catalogue-enrichment/0.1"}
QUERY = '("openneuro" OR "openfmri") AND OPEN_ACCESS:y'
KEEP = ("id", "source", "pmid", "pmcid", "doi", "title", "authorString", "journalInfo",
        "pubYear", "abstractText", "firstPublicationDate", "pubTypeList", "citedByCount")


def _get(url: str, timeout: int = 60) -> bytes:
    error: Exception = RuntimeError("no attempt")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            error = e
        except Exception as e:  # timeouts, resets
            error = e
        time.sleep(3 * (attempt + 1))
    raise error


def search(query: str, out: Path) -> None:
    cursor, n = "*", 0
    with out.open("w") as fh:
        while True:
            params = {"query": query, "resultType": "core", "format": "json",
                      "pageSize": 500, "cursorMark": cursor}
            page = json.loads(_get(SEARCH_URL + "?" + urllib.parse.urlencode(params)))
            results = page.get("resultList", {}).get("result", [])
            for r in results:
                row = {k: r.get(k) for k in KEEP if r.get(k) is not None}
                row["journal"] = (r.get("journalInfo") or {}).get("journal", {}).get("title")
                row.pop("journalInfo", None)
                fh.write(json.dumps(row) + "\n")
                n += 1
            nxt = page.get("nextCursorMark")
            print(f"{n} / {page.get('hitCount')}", flush=True)
            if not results or not nxt or nxt == cursor:
                break
            cursor = nxt
            time.sleep(0.4)
    print("done:", n, "papers ->", out)


def fulltext(search_file: Path, xml_dir: Path, pause: float) -> None:
    xml_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in search_file.open()]
    pmcids = sorted({r["pmcid"] for r in rows if r.get("pmcid")})
    missing_log = WORK / "epmc_no_fulltext.txt"
    known_missing = set(missing_log.read_text().split()) if missing_log.exists() else set()
    done = fetched = 0
    for pmcid in pmcids:
        target = xml_dir / f"{pmcid}.xml"
        if target.exists() or pmcid in known_missing:
            done += 1
            continue
        try:
            data = _get(FULLTEXT_URL.format(pmcid=pmcid))
            tmp = target.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(target)
            fetched += 1
        except urllib.error.HTTPError as e:
            if e.code == 404:
                with missing_log.open("a") as fh:
                    fh.write(pmcid + "\n")
            else:
                print("failed", pmcid, e, file=sys.stderr, flush=True)
        except Exception as e:
            print("failed", pmcid, e, file=sys.stderr, flush=True)
        done += 1
        if done % 50 == 0:
            print(f"{done} / {len(pmcids)} (new {fetched})", flush=True)
        time.sleep(pause)
    print(f"done: {len(pmcids)} PMCIDs, {fetched} newly fetched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--query", default=QUERY)
    s.add_argument("--out", type=Path, default=WORK / "epmc_search.jsonl")
    f = sub.add_parser("fulltext")
    f.add_argument("--search", type=Path, default=WORK / "epmc_search.jsonl")
    f.add_argument("--xml-dir", type=Path, default=WORK / "epmc_xml")
    f.add_argument("--pause", type=float, default=0.25)
    args = parser.parse_args()
    if args.cmd == "search":
        search(args.query, args.out)
    else:
        fulltext(args.search, args.xml_dir, args.pause)


if __name__ == "__main__":
    main()
