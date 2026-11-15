"""Find OpenNeuro accession numbers in the harvested full texts.

For every paper: which accessions it names, in which section, and the
paragraph around each mention. Also keeps paragraphs where the authors say how
they searched for or selected datasets (`selection`) and, as context, up to
six other paragraphs that name the repository (`repository`).

    python3 -I extract_mentions.py        -> work/mentions.jsonl

The XML is parsed only (no entity expansion beyond the standard five).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"

ACCESSION = re.compile(r"\bds\s?0{2}(\d{4})\b", re.I)
# openneuro.org/datasets/ds004504 or doi:10.18112/openneuro.ds004504.v1.0.2
SELECTION = re.compile(
    r"(inclusion criteri|exclusion criteri|were (selected|included|excluded|chosen)|we (selected|searched|chose|"
    r"identified|included|excluded)|selection criteri|datasets? (were|was) (selected|chosen|identified)|"
    r"search(ed)? (on|of|through|in) (openneuro|open ?fmri)|following criteri|at least \w+ (subjects|participants))",
    re.I)
REPOSITORY = re.compile(r"(openneuro|open ?fmri)", re.I)
MAX_PARAGRAPH = 1800


def _text(node: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _section_path(node: ET.Element, parents: dict) -> str:
    titles = []
    while node is not None:
        if node.tag in ("sec", "abstract", "back", "ack", "notes", "ref-list", "fn-group", "app"):
            title = node.find("title")
            label = _text(title) if title is not None else node.tag
            if node.tag == "sec" and node.get("sec-type"):
                label = f"{label} [{node.get('sec-type')}]"
            titles.append(label[:80])
        node = parents.get(node)
    return " > ".join(reversed(titles))


def paragraphs(root: ET.Element):
    parents = {child: parent for parent in root.iter() for child in parent}
    for node in root.iter():
        if node.tag in ("p", "ref", "table-wrap", "title", "article-title"):
            # skip a <p> nested in a table or reference: its container is yielded
            anc, nested = parents.get(node), False
            while anc is not None:
                if anc.tag in ("ref", "table-wrap") or (anc.tag == "p" and node.tag == "p"):
                    nested = True
                    break
                anc = parents.get(anc)
            if nested:
                continue
            text = _text(node)
            if text:
                yield node.tag, _section_path(node, parents), text


def clip(text: str, match_span) -> str:
    if len(text) <= MAX_PARAGRAPH:
        return text
    start = max(0, match_span[0] - MAX_PARAGRAPH // 2)
    return ("… " if start else "") + text[start:start + MAX_PARAGRAPH] + " …"


def main() -> None:
    meta = {}
    for line in (WORK / "epmc_search.jsonl").open():
        r = json.loads(line)
        if r.get("pmcid"):
            meta[r["pmcid"]] = r
    out = (WORK / "mentions.jsonl").open("w")
    n_files = n_with = bad = 0
    for path in sorted((WORK / "epmc_xml").glob("PMC*.xml")):
        n_files += 1
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            bad += 1
            continue
        mentions, selection, repository = [], [], []
        counts: dict = {}
        for tag, section, text in paragraphs(root):
            found = list(ACCESSION.finditer(text))
            if found:
                accs = sorted({"ds00" + m.group(1) for m in found})
                for a in accs:
                    counts[a] = counts.get(a, 0) + 1
                mentions.append({"kind": tag, "section": section, "accessions": accs,
                                 "text": clip(text, found[0].span())})
            elif tag == "p" and REPOSITORY.search(text):
                # no accession here, but the repository is named: how the data was found or chosen
                hit = SELECTION.search(text)
                entry = {"section": section, "text": clip(text, (hit or REPOSITORY.search(text)).span())}
                (selection if hit else repository).append(entry)
        m = meta.get(path.stem, {})
        if not mentions and not selection and not repository:
            continue
        n_with += 1
        out.write(json.dumps({
            "pmcid": path.stem, "doi": m.get("doi"), "pmid": m.get("pmid"), "title": m.get("title"),
            "authors": m.get("authorString"), "year": m.get("pubYear"), "journal": m.get("journal"),
            "abstract": m.get("abstractText"), "accession_counts": counts,
            "mentions": mentions, "selection": selection,
            "repository": repository[:6]}) + "\n")
    print(f"{n_files} files, {bad} unparseable, {n_with} with accession or selection text", file=sys.stderr)


if __name__ == "__main__":
    main()
