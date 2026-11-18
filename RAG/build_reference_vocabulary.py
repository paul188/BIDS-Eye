#!/usr/bin/env python3
"""Build RAG/reference_vocabulary.yaml: diagnoses and tasks from official sources.

The curated vocabulary (value_mappings.yaml) holds what OpenNeuro datasets
use. A user can name a disease or a paradigm that no dataset has yet, and the
search must still know what it is: "rasmussen" is Rasmussen encephalitis, not
an author whose name starts with "Rasmus" (2026-11-18). This file lists, from
official sources,

  diagnosis  every concept in two MeSH branches (U.S. National Library of
             Medicine, Medical Subject Headings): C10 Nervous System Diseases
             and F03 Mental Disorders. One entry per MeSH concept, so a narrower
             concept such as Rasmussen syndrome (filed under Encephalitis) is
             an entry of its own, with the concept's official terms.
  task       every task in the Cognitive Atlas (cognitiveatlas.org), the task
             list BIDS refers to through the CogAtlasID field, with its aliases.

Usage (downloads both sources when no local copies are given):

    python RAG/build_reference_vocabulary.py
    python RAG/build_reference_vocabulary.py --mesh-xml desc2025.xml --cogatlas-json tasks.json

The output is read by RAG/reference_vocabulary.py.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

_THIS_DIR = Path(__file__).parent
_OUT = _THIS_DIR / "reference_vocabulary.yaml"

MESH_YEAR = 2025
MESH_URL = f"https://nlmpubs.nlm.nih.gov/projects/mesh/{MESH_YEAR}/xmlmesh/desc{MESH_YEAR}.gz"
COGATLAS_URL = "https://www.cognitiveatlas.org/api/v-alpha/task"
MESH_BRANCHES = ("C10", "F03")   # Nervous System Diseases, Mental Disorders

# Words after a surname that make a disease name: "Rasmussen syndrome",
# "Huntington's disease". The surname alone then names the disease too.
_EPONYM_HEADS = (
    "syndrome", "disease", "encephalitis", "encephalopathy", "palsy", "ataxia",
    "chorea", "dystrophy", "neuropathy", "dementia", "disorder", "aphasia",
    "anomaly", "malformation", "sclerosis", "myopathy", "paralysis", "tumor",
    "tumour", "neuralgia", "leukodystrophy", "atrophy", "dystonia", "tremor",
)
_EPONYM_RE = re.compile(
    r"^([a-z][a-z\-]{3,})(?:'s|s')? (?:" + "|".join(_EPONYM_HEADS) + r")$"
)
# Medical adjectives and nouns, not surnames: "limbic encephalitis",
# "glycine encephalopathy", "metachromatic leukodystrophy". The word list
# below has no medical words, so these go by their ending.
_DESCRIPTIVE_ENDINGS = (
    "ic", "al", "ar", "ary", "ory", "ous", "ive", "ian", "ese", "ed", "ing", "oid",
    "in", "ine", "ion", "ness", "ia", "osis", "itis", "ella",
)
# The ending rule is a rule of thumb: it is deterministic, but MeSH does not
# mark surnames, so it errs. These lists override it, checked by hand against
# the full output of 2026-11-18 (206 words removed by ending, 239 kept).
_ALWAYS_SURNAME = frozenset({"cushing", "devic", "folling", "rubinstein", "schimmelpenning"})
_NEVER_SURNAME = frozenset({
    "brainstem", "chronobiology", "encephalon", "epileptiform", "graphomotor",
    "hypersexuality", "hypersomnolence", "locomotor", "multiinfarct", "multisystem",
    "neurotoxicity", "nonfluent", "spinodentate", "spongiform", "supinator",
    "tuberose", "vagus", "lenticulostriate", "endfolium", "geniculate", "hemicord",
})
# Lowercase English words: "down" (Down syndrome) or "pick" (Pick disease)
# alone are ordinary words, never a disease.
_DICTIONARY_PATHS = ("/usr/share/dict/words", "/usr/share/dict/american-english")


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "reference-vocabulary-builder"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def _lowercase_words() -> frozenset:
    for path in _DICTIONARY_PATHS:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="ignore").split()
        except OSError:
            continue
        return frozenset(w for w in lines if w.islower())
    sys.exit("A word list (/usr/share/dict/words) is needed to leave out everyday words.")


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("'s ", " ").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _uninvert(term: str) -> str:
    """MeSH 'Encephalitis, Rasmussen' → 'Rasmussen Encephalitis'."""
    if ", " not in term or "(" in term:
        return term
    parts = term.split(", ")
    return " ".join(reversed(parts))


def _usable(term: str) -> bool:
    """A term a user could type: no inverted or bracketed forms, not a 2-letter code."""
    if "," in term or ";" in term or "[" in term:
        return False
    letters = re.sub(r"[^A-Za-z]", "", term)
    if len(letters) < 3:
        return False
    return True


def _terms(strings: Iterable[str]) -> List[str]:
    out: List[str] = []
    for string in strings:
        term = _uninvert(string.strip())
        if not _usable(term):
            continue
        # Acronyms keep their case ("CJD"); other terms are lowercased.
        term = term if term.isupper() else term.lower()
        if term not in out:
            out.append(term)
    return out


# ── MeSH ──────────────────────────────────────────────────────────────────────

def mesh_entries(xml_bytes_or_path) -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    for _event, record in ET.iterparse(xml_bytes_or_path):
        if record.tag != "DescriptorRecord":
            continue
        trees = [t.text or "" for t in record.findall("TreeNumberList/TreeNumber")]
        if not any(t.startswith(MESH_BRANCHES) for t in trees):
            record.clear()
            continue
        descriptor_ui = record.findtext("DescriptorUI")
        descriptor = record.findtext("DescriptorName/String")
        scope = None
        preferred_key: Optional[str] = None
        concepts = record.findall("ConceptList/Concept")
        for concept in concepts:
            if concept.get("PreferredConceptYN") == "Y":
                scope = (concept.findtext("ScopeNote") or "").strip()
        for concept in concepts:
            preferred = concept.get("PreferredConceptYN") == "Y"
            name = concept.findtext("ConceptName/String") or ""
            terms = _terms(t.findtext("String") or "" for t in concept.findall("TermList/Term"))
            label = _uninvert(name if not preferred else descriptor)
            key = slug(label)
            if not key or not terms:
                continue
            if key in entries:      # same name in two descriptors: keep the first
                entries[key]["terms"] = list(dict.fromkeys(entries[key]["terms"] + terms))
                continue
            entry = {
                "label": label if label.isupper() else label[0].upper() + label[1:].lower(),
                "source": f"MeSH {descriptor_ui}" + ("" if preferred else
                                                      f" concept {concept.findtext('ConceptUI')}"),
                "terms": terms,
            }
            if preferred:
                preferred_key = key
                entry["mesh_tree"] = sorted(t for t in trees if t.startswith(MESH_BRANCHES))
                if scope:
                    entry["description"] = re.sub(r"\s+", " ", scope)[:400]
            entries[key] = entry
        # Narrower concepts name their heading.
        for concept in concepts:
            if concept.get("PreferredConceptYN") == "Y" or preferred_key is None:
                continue
            key = slug(_uninvert(concept.findtext("ConceptName/String") or ""))
            if key in entries and key != preferred_key:
                entries[key].setdefault("broader", preferred_key)
        record.clear()
    return entries


def add_eponyms(entries: Dict[str, dict], everyday: frozenset) -> None:
    """Surname alone → the one disease named after it ('rasmussen' → Rasmussen syndrome)."""
    owners: Dict[str, set] = {}
    for key, entry in entries.items():
        for term in entry["terms"]:
            match = _EPONYM_RE.match(term.lower().replace("’", "'"))
            if match:
                owners.setdefault(match.group(1), set()).add(key)
    for word, keys in owners.items():
        parts = word.split("-")
        if word in _NEVER_SURNAME or word.rstrip("s") in _NEVER_SURNAME:
            continue
        if word not in _ALWAYS_SURNAME and word.rstrip("s") not in _ALWAYS_SURNAME:
            if any(p in everyday or p.rstrip("s") in everyday for p in parts):
                continue
            if any(p.endswith(_DESCRIPTIVE_ENDINGS) for p in parts):
                continue
        # A surname with several diseases is ambiguous; one narrower concept
        # under its heading is not ("huntington" → Huntington disease).
        roots = {entries[k].get("broader", k) for k in keys}
        if len(roots) != 1:
            continue
        target = sorted(keys, key=lambda k: ("broader" not in entries[k], k))[0]
        entries[target].setdefault("eponyms", []).append(word)


# ── Cognitive Atlas ───────────────────────────────────────────────────────────

def cogatlas_entries(tasks: List[dict]) -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    for task in tasks:
        name = (task.get("name") or "").strip()
        if not name:
            continue
        aliases = [a.strip() for a in re.split(r"[,;]", task.get("alias") or "") if a.strip()]
        key = slug(name)
        terms = []
        for string in [name] + aliases:
            string = re.sub(r"\s+", " ", string)
            if len(re.sub(r"[^A-Za-z]", "", string)) < 3:
                continue
            term = string if string.isupper() else string.lower()
            if term not in terms:
                terms.append(term)
        if not key or not terms:
            continue
        entry = {"label": name, "source": f"Cognitive Atlas {task.get('id')}", "terms": terms}
        definition = re.sub(r"\s+", " ", (task.get("definition_text") or "")).strip()
        if definition:
            entry["description"] = definition[:400]
        if key in entries:
            entries[key]["terms"] = list(dict.fromkeys(entries[key]["terms"] + terms))
        else:
            entries[key] = entry
    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

_HEADER = f"""\
# Reference vocabulary: diagnoses and tasks from official sources, also those
# no OpenNeuro dataset has yet. Built by RAG/build_reference_vocabulary.py; do
# not edit by hand. The curated value_mappings.yaml decides first; this file is
# read by RAG/reference_vocabulary.py only for terms the curated one lacks.
#
# diagnosis: Medical Subject Headings (MeSH) {MESH_YEAR}, U.S. National Library of
#   Medicine, branches C10 (Nervous System Diseases) and F03 (Mental Disorders).
#   https://www.nlm.nih.gov/mesh/ — courtesy of the U.S. National Library of
#   Medicine; this is a derived subset, not the authoritative MeSH.
# task: Cognitive Atlas task list (https://www.cognitiveatlas.org), CC BY-SA 3.0.
#   Derived from the Cognitive Atlas; this section is shared under the same license.
"""


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh-xml", help=f"desc{MESH_YEAR}.xml (or .gz); downloaded if absent")
    parser.add_argument("--cogatlas-json", help="Cognitive Atlas task list JSON; downloaded if absent")
    parser.add_argument("--out", default=str(_OUT))
    args = parser.parse_args(argv)

    if args.mesh_xml:
        path = Path(args.mesh_xml)
        source = gzip.open(path) if path.suffix == ".gz" else open(path, "rb")
    else:
        print(f"Downloading {MESH_URL}", file=sys.stderr)
        source = io.BytesIO(gzip.decompress(_fetch(MESH_URL)))
    with source:
        diagnoses = mesh_entries(source)
    add_eponyms(diagnoses, _lowercase_words())

    if args.cogatlas_json:
        tasks = json.loads(Path(args.cogatlas_json).read_text(encoding="utf-8"))
    else:
        print(f"Downloading {COGATLAS_URL}", file=sys.stderr)
        tasks = json.loads(_fetch(COGATLAS_URL))
    task_entries = cogatlas_entries(tasks)

    body = yaml.safe_dump(
        {"diagnosis": dict(sorted(diagnoses.items())), "task": dict(sorted(task_entries.items()))},
        sort_keys=False, allow_unicode=True, width=100,
    )
    Path(args.out).write_text(_HEADER + "\n" + body, encoding="utf-8")
    eponyms = sum(len(e.get("eponyms", [])) for e in diagnoses.values())
    print(f"{len(diagnoses)} diagnoses ({eponyms} surnames), {len(task_entries)} tasks → {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
