#!/usr/bin/env python3
"""
backend/vocabulary/tools/mine_senses.py
----------------------
List the words that stand next to a head word in the catalogue, as material
for backend/vocabulary/senses.yaml.

    python backend/vocabulary/tools/mine_senses.py SNAPSHOT.jsonl conflict control reward
    python backend/vocabulary/tools/mine_senses.py SNAPSHOT.jsonl conflict --window 2 --top 30

SNAPSHOT.jsonl has one dataset per line with name, description, readme and
the uploader's study fields (benchmarks/recall/snapshot.sql writes it; lines
that are not JSON, such as psql headers, are skipped).

For each head word (singular and plural), every word up to --window
positions to its left or right is counted once per dataset. The output lists
the most frequent neighbours with the number of datasets and one example
phrase. "conflict" shows Simon, response, cognitive (one sense),
approach-avoidance (another) and couples, family, social (a third). Grouping
the neighbours into senses is left to the person reviewing the list: the
script only proposes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Tuple

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = frozenset("""
a an and are as at be been but by can for from had has have in into is it its of on or
our that the their there these this those to was were which while who will with we were
not no than then so such also each other more most all any both only same very using used
""".split())
_TEXT_FIELDS = ("name", "description", "readme", "dx_status", "study_domain", "study_design")


def records(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def forms(head: str) -> set:
    head = head.lower()
    return {head, head + "s", head + "es", head[:-1] + "ies" if head.endswith("y") else head}


def neighbours(words: List[str], heads: set, window: int) -> Iterator[Tuple[str, str]]:
    """(neighbour, example phrase) for every head occurrence in *words*."""
    for index, word in enumerate(words):
        if word not in heads:
            continue
        low, high = max(0, index - window), min(len(words), index + window + 1)
        example = " ".join(words[low:high])
        for position in range(low, high):
            other = words[position]
            if position != index and other not in _STOPWORDS and other not in heads and len(other) > 2:
                yield other, example


def mine(path: str, head: str, window: int) -> Tuple[Counter, Dict[str, str], int]:
    heads = forms(head)
    counts: Counter = Counter()
    examples: Dict[str, str] = {}
    datasets = 0
    for record in records(path):
        text = " ".join(str(record.get(f) or "") for f in _TEXT_FIELDS).lower()
        words = _WORD_RE.findall(text)
        found = defaultdict(str)
        for other, example in neighbours(words, heads, window):
            found.setdefault(other, example)
        if found:
            datasets += 1
        for other, example in found.items():
            counts[other] += 1
            examples.setdefault(other, example)
    return counts, examples, datasets


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot")
    parser.add_argument("heads", nargs="+")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-datasets", type=int, default=2)
    args = parser.parse_args(argv)

    for head in args.heads:
        counts, examples, datasets = mine(args.snapshot, head, args.window)
        print(f"\n== {head}: {datasets} datasets use it ==")
        print(f"{'neighbour':<22}{'datasets':>9}  example")
        for other, count in counts.most_common():
            if count < args.min_datasets:
                break
            print(f"{other:<22}{count:>9}  {examples[other][:80]}")
            args.top -= 1
            if args.top == 0:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
