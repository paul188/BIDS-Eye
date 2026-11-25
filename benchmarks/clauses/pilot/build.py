#!/usr/bin/env python3
"""Check the pilot items and write what the reviewers use.

    python build.py [--chia ../work/clinical/chia_scope/x]

Reads own.yaml and chia_items.yaml. Every span must be in its text, every tree
must parse and use only existing clauses, and two readings of one item must
differ (truth table). Writes pilot.jsonl (all items, with the line and the
original Chia relations), review.html (check the draft) and blind.html (label
without seeing it).
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FIELDS = {"diagnosis", "scan", "text", "age", "sex", "handedness", "species", "count", "institution",
          "location", "author", "licence", "funding", "sidecar", "size", "sessions", "design", "doi",
          "population"}
OPS = {"any", "all"}


def parse(tree: str):
    """The tree as a Python expression over clause names; only & | ! - and brackets."""
    node = ast.parse(tree.replace("!", "~"), mode="eval").body

    def check(n) -> None:
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.BitAnd, ast.BitOr, ast.Sub)):
            check(n.left), check(n.right)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Invert):
            check(n.operand)
        elif not isinstance(n, ast.Name):
            raise ValueError(f"not a logic tree: {tree}")
    check(node)
    return node


def names(node) -> list:
    return sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)})


def exceptions(node) -> list:
    """The "a - b" parts of a tree, innermost first."""
    found = [n for n in ast.walk(node) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)]
    return found[::-1]


def value(node, row: dict) -> bool:
    if isinstance(node, ast.Name):
        return row[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return row[ast.unparse(node)]
    if isinstance(node, ast.UnaryOp):
        return not value(node.operand, row)
    left, right = value(node.left, row), value(node.right, row)
    return left and right if isinstance(node.op, ast.BitAnd) else left or right


def tables(trees: list, variables: list) -> list:
    """The truth table of each tree, over the same rows.

    "a - b" (some a that is not b) is a variable of its own: true with a and
    without b, false without a, and either way where both hold (a person on
    metformin and another drug). Rows that break this are left out.
    """
    nodes = [parse(t) for t in trees]
    parts = {ast.unparse(e): e for n in nodes for e in exceptions(n)}
    rows = []
    for bits in itertools.product([False, True], repeat=len(variables) + len(parts)):
        row = dict(zip([*variables, *parts], bits))
        if all(row[k] <= value(e.left, row) and (value(e.left, row) and not value(e.right, row)) <= row[k]
               for k, e in parts.items()):
            rows.append(row)
    return [tuple(value(n, row) for row in rows) for n in nodes]


def spans(words) -> list:
    return [words] if isinstance(words, str) else list(words)


def check_item(item: dict, problems: list) -> None:
    text, seen = item["text"], []
    for key, clause in item["clauses"].items():
        for span in spans(clause["words"]):
            if span not in text:
                problems.append(f"{item['id']} {key}: {span!r} is not in the text")
    variables = sorted(item["clauses"])
    for reading in item["readings"]:
        try:
            used = names(parse(reading["tree"]))
        except (SyntaxError, ValueError) as exc:
            problems.append(f"{item['id']}: {exc}")
            continue
        if not set(used) <= set(variables):
            problems.append(f"{item['id']}: {reading['tree']} uses {set(used) - set(variables)}")
            continue
        seen.append(reading["tree"])
    results = tables(seen, variables)
    if len(set(results)) < len(results):
        problems.append(f"{item['id']}: two readings are the same logic")
    in_trees = {n for r in item["readings"] for tree in [r["tree"], *r.get("also", [])] for n in names(parse(tree))}
    if not in_trees <= set(variables):
        problems.append(f"{item['id']}: unknown clause {in_trees - set(variables)}")
    for key, clause in item["clauses"].items():
        if key not in in_trees and clause.get("strength") not in ("wish", "example", "optional"):
            problems.append(f"{item['id']}: clause {key} is in no reading")


def load_own() -> list:
    items = []
    for raw in yaml.safe_load((HERE / "own.yaml").read_text())["items"]:
        clauses = {}
        for key, parts in raw["clauses"].items():
            field, values, op, words, *rest = parts
            assert field in FIELDS and op in OPS, (raw["id"], key)
            clauses[key] = {"field": field, "values": [str(v) for v in values], "op": op, "words": words,
                            "attach": rest[0] if rest else "dataset", "strength": rest[1] if len(rest) > 1 else "must"}
        readings = [dict(r, representable="yes" if r["representable"] is True else
                         "no" if r["representable"] is False else r["representable"]) for r in raw["readings"]]
        items.append({**{k: raw[k] for k in ("id", "block", "source", "text")}, "set": "own", "clauses": clauses,
                      "readings": readings, "ambiguous": len(readings) > 1,
                      **{k: raw[k] for k in ("convention", "attach_alternatives", "revised") if k in raw}})
    return items


def chia_line(folder: Path, name: str, start: str):
    """The line of a Chia file that begins with *start*, and its annotated relations."""
    text = (folder / f"{name}.txt").read_text(encoding="utf-8")
    entities, relations = {}, []
    for row in (folder / f"{name}.ann").read_text(encoding="utf-8").splitlines():
        parts = row.split("\t")
        if row.startswith("T"):
            kind, rest = parts[1].split(" ", 1)
            entities[parts[0]] = (kind, int(rest.split()[0]), parts[2])
        elif row.startswith("R"):
            kind, a, b = parts[1].split()
            relations.append((kind, a.split(":")[1], b.split(":")[1]))
        elif row.startswith("*"):
            kind, *members = parts[1].split()
            relations.append((kind, *members))
    position = 0
    for line in text.split("\n"):
        begin, end = position, position + len(line)
        position = end + 1
        if line.strip().startswith(start):
            here = {k for k, v in entities.items() if begin <= v[1] < end}
            return line.strip(), [[r[0], *(entities[x][2] for x in r[1:])]
                                  for r in relations if all(x in here for x in r[1:])]
    return None, []


def load_chia(folder: Path, problems: list) -> list:
    items = []
    for raw in yaml.safe_load((HERE / "chia_items.yaml").read_text())["items"]:
        line, relations = chia_line(folder, raw["file"], raw["start"])
        if line is None:
            problems.append(f"{raw['id']}: no line {raw['start']!r} in {raw['file']}")
            continue
        items.append({
            "id": raw["id"], "set": "chia", "block": raw["block"], "source": f"Chia {raw['file']} (CC BY 4.0)",
            "header": "inclusion" if raw["file"].endswith("_inc") else "exclusion", "text": line,
            "clauses": {k: {"words": w} if not (raw["id"] == "ch-33" and k in "bc")
                        else {"words": w, "strength": "example"} for k, w in raw["atoms"].items()},
            "readings": [{"tree": t} for t in raw["trees"]], "ambiguous": len(raw["trees"]) > 1,
            **{k: raw[k] for k in ("note", "revised") if k in raw}, "chia_relations": relations})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chia", type=Path, default=HERE.parent / "work/clinical/chia_scope/x")
    args = parser.parse_args()
    problems: list = []
    items = load_own() + load_chia(args.chia, problems)
    for item in items:
        check_item(item, problems)
    if len({i["id"] for i in items}) < len(items):
        problems.append("an id occurs twice")
    for problem in problems:
        print(problem)
    for name in ("own", "chia"):
        rows = [i for i in items if i["set"] == name]
        blocks: dict = {}
        for i in rows:
            blocks[i["block"]] = blocks.get(i["block"], 0) + 1
        print(f"{name}: {len(rows)} items, {sum(i['ambiguous'] for i in rows)} ambiguous, {blocks}")
    own = [i for i in items if i["set"] == "own"]
    print("own: headline items (one reading, representable):",
          sum(not i["ambiguous"] and i["readings"][0]["representable"] == "yes" for i in own),
          "| no reading fits the plan:", sum(all(r["representable"] == "no" for r in i["readings"]) for i in own),
          "| from real wording:", sum(i["source"] != "written" for i in own))
    if problems:
        sys.exit(1)
    (HERE / "pilot.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    template = (HERE / "review_template.html").read_text()
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    bare = json.dumps([{k: i[k] for k in ("id", "set", "text", "header") if k in i} for i in items],
                      ensure_ascii=False).replace("</", "<\\/")
    (HERE / "review.html").write_text(template.replace("__ITEMS__", data).replace("__MODE__", "review"))
    (HERE / "blind.html").write_text(template.replace("__ITEMS__", bare).replace("__MODE__", "blind"))


if __name__ == "__main__":
    main()
