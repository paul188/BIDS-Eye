#!/usr/bin/env python3
"""Compare the planner's plans with the gold clauses of the own requests.

    python score_plans.py run --backend-dir ../../../backend --out plans.json [--repeats 3]
    python score_plans.py score plans.json [--errors]

run    one planner call per request and repeat (intent step only, no database)
score  each plan becomes a logic tree over the gold clauses:
         equal     the tree has the truth table of an accepted reading
         dropped   a required gold clause no part of the plan states
         invented  a part of the plan that states no gold clause
         polarity  a clause on the wrong side of a NOT
         grouping  a pair of clauses joined by AND instead of OR, or the reverse

The headline counts items with one reading that the plan can hold. Items with
several readings are scored apart (any accepted reading counts), and so are
items no plan can hold. Plan parts are matched to clauses by their words, so a
synonym that is not among a clause's values counts as dropped and invented:
read --errors before trusting a number. Attachment is not scored. The plan is
read as written; what the code does with it later is tests/test_plan_logic.py.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from math import sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import load_own, names, parse, tables  # noqa: E402

REPO = HERE.parents[2]
for _p in ("RAG", "LLM_preprocessor", "backend"):
    sys.path.append(str(REPO / _p))


# ── words ────────────────────────────────────────────────────────────────────

# Spellings of one thing. Applied to the plan's words and to the gold's alike.
ALIASES = [(re.compile(a), b) for a, b in (
    (r"\brs[- ]?fmri\b|\bresting[- ]state functional mri\b", "resting state fmri"),
    (r"\bfunctional mri\b|\bfunctional magnetic resonance imaging\b", "fmri"),
    (r"\bresting[- ]state\b|\brest[- ]state\b|\bresting\b|\brs\b", "rest"),
    (r"\bt([12])[- ]?(?:w\b|weighted\b)", r"t\1w"),
    (r"\bmp[- ]?rage\b|\bmprage[- ]like\b", "mprage"), (r"\bmp2[- ]?rage\b", "mp2rage"),
    (r"\bdiffusion[- ]?(?:mri|weighted imaging|tensor imaging|imaging)?\b|\bdti\b|\bdwi\b|\bdmri\b", "diffusion"),
    (r"\bstereo[- ]?(?:eeg|electroencephalography)\b|\bseeg\b", "seeg"),
    (r"\bintracranial eeg\b", "ieeg"), (r"\bcovid[- ]?(?:19)?\b", "covid"),
    (r"\bstructural mri\b|\banatomical mri\b", "structural"),
    (r"\bt1\b", "t1w"), (r"\bt2\b", "t2w"), (r"\bbold\b", "fmri"),
    (r"\bunited states\b|\busa?\b", "unitedstates"),
)]
FILLER = frozenset("a an and or the of with in for to from by on some any data dataset datasets scan scans image images "
                   "recording recordings patients patient participants people subjects study studies task tasks runs run "
                   "disease disorder disorders".split())


def tokens(text: str) -> frozenset:
    """The content words of a phrase, spelled one way."""
    text = text.lower().replace("'s", "").replace("’s", "").replace("_", " ")
    for pattern, replacement in ALIASES:
        text = pattern.sub(replacement, text)
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)
    words = [re.sub(r"\.0$", "", w) for w in words]
    words = [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    kept = frozenset(w for w in words if w not in FILLER)
    return kept or frozenset(words)


# ── the plan as a logic tree ─────────────────────────────────────────────────

class Atoms:
    """The conditions of one plan, numbered p1, p2, ..."""

    def __init__(self) -> None:
        self.found: dict = {}

    def add(self, field: str, text: str, low=None, high=None) -> str:
        name = f"p{len(self.found) + 1}"
        self.found[name] = {"field": field, "text": text, "low": low, "high": high}
        return name


def joined(parts, op: str) -> str:
    parts = [p for p in parts if p]
    return "" if not parts else parts[0] if len(parts) == 1 else "(" + f" {op} ".join(parts) + ")"


def negated(part: str) -> str:
    return f"!({part})" if part else ""


def formula_tree(formula, atoms: Atoms) -> str:
    """A description_search formula as a tree (RAG/text_formula.py)."""
    import text_formula as tf
    if formula is None:
        return ""
    if isinstance(formula, tf.Not):
        return negated(formula_tree(formula.child, atoms))
    if isinstance(formula, (tf.And, tf.Or)):
        op = "&" if isinstance(formula, tf.And) else "|"
        return joined([formula_tree(child, atoms) for child in formula.children], op)
    return atoms.add("text", formula.text)


def plan_tree(plan: dict, together: bool = False):
    """(tree, atoms, wishes) of a plan.

    *together*: a negated requirement with several terms excludes only the
    combination (augment._excluded_scan_lines decides by the vocabulary; the
    scorer accepts either).
    """
    import text_formula as tf
    from preprocess import _is_healthy_term

    atoms, parts = Atoms(), []

    def group_tree(group: dict) -> str:
        inner = [joined([atoms.add("population" if _is_healthy_term(t) else "diagnosis", t)
                         for t in group.get("diagnosis_terms") or []], "|")]
        inner += [atoms.add("scan", t) for t in group.get("task_terms") or []]
        inner.append(joined([atoms.add("sex", t) for t in group.get("sex") or []], "|"))
        inner.append(joined([atoms.add("handedness", t) for t in group.get("handedness") or []], "|"))
        if group.get("age_min") is not None or group.get("age_max") is not None:
            inner.append(atoms.add("age", "", group.get("age_min"), group.get("age_max")))
        if group.get("min_subjects") is not None:
            inner.append(atoms.add("count", "", group["min_subjects"]))
        inner += [atoms.add("text", f"{k} {v}") for k, v in (group.get("extra_participant_fields") or {}).items()]
        return joined(inner, "&")

    parts.append(joined([group_tree(g) for g in plan.get("groups") or []],
                        "|" if plan.get("group_operator") == "OR" else "&"))
    parts += [negated(atoms.add("diagnosis", t)) for t in plan.get("excluded_diagnosis_terms") or []]
    for need in plan.get("scan_requirements") or []:
        terms = [atoms.add("scan", t) for t in need["scan_terms"]]
        if need.get("required", True):
            parts.append(joined(terms, "|" if need.get("any_of") else "&"))
        elif together and not need.get("any_of"):
            parts.append(negated(joined(terms, "&")))
        else:
            parts += [negated(t) for t in terms]
    parts += [atoms.add("scan", t.get("phrase") or t.get("typed") or "") for t in plan.get("topics") or []]
    if plan.get("description_search"):
        parts.append(formula_tree(tf.parse_formula(plan["description_search"]), atoms))
    if plan.get("name_search"):
        parts.append(atoms.add("text", plan["name_search"]))
    parts.append(joined([atoms.add("species", s) for s in plan.get("species") or []], "|"))
    parts.append(joined([atoms.add("age", a) for a in plan.get("age_groups") or []], "|"))
    for key, field in (("author_names", "author"), ("funding_terms", "funding"), ("location_terms", "location")):
        parts += [atoms.add(field, t) for t in plan.get(key) or []]
    parts.append(joined([atoms.add("institution", t) for t in plan.get("institution_terms") or []], "|"))
    parts += [negated(atoms.add("institution", t)) for t in plan.get("excluded_institution_terms") or []]
    if plan.get("license_search"):
        parts.append(atoms.add("licence", plan["license_search"]))
    for f in plan.get("metadata_filters") or []:
        value = "" if f.get("value") is None else f" {f['value']}"
        atom = atoms.add("sidecar", f"{f['json_field']}{value}")
        parts.append(negated(atom) if f["operator"] in ("!=", "not_exists") else atom)
    if plan.get("max_size_gb") is not None or plan.get("min_size_gb") is not None:
        parts.append(atoms.add("size", "", plan.get("min_size_gb"), plan.get("max_size_gb")))
    if (plan.get("min_sessions") or 0) > 1:
        parts.append(atoms.add("sessions", "", plan["min_sessions"]))
    for key in ("study_structure", "count_comparison"):
        if plan.get(key):
            parts.append(atoms.add("design", plan[key]))
    if plan.get("doi_required") is not None:
        atom = atoms.add("doi", "doi")
        parts.append(atom if plan["doi_required"] else negated(atom))
    return joined(parts, "&"), atoms.found, list(plan.get("request_wishes") or [])


# ── matching plan parts and gold clauses ─────────────────────────────────────

NUMERIC = {"age", "count", "size", "sessions"}
# Fields whose content the plan may hold in one or the other.
SAME = [{"scan", "text", "design"}, {"diagnosis", "text", "population"}, {"location", "text"},
        {"sidecar", "text"}, {"diagnosis", "scan"}]


def fields_fit(a: str, b: str) -> bool:
    return a == b or any(a in group and b in group for group in SAME)


def number_fits(clause: dict, atom: dict) -> bool:
    """Numbers of the gold value against the bounds of the plan part."""
    for value in clause["values"]:
        wanted = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", value)]
        bounds = [float(b) for b in (atom["low"], atom["high"]) if b is not None]
        if "adult" in value and "older" not in value and not wanted:
            return atom["low"] is not None and 16 <= atom["low"] <= 21 and atom["high"] is None
        if wanted and len(wanted) == len(bounds) and all(abs(w - b) <= 1 for w, b in zip(sorted(wanted), sorted(bounds))):
            lower = any(w in value for w in ("at least", "over", "min"))
            upper = any(w in value for w in ("under", "max", "below"))
            return not (lower and atom["low"] is None) and not (upper and atom["high"] is None)
    return False


def clause_tokens(clause: dict) -> list:
    spans = clause["words"] if isinstance(clause["words"], list) else [clause["words"]]
    texts = clause["values"] + (spans if clause["strength"] != "example" or len(spans) == 1 else [])
    return [t for t in map(tokens, texts) if t]


def matches(atom: dict, clauses: dict) -> list:
    """The gold clauses a plan part states: none, one, or several ("resting-state fMRI")."""
    candidates = {k: c for k, c in clauses.items() if fields_fit(atom["field"], c["field"])}
    if atom["field"] in NUMERIC and not atom["text"]:
        return [k for k, c in candidates.items() if c["field"] == atom["field"] and number_fits(c, atom)][:1]
    mine = tokens(atom["text"])
    if not mine:
        return []
    exact = [k for k, c in candidates.items() if mine in clause_tokens(c)]
    if exact:
        # One phrase can be a value of two clauses; the same field wins.
        return sorted(exact, key=lambda k: clauses[k]["field"] != atom["field"])[:1]
    inside: dict = {}                       # the part says the clause and more
    for k, c in candidates.items():
        said = [t for t in map(tokens, c["values"]) if t and t < mine]
        if said:
            inside[k] = max(said, key=len)
    inside = {k: t for k, t in inside.items() if not any(t < other for other in inside.values())}
    if inside and frozenset().union(*inside.values()) == mine:
        return sorted(inside)
    wider = [k for k, c in candidates.items() if any(mine < t for t in map(tokens, c["values"]))]
    if len(inside) == 1 and not wider:
        # "EEG recording of sleep" for sleep: the extra words name nothing else.
        return list(inside)
    if inside:
        return sorted(inside)
    return wider[:1] if len(wider) == 1 else []


# ── comparing ────────────────────────────────────────────────────────────────

def substitute(tree: str, replace: dict) -> str:
    return re.sub(r"\bp\d+\b", lambda m: replace[m.group(0)], tree)


def signs(tree: str) -> dict:
    """variable → the polarities it has in the tree ('+', '-')."""
    found: dict = {}

    def walk(node, positive: bool) -> None:
        if isinstance(node, ast.Name):
            found.setdefault(node.id, set()).add("+" if positive else "-")
        elif isinstance(node, ast.UnaryOp):
            walk(node.operand, not positive)
        else:
            walk(node.left, positive)
            walk(node.right, positive if not isinstance(node.op, ast.Sub) else not positive)
    if tree:
        walk(parse(tree), True)
    return found


def pair_ops(tree: str) -> dict:
    """pair of variables → operator at their lowest common node, NOT pushed inwards."""
    ops: dict = {}

    def walk(node, positive: bool) -> set:
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, ast.UnaryOp):
            return walk(node.operand, not positive)
        right_sign = positive if not isinstance(node.op, ast.Sub) else not positive
        left, right = walk(node.left, positive), walk(node.right, right_sign)
        is_and = not isinstance(node.op, ast.BitOr)
        op = "&" if is_and == positive else "|"
        for a in left:
            for b in right:
                if a != b:
                    ops.setdefault(frozenset((a, b)), set()).add(op)
        return left | right
    if tree:
        walk(parse(tree), True)
    return ops


def compare(item: dict, plan: dict) -> dict:
    """The result for one plan against the best-fitting reading."""
    clauses = item["clauses"]
    best = None
    for together in (False, True):
        tree, atoms, _ = plan_tree(plan, together)
        replace, invented, stated = {}, [], set()
        for name, atom in atoms.items():
            hit = matches(atom, clauses)
            if hit:
                stated |= set(hit)
                replace[name] = joined(hit, "&")
            else:
                replace[name] = f"x{len(invented) + 1}"
                label = atom["text"] or f"{atom['low']}..{atom['high']}"
                invented.append(f"{atom['field']}:{label}")
        full = substitute(tree, replace)
        wishes = sorted(k for k in stated if clauses[k]["strength"] == "wish")
        for gold in [tree for r in item["readings"] for tree in (r["tree"], *r.get("also", []))]:
            variables = sorted(set(names(parse(gold))) | (set(names(parse(full))) if full else set()))
            same = False
            if full:
                a, b = tables([gold, full], variables)
                same = a == b
            needed = set(names(parse(gold)))
            dropped = sorted(needed - stated)
            extra = list(invented) + [f"wish used as a condition: {k}" for k in wishes]
            gold_signs, plan_signs = signs(gold), signs(full)
            polarity = sorted(k for k in needed & stated if gold_signs[k] != plan_signs.get(k, set()))
            gold_ops, plan_ops = pair_ops(gold), pair_ops(full)
            grouping = sorted("+".join(sorted(p)) for p in gold_ops
                              if p in plan_ops and len(gold_ops[p]) == 1 and plan_ops[p] != gold_ops[p]
                              and not (p & set(polarity)))
            result = {"equal": same, "reading": gold, "plan": full, "dropped": dropped, "invented": extra,
                      "polarity": polarity, "grouping": grouping, "n_needed": len(needed),
                      "n_negative": sum(v == {"-"} for v in gold_signs.values()),
                      "negative_wrong": sum(1 for k in polarity if gold_signs[k] == {"-"}),
                      "negative_dropped": sum(1 for k in dropped if gold_signs[k] == {"-"})}
            cost = (not same, len(dropped) + len(extra) + len(polarity) + len(grouping))
            if best is None or cost < best[0]:
                best = (cost, result)
    return best[1]


# ── reporting ────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if not n:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def by_item_interval(rates: list, draws: int = 4000) -> tuple:
    """Bootstrap over items: the repeats of one request are not independent."""
    if not rates:
        return (0.0, 1.0)
    rng = random.Random(1)
    means = sorted(sum(rng.choices(rates, k=len(rates))) / len(rates) for _ in range(draws))
    return (means[int(draws * 0.025)], means[int(draws * 0.975)])


def kind(item: dict) -> str:
    if item["ambiguous"]:
        return "ambiguous"
    return "headline" if item["readings"][0]["representable"] == "yes" else "not representable"


def report(items: dict, plans: dict, show: bool) -> None:
    results = {i: [compare(items[i], p) for p in runs] for i, runs in plans["plans"].items() if runs}
    errors = plans.get("errors") or {}
    print(f"{plans['label']}: {len(results)} requests, {sum(map(len, results.values()))} plans"
          + (f", {sum(errors.values())} failed calls (not scored)" if errors else ""))
    for group in ("headline", "ambiguous", "not representable"):
        chosen = {i: r for i, r in results.items() if kind(items[i]) == group}
        runs = [r for rs in chosen.values() for r in rs]
        if not runs:
            continue
        total = len(runs)
        rates = [sum(r["equal"] for r in rs) / len(rs) for rs in chosen.values()]
        every = sum(rate == 1 for rate in rates)
        low, high = by_item_interval(rates)
        wlow, whigh = wilson(every, len(rates))
        print(f"\n{group}: {len(chosen)} requests")
        if group != "not representable":
            print(f"  logic equal     {sum(r['equal'] for r in runs)} of {total} plans ({sum(rates) / len(rates):.0%}, "
                  f"{low:.0%} to {high:.0%} by request); every time {every} of {len(rates)} ({wlow:.0%} to {whigh:.0%})")
        needed = sum(r["n_needed"] for r in runs)
        negative = sum(r["n_negative"] for r in runs)
        wrong = sum(r["negative_wrong"] + r["negative_dropped"] for r in runs)
        print(f"  clauses dropped {sum(len(r['dropped']) for r in runs)} of {needed}; invented "
              f"{sum(len(r['invented']) for r in runs)}; wrong polarity {sum(len(r['polarity']) for r in runs)}; "
              f"wrongly grouped pairs {sum(len(r['grouping']) for r in runs)}")
        print(f"  excluded clauses kept as exclusions: {negative - wrong} of {negative}")
    blocks: dict = {}
    for i, rs in results.items():
        if kind(items[i]) == "headline":
            block = blocks.setdefault(items[i]["block"], [0, 0])
            block[0] += sum(r["equal"] for r in rs)
            block[1] += len(rs)
    for name, hard in (("first requests", False), ("hard requests", True)):
        rates = [sum(r["equal"] for r in rs) / len(rs) for i, rs in results.items()
                 if kind(items[i]) == "headline" and items[i]["block"].startswith("hard_") == hard]
        if rates:
            low, high = by_item_interval(rates)
            print(f"headline, {name}: {sum(rates) / len(rates):.0%} of plans ({low:.0%} to {high:.0%}), "
                  f"every time {sum(r == 1 for r in rates)} of {len(rates)}")
    print("\nheadline by block: " + ", ".join(f"{b} {k}/{n}" for b, (k, n) in sorted(blocks.items())))
    if show:
        for i, rs in results.items():
            bad = [r for r in rs if not r["equal"]]
            if bad:
                print(f"\n{i} [{kind(items[i])}] {len(rs) - len(bad)}/{len(rs)}  {items[i]['text']}")
                seen = set()
                for r in bad:
                    line = (f"    gold {r['reading']}  plan {r['plan']}  dropped {r['dropped']} invented {r['invented']} "
                            f"polarity {r['polarity']} grouping {r['grouping']}")
                    if line not in seen:
                        seen.add(line)
                        print(line)


def run(args) -> None:
    root = args.backend_dir.resolve().parent
    sys.path[:0] = [str(root / n) for n in ("backend", "LLM_preprocessor", "RAG")]
    import preprocess
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # A run that was stopped continues: complete requests are kept.
    done = json.loads(args.out.read_text()) if args.out.exists() else {}
    if done and done["label"] != args.label:
        sys.exit(f"{args.out} is another run: {done['label']}")
    plans: dict = done.get("plans", {})
    errors: dict = done.get("errors", {})
    for item in load_own():
        for _ in range(args.repeats - len(plans.get(item["id"], []))):
            try:
                plan = preprocess.preprocess_query(item["text"], api_key=key)
            except Exception as exc:  # noqa: BLE001 - one failed call must not stop the run
                errors[item["id"]] = errors.get(item["id"], 0) + 1
                print(f"{item['id']}: {type(exc).__name__}: {str(exc)[:80]}", file=sys.stderr, flush=True)
                continue
            plans.setdefault(item["id"], []).append(plan.model_dump(mode="json"))
        print(item["id"], len(plans.get(item["id"], [])), file=sys.stderr, flush=True)
        args.out.write_text(json.dumps({"label": args.label, "plans": plans, "errors": errors}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend-dir", type=Path, default=REPO / "backend")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--label", default="plans")
    s = sub.add_parser("score")
    s.add_argument("plans", type=Path)
    s.add_argument("--errors", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        report({i["id"]: i for i in load_own()}, json.loads(args.plans.read_text()), args.errors)


if __name__ == "__main__":
    main()
