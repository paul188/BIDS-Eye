#!/usr/bin/env python3
"""
benchmarks/word_forms/questions/run.py
--------------------------------------
Score the question-time word-form path on questions.yaml (README.md here).

Two parts:

1. Which words are looked up (no database). question_words() of each
   tokenizer against the judged search words and the written traps.

2. Which forms a search finally accepts (needs --evidence, made by
   fetch_evidence.py). For each search word with a truth row, the forms the
   search requires are computed as the search does: word_forms.decide() on
   the question's evidence and the model's answers, then forms_for() and
   search_variants() (services/fulltext/tsquery.py _word_forms). A row says
   whether a search for the word should also find the other form; per row
   the method is right, a wrong split (lost recall) or a wrong merge (false
   hits). Methods:
     stem   every form of the stem matches (the search without exact forms)
     lists  word_forms.yaml and word_form_pairs.json only; the model's
            answers are left out
     model  lists plus the language model's answers (--model, or the cache
            of an earlier run with --model-cache)

    python benchmarks/word_forms/questions/run.py
    python benchmarks/word_forms/questions/run.py --evidence evidence.json
    python benchmarks/word_forms/questions/run.py --evidence evidence.json --model
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path[:0] = [str(_REPO / "backend" / "vocabulary"), str(_REPO / "backend" / "llm"),
                str(_REPO / "backend")]

import word_forms as wf  # noqa: E402
from text_terms import POSTGRES_STOPWORDS  # noqa: E402

QUESTIONS = _HERE / "questions.yaml"
RESULTS = _HERE / "results"

_V1_WORD_RE = re.compile(r"[a-z]+")


def question_words_v1(question: str) -> List[str]:
    """question_words() before 2026-11-27: runs of a-z, stopwords counted in the cap."""
    words = [w for w in _V1_WORD_RE.findall(question.lower()) if len(w) >= 3]
    return list(dict.fromkeys(words))[:wf.MAX_QUESTION_WORDS]


TOKENIZERS: Dict[str, Callable[[str], List[str]]] = {
    "v1": question_words_v1,
    "current": wf.question_words,
}


# ── Part 1: which words are looked up ─────────────────────────────────────────

def score_lookup(items: Sequence[dict], tokenize: Callable[[str], List[str]]) -> dict:
    found = total = 0
    missed: List[str] = []
    looked_up = stop = 0
    trap_hits = trap_total = 0
    trap_bad: List[str] = []
    for item in items:
        words = tokenize(item["question"])
        looked_up += len(words)
        stop += sum(w in POSTGRES_STOPWORDS for w in words)
        for w in item.get("search_words", []):
            total += 1
            if w in words:
                found += 1
            else:
                missed.append(f"{item['id']}:{w}")
        for w in item.get("must_look_up", []):
            trap_total += 1
            if w in words:
                trap_hits += 1
            else:
                trap_bad.append(f"{item['id']} misses {w}")
        for w in item.get("must_not_look_up", []):
            trap_total += 1
            if w in words:
                trap_bad.append(f"{item['id']} looks up {w}")
            else:
                trap_hits += 1
    return {"search_words_found": f"{found}/{total}", "missed": missed,
            "lookups_per_question": round(looked_up / max(len(items), 1), 1),
            "stopwords_looked_up": stop,
            "trap_checks": f"{trap_hits}/{trap_total}", "trap_failures": trap_bad}


# ── Part 2: which forms a search accepts ──────────────────────────────────────

def load_evidence(path: Path) -> Tuple[Dict[str, wf.WordEvidence], dict]:
    data = json.loads(path.read_text())
    evidence = {w: wf.WordEvidence(word=w, stem=e["stem"], stem_docs=int(e["stem_docs"]),
                                   forms=tuple((f, int(n)) for f, n in e["forms"]))
                for w, e in data["words"].items()}
    return evidence, {k: v for k, v in data.items() if k != "words"}


def question_evidence(question: str, evidence: Mapping[str, wf.WordEvidence],
                      tokenize: Callable[[str], List[str]]) -> Dict[str, wf.WordEvidence]:
    """intent.word_evidence() against the fixture: the lookup of question_words()."""
    return {w: evidence[w] for w in dict.fromkeys(tokenize(question)) if w in evidence}


def required_forms(word: str, decided: Mapping[str, Sequence[str]],
                   evidence: Mapping[str, wf.WordEvidence]) -> Optional[Tuple[str, ...]]:
    """tsquery._word_forms for one plain word; None = matched by stem."""
    forms = wf.forms_for(word, decided, evidence.get)
    if not forms:
        return None
    return tuple(dict.fromkeys(f for form in forms for f in (form, *wf.search_variants(form))))


def score_rows(item: dict, decided: Mapping[str, Sequence[str]],
               evidence: Mapping[str, wf.WordEvidence], stem_only: bool) -> List[dict]:
    rows = []
    for word, truths in (item.get("truth") or {}).items():
        item_ev = evidence.get(word)
        forms = None if stem_only else required_forms(word, decided, evidence)
        for t in truths:
            known = {f for f, _ in item_ev.forms} if item_ev else set()
            other = [o for o in t["other"] if o in known]
            if not other:
                rows.append({"id": item["id"], "word": word, "other": t["other"], "result": "absent"})
                continue
            allowed = [forms is None or o in forms for o in other]
            if t["match"]:
                result = "right" if all(allowed) else "wrong_split"
            else:
                result = "right" if not any(allowed) else "wrong_merge"
            origin = "sample" if "overridden" not in t["by"] and t["by"] in SAMPLE_PAIRS else "list"
            rows.append({"id": item["id"], "word": word, "other": other, "truth": t["match"],
                         "by": t["by"], "origin": origin, "result": result})
    return rows


SAMPLE_PAIRS: set = set()


def model_choices(item: dict, q_evidence: Mapping[str, wf.WordEvidence],
                  cache: Dict[str, dict], call: bool) -> Optional[List[Tuple[str, List[str]]]]:
    """The model's (word, keep) answers for one question; None when unknown."""
    if not wf.needs_decision(q_evidence):
        return []
    hit = cache.get(item["question"])
    if hit is not None:
        return [tuple(c) for c in hit["choices"]]
    if not call:
        return None
    from preprocess.intent import preprocess_query
    choices: List[Tuple[str, List[str]]] = []
    plan = preprocess_query(item["question"], evidence=dict(q_evidence), word_choices=choices)
    cache[item["question"]] = {"choices": choices,
                               "asked": [e.word for e in wf.needs_decision(q_evidence)],
                               "summary": plan.natural_language_summary}
    return choices


def score_forms(items: Sequence[dict], evidence: Mapping[str, wf.WordEvidence],
                tokenize: Callable[[str], List[str]], cache: Dict[str, dict], call: bool) -> dict:
    results: Dict[str, List[dict]] = {"stem": [], "lists": [], "model": []}
    asked_search = asked_filler = asked_other = 0
    asked_list: List[str] = []
    no_model = 0
    for item in items:
        q_ev = question_evidence(item["question"], evidence, tokenize)
        asked = [e.word for e in wf.needs_decision(q_ev)]
        for w in asked:
            if w in item.get("search_words", []):
                asked_search += 1
            elif w in item.get("filler_words", []):
                asked_filler += 1
                asked_list.append(f"{item['id']}:{w}")
            else:
                asked_other += 1
                asked_list.append(f"{item['id']}:{w}")
        results["stem"] += score_rows(item, {}, evidence, stem_only=True)
        results["lists"] += score_rows(item, wf.decide(q_ev, []), evidence, stem_only=False)
        choices = model_choices(item, q_ev, cache, call)
        if choices is None:
            no_model += 1
        else:
            results["model"] += score_rows(item, wf.decide(q_ev, choices), evidence, stem_only=False)
    summary = {}
    for method, rows in results.items():
        if method == "model" and not (call or cache):
            continue     # no answers at all: the rows would be the lists' on easy questions only
        summary[method] = {
            origin: dict(Counter(r["result"] for r in rows if r.get("origin", "") == origin or (
                origin == "all")))
            for origin in ("all", "sample", "list")}
        summary[method]["errors"] = [
            f"{r['id']} {r['word']}->{'/'.join(r['other'])}: {r['result']} ({r['by']})"
            for r in rows if r["result"] in ("wrong_split", "wrong_merge")]
    summary["asked_to_model"] = {"search_words": asked_search, "filler_words": asked_filler,
                                 "unjudged_words": asked_other, "not_search_words": asked_list}
    if no_model and "model" in summary:
        summary["model"]["questions_without_answer"] = no_model
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--evidence", type=Path, help="fixture from fetch_evidence.py")
    ap.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default="current",
                    help="question_words used for part 2 (part 1 scores every tokenizer)")
    ap.add_argument("--model", action="store_true", help="ask the language model where the cache has no answer")
    ap.add_argument("--model-cache", type=Path, help="answers of an earlier run (JSON); default: a new file")
    ap.add_argument("--verbose", action="store_true", help="list every miss and error")
    args = ap.parse_args()

    items = yaml.safe_load(args.questions.read_text())["questions"]
    label_set = json.loads((_HERE.parent / "label_set.json").read_text())
    SAMPLE_PAIRS.update(p["id"] for p in label_set["pairs"] if p["origin"] == "sample")

    report: dict = {"questions": len(items), "lookup": {}}
    for name, tokenize in TOKENIZERS.items():
        report["lookup"][name] = score_lookup(items, tokenize)

    if args.evidence:
        evidence, meta = load_evidence(args.evidence)
        report["evidence"] = meta
        cache: Dict[str, dict] = {}
        if args.model_cache and args.model_cache.exists():
            cache = json.loads(args.model_cache.read_text())
        report["forms"] = score_forms(items, evidence, TOKENIZERS[args.tokenizer], cache, args.model)
        if args.model:
            RESULTS.mkdir(exist_ok=True)
            out = args.model_cache or RESULTS / f"model_answers_{_dt.date.today().isoformat()}.json"
            out.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
            report["model_cache"] = str(out)

    if not args.verbose:
        for part in report["lookup"].values():
            part["missed"] = f"{len(part['missed'])} (--verbose lists them)"
        forms = report.get("forms", {})
        for method in ("stem", "lists", "model"):
            if method in forms:
                forms[method]["errors"] = f"{len(forms[method]['errors'])} (--verbose lists them)"
        if "asked_to_model" in forms:
            forms["asked_to_model"]["not_search_words"] = len(forms["asked_to_model"]["not_search_words"])
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
