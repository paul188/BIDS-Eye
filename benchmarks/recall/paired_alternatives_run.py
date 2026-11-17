"""Paired measurement of description alternatives (preprocess.py,
check_description_alternatives): "dead people" also searches "post-mortem".

Each question goes through the search pipeline once (intent, vocabulary
lookup, SQL generation). The model's SQL is then rewritten and run once per
variant, which differ only in the alternatives of the 'dataset text:' filter:

  alt_off     the user's own words only, as before the change
  alt_on      the alternatives the catalogue check kept (at most
              TOPIC_MAX_ALTERNATIVE_MATCHES datasets each, 90); a dataset
              that matches only them ranks below the user's own words
  alt_flat    the same rows as alt_on, ranked without that rule
  alt_cap<N>  only the alternatives that at most N datasets match (--caps)
  alt_ratio<R>  only the alternatives that match at most R times as many
              datasets as the user's own words, or RATIO_FLOOR if that is
              more (--ratios): "genetics" (25) for "twin studies" is out

All come from one translation, so the LLM's run-to-run variation cannot
explain a difference between them. A question without alternatives is run
once; its variants are the same list.

The change only adds rows, so recall can only rise and whole-list precision
only fall. What decides is the order: compare recall@20 and nDCG@20 with

    python3 compare_pair.py alt_on.json alt_off.json ground_truth.json

Output: one <variant>.json per variant in the format of eval_recall.py; each
row also carries `alternatives` (word -> datasets matching it) and `own`.
Read-only transactions only; no re-ranker; no SQL self-correction; nothing
written to the database. The uncertain tier and paper text are as the
environment sets them (UNCERTAIN_TIER, TEXT_SEARCH_PAPERS).

    python paired_alternatives_run.py --queries /repo/benchmarks/recall/queries.yaml \\
        --out-dir /out --label "<code and database>" [--split held-out]
    python paired_alternatives_run.py --question "datasets dealing with dead people" --out-dir /out
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Every dataset, so recall is not cut by a page size.
ALL_ROWS = 5000
# An alternative may match this many datasets whatever the user's own words
# match ("dead people" matches none).
RATIO_FLOOR = 10

_LINE_RE = re.compile(r"^\s*dataset text: (?P<condition>.+?)\s*$", re.MULTILINE)
_CALL_RE = re.compile(r"\btext_match\s*\(\s*'([^']*)'\s*(?:,\s*'[^']*'\s*)?\)", re.IGNORECASE)


def description_filter(augmented_question: str) -> Optional[Tuple[str, str, List[str]]]:
    """(whole condition, the user's own text_match call, alternatives) or None."""
    line = _LINE_RE.search(augmented_question or "")
    if line is None:
        return None
    condition = line.group("condition")
    calls = list(_CALL_RE.finditer(condition))
    if len(calls) != 2 or not condition.startswith("("):
        return None
    return condition, calls[0].group(0), [a.strip() for a in calls[1].group(1).split("|") if a.strip()]


def with_alternatives(raw_sql: str, question: str, condition: str, own: str,
                      kept: List[str]) -> Tuple[str, str]:
    """The model's SQL and the augmented question with only *kept* alternatives."""
    replacement = own if not kept else f"({own} OR text_match('{' | '.join(kept)}', 'also'))"
    new_question = question.replace(condition, replacement)
    if condition in raw_sql:
        return raw_sql.replace(condition, replacement), new_question
    # The model wrote the filter in its own way: change the alternatives call.
    old_key = sorted(a.strip() for a in _CALL_RE.search(condition[len(own):]).group(1).split("|"))
    for call in _CALL_RE.finditer(raw_sql):
        if sorted(a.strip() for a in call.group(1).lower().split("|")) == old_key:
            new_call = f"text_match('{' | '.join(kept)}', 'also')" if kept else "FALSE"
            return raw_sql[:call.start()] + new_call + raw_sql[call.end():], new_question
    return raw_sql, new_question   # the model left them out; the rewriter reads the question


async def _run(questions: List[dict], out_dir: Path, backend_dir: str, label: str,
               caps: List[int], ratios: List[int]) -> None:
    sys.path.insert(0, backend_dir)
    import main  # noqa: F401  — sets up import paths exactly as the app does
    from settings import settings_obj
    from sqlalchemy import text
    from db.db import async_session_maker, engine
    from services import text_search, text_to_sql as translator, uncertain_tier
    from services.sql_rewriter import FALLBACK_SQL, build_page_sql
    from services.schema_capabilities import has_fts, refresh_available_columns
    try:
        import institution_index
    except ImportError:
        institution_index = None

    database = f"{settings_obj.METADATA_DB_HOST}:{settings_obj.METADATA_DB_PORT}/{settings_obj.METADATA_DB_NAME}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    variants = (["alt_off", "alt_on", "alt_flat"] + [f"alt_cap{cap}" for cap in caps]
                + [f"alt_ratio{ratio}" for ratio in ratios])
    metas = {v: {"label": f"{label} | description alternatives: {v} (same translation for all)",
                 "database": database, "started": started, "rerank_enabled": False,
                 "uncertain_tier": uncertain_tier.enabled()} for v in variants}
    results: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}

    await refresh_available_columns(engine)
    if institution_index is not None:
        await institution_index.refresh_from_db(engine)
    if has_fts():
        text_search.set_frequency_lookup(
            text_search.make_frequency_lookup(engine, asyncio.get_running_loop()))
    print(f"label={label!r} database={database} has_fts={has_fts()} "
          f"tier={uncertain_tier.enabled()}", file=sys.stderr, flush=True)

    # What the model wrote, before rewriting: the variants start from it.
    captured: Dict[str, Any] = {}
    rewrite = translator._rewrite_generated_sql

    def capturing(raw_sql, augmented_question, word_forms=None):
        captured.update(raw=raw_sql, question=augmented_question, forms=word_forms)
        return rewrite(raw_sql, augmented_question, word_forms)

    translator._rewrite_generated_sql = capturing

    async def rows_for(session, raw_sql, question, forms, translation,
                       flat: bool = False) -> Dict[str, Any]:
        rewritten = await asyncio.to_thread(rewrite, raw_sql, question, forms)
        rank = text_search.without_own(rewritten.rank_tsquery) if flat else rewritten.rank_tsquery
        page_sql, extra = build_page_sql(
            base_sql=rewritten.sql, scored_filters=translation.scored_filters,
            apply_relevance=translation.apply_relevance, limit=ALL_ROWS, offset=0,
            rank_tsquery=rank)
        await session.execute(text("SET TRANSACTION READ ONLY"))
        await session.execute(text("SET LOCAL statement_timeout = '600s'"))
        rows = (await session.execute(text(page_sql), {**rewritten.params, **extra})).mappings().all()
        await session.rollback()
        returned = [r["accession_id"] or str(r["id"]) for r in rows]
        return {"returned": returned, "total": len(returned), "sql": rewritten.sql,
                "rank_tsquery": rank,
                "levels": {(r["accession_id"] or str(r["id"])): int(r.get("uncertain_level") or 0)
                           for r in rows} if rows and "uncertain_level" in rows[0] else {}}

    for number, q in enumerate(questions, 1):
        t0 = time.time()
        common: Dict[str, Any] = {"id": q["id"], "question": q["question"]}
        per_variant: Dict[str, Dict[str, Any]] = {}
        captured.clear()
        try:
            translation = await translator.text_to_sql(q["question"])
            fallback = translation.sql == FALLBACK_SQL or not captured
            common.update(fallback=fallback, self_corrected=False, reranked=0,
                          explanation=translation.explanation)
            if fallback:
                raise RuntimeError("no translation (fallback sample)")
            raw, question, forms = captured["raw"], captured["question"], captured["forms"]
            found = description_filter(question)
            alternatives: Dict[str, Optional[int]] = {}
            if found:
                condition, own, words = found
                counts = await asyncio.to_thread(text_search.count_matches, words)
                alternatives = dict(zip(words, counts or [None] * len(words)))
                own_words = _CALL_RE.match(own).group(1)
                own_count = (await asyncio.to_thread(
                    text_search.count_matches, [own_words], "all") or [None])[0]
                common.update(own=own, own_count=own_count, alternatives=alternatives)
            async with async_session_maker() as session:
                per_variant["alt_on"] = await rows_for(session, raw, question, forms, translation)
                for variant in variants:
                    if variant == "alt_on":
                        continue
                    if not found:
                        per_variant[variant] = per_variant["alt_on"]
                        continue
                    if variant == "alt_flat":
                        per_variant[variant] = await rows_for(
                            session, raw, question, forms, translation, flat=True)
                        continue
                    if variant == "alt_off":
                        kept = []
                    else:
                        if variant.startswith("alt_ratio"):
                            cap = max(RATIO_FLOOR, int(variant[len("alt_ratio"):]) * (own_count or 0))
                        else:
                            cap = int(variant[len("alt_cap"):])
                        kept = [w for w in words if alternatives[w] is None or alternatives[w] <= cap]
                    if kept == words:
                        per_variant[variant] = per_variant["alt_on"]
                        continue
                    raw_v, question_v = with_alternatives(raw, question, condition, own, kept)
                    per_variant[variant] = await rows_for(session, raw_v, question_v, forms, translation)
        except Exception as exc:  # one failing question must not stop the run
            common.update(error=f"{type(exc).__name__}: {exc}"[:500])
            per_variant = {v: {"returned": []} for v in variants}
        common["seconds"] = round(time.time() - t0, 1)
        for variant in variants:
            results[variant].append({**common, **per_variant[variant]})
            (out_dir / f"{variant}.json").write_text(
                json.dumps({"meta": metas[variant], "results": results[variant]}, indent=1))
        # Progress only: per-question numbers of held-out runs must not be read.
        print(f"[{number:3d}/{len(questions)}] {common['seconds']}s"
              f"{' alternatives' if common.get('alternatives') else ''}"
              f"{' error' if common.get('error') else ''}", file=sys.stderr, flush=True)
    print(f"wrote {', '.join(v + '.json' for v in variants)} to {out_dir}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired description-alternatives measurement")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--question", action="append", default=[],
                        help="a question of your own instead of --queries (repeatable)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated query ids")
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--label", default="")
    parser.add_argument("--caps", default="30", help="comma-separated dataset limits to try")
    parser.add_argument("--ratios", default="", help="comma-separated limits relative to the "
                        "datasets the user's own words match")
    parser.add_argument("--split", choices=("development", "held-out", "all"), default="development",
                        help="held-out: only the held-out questions (once per finished change)")
    args = parser.parse_args()

    os.environ["RERANK_ENABLED"] = "false"
    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    if args.question:
        questions = [{"id": f"own_{i}", "question": q} for i, q in enumerate(args.question, 1)]
    else:
        import yaml
        questions = yaml.safe_load(args.queries.read_text())["queries"]
        # Held-out questions are scored once per finished change, never run while
        # developing (benchmarks/recall/held_out_questions.txt).
        held_out_file = args.queries.parent / "held_out_questions.txt"
        held_out = set()
        if held_out_file.exists():
            held_out = {line.strip() for line in held_out_file.read_text().splitlines()
                        if line.strip() and not line.startswith("#")}
        if args.split == "development":
            questions = [q for q in questions if q["id"] not in held_out]
        elif args.split == "held-out":
            questions = [q for q in questions if q["id"] in held_out]
        if args.only:
            wanted = set(args.only.split(","))
            questions = [q for q in questions if q["id"] in wanted]
    ratios = [int(r) for r in args.ratios.split(",") if r.strip()]
    asyncio.run(_run(questions, args.out_dir, args.backend_dir, args.label, caps, ratios))


if __name__ == "__main__":
    main()
