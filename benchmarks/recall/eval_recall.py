"""
benchmarks/recall/eval_recall.py
--------------------------------
Measure search recall against the judged query set.

Two steps, because the search needs the LLM keys and the database while the
scoring needs only files:

1. run — inside a backend container (it imports the backend code):

       docker compose -f docker-compose-hetzner.yml run --rm --no-deps \\
           -v /home/BIDS-Eye:/repo:ro -v /tmp/bidseye_recall:/out backend \\
           python /repo/benchmarks/recall/eval_recall.py run --out /out/run.json

   Every question goes through the same path as a user's request: intent
   extraction, vocabulary lookup, SQL generation and rewriting, the COUNT with
   self-correction (and fallback), and the ranked page query. The full ranked
   result list is kept, not only the first page. Nothing is written to the
   database: no user_queries row, no rate-limit row, read-only transactions.

2. score — anywhere:

       python benchmarks/recall/eval_recall.py score run.json

   Prints per-query and average recall, precision and recall@20 (the first
   page a user sees), and lists returned datasets that nobody judged, so the
   pool can be widened.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
PAGE = 20


# ── run ───────────────────────────────────────────────────────────────────────

async def _run(questions: List[dict], out: Path, backend_dir: str, label: str) -> None:
    sys.path.insert(0, backend_dir)
    import main  # noqa: F401  — sets up import paths exactly as the app does
    from settings import settings_obj
    from sqlalchemy import text
    from db.db import async_session_maker, engine
    import routers.query as router
    from services.query_cache import CachedQuery, query_cache
    from services.sql_rewriter import FALLBACK_SQL, build_page_sql
    from services.text_to_sql import text_to_sql
    from services.schema_capabilities import refresh_available_columns
    # Optional pieces: present only in newer versions of the search code. The
    # harness must also run against older versions (e.g. the live one) to
    # measure them on the same questions.
    try:
        from services.schema_capabilities import has_fts
        from services import text_search
    except ImportError:
        has_fts, text_search = (lambda: False), None
    try:
        import institution_index
    except ImportError:
        institution_index = None
    rerank_settings = getattr(router, "rerank_settings", None)
    rerank_once = getattr(router, "_rerank_once", None)

    page_params = inspect.signature(build_page_sql).parameters
    results: List[Dict[str, Any]] = []
    # Recorded with the results so a run can never be mistaken for another:
    # which code (label, e.g. the commits in the image) against which database.
    meta = {
        "label": label,
        "database": f"{settings_obj.METADATA_DB_HOST}:{settings_obj.METADATA_DB_PORT}/{settings_obj.METADATA_DB_NAME}",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rerank_enabled": bool(rerank_settings and rerank_settings()["enabled"]),
    }
    print(f"label={label!r} database={meta['database']}", file=sys.stderr, flush=True)
    # Warm up the same read-only capabilities as app startup without calling
    # main.lifespan(), which runs create_all_tables() and may issue DDL.
    await refresh_available_columns(engine)
    if institution_index is not None:
        await institution_index.refresh_from_db(engine)
    if has_fts() and text_search is not None:
        text_search.set_frequency_lookup(
            text_search.make_frequency_lookup(engine, asyncio.get_running_loop())
        )

    for number, q in enumerate(questions, 1):
        started = time.time()
        record: Dict[str, Any] = {"id": q["id"], "question": q["question"]}
        try:
            translation = await text_to_sql(q["question"])
            cached = CachedQuery(
                base_sql=translation.sql,
                params=translation.params,
                scored_filters=translation.scored_filters,
                apply_relevance=translation.apply_relevance,
                explanation=translation.explanation,
                self_corrected=translation.self_corrected,
            )
            for name in ("rank_tsquery",):  # newer versions carry more ranking input
                if hasattr(translation, name) and hasattr(cached, name):
                    setattr(cached, name, getattr(translation, name))
            query_id = query_cache.put(cached)
            async with async_session_maker() as session:
                await session.execute(text("SET TRANSACTION READ ONLY"))
                total = await router._count_with_base_correction(
                    session, query_id, cached, q["question"]
                )
                if rerank_once is not None:
                    await rerank_once(session, query_id, cached, q["question"], total)
                kwargs = dict(
                    base_sql=cached.base_sql,
                    scored_filters=cached.scored_filters,
                    apply_relevance=cached.apply_relevance,
                    limit=max(total, 1),
                    offset=0,
                )
                if "rank_tsquery" in page_params:
                    kwargs["rank_tsquery"] = getattr(cached, "rank_tsquery", None)
                if "rerank_ordered" in page_params:
                    kwargs["rerank_ordered"] = cached.rerank_ordered
                    kwargs["rerank_demoted"] = cached.rerank_demoted
                page_sql, extra = build_page_sql(**kwargs)
                rows = (await session.execute(text(page_sql), {**cached.params, **extra})).mappings().all()
                await session.rollback()
            record.update(
                total=total,
                returned=[r["accession_id"] or str(r["id"]) for r in rows],
                fallback=cached.base_sql == FALLBACK_SQL,
                self_corrected=bool(cached.self_corrected),
                reranked=len(getattr(cached, "rerank_ordered", None) or []) + len(getattr(cached, "rerank_demoted", None) or []),
                explanation=translation.explanation,
                sql=cached.base_sql,
            )
        except Exception as exc:  # one failing question must not stop the run
            record.update(error=f"{type(exc).__name__}: {exc}"[:500], returned=[])
        record["seconds"] = round(time.time() - started, 1)
        results.append(record)
        print(f"[{number:2d}/{len(questions)}] {q['id']:28} total={record.get('total')} "
              f"fallback={record.get('fallback')} reranked={record.get('reranked', 0)} "
              f"{record['seconds']}s", file=sys.stderr, flush=True)
        out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"wrote {out}", file=sys.stderr)


# ── score ─────────────────────────────────────────────────────────────────────

def _load_truth(truth_path: Path) -> Dict[str, dict]:
    return json.loads(truth_path.read_text())["queries"]


def _score(run_path: Path, truth_path: Path, show_unjudged: int) -> None:
    truth = _load_truth(truth_path)
    data = json.loads(run_path.read_text())
    meta = data.get("meta") or {}
    if meta:
        print(f"run: {meta.get('label') or '(no label)'} | database {meta.get('database')} | {meta.get('started')}\n")
    run = {r["id"]: r for r in data["results"]}
    rows = []
    for qid, t in truth.items():
        r = run.get(qid)
        if r is None:
            continue
        relevant = set(t["relevant"])
        partial = set(t.get("partially_relevant", []))
        judged = relevant | partial | set(t.get("not_relevant", [])) | set(t.get("unsure", []))
        returned = r.get("returned") or []
        hit = [a for a in returned if a in relevant]
        first_page = set(returned[:PAGE])
        judged_returned = [a for a in returned if a in judged or t["method"] == "structural"]
        lenient = relevant | partial
        rows.append({
            "id": qid,
            "group": _group(t),
            "n_relevant": len(relevant),
            "returned": len(returned),
            "recall": len(hit) / len(relevant) if relevant else None,
            # relevant + partially relevant both count as found
            "recall_lenient": len([a for a in returned if a in lenient]) / len(lenient) if lenient else None,
            "recall_at_20": len(first_page & relevant) / min(len(relevant), PAGE) if relevant else None,
            "precision": (len(hit) / len(judged_returned)) if judged_returned else None,
            "unjudged": [a for a in returned if t["method"] != "structural" and a not in judged],
            "fallback": r.get("fallback"),
            "error": r.get("error"),
        })

    def fmt(x: Optional[float]) -> str:
        return "   —" if x is None else f"{x:4.0%}"

    print(f"{'query':30} {'rel':>4} {'ret':>5} {'recall':>6} {'lenient':>7} {'R@20':>5} {'prec':>5} {'unjdg':>5}  note")
    for row in rows:
        note = "FALLBACK SAMPLE" if row["fallback"] else (row["error"] or "")
        print(f"{row['id']:30} {row['n_relevant']:4d} {row['returned']:5d} {fmt(row['recall']):>6} "
              f"{fmt(row['recall_lenient']):>7} {fmt(row['recall_at_20']):>5} {fmt(row['precision']):>5} "
              f"{len(row['unjudged']):5d}  {note[:50]}")

    def summary(subset: List[dict], label: str) -> None:
        scored = [row for row in subset if row["recall"] is not None]
        if not scored:
            return
        def avg(key: str) -> str:
            vals = [row[key] for row in scored if row[key] is not None]
            return f"{sum(vals) / len(vals):4.0%}" if vals else "   —"
        print(f"{label:22} {len(scored):4d} {avg('recall'):>7} {avg('recall_lenient'):>8} "
              f"{avg('recall_at_20'):>6} {avg('precision'):>6} {sum(row['recall'] == 0 for row in scored):>9}")

    print(f"\n{'group':22} {'n':>4} {'recall':>7} {'lenient':>8} {'R@20':>6} {'prec':>6} {'found none':>9}")
    for group in sorted({row["group"] for row in rows}):
        summary([row for row in rows if row["group"] == group], group)
    summary(rows, "ALL")
    fallbacks = sum(bool(row["fallback"]) for row in rows)
    if fallbacks:
        print(f"\n{fallbacks} question(s) fell back to the 200-dataset sample.")
    for row in rows:
        if row["recall"] is None:
            print(f"No relevant dataset exists for {row['id']}: returned {row['returned']} (all false positives)")
    if show_unjudged:
        print("\nReturned but never judged (candidates for widening the pool):")
        for row in rows:
            if row["unjudged"]:
                print(f"  {row['id']}: {', '.join(row['unjudged'][:show_unjudged])}"
                      + (f" … (+{len(row['unjudged']) - show_unjudged})" if len(row["unjudged"]) > show_unjudged else ""))


def _group(t: dict) -> str:
    """Reporting group: logic questions by operator, the rest by source."""
    qtype = t.get("type") or ""
    if qtype.startswith("logic_"):
        return qtype.replace("_", " ").upper().replace("LOGIC ", "logic: ")
    return {"user_log": "real user questions", "probe": "probes", "generated_readme": "generated from READMEs",
            "paper": "from paper abstracts", "biocaddie_style": "bioCADDIE-style"}.get(t.get("source") or "", "other")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--queries", type=Path, default=HERE / "queries.yaml")
    run.add_argument("--only", default="", help="comma-separated query ids")
    run.add_argument("--backend-dir", default="/app/backend")
    run.add_argument("--label", default="", help="what is being measured, e.g. the commits in the image")
    run.add_argument("--no-rerank", action="store_true", help="measure SQL retrieval without the LLM re-ranker")
    score = sub.add_parser("score")
    score.add_argument("run", type=Path)
    score.add_argument("--truth", type=Path, default=HERE / "ground_truth.json")
    score.add_argument("--show-unjudged", type=int, default=8)
    args = parser.parse_args()

    if args.command == "run":
        import yaml
        if args.no_rerank:
            import os
            os.environ["RERANK_ENABLED"] = "false"
        questions = yaml.safe_load(args.queries.read_text())["queries"]
        if args.only:
            wanted = set(args.only.split(","))
            questions = [q for q in questions if q["id"] in wanted]
        asyncio.run(_run(questions, args.out, args.backend_dir, args.label))
    else:
        _score(args.run, args.truth, args.show_unjudged)


if __name__ == "__main__":
    main()
