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
    from db.db import async_session_maker
    import routers.query as router
    from services.query_cache import CachedQuery, query_cache
    from services.sql_rewriter import FALLBACK_SQL, build_page_sql
    from services.text_to_sql import text_to_sql

    page_params = inspect.signature(build_page_sql).parameters
    results: List[Dict[str, Any]] = []
    # Recorded with the results so a run can never be mistaken for another:
    # which code (label, e.g. the commits in the image) against which database.
    meta = {
        "label": label,
        "database": f"{settings_obj.METADATA_DB_HOST}:{settings_obj.METADATA_DB_PORT}/{settings_obj.METADATA_DB_NAME}",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(f"label={label!r} database={meta['database']}", file=sys.stderr, flush=True)
    async with main.lifespan(main.app):
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
                    total = await router._count_with_base_correction(
                        session, query_id, cached, q["question"]
                    )
                    kwargs = dict(
                        base_sql=cached.base_sql,
                        scored_filters=cached.scored_filters,
                        apply_relevance=cached.apply_relevance,
                        limit=max(total, 1),
                        offset=0,
                    )
                    if "rank_tsquery" in page_params:
                        kwargs["rank_tsquery"] = getattr(cached, "rank_tsquery", None)
                    page_sql, extra = build_page_sql(**kwargs)
                    await session.execute(text("SET TRANSACTION READ ONLY"))
                    rows = (await session.execute(text(page_sql), {**cached.params, **extra})).mappings().all()
                    await session.rollback()
                record.update(
                    total=total,
                    returned=[r["accession_id"] or str(r["id"]) for r in rows],
                    fallback=cached.base_sql == FALLBACK_SQL,
                    self_corrected=bool(cached.self_corrected),
                    explanation=translation.explanation,
                    sql=cached.base_sql,
                )
            except Exception as exc:  # one failing question must not stop the run
                record.update(error=f"{type(exc).__name__}: {exc}"[:500], returned=[])
            record["seconds"] = round(time.time() - started, 1)
            results.append(record)
            print(f"[{number:2d}/{len(questions)}] {q['id']:28} total={record.get('total')} "
                  f"fallback={record.get('fallback')} {record['seconds']}s", file=sys.stderr, flush=True)
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
        judged = relevant | set(t.get("not_relevant", [])) | set(t.get("unsure", []))
        returned = r.get("returned") or []
        hit = [a for a in returned if a in relevant]
        first_page = set(returned[:PAGE])
        judged_returned = [a for a in returned if a in judged or t["method"] == "structural"]
        rows.append({
            "id": qid,
            "n_relevant": len(relevant),
            "returned": len(returned),
            "recall": len(hit) / len(relevant) if relevant else None,
            "recall_at_20": len(first_page & relevant) / min(len(relevant), PAGE) if relevant else None,
            "precision": (len(hit) / len(judged_returned)) if judged_returned else None,
            "unjudged": [a for a in returned if t["method"] != "structural" and a not in judged],
            "fallback": r.get("fallback"),
            "error": r.get("error"),
        })

    def fmt(x: Optional[float]) -> str:
        return "   —" if x is None else f"{x:4.0%}"

    print(f"{'query':28} {'rel':>4} {'ret':>5} {'recall':>6} {'R@20':>5} {'prec':>5} {'unjdg':>5}  note")
    for row in rows:
        note = "FALLBACK SAMPLE" if row["fallback"] else (row["error"] or "")
        print(f"{row['id']:28} {row['n_relevant']:4d} {row['returned']:5d} {fmt(row['recall']):>6} "
              f"{fmt(row['recall_at_20']):>5} {fmt(row['precision']):>5} {len(row['unjudged']):5d}  {note[:60]}")

    scored = [row for row in rows if row["recall"] is not None]
    if scored:
        avg = lambda key: sum(row[key] for row in scored if row[key] is not None) / max(1, sum(row[key] is not None for row in scored))
        print(f"\nAverage over {len(scored)} queries with relevant datasets: "
              f"recall {avg('recall'):.0%}, recall@20 {avg('recall_at_20'):.0%}, precision {avg('precision'):.0%}")
        missed = sum(row["recall"] == 0 for row in scored)
        print(f"Queries that found none of their relevant datasets: {missed}")
    empty = [row for row in rows if row["recall"] is None]
    for row in empty:
        print(f"No relevant dataset exists for {row['id']}: returned {row['returned']} (all false positives)")
    if show_unjudged:
        print("\nReturned but never judged (candidates for widening the pool):")
        for row in rows:
            if row["unjudged"]:
                print(f"  {row['id']}: {', '.join(row['unjudged'][:show_unjudged])}"
                      + (f" … (+{len(row['unjudged']) - show_unjudged})" if len(row["unjudged"]) > show_unjudged else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--queries", type=Path, default=HERE / "queries.yaml")
    run.add_argument("--only", default="", help="comma-separated query ids")
    run.add_argument("--backend-dir", default="/app/backend")
    run.add_argument("--label", default="", help="what is being measured, e.g. the commits in the image")
    score = sub.add_parser("score")
    score.add_argument("run", type=Path)
    score.add_argument("--truth", type=Path, default=HERE / "ground_truth.json")
    score.add_argument("--show-unjudged", type=int, default=8)
    args = parser.parse_args()

    if args.command == "run":
        import yaml
        questions = yaml.safe_load(args.queries.read_text())["queries"]
        if args.only:
            wanted = set(args.only.split(","))
            questions = [q for q in questions if q["id"] in wanted]
        asyncio.run(_run(questions, args.out, args.backend_dir, args.label))
    else:
        _score(args.run, args.truth, args.show_unjudged)


if __name__ == "__main__":
    main()
