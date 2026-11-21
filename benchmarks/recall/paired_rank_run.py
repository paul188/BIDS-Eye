"""Paired measurement of the full-text rank.

Each question goes through the search pipeline once (intent, vocabulary lookup,
SQL generation, rewriting, COUNT with self-correction). The resulting base SQL
is then ordered twice:

  rank_on   build_page_sql(..., rank_tsquery=<the rewriter's rank query>)
  rank_off  build_page_sql(..., rank_tsquery=None)   -- what the API did while
            _run_local dropped the value

Both orderings come from the same translation, so the LLM's run-to-run
variation cannot explain a difference between them. The two output files have
the format of benchmarks/recall/eval_recall.py, so `eval_recall.py score` reads
them. Nothing is written to the database: read-only transactions only, no
re-ranker, no user_queries row.

Run inside a backend container, like eval_recall.py:

    python paired_rank_run.py --queries /repo/benchmarks/recall/queries.yaml \
        --out-dir /out --label "<code and database>"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


async def _run(questions: List[dict], out_dir: Path, backend_dir: str, label: str) -> None:
    sys.path.insert(0, backend_dir)
    from services_layout import use_either_layout  # older backends: flat services/
    use_either_layout()
    import main  # noqa: F401  — sets up import paths exactly as the app does
    from settings import settings_obj
    from sqlalchemy import text
    from db.db import async_session_maker, engine
    try:
        from services.search_runner import count_with_base_correction
    except ImportError:
        # Search code before the router split: these lived in routers/query.py.
        from routers.query import _count_with_base_correction as count_with_base_correction
    from services.stores.translated_queries import CachedQuery, query_cache
    from services.translate.sql_rewriter import FALLBACK_SQL, build_page_sql
    from services.translate.text_to_sql import text_to_sql
    from services.catalogue.schema_capabilities import has_fts, refresh_available_columns
    from services.fulltext import word_frequencies
    try:
        import institution_index
    except ImportError:
        institution_index = None

    database = f"{settings_obj.METADATA_DB_HOST}:{settings_obj.METADATA_DB_PORT}/{settings_obj.METADATA_DB_NAME}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    variants = ("rank_on", "rank_off")
    metas = {
        v: {"label": f"{label} | text rank {'ON' if v == 'rank_on' else 'OFF'} (same translation for both)",
            "database": database, "started": started, "rerank_enabled": False}
        for v in variants
    }
    results: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}

    # Same read-only warm-up as eval_recall.py; main.lifespan() is not called
    # because it may issue DDL.
    await refresh_available_columns(engine)
    if institution_index is not None:
        await institution_index.refresh_from_db(engine)
    try:
        import worker_db
    except ImportError:  # backend before worker_db.py
        if has_fts():
            word_frequencies.set_frequency_lookup(
                word_frequencies.make_frequency_lookup(engine, asyncio.get_running_loop()))
    else:
        worker_db.install(worker_db.PooledWorkerDb(engine, asyncio.get_running_loop()))
    print(f"label={label!r} database={database} has_fts={has_fts()}", file=sys.stderr, flush=True)

    for number, q in enumerate(questions, 1):
        t0 = time.time()
        common: Dict[str, Any] = {"id": q["id"], "question": q["question"]}
        per_variant: Dict[str, Dict[str, Any]] = {v: {} for v in variants}
        try:
            translation = await text_to_sql(q["question"])
            rank_tsquery = getattr(translation, "rank_tsquery", None)
            cached = CachedQuery(
                base_sql=translation.sql,
                params=translation.params,
                scored_filters=translation.scored_filters,
                apply_relevance=translation.apply_relevance,
                rank_tsquery=rank_tsquery,
                explanation=translation.explanation,
                self_corrected=translation.self_corrected,
            )
            query_id = query_cache.put(cached)
            async with async_session_maker() as session:
                await session.execute(text("SET TRANSACTION READ ONLY"))
                total = await count_with_base_correction(
                    session, query_id, cached, q["question"]
                )
                fallback = cached.base_sql == FALLBACK_SQL
                for variant in variants:
                    page_sql, extra = build_page_sql(
                        base_sql=cached.base_sql,
                        scored_filters=cached.scored_filters,
                        apply_relevance=cached.apply_relevance,
                        limit=max(total, 1),
                        offset=0,
                        rank_tsquery=rank_tsquery if variant == "rank_on" else None,
                    )
                    rows = (await session.execute(
                        text(page_sql), {**cached.params, **extra}
                    )).mappings().all()
                    per_variant[variant] = {
                        "returned": [r["accession_id"] or str(r["id"]) for r in rows],
                        "text_rank_positive": sum(1 for r in rows if (r.get("text_rank") or 0) > 0),
                    }
                await session.rollback()
            common.update(
                total=total,
                fallback=fallback,
                self_corrected=bool(cached.self_corrected),
                reranked=0,
                apply_relevance=bool(cached.apply_relevance),
                rank_tsquery=rank_tsquery,
                scored_filters=len(cached.scored_filters or []),
                explanation=translation.explanation,
                sql=cached.base_sql,
            )
        except Exception as exc:  # one failing question must not stop the run
            common.update(error=f"{type(exc).__name__}: {exc}"[:500])
            per_variant = {v: {"returned": []} for v in variants}
        common["seconds"] = round(time.time() - t0, 1)
        for variant in variants:
            results[variant].append({**common, **per_variant[variant]})
            (out_dir / f"{variant}.json").write_text(
                json.dumps({"meta": metas[variant], "results": results[variant]}, indent=1)
            )
        print(f"[{number:2d}/{len(questions)}] {q['id']:28} total={common.get('total')} "
              f"fallback={common.get('fallback')} rank_tsquery={common.get('rank_tsquery')!r} "
              f"{common['seconds']}s {common.get('error', '')}", file=sys.stderr, flush=True)
    print(f"wrote {out_dir}/rank_on.json and rank_off.json", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired text-rank measurement")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated query ids")
    parser.add_argument("--split", choices=("development", "held-out", "all"), default="development",
                        help="'development' (default) skips the questions in held_out_questions.txt "
                             "next to the queries file")
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    import os
    import yaml
    os.environ["RERANK_ENABLED"] = "false"
    questions = yaml.safe_load(args.queries.read_text())["queries"]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    elif args.split != "all":
        listed = args.queries.parent / "held_out_questions.txt"
        lines = (line.strip() for line in listed.read_text().splitlines()) if listed.exists() else ()
        held = {line for line in lines if line and not line.startswith("#")}
        questions = [q for q in questions if (q["id"] in held) == (args.split == "held-out")]
    asyncio.run(_run(questions, args.out_dir, args.backend_dir, args.label))


if __name__ == "__main__":
    main()
