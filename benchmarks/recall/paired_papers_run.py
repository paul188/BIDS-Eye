"""Paired measurement of paper text in full-text search.

Each question goes through the search pipeline once, with paper text off
(intent, vocabulary lookup, SQL generation, rewriting, COUNT with
self-correction). The resulting SQL is then run twice:

  papers_off   as translated: text filters on search_tsv
  papers_on    the same SQL with every positive text filter on
               search_tsv_papers (text_search.with_paper_text), which is what
               the API does with TEXT_SEARCH_PAPERS=true

Ranking is identical in both (it never reads paper text), so the only
difference is which datasets match. Both come from one translation, so the
LLM's run-to-run variation cannot explain a difference between them.

Output: papers_off.json and papers_on.json in the format of eval_recall.py,
plus, per question in papers_on, `added`: the datasets only paper text found.
Read-only transactions only; no re-ranker; nothing written to the database.

Needs a database at the newest migration (backend/migrations/README.md) where
ingest/backfill/backfill_papers.py has filled paper_text. Run inside a backend
container, like eval_recall.py:

    python paired_papers_run.py --queries /repo/benchmarks/recall/queries.yaml \\
        --out-dir /out --label "<code and database>"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Every dataset, so recall is not cut by a page size.
ALL_ROWS = 5000


async def _run(questions: List[dict], out_dir: Path, backend_dir: str, label: str) -> None:
    sys.path.insert(0, backend_dir)
    from services_layout import use_either_layout  # older backends: flat services/
    use_either_layout()
    from rank_compat import rank_json, rank_kwargs, rank_of  # rank_tsquery before RankQuery
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
    from backend_capabilities import has_fts, has_paper_fts, start
    from services.fulltext import text_search, word_frequencies
    database = f"{settings_obj.METADATA_DB_HOST}:{settings_obj.METADATA_DB_PORT}/{settings_obj.METADATA_DB_NAME}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    variants = ("papers_off", "papers_on")
    metas = {
        v: {"label": f"{label} | paper text {'ON' if v == 'papers_on' else 'OFF'} (same translation for both)",
            "database": database, "started": started, "rerank_enabled": False}
        for v in variants
    }
    results: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}

    await start(engine)
    try:
        import worker_db
    except ImportError:  # backend before worker_db.py
        if has_fts():
            word_frequencies.set_frequency_lookup(
                word_frequencies.make_frequency_lookup(engine, asyncio.get_running_loop()))
    else:
        worker_db.install(worker_db.PooledWorkerDb(engine, asyncio.get_running_loop()))
    if not has_paper_fts():
        raise SystemExit("search_tsv_papers is missing: run `alembic upgrade head` in backend/ first")
    print(f"label={label!r} database={database} has_fts={has_fts()} has_paper_fts={has_paper_fts()}",
          file=sys.stderr, flush=True)

    for number, q in enumerate(questions, 1):
        t0 = time.time()
        common: Dict[str, Any] = {"id": q["id"], "question": q["question"]}
        per_variant: Dict[str, Dict[str, Any]] = {v: {} for v in variants}
        try:
            translation = await text_to_sql(q["question"])
            rank = rank_of(translation)
            cached = CachedQuery(
                base_sql=translation.sql,
                params=translation.params,
                scored_filters=translation.scored_filters,
                apply_relevance=translation.apply_relevance,
                **rank_kwargs(CachedQuery, rank),
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
                    base_sql = cached.base_sql
                    if variant == "papers_on" and not fallback:
                        base_sql = text_search.with_paper_text(base_sql)
                    page_sql, extra = build_page_sql(
                        base_sql=base_sql,
                        scored_filters=cached.scored_filters,
                        apply_relevance=cached.apply_relevance,
                        limit=ALL_ROWS,
                        offset=0,
                        **rank_kwargs(build_page_sql, rank),
                    )
                    rows = (await session.execute(
                        text(page_sql), {**cached.params, **extra}
                    )).mappings().all()
                    per_variant[variant] = {
                        "returned": [r["accession_id"] or str(r["id"]) for r in rows],
                        "uses_papers": "search_tsv_papers" in base_sql,
                    }
                await session.rollback()
            off = set(per_variant["papers_off"]["returned"])
            per_variant["papers_on"]["added"] = [
                a for a in per_variant["papers_on"]["returned"] if a not in off]
            common.update(
                total=total,
                fallback=fallback,
                self_corrected=bool(cached.self_corrected),
                reranked=0,
                apply_relevance=bool(cached.apply_relevance),
                rank_query=rank_json(rank),
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
        added = len(per_variant["papers_on"].get("added", []))
        print(f"[{number:3d}/{len(questions)}] {q['id']:28} total={common.get('total')} "
              f"off={len(per_variant['papers_off']['returned'])} +papers={added} "
              f"{common['seconds']}s {common.get('error', '')}", file=sys.stderr, flush=True)
    print(f"wrote {out_dir}/papers_off.json and papers_on.json", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired paper-text measurement")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated query ids")
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--label", default="")
    parser.add_argument("--include-held-out", action="store_true",
                        help="also run the held-out questions (once per finished change)")
    args = parser.parse_args()

    import yaml
    os.environ["RERANK_ENABLED"] = "false"
    # The translation must not use paper text; the second variant adds it.
    os.environ["TEXT_SEARCH_PAPERS"] = "false"
    questions = yaml.safe_load(args.queries.read_text())["queries"]
    # Held-out questions are scored once per finished change, never run while
    # developing (benchmarks/recall/held_out_questions.txt).
    held_out_file = args.queries.parent / "held_out_questions.txt"
    if held_out_file.exists() and not args.include_held_out:
        held_out = {line.strip() for line in held_out_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
        questions = [q for q in questions if q["id"] not in held_out]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    asyncio.run(_run(questions, args.out_dir, args.backend_dir, args.label))


if __name__ == "__main__":
    main()
