"""Paired measurement of exact word forms (backend/services/fulltext/text_search.py,
TEXT_SEARCH_EXACT; backend/vocabulary/stem_collisions.py).

Each question goes through the search pipeline once, with exact word forms
on: the intent model sees the catalogue's forms of the question's colliding
words and picks the ones meant. The SQL the model then writes is rewritten
twice:

  exact_off  as the search does without the switch: every word by its stem
  exact_on   with the decided forms (and the reviewed list) required

Both come from one translation, so the LLM's run-to-run variation cannot
explain a difference between them. The uncertain tier follows the
environment (UNCERTAIN_TIER), the same for both.

Output: exact_off.json and exact_on.json in the format of eval_recall.py
(score them with `eval_recall.py score`, or `--held-out` for held-out runs).
exact_on rows also carry
  allowed_forms  the forms decided for the question's words
  guarded        whether any search in the SQL was limited to word forms
  removed        datasets exact_off returned and exact_on did not
  added          the reverse (a NOT term that no longer excludes by stem)
and both carry `levels` (accession -> 0 confirmed, 1 probable, 2 possible).
Read-only transactions only; no re-ranker; nothing written to the database.

    python paired_exact_run.py --queries /repo/benchmarks/recall/queries.yaml \\
        --out-dir /out --label "<code and database>" [--only id1,id2] [--split held-out]
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
EXACT_ENV = "TEXT_SEARCH_EXACT"


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
        try:
            from services.search.search_runner import count_with_base_correction
        except ImportError:  # before services/search/ was a folder
            from services.search_runner import count_with_base_correction
    except ImportError:
        # Search code before the router split: these lived in routers/query.py.
        from routers.query import _count_with_base_correction as count_with_base_correction
    from services.fulltext import word_frequencies
    from services.translate import text_to_sql as translator
    from services.stores.translated_queries import CachedQuery, query_cache
    from backend_capabilities import has_exact_fts, has_fts, has_word_forms, start
    from services.translate.sql_rewriter import FALLBACK_SQL, build_page_sql
    database = f"{settings_obj.METADATA_DB_HOST}:{settings_obj.METADATA_DB_PORT}/{settings_obj.METADATA_DB_NAME}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    variants = ("exact_off", "exact_on")
    metas = {
        v: {"label": f"{label} | exact word forms {'ON' if v == 'exact_on' else 'OFF'} "
                     "(same translation for both)",
            "database": database, "started": started, "rerank_enabled": False}
        for v in variants
    }
    results: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}

    await start(engine)
    try:
        try:
            from database import worker as worker_db
        except ImportError:  # backend before database/
            import worker_db
    except ImportError:  # backend before worker_db.py
        if has_fts():
            word_frequencies.set_frequency_lookup(
                word_frequencies.make_frequency_lookup(engine, asyncio.get_running_loop()))
    else:
        worker_db.install(worker_db.PooledWorkerDb(engine, asyncio.get_running_loop()))
    if not (has_exact_fts() and has_word_forms()):
        raise SystemExit("search_tsv_exact or bids_word_forms is missing: run "
                         "`alembic upgrade head` in backend/ and "
                         "`python post_crawl.py --all --only word_forms` in ingest/openneuro-crawler/")
    print(f"label={label!r} database={database}", file=sys.stderr, flush=True)

    # The translation rewrites the model's SQL once. Keep what it was given,
    # to rewrite the same SQL again without exact word forms.
    seen: Dict[str, Any] = {}
    rewrite = translator._rewrite_generated_sql

    def recording_rewrite(raw_sql, augmented_question, allowed_forms=None):
        seen.update(raw=raw_sql, question=augmented_question, allowed_forms=allowed_forms)
        return rewrite(raw_sql, augmented_question, allowed_forms)

    translator._rewrite_generated_sql = recording_rewrite

    for number, q in enumerate(questions, 1):
        t0 = time.time()
        common: Dict[str, Any] = {"id": q["id"], "question": q["question"]}
        per_variant: Dict[str, Dict[str, Any]] = {v: {} for v in variants}
        try:
            seen.clear()
            os.environ[EXACT_ENV] = "true"
            translation = await translator.text_to_sql(q["question"])
            sqls = {"exact_on": (translation.sql, rank_of(translation))}
            if seen:
                os.environ[EXACT_ENV] = "false"
                stemmed = rewrite(seen["raw"], seen["question"], None)
                os.environ[EXACT_ENV] = "true"
                sqls["exact_off"] = (stemmed.sql, rank_of(stemmed))
            else:   # keyword fallback: nothing was rewritten, both are the same
                sqls["exact_off"] = sqls["exact_on"]
            for variant in variants:
                base_sql, rank = sqls[variant]
                cached = CachedQuery(
                    base_sql=base_sql,
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
                    page_sql, extra = build_page_sql(
                        base_sql=cached.base_sql,
                        scored_filters=cached.scored_filters,
                        apply_relevance=cached.apply_relevance,
                        limit=ALL_ROWS,
                        offset=0,
                        **rank_kwargs(build_page_sql, rank_of(cached)),
                    )
                    # The whole ranked list, not one page: give it more time
                    # than a search gets (services/query_guard.py).
                    await session.execute(text("SET LOCAL statement_timeout = '600s'"))
                    rows = (await session.execute(
                        text(page_sql), {**cached.params, **extra}
                    )).mappings().all()
                    await session.rollback()
                per_variant[variant] = {
                    "total": total,
                    "returned": [r["accession_id"] or str(r["id"]) for r in rows],
                    "levels": {(r["accession_id"] or str(r["id"])): int(r.get("uncertain_level") or 0)
                               for r in rows},
                    "fallback": cached.base_sql == FALLBACK_SQL,
                    "self_corrected": bool(cached.self_corrected),
                    "rank_query": rank_json(rank_of(cached)),
                    "sql": cached.base_sql,
                }
            off = per_variant["exact_off"]["returned"]
            on = per_variant["exact_on"]["returned"]
            per_variant["exact_on"].update(
                allowed_forms=seen.get("allowed_forms") or {},
                guarded="search_tsv_exact" in per_variant["exact_on"]["sql"],
                removed=[a for a in off if a not in set(on)],
                added=[a for a in on if a not in set(off)],
            )
            common.update(reranked=0, apply_relevance=bool(translation.apply_relevance),
                          explanation=translation.explanation)
        except Exception as exc:  # one failing question must not stop the run
            common.update(error=f"{type(exc).__name__}: {exc}"[:500])
            per_variant = {v: {"returned": []} for v in variants}
        common["seconds"] = round(time.time() - t0, 1)
        for variant in variants:
            results[variant].append({**common, **per_variant[variant]})
            (out_dir / f"{variant}.json").write_text(
                json.dumps({"meta": metas[variant], "results": results[variant]}, indent=1)
            )
        # Progress only: per-question numbers of held-out runs must not be read.
        print(f"[{number:3d}/{len(questions)}] {common['seconds']}s"
              f"{' error' if common.get('error') else ''}", file=sys.stderr, flush=True)
    print(f"wrote {out_dir}/exact_off.json and exact_on.json", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired exact-word-forms measurement")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated query ids")
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--label", default="")
    parser.add_argument("--split", choices=("development", "held-out", "all"), default="development",
                        help="held-out: only the held-out questions (once per finished change)")
    args = parser.parse_args()

    import yaml
    os.environ["RERANK_ENABLED"] = "false"
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
    asyncio.run(_run(questions, args.out_dir, args.backend_dir, args.label))


if __name__ == "__main__":
    main()
