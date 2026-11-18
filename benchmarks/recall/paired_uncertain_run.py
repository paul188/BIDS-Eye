"""Paired measurement of the uncertain tier (backend/services/fulltext/uncertain_tier.py).

Each question goes through the search pipeline once with the tier off
(intent, vocabulary lookup, SQL generation, rewriting, COUNT with
self-correction). The resulting SQL is then run twice:

  tier_off   as translated
  tier_on    uncertain_tier.apply() of the same SQL, which is what the API
             does with UNCERTAIN_TIER=true (paper text as second-tier
             evidence when the database has it)

Both come from one translation, so the LLM's run-to-run variation cannot
explain a difference between them. The confirmed rows of tier_on are the rows
of tier_off.

Output: tier_off.json and tier_on.json in the format of eval_recall.py (score
them with `eval_recall.py score`, or `--held-out` for held-out runs); tier_on
rows also carry `levels` (accession -> 0 confirmed, 1 probable, 2 possible).
Read-only transactions only; no re-ranker; nothing written to the database.

    python paired_uncertain_run.py --queries /repo/benchmarks/recall/queries.yaml \\
        --out-dir /out --label "<code and database>" [--split held-out]
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
    import main  # noqa: F401  — sets up import paths exactly as the app does
    from settings import settings_obj
    from sqlalchemy import text
    from db.db import async_session_maker, engine
    try:
        from services.search_runner import count_with_base_correction
    except ImportError:
        # Search code before the router split: these lived in routers/query.py.
        from routers.query import _count_with_base_correction as count_with_base_correction
    from services.stores.query_cache import CachedQuery, query_cache
    from services.translate.sql_rewriter import FALLBACK_SQL, build_page_sql
    from services.translate.text_to_sql import text_to_sql
    from services.catalogue.schema_capabilities import has_fts, has_paper_fts, refresh_available_columns
    from services.fulltext import text_search
    try:
        import institution_index
    except ImportError:
        institution_index = None

    database = f"{settings_obj.METADATA_DB_HOST}:{settings_obj.METADATA_DB_PORT}/{settings_obj.METADATA_DB_NAME}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    from services.fulltext import uncertain_tier
    variants = ("tier_off", "tier_on")
    metas = {
        v: {"label": f"{label} | uncertain tier {'ON' if v == 'tier_on' else 'OFF'} (same translation for both)",
            "database": database, "started": started, "rerank_enabled": False}
        for v in variants
    }
    results: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}

    await refresh_available_columns(engine)
    if institution_index is not None:
        await institution_index.refresh_from_db(engine)
    if has_fts():
        text_search.set_frequency_lookup(
            text_search.make_frequency_lookup(engine, asyncio.get_running_loop())
        )
    papers = has_paper_fts()
    print(f"label={label!r} database={database} has_fts={has_fts()} paper evidence={papers}",
          file=sys.stderr, flush=True)

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
                    base_sql = cached.base_sql
                    if variant == "tier_on" and not fallback:
                        base_sql = uncertain_tier.apply(base_sql, papers=papers)
                    page_sql, extra = build_page_sql(
                        base_sql=base_sql,
                        scored_filters=cached.scored_filters,
                        apply_relevance=cached.apply_relevance,
                        limit=ALL_ROWS,
                        offset=0,
                        rank_tsquery=rank_tsquery,
                    )
                    rows = (await session.execute(
                        text(page_sql), {**cached.params, **extra}
                    )).mappings().all()
                    per_variant[variant] = {
                        "returned": [r["accession_id"] or str(r["id"]) for r in rows],
                    }
                    if variant == "tier_on":
                        per_variant[variant]["levels"] = {
                            (r["accession_id"] or str(r["id"])): int(r.get("uncertain_level") or 0)
                            for r in rows}
                await session.rollback()
            off = set(per_variant["tier_off"]["returned"])
            per_variant["tier_on"]["added"] = [
                a for a in per_variant["tier_on"]["returned"] if a not in off]
            common.update(
                total=total,
                fallback=fallback,
                self_corrected=bool(cached.self_corrected),
                reranked=0,
                apply_relevance=bool(cached.apply_relevance),
                rank_tsquery=rank_tsquery,
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
        # Progress only: per-question numbers of held-out runs must not be read.
        print(f"[{number:3d}/{len(questions)}] {common['seconds']}s"
              f"{' error' if common.get('error') else ''}", file=sys.stderr, flush=True)
    print(f"wrote {out_dir}/tier_off.json and tier_on.json", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired uncertain-tier measurement")
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
    # The translation must not apply the tier; the second variant adds it.
    os.environ["UNCERTAIN_TIER"] = "false"
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
