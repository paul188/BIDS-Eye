#!/usr/bin/env python3
"""Check that the translation store changes no result and saves the model calls.

Each question is translated twice with the store on: first new (the entry is
kept), then again. The second translation must come from the store, ask no
model, and equal the first in everything the search runs on (SQL, parameters,
ranking input). Equal SQL on the same database is the same result, so no
search is executed and nothing is written to the database.

Runs where the backend code and its database are (the backend image):

    python store_gate.py --backend-dir /app/backend --out gate.json \
        [--recording recording.json] [--queries demand_first_run.yaml] [--prefix qf,qx,qp]

With --recording (from record_run.py) the first translation answers from the
recording and the whole check costs no model call. Without it the first
translation asks the models, and the token counts are in the output.
Exit code 1 when any question differs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recall"))

COMPARED = ("sql", "params", "explanation", "scored_filters", "apply_relevance",
            "rank_tsquery", "filter_sql")


async def _check(questions: list, recording: dict | None, out: Path) -> int:
    from db.db import engine
    import llm_client
    import preprocess
    from services.fulltext import text_search
    from services.stores import translation_store
    from services.catalogue import word_form_store
    from services.catalogue import schema_capabilities
    from services.translate import text_to_sql as t2s

    # The read-only warm-up of app startup (see eval_recall._run).
    await schema_capabilities.refresh_available_columns(engine)
    try:
        import institution_index
        await institution_index.refresh_from_db(engine)
    except ImportError:
        pass
    loop = asyncio.get_running_loop()
    if schema_capabilities.has_fts():
        text_search.set_frequency_lookup(text_search.make_frequency_lookup(engine, loop))
    if schema_capabilities.has_word_forms():
        word_form_store.set_lookup(word_form_store.make_lookup(engine, loop))
    async with engine.connect() as connection:
        translation_store.set_data_stamp(await translation_store._read_data_stamp(connection))

    calls: list = []
    current: dict = {}
    if recording is not None:
        def replay_pipeline(query, *a, **k):
            calls.append("intent")
            saved = {k: v for k, v in recording[query].items() if k not in ("model_sql", "corrected_sql", "usage", "code_commit")}
            current["question"] = query
            return preprocess.AugmentedQueryPlan.model_validate(saved)

        def replay_generate(*a, **k):
            calls.append("sql")
            sql = recording[current["question"]].get("model_sql")
            if not sql:
                raise LookupError("not recorded: this plan was compiled when it was recorded")
            return sql

        preprocess.run_pipeline = replay_pipeline
        t2s._gemini_sql_generation = replay_generate
    else:
        run_pipeline, generate = preprocess.run_pipeline, t2s._gemini_sql_generation

        def counting_pipeline(*a, **k):
            calls.append("intent")
            return run_pipeline(*a, **k)

        def counting_generate(*a, **k):
            calls.append("sql")
            return generate(*a, **k)

        preprocess.run_pipeline = counting_pipeline
        t2s._gemini_sql_generation = counting_generate

    results, failed = [], 0
    for number, q in enumerate(questions, 1):
        record: dict = {"id": q["id"], "question": q["question"]}
        try:
            calls.clear()
            usage = llm_client.start_usage_collection()
            first = await t2s.text_to_sql(q["question"])
            record["first_calls"] = list(calls)
            record["first_usage"] = list(usage)
            if first.store_entry is None:
                record["status"] = "not stored"   # fallback, or a fallback model answered
            else:
                translation_store.store.put(first.store_entry)
                calls.clear()
                second = await t2s.text_to_sql(q["question"])
                differing = [k for k in COMPARED if getattr(first, k) != getattr(second, k)]
                record["second_calls"] = list(calls)
                record["differing"] = differing
                ok = second.from_store and not calls and not differing
                record["status"] = "same" if ok else "DIFFERENT"
                failed += not ok
        except Exception as exc:  # noqa: BLE001
            record["status"] = f"error: {exc}"
        results.append(record)
        print(f"[{number}/{len(questions)}] {q['id']}: {record['status']}", file=sys.stderr, flush=True)
        out.write_text(json.dumps(results, indent=1))

    counts: dict = {}
    for record in results:
        key = record["status"].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts))
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--queries", type=Path, default=HERE / "demand_first_run.yaml")
    parser.add_argument("--prefix", default="qf,qx,qp")
    args = parser.parse_args()

    os.environ["TRANSLATION_STORE"] = "true"
    recording = json.loads(args.recording.read_text()) if args.recording else None
    if recording is not None:
        os.environ.setdefault("GEMINI_API_KEY", "replay-no-model-call")
    sys.path.insert(0, args.backend_dir)
    from services_layout import use_either_layout  # older backends: flat services/
    use_either_layout()
    import main as _app  # noqa: F401 — sets up the import paths as the app does
    from record_run import load_questions

    questions = load_questions(args.queries, args.prefix)
    if recording is not None:
        questions = [q for q in questions if q["question"] in recording]
    sys.exit(asyncio.run(_check(questions, recording, args.out)))


if __name__ == "__main__":
    main()
