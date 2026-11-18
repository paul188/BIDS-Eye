#!/usr/bin/env python3
"""Repeat a recorded run with the current search code and no model call.

Takes the recording of record_run.py. The two model stages answer from the
recording; everything after them (plan compiler, SQL rewriter, text search,
uncertain tier, ranking) is the code under --backend-dir, run against its
database read-only. The output has the format of a normal run and is scored
the same way:

    python replay_run.py --backend-dir /app/backend --recording recording.json --out replay.json
    python ../recall/eval_recall.py score replay.json \
        --truth gold/requests_truth.json --held-out-file <empty file>

A question fails with "not recorded" when the current code needs the SQL
model for a plan that was compiled when it was recorded.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recall"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=HERE / "demand_first_run.yaml")
    parser.add_argument("--prefix", default="qf,qx,qp")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    sys.path.insert(0, args.backend_dir)
    from services_layout import use_either_layout  # older backends: flat services/
    use_either_layout()
    import os
    os.environ.setdefault("GEMINI_API_KEY", "replay-no-model-call")
    import main as _app  # noqa: F401
    import eval_recall
    import preprocess
    from record_run import load_questions
    from services.translate import text_to_sql as t2s

    recording = json.loads(args.recording.read_text())
    current: dict = {}

    def replay_pipeline(query, *a, **k):
        if query not in recording:
            raise LookupError("not recorded: no plan for this question")
        current["question"] = query
        saved = {k: v for k, v in recording[query].items() if k not in ("model_sql", "corrected_sql", "usage", "code_commit")}
        return preprocess.AugmentedQueryPlan.model_validate(saved)

    def replay_generate(*a, **k):
        sql = recording.get(current.get("question"), {}).get("model_sql")
        if not sql:
            raise LookupError("not recorded: this plan was compiled when it was recorded")
        return sql

    preprocess.run_pipeline = replay_pipeline
    t2s._gemini_sql_generation = replay_generate
    # No model repairs SQL here either; the search runner holds its own reference.
    try:
        from services import search_runner as runner
    except ImportError:
        # Search code before the router split: these lived in routers/query.py.
        import routers.query as runner
    # The recorded repair, when there is one; otherwise the SQL stays as it is.
    t2s._correct_sql_with_gemini = runner._correct_sql_with_gemini = lambda sql, *a, **k: (
        recording.get(current.get("question"), {}).get("corrected_sql") or sql)
    questions = [q for q in load_questions(args.queries, args.prefix) if q["question"] in recording]
    asyncio.run(eval_recall._run(questions, args.out, args.backend_dir,
                                 args.label or f"replay of {args.recording.name}"))


if __name__ == "__main__":
    main()
