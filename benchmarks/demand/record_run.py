#!/usr/bin/env python3
"""Run the real requests through the search and record what the models said.

A normal benchmark run (../recall/eval_recall.py run), plus a second file with
the model output of every question: the QueryPlan with its resolved codes and
augmented question, the SQL the SQL model wrote (absent when the plan was
compiled without it), and its repair when the database rejected that SQL. replay_run.py repeats the run from that file without
any model call, so a change to the compiler, the rewriter or the uncertain
tier is measured for free. A change to a prompt needs a new recording.

Runs where the backend code and its database are (the backend image):

    python record_run.py --backend-dir /app/backend --out run.json --recording recording.json \
        [--queries demand_first_run.yaml] [--prefix qf,qx,qp] [--label "…"]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recall"))


def code_commit(root: Path) -> str:
    """The commit of the backend under test: the image's (see backend/Dockerfile), else git."""
    named = os.getenv("BIDSEYE_COMMIT", "").strip()
    if named:
        return named
    deployed = root / "DEPLOYED_COMMIT"
    if deployed.is_file() and deployed.read_text().split():
        return deployed.read_text().split()[0]
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                         capture_output=True, text=True)
    return git.stdout.strip() or "unknown"


def load_questions(path: Path, prefixes: str) -> list:
    import yaml
    wanted = tuple(p for p in prefixes.split(",") if p)
    return [q for q in yaml.safe_load(path.read_text())["queries"]
            if not wanted or q["id"].startswith(wanted)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend-dir", default="/app/backend")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=HERE / "demand_first_run.yaml")
    parser.add_argument("--prefix", default="qf,qx,qp", help="id prefixes to run; empty for all")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    sys.path.insert(0, args.backend_dir)
    from services_layout import use_either_layout  # older backends: flat services/
    use_either_layout()
    import main as _app  # noqa: F401 — sets up the import paths as the app does
    import eval_recall
    import preprocess
    from services.translate import text_to_sql as t2s

    recording: dict = {}
    current: dict = {}
    # The names before llm_client's profiles (2026-11-27) carried "gemini".
    generate_name = "_generate_sql" if hasattr(t2s, "_generate_sql") else "_gemini_sql_generation"
    run_pipeline, generate = preprocess.run_pipeline, getattr(t2s, generate_name)

    def save() -> None:
        args.recording.write_text(json.dumps(recording, indent=1))

    def recording_pipeline(query, *a, **k):
        import llm_client
        current["question"] = query
        # Token counts of this question's model calls; the SQL call is added
        # to the same list later (same thread, see llm_client).
        usage = llm_client.start_usage_collection()
        result = run_pipeline(query, *a, **k)
        recording[query] = result.model_dump(mode="json", exclude={"topics"})
        recording[query]["usage"] = usage
        recording[query]["code_commit"] = code_commit(Path(args.backend_dir).parent)
        save()
        return result

    def recording_generate(*a, **k):
        sql = generate(*a, **k)
        if current.get("question") in recording:
            recording[current["question"]]["model_sql"] = sql
            save()
        return sql

    # The search runner asks a model to repair SQL the database rejects; keep the repair.
    try:
        from services import search_runner as runner
    except ImportError:
        # Search code before the router split: these lived in routers/query.py.
        import routers.query as runner
    correct_name = "_correct_sql" if hasattr(runner, "_correct_sql") else "_correct_sql_with_gemini"
    correct = getattr(runner, correct_name)

    def recording_correct(sql, *a, **k):
        fixed = correct(sql, *a, **k)
        if current.get("question") in recording:
            recording[current["question"]]["corrected_sql"] = fixed
            save()
        return fixed

    setattr(runner, correct_name, recording_correct)
    setattr(t2s, correct_name, recording_correct)
    preprocess.run_pipeline = recording_pipeline
    setattr(t2s, generate_name, recording_generate)
    asyncio.run(eval_recall._run(load_questions(args.queries, args.prefix), args.out,
                                 args.backend_dir, args.label))


if __name__ == "__main__":
    main()
