#!/usr/bin/env python3
"""Generate and compare the baseline and working-tree uncertainty-tier SQL.

By default this uses ten real SQL translations captured in the 2026-11-13
recall run, applies both tier implementations, and checks each result with the
current static query guard. No database connection is opened in this mode.

Add --execute-live to run only guard-accepted SQL against the configured
database in read-only transactions. A tier rejected by the static guard is
skipped and recorded in the report. SQL, result ids and timings are written
under --out-dir for review.

Use --ssh-target to run inside the deployed backend container when PostgreSQL
is private to the production Docker network; this also applies a read-only
transaction, statement timeout and EXPLAIN row-estimate check.

Example:
    python benchmarks/recall/compare_uncertain_tier.py --execute-live \
        --out-dir /tmp/bidseye-tier-compare
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
DEFAULT_FIXTURE = ROOT / "benchmarks/recall/results/2026-11-13_167q_C_main_ranked_bad24b1.json"
CASES = (
    "u03_memory",                         # text plus broad task alternatives
    "b09_motor_eeg_stroke",               # modality, diagnosis and task
    "l12_healthy_only_wm_fmri",           # cohort plus task constraints
    "d15_reward_learning_value",           # multi-clause topic search
    "g70_multishell_diffusion_repeats",    # structured acquisition constraints
    "g78_focused_ultrasound",             # stimulation and imaging
    "p02_reward",                          # short topic query
    "g08_auditory_hallucinations_psychosis", # diagnosis and text evidence
    "u10_diffusion_children",              # age plus modality
    "g75_nonhuman_primate",                # species plus modality
)


def _load_baseline(module_name: str, temp_dir: Path) -> ModuleType:
    """Load the committed tier implementation without replacing the working file."""
    source = subprocess.run(
        ["git", "show", "HEAD:backend/services/fulltext/uncertain_tier.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    path = temp_dir / "uncertain_tier_baseline.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load baseline uncertain_tier.py from HEAD")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses resolves the defining module here
    spec.loader.exec_module(module)
    return module


def _read_fixture(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"fixture must contain a results array: {path}")
    by_id = {row.get("id"): row.get("sql") for row in rows if row.get("id") and row.get("sql")}
    missing = [qid for qid in CASES if qid not in by_id]
    if missing:
        raise ValueError(f"fixture is missing selected query SQL: {', '.join(missing)}")
    return {qid: by_id[qid] for qid in CASES}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ssh_python(target: str, code: str, timeout: float = 1800) -> dict[str, Any]:
    """Run a Python snippet inside the deployed backend container over SSH."""
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", target,
         "docker exec -i bids_backend python -"],
        input=code, text=True, capture_output=True, timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"remote backend command failed ({completed.returncode}): "
            f"{completed.stderr[-1500:]}"
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("remote backend returned no JSON result")


def _remote_capabilities(target: str) -> dict[str, Any]:
    code = r'''import asyncio, json, sys
sys.path[:0] = ["/app/backend", "/app/BIDS-SQL", "/app/RAG"]
from db.db import engine
try:
    from services.catalogue import schema_capabilities
    from services.fulltext import vectors
except ImportError:  # backend before text_search.py was split
    try:
        from services.catalogue import schema_capabilities
        from services.fulltext import text_search as vectors
    except ImportError:  # backend before services/ was grouped
        from services import schema_capabilities, text_search as vectors
async def main():
    await schema_capabilities.refresh_available_columns(engine)
    result = {
        "papers": vectors.papers_enabled(),
        "paper_fts": schema_capabilities.has_paper_fts(),
        "exact_fts": schema_capabilities.has_exact_fts(),
        "exact_paper_fts": schema_capabilities.has_exact_paper_fts(),
        "word_forms": schema_capabilities.has_word_forms(),
    }
    print(json.dumps(result))
    await engine.dispose()
asyncio.run(main())
'''
    return _ssh_python(target, code)


def _run_remote_live(cases: dict[str, dict[str, Any]], params: dict[str, Any],
                     target: str, timeout: float = 1800) -> None:
    """Plan-check and execute SQL inside the live backend container, read-only."""
    from services.translate.sql_rewriter import build_count_sql

    payload = []
    for qid, case in cases.items():
        for version in ("baseline", "working_tree"):
            item = case[version]
            if item["static_guard_reason"]:
                item["live"] = {
                    "status": "skipped_static_guard",
                    "reason": item["static_guard_reason"],
                }
                continue
            sql = item["sql"].rstrip().rstrip(";")
            payload.append({
                "id": qid, "version": version, "sql": sql,
                "count_sql": build_count_sql(sql),
                "params": params,
            })
    encoded_payload = repr(json.dumps(payload, ensure_ascii=False))
    code = r'''import asyncio, json, os, sys, time
sys.path[:0] = ["/app/backend", "/app/BIDS-SQL", "/app/RAG"]
from sqlalchemy import text
from db.db import async_session_maker, engine
items = json.loads(__PAYLOAD__)
def largest(plan):
    if isinstance(plan, str): plan = json.loads(plan)
    root = plan[0]["Plan"] if isinstance(plan, list) else plan["Plan"]
    best = (0.0, "")
    stack = [root]
    while stack:
        node = stack.pop()
        rows = float(node.get("Plan Rows") or 0)
        if rows > best[0]: best = (rows, node.get("Node Type", ""))
        stack.extend(node.get("Plans") or [])
    return best
async def main():
    results = {}
    configured = os.getenv("SEARCH_MAX_PLAN_ROWS")
    async with async_session_maker() as session:
        table_est = (await session.execute(text(
            "SELECT max(reltuples) FROM pg_class WHERE relname IN "
            "('bids_objects', 'bids_participants', 'bids_datasets')"
        ))).scalar()
    max_rows = float(configured) if configured else max(float(table_est or 0) * 3.0, 1.0e6)
    for item in items:
        key = item["id"] + "|" + item["version"]
        try:
            async with async_session_maker() as session:
                await session.execute(text("SET TRANSACTION READ ONLY"))
                await session.execute(text("SET LOCAL statement_timeout = '120s'"))
                await session.execute(text("SET LOCAL work_mem = '32MB'"))
                plan = (await session.execute(
                    text("EXPLAIN (FORMAT JSON) " + item["count_sql"]), item["params"]
                )).scalar_one()
                estimated_rows, node_type = largest(plan)
                if estimated_rows > max_rows:
                    results[key] = {
                        "status": "refused_plan", "estimated_rows": estimated_rows,
                        "node_type": node_type, "max_plan_rows": max_rows,
                    }
                    await session.rollback()
                    continue
                query = (
                    "SELECT q.id, q.accession_id, q.uncertain_level, q.uncertain_reasons "
                    "FROM (" + item["sql"] + ") AS q"
                )
                started = time.perf_counter()
                rows = (await session.execute(text(query), item["params"])).mappings().all()
                elapsed = time.perf_counter() - started
                ids = [row["accession_id"] or str(row["id"]) for row in rows]
                levels = {
                    row["accession_id"] or str(row["id"]): int(row["uncertain_level"] or 0)
                    for row in rows
                }
                results[key] = {
                    "status": "executed", "query_seconds": elapsed,
                    "row_count": len(ids), "ids": ids, "levels": levels,
                    "estimated_rows": estimated_rows, "node_type": node_type,
                    "max_plan_rows": max_rows,
                }
                await session.rollback()
        except Exception as exc:
            results[key] = {
                "status": "error", "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
    await engine.dispose()
    print(json.dumps(results))
asyncio.run(main())
'''.replace("__PAYLOAD__", encoded_payload)
    try:
        remote_results = _ssh_python(target, code, timeout=timeout)
    except Exception as exc:
        failure = {"status": "remote_error", "error_type": type(exc).__name__,
                   "error": str(exc)[:1500]}
        for case in cases.values():
            for version in ("baseline", "working_tree"):
                case[version]["live"] = dict(failure)
        return

    for qid, case in cases.items():
        for version in ("baseline", "working_tree"):
            result = remote_results.get(qid + "|" + version)
            if result is not None:
                case[version]["live"] = result
            elif "live" not in case[version]:
                case[version]["live"] = {"status": "missing_remote_result"}
    for case in cases.values():
        old_live = case["baseline"].get("live", {})
        new_live = case["working_tree"].get("live", {})
        if old_live.get("status") == new_live.get("status") == "executed":
            old_ids, new_ids = set(old_live["ids"]), set(new_live["ids"])
            case["comparison"] = {
                "same_result_set": old_ids == new_ids,
                "baseline_only": sorted(old_ids - new_ids),
                "working_tree_only": sorted(new_ids - old_ids),
                "baseline_seconds": old_live["query_seconds"],
                "working_tree_seconds": new_live["query_seconds"],
                "working_tree_over_baseline_ratio": (
                    new_live["query_seconds"] / old_live["query_seconds"]
                    if old_live["query_seconds"] else None
                ),
            }


async def _execute_live(sql: str, params: dict[str, Any], label: str,
                        allow_static_cross_join: bool = False) -> dict[str, Any]:
    """Run one tier query through the production guard and fetch all result ids."""
    # Import the app only in live mode; main.py wires the same import paths as
    # the backend service and the configured DB engine.
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    if str(ROOT / "BIDS-SQL") not in sys.path:
        sys.path.insert(0, str(ROOT / "BIDS-SQL"))
    if str(ROOT / "RAG") not in sys.path:
        sys.path.insert(0, str(ROOT / "RAG"))
    import main  # noqa: F401
    from db.db import async_session_maker, engine
    from sqlalchemy import text
    from services import query_guard
    from services.translate.sql_rewriter import build_count_sql

    outer = (
        "SELECT q.id, q.accession_id, q.uncertain_level, q.uncertain_reasons "
        f"FROM ({sql.rstrip().rstrip(';')}) AS q"
    )
    started = time.perf_counter()
    async with async_session_maker() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        await query_guard.apply_limits(session)
        if allow_static_cross_join:
            # Diagnostic-only escape hatch for the new tier's one-row terms
            # CTE. First require a successful, bounded EXPLAIN; the actual
            # statement still runs read-only with statement_timeout applied.
            plan_reason = await query_guard.check_plan(
                session, build_count_sql(sql), params)
            if plan_reason:
                raise query_guard.QueryRefusedError(plan_reason)
        else:
            await query_guard.vet(session, sql, build_count_sql(sql), params)
        guard_seconds = time.perf_counter() - started
        query_started = time.perf_counter()
        rows = (await session.execute(text(outer), params)).mappings().all()
        query_seconds = time.perf_counter() - query_started
        await session.rollback()
    total_seconds = time.perf_counter() - started
    ids = [row["accession_id"] or str(row["id"]) for row in rows]
    levels = {
        row["accession_id"] or str(row["id"]): int(row["uncertain_level"] or 0)
        for row in rows
    }
    return {
        "status": "executed", "query_seconds": query_seconds,
        "guard_and_plan_seconds": guard_seconds, "total_seconds": total_seconds,
        "row_count": len(ids), "ids": ids, "levels": levels,
        "database_label": label,
    }


async def _prepare_live_capabilities() -> bool:
    """Detect optional FTS columns before building SQL in live mode."""
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    if str(ROOT / "BIDS-SQL") not in sys.path:
        sys.path.insert(0, str(ROOT / "BIDS-SQL"))
    if str(ROOT / "RAG") not in sys.path:
        sys.path.insert(0, str(ROOT / "RAG"))
    import main  # noqa: F401
    from db.db import engine
    from services.catalogue import schema_capabilities
    from services.fulltext import vectors

    await schema_capabilities.refresh_available_columns(engine)
    papers = vectors.papers_enabled()
    # main_cli later starts a fresh asyncio loop for execution.
    await engine.dispose()
    return papers


async def _run_live(cases: dict[str, dict[str, Any]], params: dict[str, Any], label: str,
                    diagnostic_cross_join: bool) -> None:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    if str(ROOT / "BIDS-SQL") not in sys.path:
        sys.path.insert(0, str(ROOT / "BIDS-SQL"))
    if str(ROOT / "RAG") not in sys.path:
        sys.path.insert(0, str(ROOT / "RAG"))
    import main  # noqa: F401
    from db.db import async_session_maker, engine
    from sqlalchemy import text
    from services import query_guard
    from services.catalogue import schema_capabilities

    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        failure = {
            "status": "database_unavailable", "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        for case in cases.values():
            for version in ("baseline", "working_tree"):
                case[version]["live"] = dict(failure)
        print("live database unavailable; skipped SQL execution", flush=True)
        return

    await schema_capabilities.refresh_available_columns(engine)

    for qid, case in cases.items():
        for version in ("baseline", "working_tree"):
            item = case[version]
            bypass_cross_join = (
                diagnostic_cross_join and version == "working_tree"
                and item["static_guard_reason"] == "CROSS JOIN"
            )
            if item["static_guard_reason"] and not bypass_cross_join:
                item["live"] = {
                    "status": "skipped_static_guard",
                    "reason": item["static_guard_reason"],
                }
                continue
            try:
                item["live"] = await _execute_live(
                    item["sql"], params, label,
                    allow_static_cross_join=bypass_cross_join,
                )
                if bypass_cross_join:
                    item["live"]["static_guard_bypassed_for_diagnostic"] = True
            except query_guard.QueryRefusedError as exc:
                item["live"] = {"status": "refused", "reason": str(exc)}
            except Exception as exc:  # retain per-case diagnostics; keep going
                item["live"] = {
                    "status": "error", "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            print(f"{qid:38} {version:12} {item['live']['status']}", flush=True)


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                        help="saved eval run JSON with results[].id/sql")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/tmp/bidseye_uncertain_tier_compare"))
    live_mode = parser.add_mutually_exclusive_group()
    live_mode.add_argument("--execute-live", action="store_true",
                           help="run guard-accepted old/new SQL against the local configured database")
    live_mode.add_argument("--ssh-target",
                           help="run through SSH inside the deployed bids_backend container")
    parser.add_argument("--ssh-timeout-seconds", type=float, default=1800,
                        help="overall SSH runner timeout (default: 1800 seconds)")
    parser.add_argument("--diagnostic-cross-join", action="store_true",
                        help="also EXPLAIN and run working-tree SQL rejected only for CROSS JOIN; read-only")
    parser.add_argument("--database-label", default="configured DATABASE_URL",
                        help="human-readable DB name to put in the report; credentials are not logged")
    parser.add_argument("--params-json", type=Path,
                        help="optional JSON object of SQL bind parameters (default: {})")
    parser.add_argument("--papers", choices=("auto", "on", "off"), default="auto",
                        help="paper text when generating tier SQL; auto follows the current environment")
    args = parser.parse_args()

    sys.path.insert(0, str(BACKEND))
    sys.path.insert(0, str(ROOT / "RAG"))
    from services import query_guard
    from services.catalogue import schema_capabilities
    from services.fulltext import word_frequencies
    from services.fulltext import uncertain_tier as working_tier

    # Keep SQL generation deterministic and offline in the default mode. The
    # live run still compares exactly these generated statements.
    word_frequencies.set_frequency_lookup(lambda _queries, *_guards: None)
    remote_capabilities = None
    remote_capability_error = None
    if args.ssh_target:
        try:
            remote_capabilities = _remote_capabilities(args.ssh_target)
            schema_capabilities.set_paper_fts(remote_capabilities["paper_fts"])
            schema_capabilities.set_exact_fts(
                remote_capabilities["exact_fts"],
                papers=remote_capabilities["exact_paper_fts"],
                word_forms=remote_capabilities["word_forms"],
            )
        except Exception as exc:
            remote_capability_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
    if args.papers != "auto" and args.ssh_target:
        papers = args.papers == "on" and bool(
            remote_capabilities and remote_capabilities.get("paper_fts"))
    elif args.papers != "auto":
        schema_capabilities.set_paper_fts(args.papers == "on")
        papers = args.papers == "on"
    elif args.ssh_target:
        papers = bool(remote_capabilities and remote_capabilities.get("papers"))
    elif args.papers == "auto":
        papers = asyncio.run(_prepare_live_capabilities()) if args.execute_live else None
    else:
        papers = None
    base_sql = _read_fixture(args.fixture)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sql_dir = args.out_dir / "sql"
    sql_dir.mkdir(exist_ok=True)

    cases: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="bidseye-tier-baseline-") as tmp:
        baseline_tier = _load_baseline("services._uncertain_tier_baseline", Path(tmp))
        for qid, source_sql in base_sql.items():
            baseline_sql = baseline_tier.apply(source_sql, papers=papers)
            working_sql = working_tier.apply(source_sql, papers=papers)
            case: dict[str, Any] = {
                "source_sql_sha256": _sha256(source_sql),
                "source_sql_chars": len(source_sql),
            }
            for version, sql in (("baseline", baseline_sql), ("working_tree", working_sql)):
                static_reason = query_guard.check_static(sql)
                sql_path = sql_dir / f"{qid}.{version}.sql"
                sql_path.write_text(sql.rstrip() + "\n")
                case[version] = {
                    "sql_file": str(sql_path), "sql_sha256": _sha256(sql),
                    "sql_chars": len(sql), "changed_by_tier": sql != source_sql,
                    "static_guard_reason": static_reason,
                }
            case["sql_identical"] = baseline_sql == working_sql
            case["sql_size_delta_chars"] = len(working_sql) - len(baseline_sql)
            case["sql_size_ratio"] = len(working_sql) / len(baseline_sql) if baseline_sql else None
            cases[qid] = case

    params = json.loads(args.params_json.read_text()) if args.params_json else {}
    if not isinstance(params, dict):
        parser.error("--params-json must contain a JSON object")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": str(args.fixture),
        "baseline": "git HEAD:backend/services/fulltext/uncertain_tier.py",
        "working_tree": "backend/services/fulltext/uncertain_tier.py",
        "papers_mode": args.papers,
        "remote_capabilities": remote_capabilities,
        "remote_capability_error": remote_capability_error,
        "live_execution_requested": args.execute_live or bool(args.ssh_target),
        "ssh_target": args.ssh_target,
        "diagnostic_cross_join_requested": args.diagnostic_cross_join,
        "cases": cases,
    }

    if args.execute_live:
        if str(BACKEND) not in sys.path:
            sys.path.insert(0, str(BACKEND))
        from services import query_guard
        asyncio.run(_run_live(cases, params, args.database_label, args.diagnostic_cross_join))
        for case in cases.values():
            old_live = case["baseline"].get("live", {})
            new_live = case["working_tree"].get("live", {})
            if old_live.get("status") == new_live.get("status") == "executed":
                old_ids, new_ids = set(old_live["ids"]), set(new_live["ids"])
                case["comparison"] = {
                    "same_result_set": old_ids == new_ids,
                    "baseline_only": sorted(old_ids - new_ids),
                    "working_tree_only": sorted(new_ids - old_ids),
                    "baseline_seconds": old_live["query_seconds"],
                    "working_tree_seconds": new_live["query_seconds"],
                    "working_tree_over_baseline_ratio": (
                        new_live["query_seconds"] / old_live["query_seconds"]
                        if old_live["query_seconds"] else None
                    ),
                }
    elif args.ssh_target:
        if remote_capability_error:
            failure = {"status": "remote_error", "error": remote_capability_error}
            for case in cases.values():
                for version in ("baseline", "working_tree"):
                    case[version]["live"] = dict(failure)
        else:
            _run_remote_live(cases, params, args.ssh_target, args.ssh_timeout_seconds)

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"fixture: {args.fixture}")
    print(f"output:  {args.out_dir}")
    print("query                                  baseline guard   working-tree guard   SQL same")
    for qid, case in cases.items():
        old = case["baseline"]["static_guard_reason"] or "ACCEPT"
        new = case["working_tree"]["static_guard_reason"] or "ACCEPT"
        status = ""
        if args.execute_live or args.ssh_target:
            old_live = case["baseline"].get("live", {})
            new_live = case["working_tree"].get("live", {})
            status = f" | {old_live.get('status')} / {new_live.get('status')}"
            if "comparison" in case:
                comparison = case["comparison"]
                status += (f" | same={comparison['same_result_set']}"
                           f" {comparison['baseline_seconds']:.3f}s"
                           f"/{comparison['working_tree_seconds']:.3f}s")
        print(f"{qid:38} {old[:14]:14} {new[:19]:19} {case['sql_identical']}{status}")
    print(f"report:  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
