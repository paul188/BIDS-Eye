#!/usr/bin/env python3
"""Go over the stored translations (services/translation_store.py).

Run in the backend image, where the database is:

    python review_translations.py list [--status unreviewed|needs_review|verified|rejected] [--commit HASH]
    python review_translations.py commits
    python review_translations.py show KEY
    python review_translations.py approve KEY --by NAME [--sql-file corrected.sql]
    python review_translations.py reject KEY --by NAME
    python review_translations.py import FILE [FILE ...] [--skip-score SCORE.json]

`import` stores translations made elsewhere as unreviewed entries: FILE is a
capture file (TRANSLATION_CAPTURE_FILE) or a recording of
benchmarks/demand/record_run.py. Each translation is run through the current
compiler and rewriter and counted on the database; stored are those that run
and find at least one dataset. No model is asked. --skip-score leaves out the
searches a score of benchmarks/easy/score_easy.py found wrong.

`commits` counts the entries per git commit of the code that made them, and
`list --commit` shows those of one commit: what to look at again after a fix.

KEY is the start of an entry's key as `list` prints it. `list` shows the most
used entries first. `approve --sql-file` replaces the SQL with a corrected one,
written as the SQL model writes it (with {{COLS}} as the column list). The
running backend picks a decision up within five minutes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


_BAD_VERDICTS = ("MISSES", "WRONG", "EMPTY", "ERROR", "FALLBACK")


async def _import(args: argparse.Namespace, engine, ts) -> int:
    """Check every translation of the files on the database and store the good ones."""
    import asyncio as aio
    import os
    import preprocess
    import routers.query as router
    from sqlalchemy import text
    from db.db import async_session_maker
    from services import schema_capabilities, text_search, word_form_store
    from services import text_to_sql as t2s

    # What the backend sets up at startup and the rewriter reads.
    try:
        import institution_index
        await institution_index.refresh_from_db(engine)
    except ImportError:
        pass
    loop = aio.get_running_loop()
    if schema_capabilities.has_fts():
        text_search.set_frequency_lookup(text_search.make_frequency_lookup(engine, loop))
    if schema_capabilities.has_word_forms():
        word_form_store.set_lookup(word_form_store.make_lookup(engine, loop))

    skip = set()
    if args.skip_score:
        score = json.loads(Path(args.skip_score).read_text())
        skip = {ts.normalize_question(d["question"]) for d in score.values()
                if d.get("verdict") in _BAD_VERDICTS}
    translations = []
    for path in args.files:
        translations.extend(ts.read_translations(path))
    skipped = [t for t in translations if ts.normalize_question(t["question"]) in skip]
    translations = [t for t in translations if ts.normalize_question(t["question"]) not in skip]

    def no_model(*a, **k):
        raise LookupError("the current code needs the SQL model for this plan")

    # The stored plan answers stage 1 and 2; nothing below may ask a model.
    os.environ["TRANSLATION_STORE"] = "true"
    os.environ.pop("TRANSLATION_CAPTURE_FILE", None)
    t2s._gemini_sql_generation = no_model

    async def validate(question, plan, model_sql):
        async with engine.connect() as connection:
            ts.set_data_stamp(await ts._read_data_stamp(connection))
        entry = ts.candidate(question, preprocess.AugmentedQueryPlan.model_validate(plan), model_sql)
        ts.store.replace_all([entry])
        result = await aio.to_thread(t2s._run_local, question, "import-no-model-call")
        if not result.from_store:
            raise LookupError("not answered from the stored translation")
        async with async_session_maker() as session:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            try:
                total = await router._guarded_count(session, result.sql, result.params)
            finally:
                await session.rollback()
        return result.sql, total

    async with engine.begin() as connection:
        await ts.ensure_table(connection)
    async with engine.begin() as connection:
        outcome = await ts.import_translations(connection, translations, validate)
    ts.store.clear()

    for name in ("failed", "empty", "kept_reviewed"):
        for line in outcome[name]:
            print(f"{name}: {line}")
    for item in skipped:
        print(f"skipped (score): {item['question']}")
    print(f"stored {len(outcome['stored'])}, empty {len(outcome['empty'])}, "
          f"failed {len(outcome['failed'])}, already reviewed {len(outcome['kept_reviewed'])}, "
          f"skipped by score {len(skipped)}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    import main  # noqa: F401 — sets up the import paths as the app does
    from db.db import engine
    from services import schema_capabilities, translation_store as ts

    # Outside production the engine logs every statement, which buries the output.
    engine.sync_engine.echo = False
    # Approving ties an entry to the schema as the running backend sees it.
    await schema_capabilities.refresh_available_columns(engine)
    if args.command == "import":
        return await _import(args, engine, ts)
    async with engine.begin() as connection:
        if args.command == "list":
            rows = await ts.list_for_review(connection, args.status, args.commit)
            for row in rows:
                reason = f"  [{row.review_reason}]" if row.review_reason else ""
                print(f"{row.question_key[:10]}  {row.status:<12} {(row.code_commit or ts.PRE_TRACKING)[:12]:<12} "
                      f"hits={row.hits:<4} found={row.result_total}  {row.question_norm}{reason}")
            print(f"{len(rows)} entries", file=sys.stderr)
        elif args.command == "commits":
            print(f"running code: {ts.code_commit()}")
            for row in await ts.count_by_commit(connection):
                print(f"{row.code_commit:<20} {row.status:<12} {row.entries:>5}  "
                      f"{row.first:%Y-%m-%d} to {row.last:%Y-%m-%d}")
        elif args.command == "show":
            row = await ts.find(connection, args.key)
            print(f"question:  {row.question_norm}\nstatus:    {row.status}"
                  f"{' (' + row.review_reason + ')' if row.review_reason else ''}\n"
                  f"stored:    {row.created_at:%Y-%m-%d %H:%M} by commit {row.code_commit or ts.PRE_TRACKING}   hits: {row.hits}   "
                  f"datasets found: {row.result_total}\n"
                  f"reviewed:  {row.reviewed_by or '-'} {row.reviewed_at or ''}\n")
            print("plan summary:", row.plan.get("plan", {}).get("natural_language_summary"))
            print("\naugmented question:\n" + str(row.plan.get("augmented_question")))
            print("\nSQL of the model:\n" + (row.model_sql or "(none: the plan was compiled)"))
            print("\nSQL the search ran:\n" + (row.final_sql or "(not recorded)"))
            if args.plan:
                print("\nplan:\n" + json.dumps(row.plan, indent=1))
        else:
            sql = Path(args.sql_file).read_text().strip() if getattr(args, "sql_file", None) else None
            status = ts.VERIFIED if args.command == "approve" else ts.REJECTED
            row = await ts.review(connection, args.key, status, args.by, model_sql=sql)
            print(f"{status}: {row.question_norm}")
    return 0


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--status")
    listing.add_argument("--commit", help="start of a commit hash, or 'pre-tracking'")
    commands.add_parser("commits")
    show = commands.add_parser("show")
    show.add_argument("key")
    show.add_argument("--plan", action="store_true", help="also print the whole plan")
    approve = commands.add_parser("approve")
    approve.add_argument("key")
    approve.add_argument("--by", required=True)
    approve.add_argument("--sql-file")
    reject = commands.add_parser("reject")
    reject.add_argument("key")
    reject.add_argument("--by", required=True)
    importing = commands.add_parser("import")
    importing.add_argument("files", nargs="+")
    importing.add_argument("--skip-score")
    try:
        sys.exit(asyncio.run(_run(parser.parse_args())))
    except LookupError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main_cli()
