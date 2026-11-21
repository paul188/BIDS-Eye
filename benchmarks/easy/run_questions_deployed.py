"""Translate four questions with the deployed search code and list what each returns.
Read-only: no log row, no cache row; every database statement runs in a read-only transaction."""
import asyncio, json, sys, time
sys.path.insert(0, "/app/backend")
QUESTIONS = [
    "Datasets with at least 40 Parkinson's patients",
    "resting-state fMRI with adults with ADHD",
    "List all resting-state fMRI datasets",
    "Find datasets with both structural and functional MRI",
]

async def run():
    import main  # noqa: F401 — import paths as the app sets them
    from sqlalchemy import text
    from db.db import engine
    try:
        from services.translate.text_to_sql import text_to_sql
        from services.catalogue.schema_capabilities import refresh_available_columns
    except ImportError:  # backend before services/ was grouped (piped in: no helper)
        from services.text_to_sql import text_to_sql
        from services.schema_capabilities import refresh_available_columns
    engine.sync_engine.echo = False
    await refresh_available_columns(engine)
    try:
        import institution_index
        await institution_index.refresh_from_db(engine)
    except Exception:
        pass
    try:
        try:
            from services.fulltext import word_frequencies
            from services.catalogue.schema_capabilities import has_fts
        except ImportError:  # backend before text_search.py was split
            try:
                from services.fulltext import text_search as word_frequencies
                from services.catalogue.schema_capabilities import has_fts
            except ImportError:  # backend before services/ was grouped
                from services import text_search as word_frequencies
                from services.schema_capabilities import has_fts
        if has_fts():
            word_frequencies.set_frequency_lookup(
                word_frequencies.make_frequency_lookup(engine, asyncio.get_running_loop()))
    except Exception:
        pass
    for question in QUESTIONS:
        started = time.time()
        record = {"question": question}
        try:
            t = await text_to_sql(question)
            record.update(sql=t.sql, params=t.params, explanation=t.explanation,
                          self_corrected=t.self_corrected)
            columns = "accession_id" + (", uncertain_level" if "uncertain_level" in t.sql else "")
            async with engine.connect() as connection:
                await connection.execute(text("SET TRANSACTION READ ONLY"))
                await connection.execute(text("SET LOCAL statement_timeout = '300s'"))
                wrapped = f"SELECT {columns} FROM ({t.sql}) q"
                if t.params:
                    rows = (await connection.execute(text(wrapped), t.params)).all()
                else:
                    rows = (await connection.exec_driver_sql(wrapped)).all()
                await connection.rollback()
            record["returned"] = [[r[0], (r[1] if len(r) > 1 else None)] for r in rows]
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        record["seconds"] = round(time.time() - started, 1)
        print("RESULT " + json.dumps(record), flush=True)

asyncio.run(run())
