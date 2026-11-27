#!/usr/bin/env python3
"""
benchmarks/word_forms/questions/fetch_evidence.py
-------------------------------------------------
Freeze the catalogue's word forms for every word run.py can look up, so the
benchmark gives the same numbers until the fixture is fetched again.

It asks for the words of every tokenizer in run.py, with the search's own
lookup (services/fulltext/word_form_counts.py _LOOKUP_SQL), in a read-only
transaction. Nothing is written to the database.

    DATABASE_URL=postgresql+asyncpg://user:pw@host/db \\
        python benchmarks/word_forms/questions/fetch_evidence.py [--out evidence.json]

On the server, with the backend image (its code is the deployed one):
    docker compose -f docker-compose-prod.yml run --rm --no-deps -T \\
        -v /tmp/wfq:/wfq backend python /wfq/fetch_evidence.py --out /wfq/evidence.json
(copy questions.yaml, run.py and this file to /tmp/wfq first; DATABASE_URL
comes from the container's environment).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run  # noqa: E402  (sets up the backend paths)

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from services.fulltext import word_form_counts  # noqa: E402


async def fetch(url: str, words: list) -> tuple:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            rows = (await conn.execute(text(word_form_counts._LOOKUP_SQL),
                                       {"words": words, "limit": word_form_counts.FORMS_PER_WORD})).all()
            datasets = (await conn.execute(text(
                f"SELECT count(*) FROM {word_form_counts.SOURCE_TABLE}"))).scalar_one()
            await conn.rollback()
    finally:
        await engine.dispose()
    return rows, datasets


def psql_script(words: list) -> str:
    """The lookup as one psql statement printing JSON, for a database reached
    only through psql (docker exec): the words are plain a-z, so they can be
    written into the SQL as literals."""
    assert all(word_form_counts._WORD_RE.fullmatch(w) for w in words)
    array = "ARRAY[" + ",".join(f"'{w}'" for w in words) + "]::text[]"
    lookup = (word_form_counts._LOOKUP_SQL.replace("CAST(:words AS text[])", array)
              .replace(":limit", str(word_form_counts.FORMS_PER_WORD)))
    return ("\\pset tuples_only on\n\\pset format unaligned\n"
            f"SELECT json_build_object('datasets', (SELECT count(*) FROM {word_form_counts.SOURCE_TABLE}),"
            f" 'rows', (SELECT json_agg(json_build_array(t.word, t.stem, t.stem_docs, t.forms)) FROM ({lookup}) t));\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=run.QUESTIONS)
    ap.add_argument("--out", type=Path, default=_HERE / "evidence.json")
    ap.add_argument("--print-sql", action="store_true",
                    help="print the lookup as a psql script instead of running it")
    ap.add_argument("--from-psql", type=Path,
                    help="build the fixture from the output of the --print-sql script")
    args = ap.parse_args()
    items = yaml.safe_load(args.questions.read_text())["questions"]
    words = sorted({w for item in items for tokenize in run.TOKENIZERS.values()
                    for w in tokenize(item["question"]) if word_form_counts._WORD_RE.fullmatch(w)})
    if args.print_sql:
        sys.stdout.write(psql_script(words))
        return
    if args.from_psql:
        data = json.loads(args.from_psql.read_text().strip())
        rows, datasets = data["rows"] or [], data["datasets"]
    else:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise SystemExit("Set DATABASE_URL (postgresql+asyncpg://...), or use --print-sql / --from-psql")
        rows, datasets = asyncio.run(fetch(url, words))
    found = {}
    for word, stem, stem_docs, forms in rows:
        found[word] = {"stem": stem, "stem_docs": int(stem_docs or 0),
                       "forms": [list(f) for f in word_form_counts._parse_forms(forms)]}
    args.out.write_text(json.dumps({
        "fetched": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "datasets": int(datasets), "forms_per_word": word_form_counts.FORMS_PER_WORD,
        "asked": len(words), "words": found}, indent=1, sort_keys=False))
    print(f"{len(found)} of {len(words)} words have a stem; {datasets} datasets -> {args.out}")


if __name__ == "__main__":
    main()
