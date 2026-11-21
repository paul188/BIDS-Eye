"""
services/catalogue/word_form_store.py
-------------------------------------
The catalogue's word forms: which forms each stem merges, and in how many
datasets each form occurs.

bids_word_forms has one row per word form found in the datasets' own text
(search_tsv_exact; crawlers/migrate_add_exact_search.sql):

    form   the word as written, lower case      "ageing"
    stem   what the 'english' stemmer makes it  "age"
    ndoc   datasets that contain the form

refresh() rebuilds it; it runs after each crawl
(crawlers/openneuro-crawler/post_crawl.py) and from
scripts/refresh_word_forms.py. The table only has to be roughly current:
it decides whether a searched word is worth a closer look
(RAG/word_forms.py), not what a search returns.

lookup() answers for the words of one question. Like the word counts in
services/fulltext/text_search.py it is called from the translation worker
thread, so it reaches the database through worker_db.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

import worker_db
from services.catalogue import schema_capabilities

from word_forms import WordEvidence

log = logging.getLogger(__name__)

TABLE = "bids_word_forms"
SOURCE_COLUMN = "search_tsv_exact"

TABLES_SQL: Tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS bids_word_forms (
        form  TEXT PRIMARY KEY,
        stem  TEXT NOT NULL,
        ndoc  INTEGER NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_bids_word_forms_stem ON bids_word_forms (stem)",
)

# Alphabetic words only: numbers and codes ("t1w", "7t") have no word forms,
# and a form goes into tsquery literals as it is.
_FORM_PATTERN = "^[a-z]{2,40}$"
_STEM_SQL = "array_to_string(tsvector_to_array(to_tsvector('english', {})), ' ')"

_REFRESH_SQL = (
    "INSERT INTO bids_word_forms (form, stem, ndoc) "
    "SELECT s.word, l.stem, s.ndoc "
    f"FROM ts_stat('SELECT {SOURCE_COLUMN} FROM bids_datasets') AS s "
    f"CROSS JOIN LATERAL (SELECT {_STEM_SQL.format('s.word')} AS stem) AS l "
    # A stopword has no stem and can never be searched for.
    "WHERE s.word ~ :pattern AND l.stem ~ '^[a-z0-9]+$'"
)

# Forms returned per word: enough for every regular inflection of a stem.
FORMS_PER_WORD = 40
LOOKUP_TIMEOUT_SECONDS = 3.0

_LOOKUP_SQL = (
    "SELECT w.word, l.stem, "
    "       (SELECT count(*) FROM bids_datasets d "
    "         WHERE d.search_tsv @@ to_tsquery('english', w.word)) AS stem_docs, "
    "       ARRAY(SELECT f.form || ':' || f.ndoc FROM bids_word_forms f "
    "              WHERE f.stem = l.stem ORDER BY f.ndoc DESC, f.form LIMIT :limit) AS forms "
    "FROM unnest(CAST(:words AS text[])) AS w(word) "
    f"CROSS JOIN LATERAL (SELECT {_STEM_SQL.format('w.word')} AS stem) AS l "
    "WHERE l.stem <> ''"
)

_WORD_RE = re.compile(r"[a-z]{2,40}")


# ── Building the table ────────────────────────────────────────────────────────

async def ensure_tables(conn: Any) -> None:
    """Create the table if it is missing. Caller commits."""
    for statement in TABLES_SQL:
        await conn.execute(text(statement))


async def refresh(conn: Any) -> Dict[str, int]:
    """Rebuild bids_word_forms from the datasets' text. Caller commits.

    Raises when search_tsv_exact is missing: run
    crawlers/migrate_add_exact_search.sql first.

    The DELETE and the INSERT must share one transaction, or searches see an
    empty table until the INSERT ends and quietly fall back to stem matching.
    Raises on a connection that commits each statement (AUTOCOMMIT). SQLAlchemy
    cannot tell: in_transaction() is True inside begin() even then, so the
    database is asked. Two calls of pg_current_xact_id() return the same id
    only inside one transaction.
    """
    first, second = [(await conn.execute(text("SELECT pg_current_xact_id()"))).scalar_one()
                     for _ in range(2)]
    if first != second:
        raise RuntimeError("word_form_store.refresh() needs a transaction; "
                           "the connection commits each statement (AUTOCOMMIT?)")
    present = (await conn.execute(text(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = 'bids_datasets' "
        "AND column_name = :column"), {"column": SOURCE_COLUMN})).scalar_one()
    if not present:
        raise RuntimeError(f"bids_datasets.{SOURCE_COLUMN} is missing: "
                           "run crawlers/migrate_add_exact_search.sql first")
    await ensure_tables(conn)
    await conn.execute(text("DELETE FROM bids_word_forms"))
    await conn.execute(text(_REFRESH_SQL), {"pattern": _FORM_PATTERN})
    row = (await conn.execute(text(
        "SELECT count(*), count(DISTINCT stem) FROM bids_word_forms"))).one()
    return {"forms": int(row[0]), "stems": int(row[1])}


# ── Looking words up ──────────────────────────────────────────────────────────

# Given words, returns the rows of _LOOKUP_SQL, or None when unavailable.
Lookup = Callable[[List[str]], Optional[List[Sequence[Any]]]]


def _database_lookup(words: List[str]) -> List[Sequence[Any]]:
    """The rows for *words*, from the database (worker_db.current())."""
    params = {"words": words, "limit": FORMS_PER_WORD}

    async def rows(connection: Any) -> List[Sequence[Any]]:
        return list(await connection.execute(text(_LOOKUP_SQL), params))

    return worker_db.current().run(rows, LOOKUP_TIMEOUT_SECONDS)


_lookup: Optional[Lookup] = _database_lookup


def set_lookup(lookup: Optional[Lookup]) -> None:
    """Install another lookup (tests); None switches lookups off."""
    global _lookup
    _lookup = lookup


def reset_lookup() -> None:
    """Back to asking the database."""
    set_lookup(_database_lookup)


def _parse_forms(values: Sequence[str]) -> Tuple[Tuple[str, int], ...]:
    forms = []
    for value in values or ():
        form, _, count = str(value).rpartition(":")
        if form and count.isdigit():
            forms.append((form, int(count)))
    return tuple(forms)


def lookup(words: Sequence[str]) -> Dict[str, WordEvidence]:
    """The catalogue's forms for each of *words*; {} when unavailable.

    Words that are not plain lower-case words, and stopwords, are left out.
    A lookup raises worker_db.WorkerDbError when the database cannot answer;
    that gives {} and the words are searched by stem. Anything else it
    raises is a bug.
    """
    asked = [w for w in dict.fromkeys(words) if _WORD_RE.fullmatch(w)]
    if not asked or not schema_capabilities.has_word_forms():
        return {}
    if _lookup is None:
        return {}
    try:
        rows = _lookup(asked)
    except worker_db.WorkerDbError as exc:
        log.warning("Word form lookup failed (%s) — searching by stem", exc)
        return {}
    found: Dict[str, WordEvidence] = {}
    for row in rows or ():
        word, stem, stem_docs, forms = row[0], row[1], row[2], row[3]
        found[word] = WordEvidence(word=word, stem=stem, stem_docs=int(stem_docs or 0),
                                   forms=_parse_forms(forms))
    return found
