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
services/fulltext/text_search.py it is called from the translation worker thread, so
the API installs a pooled lookup at startup (make_lookup); other entry points
(one-off containers, the recall benchmark) open a connection of their own.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from services.catalogue import off_loop_query, schema_capabilities

_RAG_DIR = Path(__file__).resolve().parents[3] / "RAG"
if _RAG_DIR.exists() and str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from word_forms import WordEvidence  # noqa: E402

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

_OWN_CONNECTION = object()
_lookup: Any = _OWN_CONNECTION


def set_lookup(lookup: Optional[Lookup]) -> None:
    """Install the lookup; None switches lookups off (tests)."""
    global _lookup
    _lookup = lookup


def reset_lookup() -> None:
    """Back to a connection of its own per call (the state before startup)."""
    global _lookup
    _lookup = _OWN_CONNECTION


def _params(words: List[str]) -> Dict[str, Any]:
    return {"words": words, "limit": FORMS_PER_WORD}


def _own_connection_lookup(words: List[str]) -> Optional[List[Sequence[Any]]]:
    """Look up with a connection of its own; only off the event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        return None

    async def _rows(connection: Any) -> List[Sequence[Any]]:
        return list(await connection.execute(text(_LOOKUP_SQL), _params(words)))

    return off_loop_query.run_with_own_connection(_rows, LOOKUP_TIMEOUT_SECONDS)


def make_lookup(engine: Any, loop: asyncio.AbstractEventLoop,
                timeout: float = LOOKUP_TIMEOUT_SECONDS) -> Lookup:
    """Pooled lookup for code running in a worker thread (see
    text_search.make_frequency_lookup: the engine belongs to the event loop)."""
    statement = text(_LOOKUP_SQL)

    async def _rows(words: List[str]) -> List[Sequence[Any]]:
        async with engine.connect() as connection:
            return list(await connection.execute(statement, _params(words)))

    def lookup(words: List[str]) -> Optional[List[Sequence[Any]]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return None  # called on an event loop: blocking here would deadlock
        return off_loop_query.run_on_loop(_rows(words), loop, timeout)

    return lookup


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
    A lookup raises CatalogueLookupError when the database cannot answer;
    that gives {} and the words are searched by stem. Anything else it
    raises is a bug.
    """
    asked = [w for w in dict.fromkeys(words) if _WORD_RE.fullmatch(w)]
    if not asked or not schema_capabilities.has_word_forms():
        return {}
    run = _own_connection_lookup if _lookup is _OWN_CONNECTION else _lookup
    if run is None:
        return {}
    try:
        rows = run(asked)
    except off_loop_query.CatalogueLookupError as exc:
        log.warning("Word form lookup failed (%s) — searching by stem", exc)
        return {}
    found: Dict[str, WordEvidence] = {}
    for row in rows or ():
        word, stem, stem_docs, forms = row[0], row[1], row[2], row[3]
        found[word] = WordEvidence(word=word, stem=stem, stem_docs=int(stem_docs or 0),
                                   forms=_parse_forms(forms))
    return found
