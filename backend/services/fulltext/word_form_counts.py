"""
services/fulltext/word_form_counts.py
-------------------------------------
The catalogue's word forms: which forms each stem merges, and in how many
datasets each form occurs.

bids_word_forms has one row per word form found in the datasets' own text
(search_tsv_exact):

    form   the word as written, lower case      "ageing"
    stem   what the 'english' stemmer makes it  "age"
    ndoc   datasets that contain the form

refresh() rebuilds it; it runs after each crawl
(ingest/openneuro-crawler/post_crawl.py; where no crawler runs,
`python post_crawl.py --all --only word_forms`). It decides whether a searched word is worth a
closer look, and, for a word with forms judged different from it
(backend/vocabulary/stem_collisions.py), which forms of its stem the search requires: all the
table has, minus those. A form new since the last refresh is then missed
until the next one.

lookup() answers for the words of one question, cached_stem_counts() for one
word at a time (a reviewed word the question's decisions do not cover). Like
the word counts in services/fulltext/text_search.py they are called from the
translation worker thread, so they reach the database through database/worker.py.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from database import worker as worker_db
from services.schema.tables import datasets, word_forms
from services.fulltext import vectors
from services.fulltext.text_search import TS_CONFIG

from stem_collisions import StemCounts

log = logging.getLogger(__name__)

TABLE = word_forms.name
SOURCE_TABLE = datasets.name
SOURCE_COLUMN = datasets.c.search_tsv_exact.name

# Alphabetic words only: numbers and codes ("t1w", "7t") have no word forms,
# and a form goes into tsquery literals as it is.
_FORM_PATTERN = "^[a-z]{2,40}$"
# Stemmed like search_tsv, or the stems of the table and of the search drift apart.
_STEM_SQL = f"array_to_string(tsvector_to_array(to_tsvector('{TS_CONFIG}', {{}})), ' ')"

_REFRESH_SQL = (
    f"INSERT INTO {TABLE} (form, stem, ndoc) "
    "SELECT s.word, l.stem, s.ndoc "
    f"FROM ts_stat('SELECT {SOURCE_COLUMN} FROM {SOURCE_TABLE}') AS s "
    f"CROSS JOIN LATERAL (SELECT {_STEM_SQL.format('s.word')} AS stem) AS l "
    # A stopword has no stem and can never be searched for.
    "WHERE s.word ~ :pattern AND l.stem ~ '^[a-z0-9]+$'"
)

# Forms returned per word: enough for every regular inflection of a stem.
FORMS_PER_WORD = 40
LOOKUP_TIMEOUT_SECONDS = 3.0

_LOOKUP_SQL = (
    "SELECT w.word, l.stem, "
    f"       (SELECT count(*) FROM {SOURCE_TABLE} d "
    f"         WHERE d.search_tsv @@ to_tsquery('{TS_CONFIG}', w.word)) AS stem_docs, "
    f"       ARRAY(SELECT f.form || ':' || f.ndoc FROM {TABLE} f "
    "              WHERE f.stem = l.stem ORDER BY f.ndoc DESC, f.form LIMIT :limit) AS forms "
    "FROM unnest(CAST(:words AS text[])) AS w(word) "
    f"CROSS JOIN LATERAL (SELECT {_STEM_SQL.format('w.word')} AS stem) AS l "
    "WHERE l.stem <> ''"
)

_WORD_RE = re.compile(r"[a-z]{2,40}")


# ── Building the table ────────────────────────────────────────────────────────

async def refresh(conn: Any) -> Dict[str, int]:
    """Rebuild bids_word_forms from the datasets' text. Caller commits.

    Raises when search_tsv_exact is missing: the database is not at the newest
    migration (backend/migrations/README.md).

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
        raise RuntimeError("word_form_counts.refresh() needs a transaction; "
                           "the connection commits each statement (AUTOCOMMIT?)")
    present = (await conn.execute(text(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = :table "
        "AND column_name = :column"), {"table": SOURCE_TABLE, "column": SOURCE_COLUMN})).scalar_one()
    if not present:
        raise RuntimeError(f"{SOURCE_TABLE}.{SOURCE_COLUMN} is missing: "
                           "run `alembic upgrade head` in backend/ first")
    await conn.execute(text(f"DELETE FROM {TABLE}"))
    await conn.execute(text(_REFRESH_SQL), {"pattern": _FORM_PATTERN})
    row = (await conn.execute(text(
        f"SELECT count(*), count(DISTINCT stem) FROM {TABLE}"))).one()
    return {"forms": int(row[0]), "stems": int(row[1])}


# ── Looking words up ──────────────────────────────────────────────────────────

# Given words, returns the rows of _LOOKUP_SQL, or None when unavailable.
Lookup = Callable[[List[str]], Optional[List[Sequence[Any]]]]

# How long cached_stem_counts() keeps an answer, a missing one included: one
# search builds many queries for the same word, and a database that did not
# answer should not be asked again for each.
FORM_COUNTS_CACHE_SECONDS = 600.0
# word → (time asked, its counts or None)
_form_counts_cache: Dict[str, Tuple[float, Optional[StemCounts]]] = {}


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
    _form_counts_cache.clear()


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


def lookup(words: Sequence[str]) -> Dict[str, StemCounts]:
    """The catalogue's forms for each of *words*; {} when unavailable.

    Words that are not plain lower-case words, and stopwords, are left out.
    A lookup raises worker_db.WorkerDbError when the database cannot answer;
    that gives {} and the words are searched by stem. Anything else it
    raises is a bug.
    """
    asked = [w for w in dict.fromkeys(words) if _WORD_RE.fullmatch(w)]
    if not asked or not vectors.exact_enabled():
        return {}
    if _lookup is None:
        return {}
    try:
        rows = _lookup(asked)
    except worker_db.WorkerDbError as exc:
        log.warning("Word form lookup failed (%s) — searching by stem", exc)
        return {}
    found: Dict[str, StemCounts] = {}
    for row in rows or ():
        word, stem, stem_docs, forms = row[0], row[1], row[2], row[3]
        found[word] = StemCounts(word=word, stem=stem, stem_docs=int(stem_docs or 0),
                                   forms=_parse_forms(forms))
    return found


def cached_stem_counts(word: str) -> Optional[StemCounts]:
    """lookup() for one word, kept FORM_COUNTS_CACHE_SECONDS; None when unavailable."""
    now = time.monotonic()
    cached = _form_counts_cache.get(word)
    if cached is not None and now - cached[0] < FORM_COUNTS_CACHE_SECONDS:
        return cached[1]
    counts = lookup([word]).get(word)
    _form_counts_cache[word] = (now, counts)
    return counts
