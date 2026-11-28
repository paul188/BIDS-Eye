"""What the database under test can search, and loading the backend's counts, on any backend.

Until 79f18a2 (2026-11-21) the backend found its optional full-text columns at
startup (services/catalogue/schema_capabilities.py: has_fts, has_paper_fts,
has_exact_fts, ...). Since then the columns are required while their switch is
on (services/schema/check.py) and the switches TEXT_SEARCH_PAPERS and
TEXT_SEARCH_EXACT decide; schema_capabilities.py is gone.

The harnesses only ask whether the database has the columns, so this module
reads that from the database itself, the same on every backend version. Call
start() once after the backend folder is on sys.path and use_either_layout().
"""
from __future__ import annotations

from typing import Any, FrozenSet

COLUMNS_SQL = ("SELECT column_name FROM information_schema.columns "
               "WHERE table_schema = current_schema() AND table_name = 'bids_datasets'")
WORD_FORMS_SQL = "SELECT to_regclass('bids_word_forms') IS NOT NULL"

_columns: FrozenSet[str] = frozenset()
_word_forms = False


async def _load_backend_counts(engine: Any) -> None:
    """Load what the backend keeps in memory about the database (values, coverage, names)."""
    try:
        from services.database_stats.database_stats import database_stats
    except ImportError:  # backend before the rename to database_stats.py (bdb117c)
        pass
    else:
        await database_stats.start(engine)  # reads the institution and place names too
        return
    try:
        from services.catalogue.catalogue import catalogue
    except ImportError:  # backend before catalogue.py (79f18a2)
        try:
            from services.catalogue import schema_capabilities
        except ImportError:  # backend before services/ was grouped
            from services import schema_capabilities
        await schema_capabilities.refresh_available_columns(engine)
    else:
        await catalogue.start(engine)
    try:  # before bdb117c the institution names were read on their own
        import institution_index
    except ImportError:
        return
    await institution_index.refresh_from_db(engine)


async def start(engine: Any) -> None:
    """Load the backend's counts and read which full-text columns the database has."""
    global _columns, _word_forms
    from sqlalchemy import text
    await _load_backend_counts(engine)
    async with engine.connect() as connection:
        _columns = frozenset((await connection.execute(text(COLUMNS_SQL))).scalars())
        _word_forms = bool((await connection.execute(text(WORD_FORMS_SQL))).scalar())


def has_fts() -> bool:
    """bids_datasets has the full-text column (required since 79f18a2)."""
    return "search_tsv" in _columns


def has_paper_fts() -> bool:
    """bids_datasets has the full-text column that includes paper text."""
    return "search_tsv_papers" in _columns


def has_exact_fts() -> bool:
    """bids_datasets has the unstemmed full-text column."""
    return "search_tsv_exact" in _columns


def has_exact_paper_fts() -> bool:
    """bids_datasets has the unstemmed column that includes paper text."""
    return "search_tsv_exact_papers" in _columns


def has_word_forms() -> bool:
    """The table of the catalogue's word forms exists."""
    return _word_forms
