"""
services/schema/revision.py
---------------------------
The startup check that the database is at the newest migration.

The tables are created and changed by Alembic alone (backend/migrations/,
run as its own step before the backend starts). The backend runs no
migration; it reads which revision the database is at and stops when that is
not the newest one this code has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from exceptions.expected import ExpectedError

_BACKEND = Path(__file__).resolve().parents[2]
# The table in which Alembic records the revision of a database.
_VERSION_TABLE = "alembic_version"


# Columns the production database holds wider than BIDS-SQL's models declare
# (widened by hand before Alembic; 0001 has the wide types). The database is
# right. Remove a column here once its model says the same.
WIDER_THAN_MODEL = frozenset({
    ("bids_datasets", "name"), ("bids_datasets", "bids_version"),
    ("bids_datasets", "dataset_type"), ("bids_datasets", "source_type"),
    ("bids_datasets", "authors"), ("bids_datasets", "doi"),
    ("bids_datasets", "paper_references"), ("bids_datasets", "funding"),
    ("bids_datasets", "validation_status"),
    ("bids_objects", "datatype"), ("bids_objects", "extension"),
    ("bids_participants", "sex"), ("bids_participants", "handedness"),
})


def declared_only(obj: Any, name: str | None, type_: str, reflected: bool,
                  compare_to: Any) -> bool:
    """For Alembic's comparison of models and database: leave out what only
    the database has.

    The database holds more than the code declares (most indexes, the table
    bids_dataset_facts). --autogenerate must not propose to drop it; a drop
    is written by hand.
    """
    return not (reflected and compare_to is None)


def same_type_when_widened(context: Any, inspected_column: Any, metadata_column: Any,
                           inspected_type: Any, metadata_type: Any) -> bool | None:
    """For Alembic's comparison: the columns of WIDER_THAN_MODEL count as
    unchanged, every other column is compared as usual (None)."""
    if (metadata_column.table.name, metadata_column.name) in WIDER_THAN_MODEL:
        return False
    return None


class SchemaRevisionError(ExpectedError):
    """The database is not at the newest migration of this code.

    Raised at startup and not caught: the backend does not start until
    `alembic upgrade head` ran.
    """


def alembic_config(url: str | None = None) -> Config:
    """The Alembic settings of backend/alembic.ini, for *url* when given."""
    config = Config(_BACKEND / "alembic.ini")
    config.attributes["configure_logging"] = False
    if url is not None:
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def newest_revisions() -> set[str]:
    """The revisions no other one follows: one, unless two were added side by side."""
    return set(ScriptDirectory.from_config(alembic_config()).get_heads())


async def check_revision(engine: AsyncEngine) -> None:
    """Raise SchemaRevisionError when the database is not at the newest migration."""
    async with engine.connect() as connection:
        exists = await connection.scalar(
            text("SELECT to_regclass(:name) IS NOT NULL"), {"name": _VERSION_TABLE})
        found = (set((await connection.scalars(
            text(f"SELECT version_num FROM {_VERSION_TABLE}"))).all()) if exists else set())
    newest = newest_revisions()
    if found != newest:
        raise SchemaRevisionError(
            f"The database is at migration {_named(found)}, this code needs {_named(newest)}. "
            "Run `alembic upgrade head` in backend/ (backend/migrations/README.md).")


def _named(revisions: set[str]) -> str:
    return ", ".join(sorted(revisions)) or "none"
