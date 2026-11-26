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

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from exceptions.expected import ExpectedError

_BACKEND = Path(__file__).resolve().parents[2]
# The table in which Alembic records the revision of a database.
_VERSION_TABLE = "alembic_version"


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
