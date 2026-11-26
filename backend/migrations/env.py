"""
migrations/env.py
-----------------
What Alembic runs for every command: connects to the database and hands
Alembic the tables the code declares.

The database is the backend's own (settings_obj.DATABASE_URL), unless the
caller set sqlalchemy.url (the tests do, services/schema/revision.py).
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

import db.models  # noqa: F401  (BIDS-SQL's tables, into Base.metadata)
import models     # noqa: F401  (the backend's own tables, into Base.metadata)
from db.base import Base
from services.schema import tables  # also adds the extension columns of bids_datasets
from services.schema.revision import declared_only

config = context.config
# Not when called from code (config_file_name is set only by the alembic command):
# fileConfig would switch off the caller's loggers.
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# What --autogenerate compares the database with: the models and the extension
# columns. The tables of tables.metadata are declared for queries only (no
# keys, no NOT NULL), so their revisions are written by hand.
assert tables.datasets.metadata is Base.metadata
target_metadata = Base.metadata


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      include_object=declared_only, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        from settings import settings_obj
        url = settings_obj.DATABASE_URL
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    # alembic upgrade head --sql: print the statements, run nothing.
    context.configure(url="postgresql://", target_metadata=target_metadata,
                      literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(_run_online())
