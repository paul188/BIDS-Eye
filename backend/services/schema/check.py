"""
services/schema/check.py
------------------------
The startup check that the database has what services/schema/tables.py
declares.
"""

from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from exceptions.expected import ExpectedError
from services.schema import tables
from services.switches import Switch


class MissingSchemaError(ExpectedError):
    """The database lacks a table or column that tables.py declares.

    Raised at startup and not caught: the backend does not start until the
    migration ran.
    """


async def check_schema(connection: AsyncConnection | AsyncSession) -> None:
    """Raise MissingSchemaError when the database lacks a declared table or column.

    Tables and columns that name a flag (tables.FLAG) are needed only while
    that flag is on.
    """
    declared = tables.MIGRATED_TABLES
    rows = await connection.execute(
        text("SELECT table_name, column_name FROM information_schema.columns "
             "WHERE table_schema = current_schema() AND table_name IN :names")
        .bindparams(bindparam("names", expanding=True)),
        {"names": [table.name for table in declared]})
    present = {(table_name, column_name) for table_name, column_name in rows}
    missing: list[str] = []
    for table in declared:
        if not _needed(table.info):
            continue
        for column in table.columns:
            migration = column.info.get(tables.MIGRATION) or table.info.get(tables.MIGRATION)
            if migration is None:
                continue  # a column BIDS-SQL creates itself
            if not _needed(column.info) or (table.name, column.name) in present:
                continue
            missing.append(f"{table.name}.{column.name} ({migration})")
    if missing:
        raise MissingSchemaError(
            "The database lacks " + ", ".join(missing)
            + ". Run those migrations, or switch off the setting that needs them.")


def _needed(info: Mapping[str, object]) -> bool:
    flag = info.get(tables.FLAG)
    return flag is None or Switch(str(flag)).is_on()
