"""
services/schema/check.py
------------------------
The startup check that the database has what services/schema/tables.py
declares.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import Column, bindparam, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from exceptions.expected import ExpectedError
from services.schema import tables
from services.switches import Switch


class MissingSchemaError(ExpectedError):
    """The database lacks a table or column that tables.py declares, or has it
    with another type.

    Raised at startup and not caught: the backend does not start until the
    migration ran.
    """


async def check_schema(connection: AsyncConnection | AsyncSession) -> None:
    """Raise MissingSchemaError when the database lacks a declared table or column,
    or has a column with another type.

    Tables and columns that name a flag (tables.FLAG) are needed only while
    that flag is on.
    """
    declared = tables.MIGRATED_TABLES
    rows = await connection.execute(
        text("SELECT table_name, column_name, CASE data_type "
             "WHEN 'ARRAY' THEN ltrim(udt_name, '_') || '[]' "
             "WHEN 'character varying' THEN 'text' ELSE data_type END "
             "FROM information_schema.columns "
             "WHERE table_schema = current_schema() AND table_name IN :names")
        .bindparams(bindparam("names", expanding=True)),
        {"names": [table.name for table in declared]})
    present = {(table_name, column_name): type_ for table_name, column_name, type_ in rows}
    missing: list[str] = []
    mistyped: list[str] = []
    for table in declared:
        if not _needed(table.info):
            continue
        for column in table.columns:
            migration = column.info.get(tables.MIGRATION) or table.info.get(tables.MIGRATION)
            if migration is None:
                continue  # a column of BIDS-SQL's own model
            if not _needed(column.info):
                continue
            found = present.get((table.name, column.name))
            if found is None:
                missing.append(f"{table.name}.{column.name} (revision {migration})")
            elif found != sql_type(column):
                mistyped.append(f"{table.name}.{column.name} is {found}, "
                                f"declared {sql_type(column)} (revision {migration})")
    if missing:
        raise MissingSchemaError(
            "The database lacks " + ", ".join(missing)
            + ". Run `alembic upgrade head` in backend/, or switch off the setting that needs them.")
    if mistyped:
        raise MissingSchemaError("Columns with another type than tables.py declares: "
                                 + ", ".join(mistyped) + ".")


def sql_type(column: Column[Any]) -> str:
    """The declared type of *column* as PostgreSQL names it, e.g. 'double precision'."""
    return column.type.compile(dialect=postgresql.dialect()).lower()


def _needed(info: Mapping[str, object]) -> bool:
    flag = info.get(tables.FLAG)
    return flag is None or Switch(str(flag)).is_on()
