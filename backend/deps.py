"""
deps.py
-------
FastAPI dependency providers shared across routers.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.db import async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session, rolling back on error."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# What a route writes to get a session: ``session: AsyncSessionDep``.
AsyncSessionDep = Annotated[AsyncSession, Depends(get_db)]
