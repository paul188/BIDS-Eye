"""
deps.py
-------
FastAPI dependency providers shared across routers.
"""

from typing import AsyncGenerator

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
