from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class MessageUsage(Base):
    """Tracks per-user daily message counts for rate limiting."""

    __tablename__ = "message_usage"

    github_login: Mapped[str] = mapped_column(String(100), primary_key=True)
    usage_date: Mapped[date] = mapped_column(Date, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class User(Base):
    """One row per GitHub user who has ever logged in."""

    __tablename__ = "users"

    github_login: Mapped[str] = mapped_column(String(100), primary_key=True)
    github_email: Mapped[str | None] = mapped_column(String(254), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    queries: Mapped[list[UserQuery]] = relationship(back_populates="user")


class UserQuery(Base):
    """One row per query a user has submitted."""

    __tablename__ = "user_queries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    github_login: Mapped[str] = mapped_column(
        ForeignKey("users.github_login"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    user: Mapped[User] = relationship(back_populates="queries")
