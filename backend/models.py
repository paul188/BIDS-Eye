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


class QueryInterpretation(Base):
    """How one question was translated into SQL (services/interpretation_store.py)."""

    __tablename__ = "query_interpretations"

    # sha256 of pipeline version + normalized question
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    pipeline_version: Mapped[str] = mapped_column(String(16), index=True)
    question: Mapped[str] = mapped_column(Text)
    question_normalized: Mapped[str] = mapped_column(Text, index=True)
    # JSON: sql, params, scored_filters, apply_relevance, rank_tsquery,
    # explanation, filter_sql, relaxed_sql, relaxed
    payload: Mapped[str] = mapped_column(Text)
    total_at_creation: Mapped[int] = mapped_column(Integer)
    hits: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Filled by a later offline check of the translation against the database:
    # 'ok', 'wrong' (never served again) or NULL (not checked).
    review_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class QueryLog(Base):
    """Every search that was run: the question, its translation and what it found.

    Permanent, unlike services/query_cache.py (in memory, 30 minutes) and
    unlike the interpretation cache, which keeps one translation per question
    and none that found nothing. No user is recorded here.
    """

    __tablename__ = "query_log"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    pipeline_version: Mapped[str] = mapped_column(String(16), index=True)
    # JSON, as in query_interpretations.payload
    payload: Mapped[str] = mapped_column(Text)
    total: Mapped[int] = mapped_column(Integer)
    from_cache: Mapped[bool] = mapped_column(default=False)
    relaxed: Mapped[bool] = mapped_column(default=False)
    self_corrected: Mapped[bool] = mapped_column(default=False)
    fallback: Mapped[bool] = mapped_column(default=False)


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
