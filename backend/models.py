from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, String, Text
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


class QueryTranslation(Base):
    """A question asked before, with what the models made of it.

    Read and written by services/translation_store.py; see there for when an
    entry is used.
    """

    __tablename__ = "query_translations"

    # sha256 of question_norm: questions can be longer than an index key.
    question_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    question_norm: Mapped[str] = mapped_column(Text)
    # unreviewed / verified / needs_review / rejected
    status: Mapped[str] = mapped_column(String(16), default="unreviewed", index=True)
    plan: Mapped[dict] = mapped_column(JSON)
    model_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline_fp: Mapped[str] = mapped_column(String(64))
    vocabulary_fp: Mapped[str] = mapped_column(String(64))
    schema_fp: Mapped[str] = mapped_column(String(64))
    data_stamp: Mapped[str] = mapped_column(String(64))
    # Git commit of the code that made the translation ("unknown" when the
    # process could not tell); see translation_store.code_commit().
    code_commit: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # For the reviewer.
    final_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hits: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
