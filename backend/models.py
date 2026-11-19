"""The app's own tables, next to the dataset catalogue in the same database.

None of them holds anything about a user: no logins, no e-mail addresses, no
per-user counts (the rate limits count in memory, rate_limits.py). The tables
users, message_usage and user_queries did until November 2026 and were removed
for the site's privacy notice (frontend/src/components/CookieBanner.vue).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class QueryInterpretation(Base):
    """How one question was translated into SQL (services/stores/interpretation_store.py)."""

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

    Permanent, unlike services/stores/query_cache.py (in memory, 30 minutes) and
    unlike the interpretation cache, which keeps one translation per question
    and none that found nothing.

    Anonymous, and the only record of a search: no column and no other table
    links a question to who asked it. The site's privacy notice promises this
    (frontend/src/components/CookieBanner.vue); a per-user search table
    (user_queries) was removed for it in November 2026.
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
