"""The app's own tables, next to the dataset catalogue in the same database.

Two of them hold something about a user, and only for users who asked for
it: search_history_consents and search_history, the searches of people who
turned on "Remember my searches and search results"
(services/stores/search_history.py). Everything else is anonymous: no logins,
no e-mail addresses, no per-user counts (the rate limits count in memory,
rate_limits.py). The tables users, message_usage and user_queries were removed
in November 2026 for the site's privacy notice
(frontend/src/components/CookieBanner.vue), which describes both.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class QueryInterpretation(Base):
    """How one question was translated into SQL (services/stores/interpretations.py)."""

    __tablename__ = "query_interpretations"

    # sha256 of pipeline version + normalized question
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    pipeline_version: Mapped[str] = mapped_column(String(16), index=True)
    question: Mapped[str] = mapped_column(Text)
    question_normalized: Mapped[str] = mapped_column(Text, index=True)
    # JSON: sql, params, scored_filters, apply_relevance, rank_query,
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

    Permanent, unlike services/stores/translated_queries.py (in memory, 30 minutes) and
    unlike the interpretation cache, which keeps one translation per question
    and none that found nothing.

    Anonymous: no column links a row to who asked, and search_history rows
    do not point here. The site's privacy notice promises this
    (frontend/src/components/CookieBanner.vue), so this table needs no storage
    period and stays when a user deletes their history.
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


class SearchHistoryConsent(Base):
    """A user who turned on "Remember my searches and search results".

    The row is the consent (GDPR Art. 6(1)(a)): searches are kept in
    search_history only while it exists, and deleting it deletes them. The
    user is GitHub's account number, not the login: a login can be renamed,
    and a released one can be taken by someone else.
    """

    __tablename__ = "search_history_consents"

    github_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    consented_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Last search saved or history read; the consent ends RETENTION_DAYS after it.
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class SearchHistoryEntry(Base):
    """One search of a user with a SearchHistoryConsent row, as it was answered."""

    __tablename__ = "search_history"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    github_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("search_history_consents.github_id", ondelete="CASCADE"),
        index=True,
    )
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    total: Mapped[int] = mapped_column(Integer)
    # JSON of the QueryResponse the user got: translation, chips, first page.
    answer: Mapped[str] = mapped_column(Text)
