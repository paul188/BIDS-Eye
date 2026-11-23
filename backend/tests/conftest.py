from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Every test starts with empty rate-limit counts (rate_limits.py keeps them
    in memory for the whole process)."""
    import rate_limits

    rate_limits.reset()
    yield
    rate_limits.reset()


@pytest.fixture(autouse=True)
def _empty_database_stats(monkeypatch):
    """Every test starts with empty database_stats, freshly loaded: no usable
    optional column, no summaries, no coded values, no names. A test may assign its
    own parts (database_stats.coverage = ColumnCoverage.full([...])); they are
    put back afterwards. Preprocessing answers its DB back-check from
    database_stats.values, so values left by one test would change the next."""
    import time

    from services.database_stats.database_stats import ColumnCoverage, Names, Summaries, ValueCounts, database_stats

    monkeypatch.setattr(database_stats, "coverage", ColumnCoverage())
    monkeypatch.setattr(database_stats, "summaries", Summaries())
    monkeypatch.setattr(database_stats, "values", ValueCounts())
    monkeypatch.setattr(database_stats, "names", Names())
    # As after startup: a search does not reload before REFRESH_SECONDS.
    monkeypatch.setattr(database_stats, "_last_attempt", time.monotonic())
