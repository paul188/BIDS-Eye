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
def _empty_catalogue(monkeypatch):
    """Every test starts with an empty catalogue, freshly loaded: no usable
    optional column, no summaries, no coded values. A test may assign its
    own parts (catalogue.coverage = ColumnCoverage.full([...])); they are
    put back afterwards. Preprocessing answers its DB back-check from
    catalogue.values, so values left by one test would change the next."""
    import time

    from services.catalogue.catalogue import ColumnCoverage, Summaries, ValueCounts, catalogue

    monkeypatch.setattr(catalogue, "coverage", ColumnCoverage())
    monkeypatch.setattr(catalogue, "summaries", Summaries())
    monkeypatch.setattr(catalogue, "values", ValueCounts())
    # As after startup: a search does not reload before REFRESH_SECONDS.
    monkeypatch.setattr(catalogue, "_last_attempt", time.monotonic())
