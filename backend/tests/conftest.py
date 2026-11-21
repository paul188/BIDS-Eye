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
def _empty_catalogue_values(monkeypatch):
    """Every test starts with nothing cached in catalogue_values, freshly
    loaded. A test that loads values (through schema_capabilities) would
    otherwise leave them for the next, and preprocessing answers its DB
    back-check from that cache."""
    import time

    from services.catalogue import catalogue_values

    monkeypatch.setattr(catalogue_values, "_counts", {})
    # As after startup: a search does not reload before REFRESH_SECONDS.
    monkeypatch.setattr(catalogue_values, "_last_attempt", time.monotonic())
