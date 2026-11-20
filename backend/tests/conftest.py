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
