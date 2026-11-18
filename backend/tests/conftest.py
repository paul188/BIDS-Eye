from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "backend",
    ROOT / "LLM_preprocessor",
    ROOT / "RAG",
    ROOT / "BIDS-SQL",
):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))



import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Every test starts with empty rate-limit counts (rate_limits.py keeps them
    in memory for the whole process)."""
    import rate_limits

    rate_limits.reset()
    yield
    rate_limits.reset()
