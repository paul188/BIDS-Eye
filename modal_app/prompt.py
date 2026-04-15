"""
modal_app/prompt.py
-------------------
Re-exports SYSTEM from training_data_generation/constants.py.
In the Modal container, constants.py is baked in as /app/constants.py.
Locally (dev / tests), we resolve it relative to the repo root.
"""

import sys
from pathlib import Path

# Try repo-relative import first (local dev), then container path.
for _p in [
    Path(__file__).resolve().parents[1] / "training_data_generation",
    Path("/app"),
]:
    if (_p / "constants.py").exists():
        sys.path.insert(0, str(_p))
        break

from constants import SYSTEM as SYSTEM_PROMPT  # noqa: E402

__all__ = ["SYSTEM_PROMPT"]
