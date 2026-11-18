from __future__ import annotations

import os
import sys
from pathlib import Path

# routers.auth refuses to import without a JWT secret; tests only need a placeholder.
os.environ.setdefault("JWT_SECRET", "tests-only-not-a-secret")

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "backend",
    ROOT / "LLM_preprocessor",
    ROOT / "RAG",
    ROOT / "BIDS-SQL",
):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

