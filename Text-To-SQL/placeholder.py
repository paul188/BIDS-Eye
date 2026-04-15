"""
Text-To-SQL/placeholder.py
--------------------------
Text-to-SQL translation layer — calls the deployed Modal inference function.

Requires env vars (set in .env):
    MODAL_TOKEN_ID
    MODAL_TOKEN_SECRET

If Modal is unreachable or not configured, falls back to returning all datasets
so the frontend never shows an empty error state during local development.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from schemas import TextToSQLResult  # noqa: E402

_FALLBACK_SQL = """
SELECT
    d.id,
    d.name,
    d.accession_id,
    d.bids_version,
    d.dataset_type,
    d.source_type,
    d.remote_url,
    d.validation_status,
    COUNT(DISTINCT o.subject) AS subject_count
FROM bids_datasets d
LEFT JOIN bids_objects o
       ON o.dataset_id = d.id
      AND o.subject IS NOT NULL
GROUP BY d.id
ORDER BY d.name
LIMIT 50
""".strip()


async def text_to_sql(question: str) -> TextToSQLResult:
    """
    Translate a natural-language question into SQL via the Modal-hosted
    Phi-3-mini + QLoRA inference function.

    Falls back to returning all datasets when Modal is not configured.
    """
    token_id = os.getenv("MODAL_TOKEN_ID")
    token_secret = os.getenv("MODAL_TOKEN_SECRET")

    if not token_id or not token_secret:
        return TextToSQLResult(
            sql=_FALLBACK_SQL,
            params={},
            explanation=(
                "[fallback] MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set. "
                "Returning all datasets."
            ),
        )

    try:
        import modal

        fn = modal.Function.lookup("bids-eye", "TextToSQLModel.generate")
        result: dict = await fn.remote.aio(question)
        return TextToSQLResult(
            sql=result["sql"],
            params={},
            explanation=result.get("explanation"),
        )

    except Exception as exc:
        return TextToSQLResult(
            sql=_FALLBACK_SQL,
            params={},
            explanation=f"[fallback] Modal call failed: {exc}",
        )
