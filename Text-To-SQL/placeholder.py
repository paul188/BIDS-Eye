"""
Text-To-SQL/placeholder.py
--------------------------
Placeholder for the Text-To-SQL translation layer.

Replace `text_to_sql()` with your LLM-based implementation when ready.
The function receives the user's natural-language question and must return
a TextToSQLResult whose `sql` selects columns compatible with DatasetSchema.

Required SQL output columns (at minimum):
    id, name, accession_id, bids_version, dataset_type,
    source_type, remote_url, validation_status, subject_count
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing schemas from the backend package when run standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from schemas import TextToSQLResult  # noqa: E402


def text_to_sql(question: str) -> TextToSQLResult:
    """
    Convert a natural-language question into a TextToSQLResult.

    TODO: Replace this body with an LLM call, for example:

        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            system=BIDS_SQL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        sql = _parse_sql_block(response.content[0].text)
        return TextToSQLResult(sql=sql, explanation=response.content[0].text)

    Current behaviour: returns ALL datasets regardless of the question so
    the frontend works end-to-end before the LLM layer is wired in.
    """
    sql = """
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

    return TextToSQLResult(
        sql=sql,
        params={},
        explanation=(
            f"[placeholder] Question received: {question!r}. "
            "Returning all datasets. Replace text_to_sql() with your LLM implementation."
        ),
    )
