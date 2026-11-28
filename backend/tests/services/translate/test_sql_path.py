"""query_log says who wrote the SQL, without the words of the question."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlglot")

from routers.schemas import TextToSQLResult  # noqa: E402
from services.translate import text_to_sql  # noqa: E402
from services.translate.participant_query_compiler import _line_kind  # noqa: E402


@pytest.mark.parametrize("line, kind", [
    ('  institution "Secret Clinic": d.x', "institution"),
    ('  not institution "Secret Clinic": d.x', "not institution"),
    ("  author names (exact DB strings): Secret", "author names"),
    ("  Secret words", "unknown"),
    ("  secret words without a label", "unknown"),
])
def test_a_declined_line_is_named_by_its_label(line, kind):
    assert _line_kind(line) == kind


def test_the_api_does_not_send_the_path():
    assert "sql_path" not in TextToSQLResult(sql="SELECT 1", sql_path="compiled").model_dump()


def test_a_keyword_search_says_so():
    assert text_to_sql.keyword_fallback("sleep and dreams", "").sql_path == "keyword"
