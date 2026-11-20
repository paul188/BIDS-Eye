from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "crawlers" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

import constants  # noqa: E402
from remote import openneuro_metadata as meta  # noqa: E402


def test_extract_fields_from_full_record() -> None:
    record = {
        "id": "ds004998",
        "metadata": {
            "dxStatus": "  Parkinson's   Disease ",
            "studyDomain": "Basal Ganglia in Parkinson",
            "studyDesign": "",
            "species": "Macaque ",
            "ages": [61, 55.5, None, 70, 999, -3, True],
        },
        "latestSnapshot": {
            "readme": "MEG and LFP recordings.\x00 See https://doi.org/10.1/abc).\n\n\n\nEnd",
            "size": "173691728283",
            "summary": {"size": 173691728283},
        },
    }
    fields = meta.extract_fields(record)

    assert fields["dx_status"] == "Parkinson's Disease"
    assert fields["study_domain"] == "Basal Ganglia in Parkinson"
    assert fields["study_design"] is None
    assert fields["species"] == "Macaque"
    # implausible ages and booleans are ignored
    assert (fields["age_min"], fields["age_max"]) == (55.5, 70.0)
    assert fields["size_bytes"] == 173691728283
    assert "\x00" not in fields["readme_text"]
    assert "\n\n\n" not in fields["readme_text"]
    assert fields["readme_urls"] == ["https://doi.org/10.1/abc"]


def test_extract_fields_tolerates_missing_record() -> None:
    fields = meta.extract_fields(None)
    assert set(meta.METADATA_COLUMNS) <= set(fields)
    assert all(fields[name] is None for name in meta.METADATA_COLUMNS)
    assert fields["readme_urls"] == []

    partial = meta.extract_fields({"metadata": None, "latestSnapshot": {"summary": None, "size": "n/a"}})
    assert partial["size_bytes"] is None and partial["age_min"] is None


def test_size_falls_back_to_snapshot_size() -> None:
    fields = meta.extract_fields({"latestSnapshot": {"size": 2048, "summary": None}})
    assert fields["size_bytes"] == 2048


def test_clean_readme_strips_markup_and_caps_length() -> None:
    text, urls = meta.clean_readme(
        "<h1>Title</h1> ![plot](data:image/png;base64,AAAA) body http://a.example/x."
    )
    assert text == "Title   body http://a.example/x."
    assert urls == ["http://a.example/x"]

    long_text, _ = meta.clean_readme("a" * (meta.README_MAX_CHARS + 5000) + "TAIL")
    assert len(long_text) <= meta.README_MAX_CHARS + 3
    assert long_text.endswith("TAIL")          # the end, where references live, is kept

    assert meta.clean_readme("") == (None, [])
    assert meta.clean_readme("<br>") == (None, [])


def test_decode_readme_bytes_handles_utf16() -> None:
    sample = "Resting-state data from 20 patients"
    assert meta.decode_readme_bytes(sample.encode("utf-8")) == sample
    assert meta.decode_readme_bytes(sample.encode("utf-16")) == sample          # with BOM
    assert meta.decode_readme_bytes(sample.encode("utf-16-le")) == sample       # without BOM


def test_migration_file_matches_metadata_columns() -> None:
    sql = (ROOT / "crawlers" / "migrate_add_openneuro_metadata.sql").read_text()
    added = dict(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)\s+([A-Z][A-Z ]*[A-Z])", sql))
    assert added == {**meta.METADATA_COLUMNS, **meta.STATUS_COLUMNS}


def test_every_metadata_column_is_known_to_the_prompt_layer() -> None:
    """A column the crawler fills must be describable to the SQL model."""
    assert set(meta.METADATA_COLUMNS) <= set(constants.OPTIONAL_DATASET_COLUMNS)


def test_ensure_columns_sql_is_idempotent_ddl() -> None:
    statements = meta.ensure_columns_sql()
    assert len(statements) == len(meta.CRAWLER_MANAGED_COLUMNS)
    assert all("ADD COLUMN IF NOT EXISTS" in statement for statement in statements)
    assert any("institutions TEXT[]" in statement for statement in statements)


def test_deprecation_reason_covers_unavailable_and_flagged_datasets() -> None:
    assert meta.deprecation_reason({"latestSnapshot": {"deprecated": None}}) is None
    assert meta.deprecation_reason(None) is None
    assert meta.deprecation_reason(None, meta.REMOVED_REASON) == meta.REMOVED_REASON
    assert (
        meta.deprecation_reason({"latestSnapshot": {"deprecated": {"reason": " replaced by  ds009 "}}})
        == "Deprecated on OpenNeuro: replaced by ds009"
    )
    assert (
        meta.deprecation_reason({"latestSnapshot": {"deprecated": {"reason": None}}})
        == "Deprecated on OpenNeuro"
    )


def test_unavailable_reason_reads_the_three_definite_api_answers() -> None:
    reason = meta.unavailable_reason
    assert reason("Dataset ds001019 does not exist.") == "No longer available on OpenNeuro"
    assert reason("Dataset ds004929 has been deleted. Reason: duplicate dataset.") == "Deleted on OpenNeuro: duplicate dataset"
    assert reason("Dataset ds004167 has been deleted. Reason: .") == "Deleted on OpenNeuro"
    assert reason("Dataset ds004174 has been deleted. Reason: undefined.") == "Deleted on OpenNeuro"
    assert reason("You do not have access to read this dataset.") == "No longer public on OpenNeuro"
    # anything else is not a definite answer
    assert reason("Internal server error") is None
    assert reason("") is None


def test_fetch_batch_separates_unavailable_from_unknown(monkeypatch) -> None:
    """Only a definite API answer may mark a dataset; other errors stay unknown."""
    def fake_post(query, variables=None, **_):
        assert variables == {"id0": "ds000001", "id1": "ds001019", "id2": "ds000777", "id3": "ds000888"}
        return {
            "data": {"d0": {"id": "ds000001"}, "d1": None, "d2": None, "d3": None},
            "errors": [
                {"message": "Dataset ds001019 does not exist.", "path": ["d1"]},
                {"message": "You do not have access to read this dataset.", "path": ["d2"]},
                {"message": "Unexpected error value", "path": ["d3"]},
            ],
        }

    monkeypatch.setattr(meta, "post_graphql", fake_post)
    records, unavailable = meta.fetch_batch(["ds000001", "ds001019", "ds000777", "ds000888"])

    assert records["ds000001"] == {"id": "ds000001"}
    assert unavailable == {
        "ds001019": "No longer available on OpenNeuro",
        "ds000777": "No longer public on OpenNeuro",
    }


def test_status_column_is_not_shown_to_the_sql_model() -> None:
    assert not set(meta.STATUS_COLUMNS) & set(constants.OPTIONAL_DATASET_COLUMNS)


class _Body:
    """A urlopen answer whose read raises *error* or returns *data*."""

    def __init__(self, data=b"", error=None):
        self.data, self.error = data, error

    def read(self, *args):
        if self.error is not None:
            raise self.error
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _answers(monkeypatch, *answers):
    queue = list(answers)

    def urlopen(request, timeout):
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(meta.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)
    return queue


def test_a_body_cut_short_or_not_utf8_is_retried(monkeypatch):
    import http.client

    _answers(monkeypatch, _Body(error=http.client.IncompleteRead(b"{")),
             _Body(b'{"data": "\xff"}'), _Body(b'{"data": {"ok": 1}}'))
    assert meta.post_graphql("{ x }") == {"data": {"ok": 1}}


def test_every_attempt_failing_raises_openneuro_api_error_from_the_last(monkeypatch):
    import http.client

    _answers(monkeypatch, *[_Body(error=http.client.IncompleteRead(b"{"))] * 3)
    with pytest.raises(meta.OpenNeuroAPIError) as caught:
        meta.post_graphql("{ x }")
    assert isinstance(caught.value.__cause__, http.client.IncompleteRead)


def test_a_bug_is_not_retried(monkeypatch):
    _answers(monkeypatch, TypeError("a bug"))
    with pytest.raises(TypeError):
        meta.post_graphql("{ x }")
