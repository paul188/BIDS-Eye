"""backend/vocabulary/tools/cognitive_atlas_api.py: a missing page is None, an API that cannot
be read is CognitiveAtlasFetchError, never None."""
from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "vocabulary" / "tools"))
import cognitive_atlas_api as ca  # noqa: E402


def _http_error(code: int, headers=None):
    return urllib.error.HTTPError("https://ca/x", code, "error", headers or {}, None)


@pytest.fixture
def answers(monkeypatch):
    queue = []

    def urlopen(url, timeout):
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return io.BytesIO(answer)

    monkeypatch.setattr(ca.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ca.time, "sleep", lambda s: None)
    return queue


def test_a_missing_page_is_none(answers):
    answers.append(_http_error(404))
    assert ca.get_json("https://ca/x") is None


def test_rate_limits_and_server_errors_are_retried(answers):
    answers.extend([_http_error(429, {"Retry-After": "soon"}), _http_error(503), b'{"id": 1}'])
    assert ca.get_json("https://ca/x") == {"id": 1}


def test_an_api_that_cannot_be_read_raises(answers):
    answers.extend([urllib.error.URLError("timed out")] * 3)
    with pytest.raises(ca.CognitiveAtlasFetchError, match="timed out"):
        ca.get_json("https://ca/x", retries=3)


def test_a_refused_request_is_not_retried(answers):
    answers.extend([_http_error(403), b"{}"])
    with pytest.raises(ca.CognitiveAtlasFetchError):
        ca.get_json("https://ca/x")
    assert answers == [b"{}"]
