"""Parsing GitHub's answers in the sign-in (services/github_oauth.py)."""
from __future__ import annotations

import pytest

from services.github_oauth import _github_login


class Answer:
    def __init__(self, payload, status: int = 200):
        self.payload, self.status_code = payload, status

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_login() -> None:
    assert _github_login(Answer({"login": "octocat"})) == "octocat"


@pytest.mark.parametrize("answer", [
    Answer({"message": "Bad credentials"}, 401),
    Answer({"message": "API rate limit exceeded"}, 200),
    Answer({"login": ""}),
    Answer({"login": None}),
    Answer(["octocat"]),
    Answer(ValueError("not JSON")),
])
def test_no_login_means_no_shared_unknown_account(answer: Answer) -> None:
    assert _github_login(answer) == ""
