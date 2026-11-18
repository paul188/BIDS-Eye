"""Parsing GitHub's answers in the sign-in callback (routers/auth.py)."""
from __future__ import annotations

import pytest

from routers.auth import _github_login, _primary_email


class Answer:
    def __init__(self, payload, status: int = 200):
        self.payload, self.status_code = payload, status

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


EMAILS = [
    {"email": "old@example.org", "primary": False, "verified": True},
    {"email": "me@example.org", "primary": True, "verified": True},
]


def test_primary_verified_address() -> None:
    assert _primary_email(Answer(EMAILS)) == "me@example.org"


def test_unverified_primary_is_not_used() -> None:
    assert _primary_email(Answer([{"email": "x@example.org", "primary": True, "verified": False}])) is None


@pytest.mark.parametrize("answer", [
    Answer({"message": "Resource not accessible by integration"}, 403),   # no user:email scope
    Answer({"message": "Bad credentials"}, 200),                         # object, not a list
    Answer(ValueError("not JSON"), 200),
    Answer(["not an object", None, {"primary": True, "verified": True}]),  # malformed entries
    Answer(EMAILS, 500),
])
def test_unusable_email_answers_give_no_address(answer: Answer) -> None:
    assert _primary_email(answer) is None


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
