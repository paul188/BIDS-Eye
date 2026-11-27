"""Login session and API security middleware.

GitHub and the database are faked. Everything else is the real stack from
middleware.install_security_middleware: CORS, CSRF, the signed HttpOnly
session cookie and RequireLogin, in front of the real auth router.
"""
from __future__ import annotations

import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import time_machine
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from security import middleware
from exceptions.handlers import add_error_handlers
from routers import auth

SECRET = "test-secret-" + "x" * 40
CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"
REVOKE_PATH = f"/applications/{CLIENT_ID}/token"
HOST = "testserver"
CALLBACK = f"https://{HOST}/api/auth/github/callback"
_REAL_ASYNC_CLIENT = httpx.AsyncClient  # before any test patches it


def _github(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/login/oauth/access_token":
        return httpx.Response(200, json={"access_token": "gh-token"})
    if request.url.path == "/user":
        return httpx.Response(200, json={"login": "octo", "id": 583231})
    if request.url.path == "/user/emails":
        return httpx.Response(200, json=[{"email": "octo@example.org", "primary": True, "verified": True}])
    if request.url.path == REVOKE_PATH and request.method == "DELETE":
        return httpx.Response(204)
    return httpx.Response(404)


def _set_github_env(monkeypatch: pytest.MonkeyPatch, org: str | None = None) -> None:
    """A complete sign-in configuration: open to every account, or to one organization."""
    monkeypatch.setenv("SESSION_SECRET", SECRET)
    monkeypatch.setenv("GITHUB_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setenv("GITHUB_REDIRECT_URI", CALLBACK)
    if org is None:
        monkeypatch.delenv("GITHUB_ORG", raising=False)
        monkeypatch.setenv("ALLOW_ANY_GITHUB_USER", "true")
    else:
        monkeypatch.setenv("GITHUB_ORG", org)
        monkeypatch.delenv("ALLOW_ANY_GITHUB_USER", raising=False)


def _answer_with(monkeypatch: pytest.MonkeyPatch, answer) -> list[httpx.Request]:
    """Let ``answer`` play GitHub; returns the requests it gets, in order."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(request)

    monkeypatch.setattr(
        auth.httpx, "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(record), **kw),
    )
    return seen


def _session_data(client: TestClient) -> dict:
    """What the signed session cookie holds (the payload is base64 JSON before the first dot)."""
    import base64
    import json
    payload = client.cookies[auth.SESSION_COOKIE].split(".")[0]
    return json.loads(base64.b64decode(payload + "=" * (-len(payload) % 4)))


def _build_app(
    monkeypatch: pytest.MonkeyPatch, frontend_url: str = "*", org: str | None = None
) -> FastAPI:
    _set_github_env(monkeypatch, org)
    monkeypatch.setenv("FRONTEND_URL", frontend_url)

    app = FastAPI()
    middleware.install_security_middleware(app)
    add_error_handlers(app)
    app.include_router(auth.auth_router, prefix="/api")

    @app.get("/api/protected")
    def protected(user: dict = Depends(auth.require_auth)) -> dict:
        return {"sub": user["sub"]}

    # No Depends(require_auth): RequireLogin must still guard it.
    @app.post("/api/forgot-the-dependency")
    def forgot() -> dict:
        return {"ok": True}

    return app


@pytest.fixture
def github_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """The paths the backend asked GitHub for, in order."""
    calls: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _github(request)

    monkeypatch.setattr(
        auth.httpx, "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(answer), **kw),
    )
    return calls


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, github_calls: list) -> FastAPI:
    return _build_app(monkeypatch)


def _browser(app: FastAPI) -> TestClient:
    # https: the cookies are Secure, so the client only sends them back over https.
    return TestClient(app, base_url=f"https://{HOST}")


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return _browser(app)


def _start_login(client: TestClient) -> str:
    resp = client.get("/api/auth/github", follow_redirects=False)
    assert resp.status_code in (302, 307)
    location = urlsplit(resp.headers["location"])
    assert location.netloc == "github.com"
    return parse_qs(location.query)["state"][0]


def _callback(client: TestClient, state: str) -> httpx.Response:
    return client.get(f"/api/auth/github/callback?state={state}&code=abc", follow_redirects=False)


def _login(client: TestClient) -> httpx.Response:
    return _callback(client, _start_login(client))


def _csrf(client: TestClient) -> dict:
    return {middleware.CSRF_HEADER: client.cookies[middleware.CSRF_COOKIE]}


# ── Session cookie ───────────────────────────────────────────────────────────


def test_callback_sets_httponly_cookie_and_keeps_token_out_of_url(client: TestClient) -> None:
    resp = _login(client)

    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/"

    session_cookie = next(
        c for c in resp.headers.get_list("set-cookie") if c.startswith(f"{auth.SESSION_COOKIE}=")
    )
    flags = session_cookie.lower()
    assert "httponly" in flags
    assert "secure" in flags
    assert "samesite=lax" in flags
    assert "path=/api" in flags


def test_sign_in_asks_and_keeps_only_the_login(client: TestClient, github_calls: list) -> None:
    # The privacy notice (frontend/src/components/CookieBanner.vue): no e-mail
    # address is asked for, and the session holds the login, GitHub's number
    # for the account and the sign-in time only.
    # Nothing is written to the database (auth.py has no database code at all).
    resp = client.get("/api/auth/github", follow_redirects=False)
    scope = parse_qs(urlsplit(resp.headers["location"]).query)["scope"][0]
    assert scope == "read:org"

    _callback(client, parse_qs(urlsplit(resp.headers["location"]).query)["state"][0])
    # The last call has GitHub delete the access token again.
    assert github_calls == ["/login/oauth/access_token", "/user", REVOKE_PATH]
    assert set(_session_data(client)) == {"user"}
    assert set(_session_data(client)["user"]) == {"sub", "uid", "iat"}
    assert _session_data(client)["user"]["uid"] == 583231
    assert not hasattr(auth, "_remember_user")


def test_cookie_signs_in_and_logout_signs_out(client: TestClient) -> None:
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/protected").status_code == 401

    _login(client)
    assert client.get("/api/auth/me").json() == {"login": "octo"}
    assert client.get("/api/protected").json() == {"sub": "octo"}

    resp = client.post("/api/auth/logout", headers=_csrf(client))
    assert resp.status_code == 200
    assert any(
        c.startswith(f"{auth.SESSION_COOKIE}=null") or "expires=thu, 01 jan 1970" in c.lower()
        for c in resp.headers.get_list("set-cookie")
    )
    assert client.get("/api/auth/me").status_code == 401


def test_bearer_header_is_not_accepted(client: TestClient) -> None:
    resp = client.get("/api/protected", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 401


def test_tampered_cookie_is_rejected(client: TestClient) -> None:
    _login(client)
    value = client.cookies[auth.SESSION_COOKIE]
    client.cookies.clear()

    def call_with(cookie_value: str) -> int:
        headers = {"Cookie": f"{auth.SESSION_COOKIE}={cookie_value}"}
        return client.get("/api/protected", headers=headers).status_code

    assert call_with(value) == 200
    assert call_with(value[:-2] + ("aa" if not value.endswith("aa") else "bb")) == 401


def test_session_ends_after_its_lifetime_even_when_used(client: TestClient) -> None:
    half = auth._SESSION_MAX_AGE / 2 + 60
    with time_machine.travel(time.time(), tick=False) as clock:
        _login(client)
        clock.shift(half)
        assert client.get("/api/protected").status_code == 200
        clock.shift(half)  # SESSION_DAYS and two minutes after the login
        assert client.get("/api/protected").status_code == 401


@pytest.mark.parametrize("age", [None, auth._SESSION_MAX_AGE + 60])
def test_a_session_past_its_login_time_is_refused(age: float | None) -> None:
    """The second guard, without the middleware's own max_age in front of it."""
    user = {"sub": "octo"} if age is None else {"sub": "octo", "iat": time.time() - age}
    session = {"user": user}
    assert auth.session_user(session) is None
    assert session == {}  # emptied, so the middleware deletes the cookie


# ── OAuth state ──────────────────────────────────────────────────────────────


def test_state_from_another_browser_is_rejected(app: FastAPI) -> None:
    attacker, victim = _browser(app), _browser(app)
    state = _start_login(attacker)
    # The attacker's callback link, opened in the victim's browser.
    resp = _callback(victim, state)
    assert resp.status_code == 400
    assert victim.get("/api/auth/me").status_code == 401


def test_state_is_single_use(client: TestClient) -> None:
    state = _start_login(client)
    assert _callback(client, state).status_code in (302, 307)
    client.post("/api/auth/logout", headers=_csrf(client))
    assert _callback(client, state).status_code == 400


def test_wrong_or_expired_state_is_rejected(client: TestClient) -> None:
    _start_login(client)
    assert _callback(client, "forged").status_code == 400

    state = _start_login(client)
    with time_machine.travel(time.time() + auth._OAUTH_STATE_SECONDS + 1):
        assert _callback(client, state).status_code == 400


def test_login_started_on_other_host_moves_to_callback_host_first(client: TestClient) -> None:
    resp = client.get(
        "/api/auth/github", headers={"host": "www.testserver"}, follow_redirects=False
    )
    assert resp.headers["location"] == f"https://{HOST}/api/auth/github?hop=1"
    assert auth.SESSION_COOKIE not in resp.headers.get("set-cookie", "")

    # At most one hop, even if the host still looks different (e.g. a proxy).
    resp = client.get(
        "/api/auth/github?hop=1", headers={"host": "www.testserver"}, follow_redirects=False
    )
    assert urlsplit(resp.headers["location"]).netloc == "github.com"


def test_missing_github_login_is_an_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_login(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/user":
            return httpx.Response(200, json={})
        return _github(request)

    monkeypatch.setattr(
        auth.httpx, "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(no_login), **kw),
    )
    assert _login(client).status_code == 502
    assert client.get("/api/auth/me").status_code == 401


def test_cancel_at_github_returns_to_the_login_page(client: TestClient, github_calls: list) -> None:
    state = _start_login(client)
    resp = client.get(
        f"/api/auth/github/callback?error=access_denied&state={state}", follow_redirects=False
    )
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == f"/?login={auth.LOGIN_CANCELLED}"
    assert github_calls == []  # no empty code sent to the token endpoint
    assert client.get("/api/auth/me").status_code == 401
    # The state is used up like after any other callback.
    assert _callback(client, state).status_code == 400


def test_other_github_error_returns_to_the_login_page(client: TestClient) -> None:
    state = _start_login(client)
    resp = client.get(
        f"/api/auth/github/callback?error=application_suspended&state={state}",
        follow_redirects=False,
    )
    assert resp.headers["location"] == f"/?login={auth.LOGIN_REFUSED}"
    assert client.get("/api/auth/me").status_code == 401


@pytest.mark.parametrize("failure", [httpx.ConnectTimeout, httpx.ConnectError])
def test_unreachable_github_is_a_502(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, failure: type[httpx.HTTPError]
) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise failure("no route", request=request)

    _answer_with(monkeypatch, down)
    client = _browser(app)
    resp = _login(client)
    assert resp.status_code == 502
    assert resp.json() == {"detail": auth.GitHubUnreachableError.detail}
    assert client.get("/api/auth/me").status_code == 401


# ── Access token ─────────────────────────────────────────────────────────────


def test_access_token_is_revoked_with_the_app_credentials(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64
    import json

    seen = _answer_with(monkeypatch, _github)
    _login(_browser(app))

    revoke = seen[-1]
    assert (revoke.method, revoke.url.path) == ("DELETE", REVOKE_PATH)
    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    assert revoke.headers["authorization"] == f"Basic {basic}"
    assert json.loads(revoke.content) == {"access_token": "gh-token"}


@pytest.mark.parametrize("failure", [500, httpx.ReadTimeout])
def test_failed_revocation_does_not_undo_the_sign_in(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, failure
) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == REVOKE_PATH:
            if isinstance(failure, int):
                return httpx.Response(failure)
            raise failure("slow", request=request)
        return _github(request)

    _answer_with(monkeypatch, answer)
    client = _browser(app)
    assert _login(client).headers["location"] == "/"
    assert client.get("/api/auth/me").json() == {"login": "octo"}


# ── Organization membership ──────────────────────────────────────────────────

ORG = "some org"  # the space shows the name is quoted in the URL
MEMBERS_PATH = "/orgs/some org/members/octo"


def _org_client(monkeypatch: pytest.MonkeyPatch, membership_status: int):
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == MEMBERS_PATH:
            return httpx.Response(membership_status)
        return _github(request)

    seen = _answer_with(monkeypatch, answer)
    return _browser(_build_app(monkeypatch, org=ORG)), seen


def test_org_member_signs_in(monkeypatch: pytest.MonkeyPatch) -> None:
    client, seen = _org_client(monkeypatch, 204)
    assert _login(client).headers["location"] == "/"
    assert client.get("/api/auth/me").json() == {"login": "octo"}
    assert b"/orgs/some%20org/members/octo" in seen[2].url.raw_path


@pytest.mark.parametrize("status", [404, 302])
def test_non_member_is_refused(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    client, seen = _org_client(monkeypatch, status)
    resp = _login(client)
    assert resp.status_code == 403
    assert client.get("/api/auth/me").status_code == 401
    # Revoked although the sign-in was refused.
    assert seen[-1].url.path == REVOKE_PATH


@pytest.mark.parametrize("status", [403, 429, 500, 503])
def test_unanswered_membership_check_is_not_called_non_membership(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    # 5xx: GitHub is down. 403: the organization restricts third-party apps and
    # has not approved this one. Neither says the user is not a member.
    client, _ = _org_client(monkeypatch, status)
    resp = _login(client)
    assert resp.status_code == 502
    assert resp.json() == {"detail": auth.GitHubOrgCheckError.detail}
    assert client.get("/api/auth/me").status_code == 401


# ── Middleware: login by default, CSRF, CORS ─────────────────────────────────


def test_api_routes_need_login_without_depends(client: TestClient) -> None:
    assert client.post("/api/forgot-the-dependency").status_code == 401
    assert client.get("/api/does-not-exist").status_code == 401
    _login(client)
    assert client.post("/api/forgot-the-dependency", headers=_csrf(client)).status_code == 200


def test_public_paths_work_signed_out(app: FastAPI) -> None:
    fresh = _browser(app)
    assert fresh.post("/api/auth/logout").status_code == 200
    assert fresh.get("/api/auth/github", follow_redirects=False).status_code in (302, 307)
    # Starting a login puts the OAuth state in a session cookie, so from here
    # on this browser's POSTs need the CSRF header like any signed-in one.
    assert fresh.post("/api/auth/logout").status_code == 403
    assert fresh.post("/api/auth/logout", headers=_csrf(fresh)).status_code == 200


def test_post_needs_csrf_header_once_signed_in(client: TestClient) -> None:
    _login(client)
    assert middleware.CSRF_COOKIE in client.cookies

    assert client.post("/api/forgot-the-dependency").status_code == 403
    assert client.post(
        "/api/forgot-the-dependency", headers={middleware.CSRF_HEADER: "made-up"}
    ).status_code == 403
    assert client.post("/api/auth/logout").status_code == 403  # still signed in
    assert client.get("/api/auth/me").status_code == 200

    assert client.post("/api/auth/logout", headers=_csrf(client)).status_code == 200


def test_no_credentialed_cors_for_wildcard_frontend(client: TestClient) -> None:
    _login(client)
    resp = client.get("/api/auth/me", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
    assert "access-control-allow-credentials" not in resp.headers


def test_cors_only_for_the_configured_frontend(
    monkeypatch: pytest.MonkeyPatch, github_calls: list
) -> None:
    client = _browser(_build_app(monkeypatch, frontend_url="https://app.example"))
    _login(client)

    ok = client.get("/api/auth/me", headers={"Origin": "https://app.example"})
    assert ok.headers["access-control-allow-origin"] == "https://app.example"
    assert ok.headers["access-control-allow-credentials"] == "true"

    evil = client.get("/api/auth/me", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


# ── Signing key ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value", [None, "", "   ", "change-me-in-production", "change-me-before-deploying"]
)
def test_missing_or_placeholder_secret_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    _set_github_env(monkeypatch)
    if value is None:
        monkeypatch.delenv("SESSION_SECRET", raising=False)
    else:
        monkeypatch.setenv("SESSION_SECRET", value)
    with pytest.raises(ValidationError, match="SESSION_SECRET") as refused:
        middleware.install_security_middleware(FastAPI())
    # A rejected value never reaches the logs.
    if value and value.strip():
        assert value not in str(refused.value)


def test_real_secret_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_github_env(monkeypatch)
    assert auth.AuthSettings().SESSION_SECRET.get_secret_value() == SECRET


# ── Who may sign in ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("org", [None, "", "   "])
@pytest.mark.parametrize("allow_any", [None, "false"])
def test_missing_org_refuses_to_start_unless_open_is_asked_for(
    monkeypatch: pytest.MonkeyPatch, org: str | None, allow_any: str | None
) -> None:
    # A lost or blank GITHUB_ORG must not open the site to every GitHub account.
    _set_github_env(monkeypatch)
    for name, value in (("GITHUB_ORG", org), ("ALLOW_ANY_GITHUB_USER", allow_any)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match="ALLOW_ANY_GITHUB_USER"):
        middleware.install_security_middleware(FastAPI())


def test_org_or_explicit_open_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_github_env(monkeypatch)
    settings = auth.AuthSettings()
    assert (settings.GITHUB_ORG, settings.ALLOW_ANY_GITHUB_USER) == ("", True)

    _set_github_env(monkeypatch, org=" my-lab ")
    settings = auth.AuthSettings()
    assert (settings.GITHUB_ORG, settings.ALLOW_ANY_GITHUB_USER) == ("my-lab", False)


def test_org_together_with_open_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_github_env(monkeypatch, org="my-lab")
    monkeypatch.setenv("ALLOW_ANY_GITHUB_USER", "true")
    with pytest.raises(ValidationError, match="contradict"):
        auth.AuthSettings()


@pytest.mark.parametrize("name", ["GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_oauth_app_credentials_refuse_to_start(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str | None
) -> None:
    _set_github_env(monkeypatch)
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=name):
        middleware.install_security_middleware(FastAPI())


def test_client_secret_stays_out_of_error_and_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_github_env(monkeypatch)
    assert CLIENT_SECRET not in repr(auth.AuthSettings())
