"""Login session and API security middleware.

GitHub and the database are faked. Everything else is the real stack from
api_security.install_security_middleware: CORS, CSRF, the signed HttpOnly
session cookie and RequireLogin, in front of the real auth router.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import api_security
from routers import auth

SECRET = "test-secret-" + "x" * 40
HOST = "testserver"
CALLBACK = f"https://{HOST}/api/auth/github/callback"
_REAL_ASYNC_CLIENT = httpx.AsyncClient  # before any test patches it


def _github(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/login/oauth/access_token":
        return httpx.Response(200, json={"access_token": "gh-token"})
    if request.url.path == "/user":
        return httpx.Response(200, json={"login": "octo"})
    if request.url.path == "/user/emails":
        return httpx.Response(200, json=[{"email": "octo@example.org", "primary": True, "verified": True}])
    return httpx.Response(404)


def _build_app(monkeypatch: pytest.MonkeyPatch, frontend_url: str = "*") -> FastAPI:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("FRONTEND_URL", frontend_url)
    monkeypatch.setenv("GITHUB_REDIRECT_URI", CALLBACK)
    monkeypatch.delenv("GITHUB_ORG", raising=False)

    app = FastAPI()
    api_security.install_security_middleware(app)
    app.include_router(auth.router, prefix="/api")

    @app.get("/api/protected")
    def protected(user: dict = Depends(auth.require_auth)) -> dict:
        return {"sub": user["sub"]}

    # No Depends(require_auth): RequireLogin must still guard it.
    @app.post("/api/forgot-the-dependency")
    def forgot() -> dict:
        return {"ok": True}

    return app


@pytest.fixture
def remembered(monkeypatch: pytest.MonkeyPatch) -> list:
    monkeypatch.setattr(
        auth.httpx, "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(_github), **kw),
    )
    calls: list[tuple[str, str | None]] = []

    async def remember(login: str, email: str | None) -> None:
        calls.append((login, email))

    monkeypatch.setattr(auth, "_remember_user", remember)
    return calls


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, remembered: list) -> FastAPI:
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
    return {api_security.CSRF_HEADER: client.cookies[api_security.CSRF_COOKIE]}


# ── Session cookie ───────────────────────────────────────────────────────────


def test_callback_sets_httponly_cookie_and_keeps_token_out_of_url(
    client: TestClient, remembered: list
) -> None:
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
    assert remembered == [("octo", "octo@example.org")]


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


def test_session_ends_after_its_lifetime_even_when_used(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _login(client)
    login_time = auth._now()
    # Still in use (so the cookie keeps being re-issued) but past the cap.
    monkeypatch.setattr(auth, "_now", lambda: login_time + auth._SESSION_MAX_AGE + 60)
    assert client.get("/api/protected").status_code == 401
    monkeypatch.setattr(auth, "_now", lambda: login_time)
    assert client.get("/api/protected").status_code == 401  # session was cleared


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


def test_wrong_or_expired_state_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_login(client)
    assert _callback(client, "forged").status_code == 400

    state = _start_login(client)
    later = auth._now() + auth._OAUTH_STATE_SECONDS + 1
    monkeypatch.setattr(auth, "_now", lambda: later)
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
    assert api_security.CSRF_COOKIE in client.cookies

    assert client.post("/api/forgot-the-dependency").status_code == 403
    assert client.post(
        "/api/forgot-the-dependency", headers={api_security.CSRF_HEADER: "made-up"}
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
    monkeypatch: pytest.MonkeyPatch, remembered: list
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
    if value is None:
        monkeypatch.delenv("JWT_SECRET", raising=False)
    else:
        monkeypatch.setenv("JWT_SECRET", value)
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        api_security.install_security_middleware(FastAPI())


def test_real_secret_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    assert auth.session_secret() == SECRET
