"""The central rate limiter (rate_limits.py): one budget per kind of request."""
from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

import rate_limits
from exceptions.handlers import install_error_handlers
from rate_limits import DAY, MINUTE, Category, rate_limit

# Noon UTC on some day: far from both a minute and a day boundary.
NOON = datetime(2026, 11, 18, 12, 0, 30, tzinfo=timezone.utc).timestamp()


class _SignedIn:
    """Stands in for RequireLoginMiddleware: the X-Test-User header is the user."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            user = dict(scope["headers"]).get(b"x-test-user")
            if user:
                scope.setdefault("state", {})["user"] = {"sub": user.decode()}
        await self.app(scope, receive, send)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [NOON]
    monkeypatch.setattr(rate_limits, "_now", lambda: now[0])
    rate_limits.reset()
    yield now
    rate_limits.reset()


@pytest.fixture
def client(clock: list[float]) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(_SignedIn)

    @app.post("/api/search", dependencies=[rate_limit(Category.SEARCH)])
    def search() -> dict:
        return {"ok": True}

    @app.post("/api/page", dependencies=[rate_limit(Category.PAGE)])
    def page() -> dict:
        return {"ok": True}

    @app.get("/api/me", dependencies=[rate_limit(Category.LIGHT)])
    def me() -> dict:
        return {"ok": True}

    @app.get("/api/login", dependencies=[rate_limit(Category.LOGIN)])
    def login(request: Request) -> dict:
        return {"ok": True}

    return TestClient(app)


def _as(user: str) -> dict:
    return {"x-test-user": user}


def _limit(category: Category, seconds: int) -> int:
    return next(w.limit for w in rate_limits.LIMITS[category] if w.seconds == seconds)


def test_categories_have_separate_budgets(client: TestClient, clock: list[float]) -> None:
    per_minute = _limit(Category.SEARCH, MINUTE)
    for _ in range(per_minute):
        assert client.post("/api/search", headers=_as("ana")).status_code == 200

    refused = client.post("/api/search", headers=_as("ana"))
    assert refused.status_code == 429
    detail = refused.json()["detail"]
    assert (detail["category"], detail["window"], detail["limit"]) == ("search", "minute", per_minute)
    assert 1 <= int(refused.headers["retry-after"]) <= 60

    # Searching is used up for this minute; refreshing and paging are not.
    assert client.get("/api/me", headers=_as("ana")).status_code == 200
    assert client.post("/api/page", headers=_as("ana")).status_code == 200

    clock[0] += MINUTE
    assert client.post("/api/search", headers=_as("ana")).status_code == 200


def test_light_requests_get_a_much_larger_budget(client: TestClient) -> None:
    light = _limit(Category.LIGHT, MINUTE)
    assert light > _limit(Category.PAGE, MINUTE) > _limit(Category.SEARCH, MINUTE)
    for _ in range(light):
        assert client.get("/api/me", headers=_as("ana")).status_code == 200
    assert client.get("/api/me", headers=_as("ana")).status_code == 429


def test_daily_search_cap_resets_at_utc_midnight(client: TestClient, clock: list[float]) -> None:
    daily = _limit(Category.SEARCH, DAY)
    assert daily == 20
    for i in range(daily):
        clock[0] = NOON + i * MINUTE  # spread out: the minute window never fills
        assert client.post("/api/search", headers=_as("ana")).status_code == 200

    clock[0] = NOON + daily * MINUTE
    refused = client.post("/api/search", headers=_as("ana"))
    assert refused.status_code == 429
    detail = refused.json()["detail"]
    assert (detail["window"], detail["limit"]) == ("day", daily)
    assert detail["retry_after"] == "2026-11-19T00:00:00+00:00"

    clock[0] = datetime(2026, 11, 19, 0, 0, 1, tzinfo=timezone.utc).timestamp()
    assert client.post("/api/search", headers=_as("ana")).status_code == 200


def test_refused_request_uses_nothing(client: TestClient, clock: list[float]) -> None:
    per_minute = _limit(Category.SEARCH, MINUTE)
    for _ in range(per_minute):
        client.post("/api/search", headers=_as("ana"))
    for _ in range(5):
        assert client.post("/api/search", headers=_as("ana")).status_code == 429
    # The five refusals did not count against the day: only per_minute searches did.
    key = ("user:ana", Category.SEARCH, DAY)
    assert rate_limits._counts[key][1] == per_minute


def test_admins_skip_only_the_daily_cap(
    client: TestClient, clock: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rate_limits, "ADMIN_LOGINS", frozenset({"boss"}))
    daily = _limit(Category.SEARCH, DAY)
    for i in range(daily + 5):
        clock[0] = NOON + i * MINUTE
        assert client.post("/api/search", headers=_as("boss")).status_code == 200

    per_minute = _limit(Category.SEARCH, MINUTE)
    for _ in range(per_minute - 1):
        client.post("/api/search", headers=_as("boss"))
    assert client.post("/api/search", headers=_as("boss")).status_code == 429


def test_each_user_has_their_own_budget(client: TestClient) -> None:
    for _ in range(_limit(Category.SEARCH, MINUTE)):
        client.post("/api/search", headers=_as("ana"))
    assert client.post("/api/search", headers=_as("ana")).status_code == 429
    assert client.post("/api/search", headers=_as("ben")).status_code == 200


def test_signed_out_requests_count_per_client_address(client: TestClient) -> None:
    per_minute = _limit(Category.LOGIN, MINUTE)
    for _ in range(per_minute):
        assert client.get("/api/login", headers={"x-real-ip": "203.0.113.5"}).status_code == 200
    assert client.get("/api/login", headers={"x-real-ip": "203.0.113.5"}).status_code == 429
    assert client.get("/api/login", headers={"x-real-ip": "198.51.100.7"}).status_code == 200


def test_old_windows_are_pruned(client: TestClient, clock: list[float]) -> None:
    client.get("/api/me", headers=_as("ana"))
    clock[0] += 2 * MINUTE
    rate_limits._prune(clock[0])
    assert ("user:ana", Category.LIGHT, MINUTE) not in rate_limits._counts


def test_every_api_route_declares_exactly_one_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """New routes cannot be added without a category: the app is built as in production."""
    monkeypatch.setenv("JWT_SECRET", "test-secret-" + "x" * 40)
    monkeypatch.setenv("CRAWLER_ENABLED", "true")
    for name in ("METADATA_DB_NAME", "METADATA_DB_USER", "METADATA_DB_PASSWORD", "METADATA_DB_HOST"):
        monkeypatch.setenv(name, "x")
    main = importlib.import_module("main")
    if not any(r.path.startswith("/api/crawler") for r in main.app.routes):
        pytest.skip("main was imported earlier without CRAWLER_ENABLED")

    api_routes = [r for r in main.app.routes if isinstance(r, APIRoute) and r.path.startswith("/api")]
    assert len(api_routes) >= 7
    for route in api_routes:
        limits = [d for d in route.dependencies if isinstance(d.dependency, rate_limits.RateLimit)]
        assert len(limits) == 1, f"{sorted(route.methods)} {route.path} declares {len(limits)} rate limits"
