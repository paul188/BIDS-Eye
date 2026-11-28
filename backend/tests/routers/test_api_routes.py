"""Rules every route of the real app keeps, read from main.app itself, so a
new route is checked without a test of its own.

The login is asked for twice: RequireLoginMiddleware refuses a signed-out
request before routing (security/middleware.py), and each route names require_auth.
Either alone would do; a route that loses one is still closed.
"""
from __future__ import annotations

import importlib
import re

import pytest
from fastapi import FastAPI
from fastapi.dependencies.utils import get_flat_dependant
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from deps import require_auth
from security.middleware import PUBLIC_API_PATHS
from security.sessions import NotSignedInError

# Who is asking comes from the session only. A route that took one of these
# from the request would let a user name someone else.
IDENTITY_NAMES = {"uid", "user", "user_id", "github_id", "github_login", "sub"}


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """main.app, built as in production."""
    monkeypatch.setenv("SESSION_SECRET", "test-secret-" + "x" * 40)
    monkeypatch.setenv("GITHUB_CLIENT_ID", "x")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "x")
    monkeypatch.setenv("ALLOW_ANY_GITHUB_USER", "true")
    monkeypatch.delenv("GITHUB_ORG", raising=False)
    for name in ("METADATA_DB_NAME", "METADATA_DB_USER", "METADATA_DB_PASSWORD", "METADATA_DB_HOST"):
        monkeypatch.setenv(name, "x")
    return importlib.import_module("main").app


def _api_routes(app: FastAPI) -> list[APIRoute]:
    routes = [r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api")]
    assert len(routes) >= 6
    return routes


def _closed_routes(app: FastAPI) -> list[APIRoute]:
    return [r for r in _api_routes(app) if r.path not in PUBLIC_API_PATHS]


def _asks_for_login(route: APIRoute) -> bool:
    return any(d.call is require_auth for d in get_flat_dependant(route.dependant).dependencies)


def test_a_signed_out_request_is_refused_at_every_route(app: FastAPI) -> None:
    client = TestClient(app)   # not entered: no startup, no database
    closed = _closed_routes(app)
    assert closed
    for route in closed:
        path = re.sub(r"\{[^}]+\}", "x", route.path)
        for method in route.methods:
            reply = client.request(method, path)
            assert reply.status_code == NotSignedInError.status_code == 401, f"{method} {route.path}"
            assert reply.json() == {"detail": NotSignedInError.detail}


def test_every_closed_route_names_the_login_itself(app: FastAPI) -> None:
    missing = [f"{sorted(r.methods)} {r.path}" for r in _closed_routes(app) if not _asks_for_login(r)]
    assert missing == []


def test_the_public_paths_are_routes_that_ask_for_no_login(app: FastAPI) -> None:
    by_path = {r.path: r for r in _api_routes(app)}
    assert PUBLIC_API_PATHS <= set(by_path), "a public path that is no route"
    assert [path for path in PUBLIC_API_PATHS if _asks_for_login(by_path[path])] == []


def test_no_route_takes_the_asker_from_the_request(app: FastAPI) -> None:
    spec = app.openapi()
    schemas = spec.get("components", {}).get("schemas", {})
    taken = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            names = {p["name"] for p in operation.get("parameters", [])}
            for content in operation.get("requestBody", {}).get("content", {}).values():
                ref = content.get("schema", {}).get("$ref", "")
                names |= set(schemas.get(ref.rsplit("/", 1)[-1], {}).get("properties", {}))
            taken += [f"{method.upper()} {path}: {name}" for name in sorted(names & IDENTITY_NAMES)]
    assert taken == []
