"""
The API must run as one process. Three things keep their state in a dict in
memory and are only correct when every request reaches the same process:

- the query cache (services/stores/translated_queries.py): paging a query_id made by
  another process fails,
- the rate-limit counts (security/rate_limits.py): each process would count on its own,
- the searches running now (services/search/search.py ``_in_flight``): a repeated
  question on another process would start a second search.

These tests read the Dockerfile and the Compose files and fail when one of them
would start a second API process: uvicorn ``--workers`` or ``WEB_CONCURRENCY``
above 1, Compose ``replicas``/``scale`` above 1, or a second service serving
``main:app``. Moving to several processes means moving those three stores to a
shared one (Redis or the database) first, and then changing these tests.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "backend" / "Dockerfile"
COMPOSE_FILES = sorted(ROOT.glob("docker-compose*.yml"))
# The server runs ``docker compose -f docker-compose-prod.yml``; its fixed container_name is
# also what makes ``docker compose up --scale backend=2`` fail there.
PRODUCTION_COMPOSE = ROOT / "docker-compose-prod.yml"
ENV_EXAMPLE = ROOT / ".env.example"

if not DOCKERFILE.is_file() or not COMPOSE_FILES:
    pytest.skip("Dockerfile/Compose files are not next to this checkout (e.g. inside the image)",
                allow_module_level=True)


def _dockerfile_cmd() -> list[str]:
    """The last CMD of backend/Dockerfile, as an argument list."""
    cmds = [line.strip()[len("CMD"):].strip()
            for line in DOCKERFILE.read_text().splitlines()
            if line.strip().upper().startswith("CMD ")]
    assert cmds, "backend/Dockerfile has no CMD"
    cmd = cmds[-1]
    return json.loads(cmd) if cmd.startswith("[") else shlex.split(cmd)


def _workers(args: list[str]) -> int:
    """The worker count uvicorn would start with these arguments (1 without --workers)."""
    for i, arg in enumerate(args):
        if arg == "--workers":
            return int(args[i + 1])
        if arg.startswith("--workers="):
            return int(arg.split("=", 1)[1])
    return 1


def _as_args(command) -> list[str]:
    return command if isinstance(command, list) else shlex.split(command)


def _environment(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        env = dict(item.split("=", 1) if "=" in item else (item, "") for item in env)
    return {str(k): str(v) for k, v in env.items()}


def _api_services(compose: dict) -> dict[str, dict]:
    """Services that serve main:app: built from backend/Dockerfile, or with a command naming main:app."""
    found = {}
    for name, service in (compose.get("services") or {}).items():
        build = service.get("build") or {}
        dockerfile = build.get("dockerfile", "") if isinstance(build, dict) else ""
        command = service.get("command") or []
        if dockerfile.endswith("backend/Dockerfile") or "main:app" in " ".join(_as_args(command)):
            found[name] = service
    return found


def test_dockerfile_starts_one_uvicorn_process():
    args = _dockerfile_cmd()
    assert args[0] == "uvicorn", f"backend/Dockerfile must start uvicorn directly, not {args[0]!r}"
    assert "main:app" in args
    assert _workers(args) == 1, f"backend/Dockerfile CMD starts {_workers(args)} workers"


def test_dockerfile_sets_no_worker_count():
    text = DOCKERFILE.read_text()
    assert not re.search(r"\bWEB_CONCURRENCY\b", text), "uvicorn reads WEB_CONCURRENCY as --workers"


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: p.name)
def test_compose_runs_one_api_process(path: Path):
    compose = yaml.safe_load(path.read_text())
    api = _api_services(compose)
    assert len(api) == 1, f"{path.name}: expected one service serving main:app, found {sorted(api)}"

    (name, service), = api.items()
    replicas = (service.get("deploy") or {}).get("replicas", 1)
    assert int(replicas) == 1, f"{path.name}: {name} has deploy.replicas {replicas}"
    assert int(service.get("scale", 1)) == 1, f"{path.name}: {name} has scale {service['scale']}"

    command = service.get("command")
    if command:
        args = _as_args(command)
        assert args[0] == "uvicorn", f"{path.name}: {name} must start uvicorn directly, not {args[0]!r}"
        assert _workers(args) == 1, f"{path.name}: {name} command starts {_workers(args)} workers"

    concurrency = _environment(service).get("WEB_CONCURRENCY")
    assert concurrency in (None, "", "1"), f"{path.name}: {name} sets WEB_CONCURRENCY={concurrency}"


def test_production_backend_cannot_be_scaled():
    compose = yaml.safe_load(PRODUCTION_COMPOSE.read_text())
    service = compose["services"]["backend"]
    assert service.get("container_name"), (
        "the production backend needs a fixed container_name, which makes --scale backend=N fail"
    )


def test_env_example_sets_no_worker_count():
    if not ENV_EXAMPLE.is_file():
        pytest.skip(".env.example is missing")
    for line in ENV_EXAMPLE.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "WEB_CONCURRENCY":
            assert value.strip() in ("", "1"), f".env.example sets WEB_CONCURRENCY={value.strip()}"
