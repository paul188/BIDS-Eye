"""One rate limiter for the whole API, with a budget per kind of request.

Every /api route declares what it costs by depending on exactly one limiter
(tests/test_rate_limits.py checks that no route is left out):

    @router.post("/page", dependencies=[rate_limit(Category.PAGE)])

A search runs a language-model call and costs money, a page of results runs
SQL, a page refresh or a status poll costs nearly nothing, so each category
has its own windows instead of one shared quota:

    LIGHT   120 per minute         /auth/me, /auth/logout, /crawler/status
    LOGIN    10 per minute         /auth/github, /auth/github/callback (call GitHub)
    PAGE     60 per minute         /query/page
    SEARCH   10 per minute and DAILY_MESSAGE_LIMIT (20) per UTC day   POST /query

Requests are counted per signed-in user, or per client address for the few
public routes. ADMIN_GITHUB_LOGINS are exempt from the daily search cap only.

Counts live in memory: one uvicorn process serves the API (backend/Dockerfile,
like the query cache and the in-flight searches in routers/query.py), so a
dict sees every request (tests/test_single_process.py fails if the Dockerfile or
a Compose file would start more). A restart starts all counts from zero.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from fastapi import Depends, Request

from exceptions.domain import DomainError

MINUTE = 60
DAY = 24 * 3600


class Category(Enum):
    LIGHT = "light"
    LOGIN = "login"
    PAGE = "page"
    SEARCH = "search"


@dataclass(frozen=True)
class Window:
    limit: int
    seconds: int
    admins_exempt: bool = False

    @property
    def name(self) -> str:
        return "day" if self.seconds == DAY else "minute" if self.seconds == MINUTE else f"{self.seconds}s"


def _daily_search_limit() -> int:
    return int(os.getenv("DAILY_MESSAGE_LIMIT", "20"))


def _admin_logins() -> frozenset[str]:
    raw = os.getenv("ADMIN_GITHUB_LOGINS", "paul188")
    return frozenset(u.strip() for u in raw.split(",") if u.strip())


LIMITS: dict[Category, tuple[Window, ...]] = {
    Category.LIGHT: (Window(120, MINUTE),),
    Category.LOGIN: (Window(10, MINUTE),),
    Category.PAGE: (Window(60, MINUTE),),
    Category.SEARCH: (
        Window(10, MINUTE),
        Window(_daily_search_limit(), DAY, admins_exempt=True),
    ),
}
ADMIN_LOGINS = _admin_logins()


class RateLimitError(DomainError, status_code=429, detail={"code": "rate_limit_exceeded"}):
    """A window of the request's category is used up.

    ``detail`` says which (code, category, window, limit, retry_after; read by
    the frontend's class of the same name in api/index.ts) and the
    Retry-After header when, so both are passed when it is raised.
    """


# (client, category, window seconds) -> (window number, requests counted in it).
# Fixed windows aligned to the epoch, so the day window resets at UTC midnight.
_counts: dict[tuple[str, Category, int], tuple[int, int]] = {}
_PRUNE_EVERY = 1000
_calls_since_prune = 0


def _client(request: Request) -> tuple[str, bool]:
    """Who to count against, and whether that is an admin."""
    user = getattr(request.state, "user", None)
    if isinstance(user, dict) and user.get("sub"):
        return f"user:{user['sub']}", user["sub"] in ADMIN_LOGINS
    # nginx overwrites X-Real-IP with the real address; the backend's port is
    # not published, so nothing else can reach it to set the header.
    address = request.headers.get("x-real-ip") or (request.client.host if request.client else "?")
    return f"ip:{address}", False


def _prune(now: float) -> None:
    stale = [key for key, (number, _) in _counts.items() if number < int(now // key[2])]
    for key in stale:
        del _counts[key]


class RateLimit:
    """The dependency behind rate_limit(category). Async on purpose: it runs on
    the event loop, so checking and counting cannot interleave between requests."""

    def __init__(self, category: Category) -> None:
        self.category = category

    async def __call__(self, request: Request) -> None:
        global _calls_since_prune
        client, is_admin = _client(request)
        now = time.time()
        windows = [w for w in LIMITS[self.category] if not (is_admin and w.admins_exempt)]

        # Check every window before counting, so a refused request uses nothing.
        for window in windows:
            number = int(now // window.seconds)
            seen_number, count = _counts.get((client, self.category, window.seconds), (number, 0))
            if seen_number == number and count >= window.limit:
                reset_at = (number + 1) * window.seconds
                raise RateLimitError(
                    f"{client} used {window.limit} {self.category.value} per {window.name}",
                    detail={
                        "code": "rate_limit_exceeded",
                        "category": self.category.value,
                        "window": window.name,
                        "limit": window.limit,
                        "retry_after": datetime.fromtimestamp(reset_at, timezone.utc).isoformat(),
                    },
                    headers={"Retry-After": str(max(1, int(reset_at - now) + 1))},
                )

        for window in windows:
            number = int(now // window.seconds)
            key = (client, self.category, window.seconds)
            seen_number, count = _counts.get(key, (number, 0))
            _counts[key] = (number, count + 1 if seen_number == number else 1)

        _calls_since_prune += 1
        if _calls_since_prune >= _PRUNE_EVERY:
            _calls_since_prune = 0
            _prune(now)


def rate_limit(category: Category):
    """The Depends(...) a route lists in its dependencies to declare its cost."""
    return Depends(RateLimit(category))


def reset() -> None:
    """Forget all counts (tests)."""
    _counts.clear()
