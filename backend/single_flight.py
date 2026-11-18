"""
single_flight.py
----------------
One running call per key: a second call with the same key while the first
runs does not start another, it waits for the first one's result or error.
routers/query.py uses it so that a double submit, a retry or a second tab
gets the running search's answer instead of starting (and paying for) a
second search.

The running calls live in a dict in memory, so this holds within one process
only (tests/test_single_process.py keeps the API at one).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Hashable
from typing import Generic, TypeVar

T = TypeVar("T")


class Interrupted(Exception):
    """The running call this one waited for was cancelled before it finished
    (its client went away, or the server is shutting down)."""


class SingleFlight(Generic[T]):
    def __init__(self) -> None:
        self._running: dict[Hashable, asyncio.Future[T]] = {}

    def __len__(self) -> int:
        """How many calls are running now."""
        return len(self._running)

    async def run(self, key: Hashable, start: Callable[[], Awaitable[T]]) -> T:
        """The result of ``start()``, or of the call with this key already running.

        An error of the running call is raised to every caller waiting for it.
        A waiting caller that is cancelled itself stops waiting without
        cancelling the running call; a running call that is cancelled raises
        Interrupted to the callers waiting for it.
        """
        running = self._running.get(key)
        if running is not None:
            try:
                return await asyncio.shield(running)
            except asyncio.CancelledError:
                if running.cancelled():
                    raise Interrupted from None
                raise

        answer: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        # Marks an error as seen when no second call was waiting for it.
        answer.add_done_callback(lambda f: f.cancelled() or f.exception())
        self._running[key] = answer
        try:
            result = await start()
        except asyncio.CancelledError:
            answer.cancel()
            raise
        except BaseException as exc:
            answer.set_exception(exc)
            raise
        else:
            answer.set_result(result)
            return result
        finally:
            self._running.pop(key, None)
