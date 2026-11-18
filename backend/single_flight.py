"""
single_flight.py
----------------
One running call per key: a second call with the same key while the first
runs does not start another, it waits for the first one's result or error.
services/search.py uses it so that a double submit, a retry, a second tab or
another user asking the same question gets the running search's answer
instead of starting (and paying for) a second search.

If the running call is cancelled (in practice only when the server shuts
down; a client that goes away does not cancel its request), a caller that was
waiting for it and is not cancelled itself runs the call instead: it did
nothing wrong and should get its answer. The first such caller takes over the
key; the others wait for it.

The running calls live in a dict in memory, so this holds within one process
only (tests/test_single_process.py keeps the API at one).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Hashable
from typing import Generic, TypeVar

T = TypeVar("T")

# How often one caller waits for someone else's call before it stops waiting
# and runs its own, unshared. Every wait after the first needs another
# cancelled call, so this is never reached in practice; it guarantees that
# run() ends whatever state the dict is in.
MAX_WAITS = 3


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
        cancelling the running call; a running call that is cancelled is run
        again by the callers that waited for it (one runs it, the rest wait).
        """
        for _ in range(MAX_WAITS):
            running = self._running.get(key)
            if running is None:
                return await self._run_shared(key, start)
            try:
                return await asyncio.shield(running)
            except asyncio.CancelledError:
                if not running.cancelled() or _this_task_is_cancelled():
                    raise
                # The running call was cancelled, this caller was not: look
                # again; the first caller back finds the key free and runs it.
        return await start()

    async def _run_shared(self, key: Hashable, start: Callable[[], Awaitable[T]]) -> T:
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
            # Only its own entry. Today nothing can take the key over between
            # resolving *answer* and this line (no await in between); the
            # check keeps it so if that ever changes.
            if self._running.get(key) is answer:
                del self._running[key]


def _this_task_is_cancelled() -> bool:
    """Whether the current task was asked to cancel (not just woken by a
    cancelled future it awaited)."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
