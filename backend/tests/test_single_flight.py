"""single_flight.py: one running call per key. The router side (users,
normalised questions) is in test_query_in_flight.py."""
from __future__ import annotations

import asyncio
import threading

import pytest

import single_flight
from single_flight import SingleFlight


def _slow(runs: list[str], value: str, *, gate: asyncio.Event | None = None, fail: Exception | None = None):
    async def start():
        runs.append(value)
        if gate is not None:
            await gate.wait()
        else:
            await asyncio.sleep(0.02)
        if fail is not None:
            raise fail
        return value
    return lambda: start()


def test_a_second_call_with_the_same_key_waits_for_the_first():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []

    async def scenario():
        return await asyncio.gather(flight.run("k", _slow(runs, "a")), flight.run("k", _slow(runs, "b")))

    assert asyncio.run(scenario()) == ["a", "a"]
    assert runs == ["a"]
    assert len(flight) == 0


def test_other_keys_run_on_their_own():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []

    async def scenario():
        return await asyncio.gather(flight.run("k", _slow(runs, "a")), flight.run("j", _slow(runs, "b")))

    assert asyncio.run(scenario()) == ["a", "b"]
    assert runs == ["a", "b"]


def test_an_error_reaches_every_waiting_call():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []
    boom = ValueError("boom")

    async def scenario():
        return await asyncio.gather(flight.run("k", _slow(runs, "a", fail=boom)),
                                    flight.run("k", _slow(runs, "b")), return_exceptions=True)

    assert asyncio.run(scenario()) == [boom, boom]
    assert len(flight) == 0


def _bounded(scenario):
    """Run *scenario* and fail, instead of hanging, if it does not end. A loop
    over an already cancelled future never yields to the event loop, so an
    asyncio timeout could not stop it; a thread with a join timeout can."""
    outcome: dict = {}

    def target():
        try:
            outcome["value"] = asyncio.run(scenario())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "SingleFlight.run() did not end"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def test_a_waiting_call_runs_it_itself_when_the_running_one_is_cancelled():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []
    gate = asyncio.Event()

    async def scenario():
        first = asyncio.create_task(flight.run("k", _slow(runs, "a", gate=gate)))
        await asyncio.sleep(0)
        second = asyncio.create_task(flight.run("k", _slow(runs, "b")))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        return await second

    assert _bounded(scenario) == "b"
    assert runs == ["a", "b"]
    assert len(flight) == 0


def test_several_waiting_calls_share_one_takeover():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []
    gate = asyncio.Event()

    async def scenario():
        first = asyncio.create_task(flight.run("k", _slow(runs, "a", gate=gate)))
        await asyncio.sleep(0)
        waiting = [asyncio.create_task(flight.run("k", _slow(runs, v))) for v in "bcd"]
        await asyncio.sleep(0)
        first.cancel()
        return await asyncio.gather(*waiting)

    results = _bounded(scenario)
    assert runs == ["a", "b"]           # one new run, not three
    assert results == ["b", "b", "b"]
    assert len(flight) == 0


def test_a_cancelled_waiting_call_does_not_take_over():
    """Both cancelled together: the waiting call ends cancelled, it does not
    start the call again."""
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []
    gate = asyncio.Event()

    async def scenario():
        first = asyncio.create_task(flight.run("k", _slow(runs, "a", gate=gate)))
        await asyncio.sleep(0)
        second = asyncio.create_task(flight.run("k", _slow(runs, "b")))
        await asyncio.sleep(0)
        second.cancel()
        first.cancel()
        for task in (first, second):
            with pytest.raises(asyncio.CancelledError):
                await task

    _bounded(scenario)
    assert runs == ["a"]
    assert len(flight) == 0


def test_a_call_that_keeps_finding_a_cancelled_one_stops_waiting():
    """A cancelled entry left in the dict (run() never leaves one) would make
    every wait end at once; the cap ends the loop and the call runs itself."""
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []

    async def scenario():
        stale = asyncio.get_running_loop().create_future()
        stale.cancel()
        flight._running["k"] = stale
        return await flight.run("k", _slow(runs, "b"))

    assert _bounded(scenario) == "b"
    assert runs == ["b"]
    assert single_flight.MAX_WAITS >= 1


def test_a_call_after_the_takeover_waits_for_it():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []
    gate_a, gate_b = asyncio.Event(), asyncio.Event()

    async def scenario():
        first = asyncio.create_task(flight.run("k", _slow(runs, "a", gate=gate_a)))
        await asyncio.sleep(0)
        second = asyncio.create_task(flight.run("k", _slow(runs, "b", gate=gate_b)))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.sleep(0)
        assert len(flight) == 1          # the takeover holds the key
        third = asyncio.create_task(flight.run("k", _slow(runs, "c")))
        await asyncio.sleep(0)
        gate_b.set()
        return await asyncio.gather(second, third)

    assert _bounded(scenario) == ["b", "b"]
    assert runs == ["a", "b"]
    assert len(flight) == 0


def test_a_cancelled_waiting_call_leaves_the_running_one_alone():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []
    gate = asyncio.Event()

    async def scenario():
        first = asyncio.create_task(flight.run("k", _slow(runs, "a", gate=gate)))
        await asyncio.sleep(0)
        second = asyncio.create_task(flight.run("k", _slow(runs, "b")))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        gate.set()
        return await first

    assert asyncio.run(scenario()) == "a"
    assert runs == ["a"]
    assert len(flight) == 0


def test_after_a_call_finished_the_key_runs_again():
    flight: SingleFlight[str] = SingleFlight()
    runs: list[str] = []
    asyncio.run(flight.run("k", _slow(runs, "a")))
    asyncio.run(flight.run("k", _slow(runs, "b")))
    assert runs == ["a", "b"]
