"""single_flight.py: one running call per key. The router side (users,
normalised questions, the 503) is in test_query_in_flight.py."""
from __future__ import annotations

import asyncio

import pytest

from single_flight import Interrupted, SingleFlight


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


def test_a_cancelled_running_call_interrupts_the_waiting_ones():
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
        with pytest.raises(Interrupted):
            await second

    asyncio.run(scenario())
    assert runs == ["a"]
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
