"""Feature switches are true or false, nothing else."""
import pytest

from services import switches
from services.switches import InvalidSwitchError, Switch


@pytest.fixture(autouse=True)
def _all_unset(monkeypatch):
    for switch in Switch:
        monkeypatch.delenv(switch.value, raising=False)


@pytest.mark.parametrize("switch", list(Switch))
def test_true_is_on_false_and_unset_are_off(monkeypatch, switch):
    assert switch.is_on() is False
    monkeypatch.setenv(switch.value, "true")
    assert switch.is_on() is True
    monkeypatch.setenv(switch.value, "false")
    assert switch.is_on() is False


@pytest.mark.parametrize("value", ["1", "0", "yes", "no", "on", "off", "True", "FALSE", "ture", " true", ""])
def test_any_other_value_is_an_error_that_names_the_switch(monkeypatch, value):
    monkeypatch.setenv("TEXT_SEARCH_PAPERS", value)
    with pytest.raises(InvalidSwitchError, match="TEXT_SEARCH_PAPERS must be true or false"):
        Switch.TEXT_SEARCH_PAPERS.is_on()


@pytest.mark.parametrize("switch", list(Switch))
def test_the_startup_check_reads_every_switch(monkeypatch, switch):
    switches.check()
    monkeypatch.setenv(switch.value, "1")
    with pytest.raises(InvalidSwitchError, match=switch.value):
        switches.check()


def test_startup_stops_on_a_bad_switch_before_anything_else(monkeypatch):
    import asyncio

    import main

    async def unexpected():
        raise AssertionError("the database was reached before the switches were checked")

    monkeypatch.setattr(main, "create_all_tables", unexpected)
    monkeypatch.setenv("UNCERTAIN_TIER", "on")

    async def start():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(InvalidSwitchError, match="UNCERTAIN_TIER"):
        asyncio.run(start())
