"""Feature switches are true or false, nothing else."""
import pytest
from pydantic import ValidationError

from config.settings import Settings, settings_obj
from services.switches import Switch


@pytest.fixture(autouse=True)
def _all_unset(monkeypatch):
    for switch in Switch:
        monkeypatch.setattr(settings_obj, switch.value, False)


@pytest.mark.parametrize("switch", list(Switch))
def test_true_is_on_false_and_unset_are_off(monkeypatch, switch):
    assert switch.is_on() is False
    monkeypatch.setattr(settings_obj, switch.value, True)
    assert switch.is_on() is True


@pytest.mark.parametrize("switch", list(Switch))
def test_every_switch_is_a_setting_read_from_true_or_false(monkeypatch, switch):
    monkeypatch.setenv(switch.value, "true")
    assert getattr(Settings(), switch.value) is True
    monkeypatch.setenv(switch.value, "false")
    assert getattr(Settings(), switch.value) is False
    monkeypatch.delenv(switch.value)
    assert getattr(Settings(), switch.value) is False


@pytest.mark.parametrize("value", ["1", "0", "yes", "no", "on", "off", "True", "FALSE", "ture", " true", ""])
def test_any_other_value_stops_the_start_and_names_the_switch(monkeypatch, value):
    # Settings() is what config/settings.py builds at import, before the app exists.
    monkeypatch.setenv("TEXT_SEARCH_PAPERS", value)
    with pytest.raises(ValidationError, match="TEXT_SEARCH_PAPERS"):
        Settings()
