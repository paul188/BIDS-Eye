"""Every Gemini call carries a fixed seed, so a repeated question repeats its sample."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend" / "llm"))

import llm_client  # noqa: E402
from google import genai  # noqa: E402


@pytest.fixture
def sent_configs(monkeypatch):
    """The GenerateContentConfig of each call, captured instead of sent."""
    configs = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            configs.append(config)
            return SimpleNamespace(text="ok", usage_metadata=None)

    class FakeClient:
        def __init__(self, **_):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    return configs


def _call():
    return llm_client._call_gemini("gemini-2.5-flash", "q", None, 0.0, "key")


def test_default_seed_is_sent(monkeypatch, sent_configs):
    monkeypatch.delenv("GEMINI_SEED", raising=False)
    assert _call() == "ok"
    assert sent_configs[0].seed == 0
    assert sent_configs[0].temperature == 0.0


def test_seed_from_the_environment(monkeypatch, sent_configs):
    monkeypatch.setenv("GEMINI_SEED", "42")
    _call()
    assert sent_configs[0].seed == 42


@pytest.mark.parametrize("value", ["none", "off", ""])
def test_seed_can_be_switched_off(monkeypatch, sent_configs, value):
    monkeypatch.setenv("GEMINI_SEED", value)
    _call()
    assert sent_configs[0].seed is None


def test_a_bad_seed_falls_back_to_the_default(monkeypatch, sent_configs):
    monkeypatch.setenv("GEMINI_SEED", "abc")
    _call()
    assert sent_configs[0].seed == 0
