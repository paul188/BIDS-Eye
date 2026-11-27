"""What the provider adapters send, from the model's capabilities in the config.

Gemini calls carry the configured seed, so a repeated question repeats its sample.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend" / "llm"))

import llm_client  # noqa: E402
from google import genai  # noqa: E402

SCHEMA = {"type": "object"}


@pytest.fixture
def sent(monkeypatch):
    """The (contents, GenerateContentConfig) of each Gemini call, captured instead of sent."""
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append((contents, config))
            return SimpleNamespace(text="ok", usage_metadata=None)

    class FakeClient:
        def __init__(self, **_):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    return calls


def _gemini(provider_seed=0, **model):
    spec = llm_client.ModelSpec(name="gemini-test", provider="gemini", **{"seed": True, **model})
    return llm_client.GeminiProvider(["GEMINI_API_KEY"], provider_seed), spec


def test_the_configured_seed_is_sent(sent):
    provider, spec = _gemini(provider_seed=0)
    assert provider.generate(spec, llm_client.LLMRequest("q"), timeout=12).text == "ok"
    assert sent[0][1].seed == 0
    assert sent[0][1].temperature == 0.0


def test_no_seed_when_the_config_sets_none_or_the_model_takes_none(sent):
    provider, spec = _gemini(provider_seed=None)
    provider.generate(spec, llm_client.LLMRequest("q"), timeout=12)
    provider, spec = _gemini(provider_seed=42, seed=False)
    provider.generate(spec, llm_client.LLMRequest("q"), timeout=12)
    assert [c[1].seed for c in sent] == [None, None]


def test_the_shipped_seed_is_zero():
    cfg = llm_client.load_config(llm_client.DEFAULT_CONFIG_PATH)
    assert cfg.providers["gemini"].seed == 0
    assert cfg.models["gemini-2.5-flash"].seed


def test_thinking_level_only_for_models_that_take_one(sent):
    req = llm_client.LLMRequest("q", thinking_level="low")
    provider, level = _gemini(thinking="level")
    provider.generate(level, req, timeout=12)
    provider, none = _gemini(thinking="none")
    provider.generate(none, req, timeout=12)
    assert sent[0][1].thinking_config is not None
    assert sent[1][1].thinking_config is None


def test_a_native_schema_goes_to_the_api_a_prompt_schema_into_the_prompt(sent):
    req = llm_client.LLMRequest("q", json_schema=SCHEMA, max_tokens=99)
    provider, native = _gemini(json_schema="native")
    provider.generate(native, req, timeout=12)
    provider, prompt = _gemini(json_schema="prompt")
    provider.generate(prompt, req, timeout=12)
    (contents0, cfg0), (contents1, cfg1) = sent
    assert cfg0.response_json_schema == SCHEMA and contents0 == "q"
    assert cfg1.response_json_schema is None and "JSON Schema" in contents1
    assert cfg0.max_output_tokens == cfg1.max_output_tokens == 99


@pytest.fixture
def anthropic_sent(monkeypatch):
    import anthropic
    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], usage=None)

    class FakeAnthropic:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    return calls


def test_anthropic_leaves_out_temperature_where_the_model_rejects_it(anthropic_sent):
    provider = llm_client.AnthropicProvider(["ANTHROPIC_API_KEY"])
    req = llm_client.LLMRequest("q", temperature=0.2, json_schema=SCHEMA)
    provider.generate(llm_client.ModelSpec("claude-a", "anthropic"), req, timeout=20)
    provider.generate(llm_client.ModelSpec("claude-b", "anthropic", temperature=False), req, timeout=20)
    client0, call0, client1, call1 = anthropic_sent
    assert client0 == {"api_key": "key", "timeout": 20, "max_retries": 0}
    assert call0["temperature"] == 0.2 and "temperature" not in call1
    assert "JSON Schema" in call0["messages"][0]["content"]
