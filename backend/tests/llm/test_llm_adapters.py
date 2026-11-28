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


def test_gemini_drops_the_schema_keywords_it_does_not_accept(sent):
    schema = {"type": "object", "properties": {"n": {"type": "integer", "default": 3}},
              "default": {}}
    provider, spec = _gemini(json_schema="native")
    provider.generate(spec, llm_client.LLMRequest("q", json_schema=schema), timeout=12)
    assert sent[0][1].response_json_schema == {"type": "object",
                                               "properties": {"n": {"type": "integer"}}}


# ── OpenAI-compatible servers (local models) ──────────────────────────────────

class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture
def posted(monkeypatch):
    import httpx
    calls, answers = [], []

    def post(url, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return answers.pop(0)
    monkeypatch.setattr(httpx, "post", post)
    return calls, answers


def _ok(text="ok", usage=None):
    return _Response(200, {"choices": [{"message": {"content": text}}], "usage": usage})


def test_openai_compatible_sends_a_chat_completion(posted):
    calls, answers = posted
    answers.append(_ok(usage={"prompt_tokens": 10, "completion_tokens": 2,
                              "prompt_tokens_details": {"cached_tokens": 4}}))
    provider = llm_client.OpenAICompatibleProvider("box", "http://gpu:11434/v1/", seed=7)
    spec = llm_client.ModelSpec("qwen3:32b", "box", seed=True, json_schema="native")
    req = llm_client.LLMRequest("q", system="s", temperature=0.1, max_tokens=50, json_schema=SCHEMA)
    result = provider.generate(spec, req, timeout=30)

    assert result.text == "ok"
    assert result.usage == {"input_tokens": 10, "cached_tokens": 4, "output_tokens": 2,
                            "thinking_tokens": 0}
    call = calls[0]
    assert call["url"] == "http://gpu:11434/v1/chat/completions" and call["timeout"] == 30
    assert call["headers"] == {}   # a local server needs no key
    body = call["json"]
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert (body["model"], body["temperature"], body["max_tokens"], body["seed"]) == (
        "qwen3:32b", 0.1, 50, 7)
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_openai_compatible_key_and_prompt_schema(posted, monkeypatch):
    calls, answers = posted
    answers.append(_ok())
    monkeypatch.setenv("BOX_KEY", "secret")
    provider = llm_client.OpenAICompatibleProvider("box", "https://api.example/v1", ["BOX_KEY"])
    spec = llm_client.ModelSpec("m", "box", temperature=False)
    provider.generate(spec, llm_client.LLMRequest("q", json_schema=SCHEMA), timeout=5)
    body = calls[0]["json"]
    assert calls[0]["headers"] == {"Authorization": "Bearer secret"}
    assert "temperature" not in body and "response_format" not in body
    assert "JSON Schema" in body["messages"][0]["content"]


def test_openai_compatible_needs_its_key_when_it_names_one(monkeypatch):
    monkeypatch.delenv("BOX_KEY", raising=False)
    assert llm_client.OpenAICompatibleProvider("box", "http://x").available()
    assert not llm_client.OpenAICompatibleProvider("box", "http://x", ["BOX_KEY"]).available()


def test_openai_compatible_errors(posted):
    import httpx
    calls, answers = posted
    provider = llm_client.OpenAICompatibleProvider("box", "http://x")
    spec = llm_client.ModelSpec("m", "box")
    for status, transient in ((503, True), (429, True), (400, False)):
        answers.append(_Response(status, {"error": "x"}))
        with pytest.raises(llm_client.OpenAICompatibleError) as info:
            provider.generate(spec, llm_client.LLMRequest("q"), timeout=5)
        assert provider.is_transient(info.value) is transient
    answers.append(_Response(200, ValueError("not JSON")))
    with pytest.raises(llm_client.OpenAICompatibleError):
        provider.generate(spec, llm_client.LLMRequest("q"), timeout=5)
    assert provider.is_transient(httpx.ConnectError("refused"))   # server not up yet
