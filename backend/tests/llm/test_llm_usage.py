"""Token usage of every model call is logged and can be collected per search."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend" / "llm"))

import llm_client  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402


def _gemini_response(text="ok", **counts):
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(**counts))


def _fake_gemini(monkeypatch, response):
    def call(model, prompt, system, temperature, api_key, **_):
        llm_client._note_gemini_usage(response)
        return response.text
    monkeypatch.setattr(llm_client, "_call_gemini", call)


def test_usage_is_collected_with_purpose_and_model(monkeypatch):
    _fake_gemini(monkeypatch, _gemini_response(
        prompt_token_count=4500, cached_content_token_count=4000,
        candidates_token_count=300, thoughts_token_count=900))
    collected = llm_client.start_usage_collection()

    llm_client.llm_generate("q", api_key="k", models=["gemini-2.5-flash"],
                            use_fallback=False, purpose="intent")

    assert len(collected) == 1
    entry = collected[0]
    assert entry["purpose"] == "intent"
    assert (entry["provider"], entry["model"]) == ("gemini", "gemini-2.5-flash")
    assert entry["input_tokens"] == 4500
    assert entry["cached_tokens"] == 4000
    assert entry["output_tokens"] == 300
    assert entry["thinking_tokens"] == 900


def test_missing_counts_are_zero(monkeypatch):
    # The API leaves out counts that do not apply (no thinking, nothing cached).
    _fake_gemini(monkeypatch, _gemini_response(
        prompt_token_count=10, candidates_token_count=2,
        thoughts_token_count=None))
    collected = llm_client.start_usage_collection()

    llm_client.llm_generate("q", api_key="k", models=["m"], use_fallback=False)

    assert collected[0]["thinking_tokens"] == 0
    assert collected[0]["cached_tokens"] == 0
    assert collected[0]["purpose"] == "unlabelled"


def test_a_call_without_usage_data_adds_nothing(monkeypatch):
    monkeypatch.setattr(llm_client, "_call_gemini", lambda *a, **k: "ok")
    collected = llm_client.start_usage_collection()

    assert llm_client.llm_generate("q", api_key="k", models=["m"], use_fallback=False) == "ok"
    assert collected == []


def test_a_failed_tier_does_not_leak_its_usage_into_the_next(monkeypatch):
    def call(model, *a, **k):
        if model == "first":
            llm_client._note_gemini_usage(_gemini_response(prompt_token_count=99))
            raise genai_errors.ClientError(400, {"error": {"status": "INVALID_ARGUMENT"}})
        return "ok"   # second tier reports no usage
    monkeypatch.setattr(llm_client, "_call_gemini", call)
    collected = llm_client.start_usage_collection()

    llm_client.llm_generate("q", api_key="k", models=["first", "second"], use_fallback=False)

    assert collected == []


def test_anthropic_input_includes_cached_tokens():
    llm_client._note_anthropic_usage(SimpleNamespace(usage=SimpleNamespace(
        input_tokens=100, cache_read_input_tokens=900,
        cache_creation_input_tokens=0, output_tokens=50)))
    usage = llm_client._last_call.usage
    llm_client._last_call.usage = None
    assert usage["input_tokens"] == 1000
    assert usage["cached_tokens"] == 900


def _server_error():
    return genai_errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}})


def test_a_bug_in_a_provider_call_is_raised_not_taken_for_an_outage(monkeypatch):
    def call(*a, **k):
        raise TypeError("generate_content() got an unexpected keyword argument 'timeout'")
    monkeypatch.setattr(llm_client, "_call_gemini", call)
    import pytest
    with pytest.raises(TypeError):
        llm_client.llm_generate("q", api_key="k", models=["first", "second"], use_fallback=False)


def test_a_server_error_is_retried_on_the_same_tier(monkeypatch):
    calls = []

    def call(model, *a, **k):
        calls.append(model)
        if len(calls) == 1:
            raise _server_error()
        return "ok"
    monkeypatch.setattr(llm_client, "_call_gemini", call)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    assert llm_client.llm_generate("q", api_key="k", models=["first", "second"], use_fallback=False) == "ok"
    assert calls == ["first", "first"]


def test_a_client_error_that_mentions_a_timeout_is_not_retried(monkeypatch):
    # The old check looked for words such as "timeout" in the message.
    calls = []

    def call(model, *a, **k):
        calls.append(model)
        if model == "first":
            raise genai_errors.ClientError(400, {"error": {"status": "INVALID_ARGUMENT",
                                                          "message": "timeout must be positive"}})
        return "ok"
    monkeypatch.setattr(llm_client, "_call_gemini", call)
    assert llm_client.llm_generate("q", api_key="k", models=["first", "second"], use_fallback=False) == "ok"
    assert calls == ["first", "second"]


def test_a_timeout_is_transient():
    import httpx
    assert llm_client._gemini_is_transient(httpx.ReadTimeout("read timed out"))
    assert not llm_client._gemini_is_transient(genai_errors.ClientError(403, {"error": {}}))


def test_no_gemini_key_skips_to_claude_without_calling_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    gemini_calls = []
    monkeypatch.setattr(llm_client, "_call_gemini", lambda *a, **k: gemini_calls.append(a) or "x")
    monkeypatch.setattr(llm_client, "_call_anthropic", lambda *a, **k: "from claude")
    assert llm_client.llm_generate("q", models=["first"]) == "from claude"
    assert gemini_calls == []
