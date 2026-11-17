"""Token usage of every model call is logged and can be collected per search."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LLM_preprocessor"))

import llm_client  # noqa: E402


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
            raise RuntimeError("400 bad request")
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
