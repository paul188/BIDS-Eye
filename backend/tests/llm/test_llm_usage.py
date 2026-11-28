"""The cascade loop of llm_client: profiles, usage, retries, failover and the deadline.

The loop runs against a config whose providers are fakes; the real adapters
are tested apart (usage parsing here, the requests they send in
test_gemini_seed.py).
"""
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend" / "llm"))

import llm_client  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402


class FakeProvider:
    """A provider whose answers come from *answer(model_name, req, timeout)*."""

    def __init__(self, name, answer=lambda *a: "ok", available=True):
        self.name = name
        self.api_key_env = ("FAKE_KEY",)
        self._answer = answer
        self._available = available
        self.calls = []

    def available(self):
        return self._available

    def errors(self):
        return llm_client.GeminiProvider(["X"]).errors()

    def is_transient(self, exc):
        return llm_client.GeminiProvider(["X"]).is_transient(exc)

    def generate(self, model, req, timeout):
        self.calls.append((model.name, req, timeout))
        result = self._answer(model.name, req, timeout)
        return result if isinstance(result, llm_client.LLMResult) else llm_client.LLMResult(result)


_MODELS = {name: {"provider": "gemini"} for name in
           ("gemini-a", "gemini-b", "gemini-c", "gemini-2.5-flash")}
_MODELS["claude-x"] = {"provider": "anthropic"}


def _install(monkeypatch, profiles, gemini=None, anthropic=None):
    """Use *profiles* with fake providers; returns (gemini, anthropic) fakes."""
    cfg = llm_client.parse_config({
        "providers": {"gemini": {"api_key_env": "X"}, "anthropic": {"api_key_env": "Y"}},
        "models": _MODELS,
        "profiles": profiles,
    })
    gemini = gemini or FakeProvider("gemini")
    anthropic = anthropic or FakeProvider("anthropic")
    monkeypatch.setattr(llm_client, "_config",
                        dataclasses.replace(cfg, providers={"gemini": gemini, "anthropic": anthropic}))
    return gemini, anthropic


def _profile(*models, deadline=100, **tier):
    return {"deadline_seconds": deadline,
            "tiers": [{"model": m, **tier} for m in models]}


def _server_error():
    return genai_errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}})


def _client_error(message=""):
    return genai_errors.ClientError(400, {"error": {"status": "INVALID_ARGUMENT", "message": message}})


def _raise(exc):
    def answer(*a):
        raise exc
    return answer


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)


# ── JSON answers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("answer", [
    '{"a": 1}', '```json\n{"a": 1}\n```', '```\n{"a": 1}\n```  ',
    '```json {"a": 1}```', '```json\n{"a": 1}',
])
def test_a_json_answer_comes_without_its_code_fence(monkeypatch, answer):
    _install(monkeypatch, {"p": _profile("gemini-a")},
             gemini=FakeProvider("gemini", lambda *a: answer))
    assert llm_client.llm_generate("q", profile="p", json_schema={"type": "object"}) == '{"a": 1}'


def test_a_text_answer_keeps_its_code_fence(monkeypatch):
    fenced = "```sql\nSELECT 1\n```"
    _install(monkeypatch, {"p": _profile("gemini-a")},
             gemini=FakeProvider("gemini", lambda *a: fenced))
    assert llm_client.llm_generate("q", profile="p") == fenced


# ── usage ────────────────────────────────────────────────────────────────────

def test_usage_is_collected_with_purpose_and_model(monkeypatch):
    usage = {"input_tokens": 4500, "cached_tokens": 4000, "output_tokens": 300, "thinking_tokens": 900}
    _install(monkeypatch, {"p": _profile("gemini-2.5-flash")},
             gemini=FakeProvider("gemini", lambda *a: llm_client.LLMResult("ok", usage)))
    collected = llm_client.start_usage_collection()

    llm_client.llm_generate("q", profile="p", purpose="intent")

    assert len(collected) == 1
    entry = collected[0]
    assert entry["purpose"] == "intent"
    assert (entry["provider"], entry["model"]) == ("gemini", "gemini-2.5-flash")
    assert {k: entry[k] for k in usage} == usage


def test_the_purpose_defaults_to_the_profile(monkeypatch):
    usage = {"input_tokens": 1, "cached_tokens": 0, "output_tokens": 1, "thinking_tokens": 0}
    _install(monkeypatch, {"sql": _profile("gemini-a")},
             gemini=FakeProvider("gemini", lambda *a: llm_client.LLMResult("ok", usage)))
    collected = llm_client.start_usage_collection()
    llm_client.llm_generate("q", profile="sql")
    assert collected[0]["purpose"] == "sql"


def test_a_call_without_usage_data_adds_nothing(monkeypatch):
    _install(monkeypatch, {"p": _profile("gemini-a")})
    collected = llm_client.start_usage_collection()
    assert llm_client.llm_generate("q", profile="p") == "ok"
    assert collected == []


def test_gemini_usage_missing_counts_are_zero():
    # The API leaves out counts that do not apply (no thinking, nothing cached).
    usage = llm_client.GeminiProvider._usage(SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=10, candidates_token_count=2, thoughts_token_count=None)))
    assert usage == {"input_tokens": 10, "cached_tokens": 0, "output_tokens": 2, "thinking_tokens": 0}


def test_anthropic_input_includes_cached_tokens():
    usage = llm_client.AnthropicProvider._usage(SimpleNamespace(usage=SimpleNamespace(
        input_tokens=100, cache_read_input_tokens=900,
        cache_creation_input_tokens=0, output_tokens=50)))
    assert usage["input_tokens"] == 1000
    assert usage["cached_tokens"] == 900


# ── errors and retries ───────────────────────────────────────────────────────

def test_a_bug_in_a_provider_call_is_raised_not_taken_for_an_outage(monkeypatch):
    _install(monkeypatch, {"p": _profile("gemini-a", "gemini-b")}, gemini=FakeProvider(
        "gemini", _raise(TypeError("generate_content() got an unexpected keyword argument"))))
    with pytest.raises(TypeError):
        llm_client.llm_generate("q", profile="p")


def test_a_server_error_is_retried_on_the_same_tier(monkeypatch):
    calls = []

    def answer(model, *a):
        calls.append(model)
        if len(calls) == 1:
            raise _server_error()
        return "ok"
    _install(monkeypatch, {"p": _profile("gemini-a", "gemini-b", attempts=2, waits=[2])},
             gemini=FakeProvider("gemini", answer))
    assert llm_client.llm_generate("q", profile="p") == "ok"
    assert calls == ["gemini-a", "gemini-a"]


def test_a_client_error_that_mentions_a_timeout_is_not_retried(monkeypatch):
    # The old check looked for words such as "timeout" in the message.
    calls = []

    def answer(model, *a):
        calls.append(model)
        if model == "gemini-a":
            raise _client_error("timeout must be positive")
        return "ok"
    _install(monkeypatch, {"p": _profile("gemini-a", "gemini-b", attempts=2, waits=[2])},
             gemini=FakeProvider("gemini", answer))
    assert llm_client.llm_generate("q", profile="p") == "ok"
    assert calls == ["gemini-a", "gemini-b"]


def test_an_empty_answer_moves_on_without_a_retry(monkeypatch):
    gemini, _ = _install(monkeypatch, {"p": _profile("gemini-a", "gemini-b", attempts=2, waits=[2])},
                         gemini=FakeProvider("gemini", lambda m, *a: "" if m == "gemini-a" else "ok"))
    assert llm_client.llm_generate("q", profile="p") == "ok"
    assert [c[0] for c in gemini.calls] == ["gemini-a", "gemini-b"]


def test_a_timeout_is_transient():
    import httpx
    gemini = llm_client.GeminiProvider(["X"])
    assert gemini.is_transient(httpx.ReadTimeout("read timed out"))
    assert not gemini.is_transient(genai_errors.ClientError(403, {"error": {}}))


def test_all_tiers_failing_names_each(monkeypatch):
    _install(monkeypatch, {"p": _profile("gemini-a", "gemini-b")},
             gemini=FakeProvider("gemini", _raise(_client_error())))
    with pytest.raises(llm_client.LLMAllFailedError) as info:
        llm_client.llm_generate("q", profile="p")
    assert "gemini/gemini-a" in str(info.value) and "gemini/gemini-b" in str(info.value)


# ── profiles and tiers ───────────────────────────────────────────────────────

def test_the_profile_sets_the_request_and_a_call_may_set_the_thinking_level(monkeypatch):
    gemini, _ = _install(monkeypatch, {"p": {
        "deadline_seconds": 10, "temperature": 0.3, "max_tokens": 99, "thinking_level": "low",
        "tiers": [{"model": "gemini-a"}]}})
    schema = {"type": "object"}
    llm_client.llm_generate("p", profile="p", system="s", json_schema=schema)
    llm_client.llm_generate("p", profile="p", thinking_level="high")
    assert gemini.calls[0][1] == llm_client.LLMRequest("p", "s", 0.3, 99, schema, "low")
    assert gemini.calls[1][1] == llm_client.LLMRequest("p", None, 0.3, 99, None, "high")


def test_a_tier_sets_its_own_values_for_its_model(monkeypatch):
    gemini, claude = _install(monkeypatch, {"p": {
        "deadline_seconds": 10, "temperature": 0.0, "max_tokens": 1024, "thinking_level": "low",
        "tiers": [{"model": "gemini-a", "max_tokens": 8192, "temperature": 0.7},
                  {"model": "claude-x", "thinking_level": "high"}]}},
        gemini=FakeProvider("gemini", _raise(_client_error())))
    llm_client.llm_generate("q", profile="p")
    g, c = gemini.calls[0][1], claude.calls[0][1]
    assert (g.temperature, g.max_tokens, g.thinking_level) == (0.7, 8192, "low")
    assert (c.temperature, c.max_tokens, c.thinking_level) == (0.0, 1024, "high")


def test_a_model_override_does_not_inherit_the_replaced_models_values(monkeypatch):
    gemini, _ = _install(monkeypatch, {"p": {
        "deadline_seconds": 10, "max_tokens": 1024,
        "tiers": [{"model": "gemini-a", "max_tokens": 8192, "timeout": 7}]}})
    llm_client.llm_generate("q", profile="p", models=["gemini-b"])
    model, req, timeout = gemini.calls[0]
    assert (model, req.max_tokens, timeout) == ("gemini-b", 1024, 7)


def test_profile_available(monkeypatch):
    _install(monkeypatch, {"p": _profile("gemini-a", "claude-x")},
             gemini=FakeProvider("gemini", available=False))
    assert llm_client.profile_available("p")
    _install(monkeypatch, {"p": _profile("gemini-a")}, gemini=FakeProvider("gemini", available=False))
    assert not llm_client.profile_available("p")
    with pytest.raises(llm_client.LLMConfigError):
        llm_client.profile_available("nope")


def test_an_unknown_profile_or_model_is_refused(monkeypatch):
    _install(monkeypatch, {"p": _profile("gemini-a")})
    with pytest.raises(llm_client.LLMConfigError):
        llm_client.llm_generate("q", profile="nope")
    with pytest.raises(llm_client.LLMConfigError):
        llm_client.llm_generate("q", profile="p", models=["gpt-5"])


def test_models_run_under_the_profiles_policy(monkeypatch):
    # Model i takes tier i's settings; models beyond the profile take the last tier's.
    gemini, _ = _install(monkeypatch, {"p": {"deadline_seconds": 100, "tiers": [
        {"model": "gemini-a", "timeout": 30}, {"model": "gemini-b", "timeout": 5}]}},
        gemini=FakeProvider("gemini", _raise(_client_error())))
    with pytest.raises(llm_client.LLMAllFailedError):
        llm_client.llm_generate("q", profile="p", models=["gemini-c", "gemini-2.5-flash", "gemini-a"])
    assert [(c[0], c[2]) for c in gemini.calls] == [
        ("gemini-c", 30), ("gemini-2.5-flash", 5), ("gemini-a", 5)]


def test_an_unconfigured_provider_is_skipped_without_a_call(monkeypatch):
    gemini, _ = _install(monkeypatch, {"p": _profile("gemini-a", "claude-x")},
                         gemini=FakeProvider("gemini", available=False),
                         anthropic=FakeProvider("anthropic", lambda *a: "from claude"))
    assert llm_client.llm_generate("q", profile="p") == "from claude"
    assert gemini.calls == []


def test_an_optional_tier_without_a_key_leaves_no_failure(monkeypatch):
    _install(monkeypatch, {"p": {"deadline_seconds": 10, "tiers": [
        {"model": "gemini-a"}, {"model": "claude-x", "optional": True}]}},
        gemini=FakeProvider("gemini", _raise(_client_error())),
        anthropic=FakeProvider("anthropic", available=False))
    with pytest.raises(llm_client.LLMAllFailedError) as info:
        llm_client.llm_generate("q", profile="p")
    assert "anthropic" not in str(info.value)


def test_no_tier_configured_fails(monkeypatch):
    _install(monkeypatch, {"p": _profile("gemini-a")}, gemini=FakeProvider("gemini", available=False))
    with pytest.raises(llm_client.LLMAllFailedError, match="no gemini key"):
        llm_client.llm_generate("q", profile="p")


# ── config ───────────────────────────────────────────────────────────────────

def test_the_shipped_config_is_valid():
    cfg = llm_client.load_config(llm_client.DEFAULT_CONFIG_PATH)
    assert {"intent", "sql", "rerank", "readme_profile", "dataset_description",
            "word_forms_benchmark"} <= set(cfg.profiles)
    sql = cfg.profiles["sql"]
    assert sql.tiers[0].model == "gemini-2.5-flash"
    assert cfg.models[sql.tiers[-1].model].provider == "anthropic" and sql.tiers[-1].optional
    assert not cfg.models["claude-opus-4-8"].temperature


def test_the_callers_profiles_and_default_models_are_defined():
    from services.indexing import dataset_description, readme_profile_runner
    cfg = llm_client.load_config(llm_client.DEFAULT_CONFIG_PATH)
    for model in (dataset_description.DEFAULT_MODEL, dataset_description.FALLBACK_MODEL,
                  readme_profile_runner.DEFAULT_MODEL):
        assert model in cfg.models


@pytest.mark.parametrize("change, message", [
    (lambda d: d["profiles"]["p"]["tiers"][0].update(model="gemini-typo"), "not defined"),
    (lambda d: d["profiles"]["p"]["tiers"][0].update(timout=5), "unknown setting"),
    (lambda d: d["profiles"]["p"]["tiers"][0].update(attempts=2), "wait before each retry"),
    (lambda d: d["profiles"]["p"].pop("deadline_seconds"), "deadline_seconds"),
    (lambda d: d["models"]["gemini-a"].update(provider="openai"), "not defined"),
    (lambda d: d["models"]["gemini-a"].update(json_schema="strict"), "native or prompt"),
    (lambda d: d["providers"].update(openai={"api_key_env": "K"}), "unknown type"),
    (lambda d: d["providers"].update(box={"type": "openai_compatible"}), "base_url"),
    (lambda d: d["profiles"]["p"]["tiers"][0].update(max_token=5), "unknown setting"),
])
def test_a_bad_config_is_refused(change, message):
    data = {"providers": {"gemini": {"api_key_env": "X"}},
            "models": {"gemini-a": {"provider": "gemini"}},
            "profiles": {"p": {"deadline_seconds": 10, "tiers": [{"model": "gemini-a"}]}}}
    change(data)
    with pytest.raises(llm_client.LLMConfigError, match=message):
        llm_client.parse_config(data)


def test_providers_are_named_entries_of_a_type():
    cfg = llm_client.parse_config({
        "providers": {"gemini": {"api_key_env": "X"},
                      "box": {"type": "openai_compatible", "base_url": "http://gpu:11434/v1/"},
                      "claude": {"type": "anthropic", "api_key_env": "Y"}},
        "models": {"qwen3:32b": {"provider": "box", "json_schema": "native"},
                   "claude-x": {"provider": "claude"}},
        "profiles": {"p": {"deadline_seconds": 10, "tiers": [{"model": "qwen3:32b"}]}}})
    box = cfg.providers["box"]
    assert isinstance(box, llm_client.OpenAICompatibleProvider)
    assert (box.name, box.base_url) == ("box", "http://gpu:11434/v1")
    assert isinstance(cfg.providers["claude"], llm_client.AnthropicProvider)
    assert cfg.providers["claude"].name == "claude"


def test_llm_config_path_points_at_another_file(monkeypatch, tmp_path):
    other = tmp_path / "llm.yaml"
    other.write_text("providers: {gemini: {api_key_env: X}}\n"
                     "models: {gemini-z: {provider: gemini}}\n"
                     "profiles: {sql: {deadline_seconds: 5, tiers: [{model: gemini-z}]}}\n")
    monkeypatch.setenv("LLM_CONFIG_PATH", str(other))
    monkeypatch.setattr(llm_client, "_config", None)
    assert llm_client.config_path() == other
    assert llm_client.config().profiles["sql"].tiers[0].model == "gemini-z"


def test_check_config_warns_about_a_required_tier_without_a_key(monkeypatch, caplog):
    _install(monkeypatch, {"p": {"deadline_seconds": 10, "tiers": [
        {"model": "gemini-a"}, {"model": "claude-x", "optional": True}]}},
        gemini=FakeProvider("gemini", available=False),
        anthropic=FakeProvider("anthropic", available=False))
    llm_client.check_config()
    assert "gemini-a cannot be called" in caplog.text
    assert "claude-x" not in caplog.text


# ── deadline ─────────────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(llm_client.time, "monotonic", c)
    return c


def _slow(clock, exc):
    def answer(model, req, timeout):
        clock.now += timeout
        raise exc
    return answer


def test_a_later_tier_keeps_its_reserve(monkeypatch, clock):
    gemini, claude = _install(monkeypatch, {"p": {"deadline_seconds": 40, "tiers": [
        {"model": "gemini-a", "attempts": 2, "waits": [2], "timeout": 12},
        {"model": "gemini-b", "timeout": 12},
        {"model": "claude-x", "timeout": 20, "reserve_seconds": 20}]}},
        gemini=FakeProvider("gemini", _slow(clock, _server_error())))

    assert llm_client.llm_generate("q", profile="p") == "ok"
    # 40 s, 20 s kept for Claude: Gemini gets 12 + 8, then nothing.
    assert [(c[0], round(c[2])) for c in gemini.calls] == [("gemini-a", 12), ("gemini-a", 8)]
    assert [c[2] for c in claude.calls] == [20]


def test_reserves_add_up(monkeypatch, clock):
    gemini, _ = _install(monkeypatch, {"p": {"deadline_seconds": 30, "tiers": [
        {"model": "gemini-a", "timeout": 30},
        {"model": "gemini-b", "timeout": 30, "reserve_seconds": 5},
        {"model": "gemini-c", "timeout": 30, "reserve_seconds": 10}]}},
        gemini=FakeProvider("gemini", _slow(clock, _client_error())))
    with pytest.raises(llm_client.LLMAllFailedError):
        llm_client.llm_generate("q", profile="p")
    assert [(c[0], c[2]) for c in gemini.calls] == [
        ("gemini-a", 15), ("gemini-b", 5), ("gemini-c", 10)]


def test_the_deadline_is_checked_before_every_attempt(monkeypatch, clock):
    gemini, _ = _install(monkeypatch, {"p": {"deadline_seconds": 30, "tiers": [
        {"model": "gemini-a", "attempts": 2, "waits": [2], "timeout": 12},
        {"model": "gemini-b", "timeout": 12},
        {"model": "gemini-c", "timeout": 12, "reserve_seconds": 12}]}},
        gemini=FakeProvider("gemini", _slow(clock, _server_error())))
    with pytest.raises(llm_client.LLMAllFailedError, match="time budget spent"):
        llm_client.llm_generate("q", profile="p")
    assert [(c[0], round(c[2])) for c in gemini.calls] == [
        ("gemini-a", 12), ("gemini-a", 6), ("gemini-c", 12)]


def test_an_unavailable_tier_keeps_no_reserve(monkeypatch, clock):
    gemini, _ = _install(monkeypatch, {"p": {"deadline_seconds": 30, "tiers": [
        {"model": "gemini-a", "timeout": 30},
        {"model": "claude-x", "timeout": 20, "optional": True, "reserve_seconds": 20}]}},
        gemini=FakeProvider("gemini", _slow(clock, _client_error())),
        anthropic=FakeProvider("anthropic", available=False))
    with pytest.raises(llm_client.LLMAllFailedError):
        llm_client.llm_generate("q", profile="p")
    assert [c[2] for c in gemini.calls] == [30]
