"""Offline tools catch only the failures they expect and raise the rest
(backend/vocabulary/tools/)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auto_map = _load(ROOT / "backend" / "vocabulary" / "tools" / "auto_map_unmapped.py",
                 "auto_map_unmapped")


def _fake_llm(monkeypatch, answer):
    """llm_client.llm_generate replaced; returns the calls' keyword arguments."""
    import llm_client
    calls = []

    def generate(prompt, **kwargs):
        calls.append(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr(llm_client, "llm_generate", generate)
    return calls


def test_auto_map_asks_its_profile_with_the_decision_schema(monkeypatch):
    calls = _fake_llm(monkeypatch, "[]")
    assert auto_map._ask_model("p") == "[]"
    assert calls[0]["profile"] == "vocabulary_mapping"
    assert calls[0]["json_schema"]["items"]["properties"]["action"]["enum"][0] == "filter"
    auto_map._ask_model("p", ["gemini-3.1-flash"])
    assert calls[1]["models"] == ["gemini-3.1-flash"]


def test_auto_map_lets_a_failed_call_and_a_bug_through(monkeypatch):
    import llm_client
    _fake_llm(monkeypatch, llm_client.LLMAllFailedError("no model answered"))
    with pytest.raises(llm_client.LLMAllFailedError):
        auto_map._ask_model("p")
    _fake_llm(monkeypatch, TypeError("unexpected keyword"))
    with pytest.raises(TypeError):
        auto_map._ask_model("p")


expand_synonyms = _load(ROOT / "backend" / "vocabulary" / "tools" / "expand_synonyms.py", "expand_synonyms")


def test_expand_synonyms_asks_its_profile_and_parses_the_list(monkeypatch):
    calls = _fake_llm(monkeypatch, 'Here: ["motor imagery", "imagined movement"]')
    assert expand_synonyms._ask_model("p", system="s") == ["motor imagery", "imagined movement"]
    assert calls[0]["profile"] == "synonym_expansion" and calls[0]["system"] == "s"


def test_expand_synonyms_lets_a_bug_through(monkeypatch):
    _fake_llm(monkeypatch, TypeError("unexpected keyword"))
    with pytest.raises(TypeError):
        expand_synonyms._ask_model("p")


def test_a_failed_crossref_fetch_is_not_cached(monkeypatch):
    import urllib.error

    monkeypatch.setattr(expand_synonyms.time, "sleep", lambda s: None)

    def unreachable(request, timeout):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(expand_synonyms.urllib.request, "urlopen", unreachable)
    assert expand_synonyms._fetch_crossref_abstract("10.1/x") is None
    assert "10.1/x" not in expand_synonyms._crossref_cache


from services.indexing import readme_profile_runner as profiles  # noqa: E402


def test_an_unusable_answer_or_failed_call_is_returned_a_bug_is_raised(monkeypatch):
    import llm_client

    source = SimpleNamespace(accession_id="ds000001")
    monkeypatch.setattr(profiles, "_ask_model", lambda *a: "not json")
    monkeypatch.setattr(profiles.rp, "parse_response",
                        lambda raw, source: (_ for _ in ()).throw(ValueError("no JSON")))
    _, profile, error = profiles.extract_safe(source, "m", None)
    assert profile is None and isinstance(error, profiles.ProfileAnswerError)

    def all_failed(*a):
        raise llm_client.LLMAllFailedError("All LLM tiers failed")
    monkeypatch.setattr(profiles, "_ask_model", all_failed)
    assert isinstance(profiles.extract_safe(source, "m", None)[2], llm_client.LLMAllFailedError)

    def bug(*a):
        raise KeyError("prompt field")
    monkeypatch.setattr(profiles, "_ask_model", bug)
    with pytest.raises(KeyError):
        profiles.extract_safe(source, "m", None)
