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
    """llm_client.llm_generate replaced (every profile callable); returns the calls' keyword arguments."""
    import llm_client
    calls = []

    def generate(prompt, **kwargs):
        calls.append(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr(llm_client, "llm_generate", generate)
    monkeypatch.setattr(llm_client, "profile_available", lambda profile: True)
    return calls


def _generate(tmp_path, model=None):
    """auto_map.generate_decisions over one unmapped value, nothing applied."""
    unmapped = {"diagnosis": {"tle": {"count": 1, "datasets": ["ds000001"]}}}
    return auto_map.generate_decisions(
        unmapped, {"diagnosis": {}}, tmp_path / "m.yaml", tmp_path / "ck.json",
        tmp_path / "log.txt", ["diagnosis"], batch_size=10, model=model, resume=False,
        generate_only=True, dataset_context={})


def test_auto_map_asks_its_profile_with_the_decision_schema(monkeypatch, tmp_path):
    calls = _fake_llm(monkeypatch, "[]")
    assert _generate(tmp_path) == {"diagnosis": []}
    assert calls[0]["profile"] == "vocabulary_mapping"
    assert calls[0]["json_schema"]["items"]["properties"]["action"]["enum"][0] == "filter"
    assert calls[0]["models"] is None
    _generate(tmp_path, model="gemini-3.1-flash")
    assert calls[1]["models"] == ["gemini-3.1-flash"]


def test_auto_map_skips_a_failed_call_and_raises_a_bug(monkeypatch, tmp_path):
    import llm_client
    _fake_llm(monkeypatch, llm_client.LLMAllFailedError("no model answered"))
    assert _generate(tmp_path) == {"diagnosis": []}
    _fake_llm(monkeypatch, TypeError("unexpected keyword"))
    with pytest.raises(TypeError):
        _generate(tmp_path)


expand_synonyms = _load(ROOT / "backend" / "vocabulary" / "tools" / "expand_synonyms.py", "expand_synonyms")


def test_expand_synonyms_finds_the_list_in_a_chatty_answer():
    answer = 'Here: ["motor imagery", "imagined movement"]'
    assert expand_synonyms._parse_llm_response(answer) == ["motor imagery", "imagined movement"]


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
    monkeypatch.setattr(profiles.rp, "build_prompt", lambda source: "prompt")
    monkeypatch.setattr(llm_client, "llm_generate", lambda *a, **k: "not json")
    monkeypatch.setattr(profiles.rp, "parse_response",
                        lambda raw, source: (_ for _ in ()).throw(ValueError("no JSON")))
    _, profile, error = profiles.extract_safe(source, "m", None)
    assert profile is None and isinstance(error, profiles.ProfileAnswerError)

    def all_failed(*a, **k):
        raise llm_client.LLMAllFailedError("All LLM tiers failed")
    monkeypatch.setattr(llm_client, "llm_generate", all_failed)
    assert isinstance(profiles.extract_safe(source, "m", None)[2], llm_client.LLMAllFailedError)

    def bug(*a, **k):
        raise KeyError("prompt field")
    monkeypatch.setattr(llm_client, "llm_generate", bug)
    with pytest.raises(KeyError):
        profiles.extract_safe(source, "m", None)
