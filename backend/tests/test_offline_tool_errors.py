"""Offline tools catch only the failures they expect and raise the rest
(RAG/tools/, scripts/)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Client:
    def __init__(self, answers):
        self.answers = list(answers)
        self.models = self
        self.calls = []

    def generate_content(self, model, config, contents):
        self.calls.append(model)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return SimpleNamespace(text=answer)


expand_synonyms = _load(ROOT / "RAG" / "tools" / "expand_synonyms.py", "expand_synonyms")


def _gemini_answers(monkeypatch, answers):
    from google import genai

    client = _Client(answers)
    monkeypatch.setattr(genai, "Client", lambda api_key: client)
    monkeypatch.setattr(expand_synonyms.time, "sleep", lambda s: None)
    return client


def test_expand_synonyms_turns_an_api_error_into_its_own_error(monkeypatch):
    from google.genai import errors

    client = _gemini_answers(monkeypatch, [errors.ServerError(503, {"error": {}}),
                                           errors.ClientError(400, {"error": {}})])
    with pytest.raises(expand_synonyms.SynonymLLMError):
        expand_synonyms._call_gemini("p", "key")
    assert len(client.calls) == 2


def test_expand_synonyms_raises_a_bug(monkeypatch):
    _gemini_answers(monkeypatch, [TypeError("unexpected keyword")])
    with pytest.raises(TypeError):
        expand_synonyms._call_gemini("p", "key")


def test_a_failed_crossref_fetch_is_not_cached(monkeypatch):
    import urllib.error

    monkeypatch.setattr(expand_synonyms.time, "sleep", lambda s: None)

    def unreachable(request, timeout):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(expand_synonyms.urllib.request, "urlopen", unreachable)
    assert expand_synonyms._fetch_crossref_abstract("10.1/x") is None
    assert "10.1/x" not in expand_synonyms._crossref_cache


profiles = _load(ROOT / "scripts" / "extract_readme_profiles.py", "extract_readme_profiles")


def test_an_unusable_answer_or_failed_call_is_returned_a_bug_is_raised(monkeypatch):
    import llm_client

    source = SimpleNamespace(accession_id="ds000001")
    monkeypatch.setattr(profiles, "_ask_model", lambda *a: "not json")
    monkeypatch.setattr(profiles.rp, "parse_response",
                        lambda raw, source: (_ for _ in ()).throw(ValueError("no JSON")))
    _, profile, error = profiles._extract_safe(source, "m", None)
    assert profile is None and isinstance(error, profiles.ProfileAnswerError)

    def all_failed(*a):
        raise llm_client.LLMAllFailedError("All LLM tiers failed")
    monkeypatch.setattr(profiles, "_ask_model", all_failed)
    assert isinstance(profiles._extract_safe(source, "m", None)[2], llm_client.LLMAllFailedError)

    def bug(*a):
        raise KeyError("prompt field")
    monkeypatch.setattr(profiles, "_ask_model", bug)
    with pytest.raises(KeyError):
        profiles._extract_safe(source, "m", None)
