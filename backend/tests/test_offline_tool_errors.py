"""Offline tools catch only the failures they expect and raise the rest
(DB Preprocessing/, RAG/tools/, scripts/)."""
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


auto_map = _load(ROOT / "DB Preprocessing" / "Generate_Value_Mappings" / "auto_map_unmapped.py",
                 "auto_map_unmapped")


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


def test_auto_map_retries_an_overloaded_model_and_skips_a_missing_one(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr(auto_map.time, "sleep", lambda s: None)
    client = _Client([errors.ClientError(404, {"error": {"status": "NOT_FOUND"}}),
                      errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}}),
                      "[]"])
    assert auto_map._call_gemini(client, "", "p") == "[]"
    assert client.calls == [auto_map._MODEL_IDS[0], auto_map._MODEL_IDS[1], auto_map._MODEL_IDS[1]]


def test_auto_map_raises_its_own_error_when_no_model_answers():
    from google.genai import errors

    client = _Client([errors.ClientError(400, {"error": {}})] * len(auto_map._MODEL_IDS))
    with pytest.raises(auto_map.GeminiCascadeError):
        auto_map._call_gemini(client, "", "p")


def test_auto_map_raises_a_bug():
    with pytest.raises(TypeError):
        auto_map._call_gemini(_Client([TypeError("unexpected keyword")]), "", "p")
