"""describe_one (services/indexing/dataset_description.py): a model that fails
on every tier is recorded per dataset; a bug stops the batch."""
from __future__ import annotations

import pytest

from llm_client import LLMAllFailedError
from services.indexing import dataset_description as dd
from tests.test_dataset_description import _source


def test_a_failed_model_call_is_recorded_in_the_outcome(monkeypatch):
    def failing(*args, **kwargs):
        raise LLMAllFailedError("All LLM tiers failed")
    monkeypatch.setattr(dd, "_ask", failing)
    outcome = dd.describe_one(_source(), True, None, True)
    assert outcome.relation_error.startswith("model call failed")


def test_a_bug_is_not_recorded_as_a_failed_model_call(monkeypatch):
    def buggy(*args, **kwargs):
        raise KeyError("prompt field")
    monkeypatch.setattr(dd, "_ask", buggy)
    with pytest.raises(KeyError):
        dd.describe_one(_source(), True, None, True)
