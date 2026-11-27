"""Vocabulary files that cannot be read are logged and the search goes on
without them (backend/vocabulary/reference_vocabulary.py, backend/vocabulary/cohort_count.py); a missing
optional file stays silent."""
from __future__ import annotations

import logging

import pytest

import cohort_count
import reference_vocabulary


def test_a_broken_reference_vocabulary_is_logged_and_left_out(monkeypatch, tmp_path, caplog):
    broken = tmp_path / "reference_vocabulary.yaml"
    broken.write_text("diagnosis: [unclosed\n")
    monkeypatch.setattr(reference_vocabulary, "PATH", broken)
    reference_vocabulary._index.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            assert reference_vocabulary.match("diagnosis", "rasmussen") == []
        assert "not read" in caplog.text
    finally:
        reference_vocabulary._index.cache_clear()


def test_a_missing_reference_vocabulary_is_silent(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(reference_vocabulary, "PATH", tmp_path / "absent.yaml")
    reference_vocabulary._index.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            assert reference_vocabulary.match("diagnosis", "rasmussen") == []
        assert caplog.text == ""
    finally:
        reference_vocabulary._index.cache_clear()


def test_an_unreadable_cohort_vocabulary_is_logged(monkeypatch, tmp_path, caplog):
    broken = tmp_path / "value_mappings.yaml"
    broken.write_text("diagnosis: {a: [\n")
    monkeypatch.setattr(cohort_count, "_VOCABULARY", broken)
    cohort_count._vocabulary.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            assert cohort_count._vocabulary() == {}
        assert "only coded diagnoses" in caplog.text
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        monkeypatch.setattr(cohort_count, "_VOCABULARY", empty)
        cohort_count._vocabulary.cache_clear()
        assert cohort_count._vocabulary() == {}  # an empty file is no error
    finally:
        cohort_count._vocabulary.cache_clear()


def test_a_word_forms_list_that_is_no_mapping_is_ignored(monkeypatch, tmp_path, caplog):
    import word_forms

    odd = tmp_path / "word_forms.yaml"
    odd.write_text("- just\n- a list\n")
    monkeypatch.setattr(word_forms, "_PATH", odd)
    word_forms._reviewed.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            assert word_forms._reviewed() == {}
        assert "not a mapping" in caplog.text
    finally:
        word_forms._reviewed.cache_clear()


def test_a_failed_embedding_model_is_tried_again_later(monkeypatch):
    import retriever
    from yaml_to_llamaindex import EmbeddingIndexError

    pytest.importorskip("sentence_transformers")
    calls = []

    def build(model_name, yaml_path):
        calls.append(model_name)
        raise EmbeddingIndexError("model download failed")

    monkeypatch.setattr(retriever, "build_embedding_index", build)
    monkeypatch.setattr(retriever, "_emb_indices", {})
    monkeypatch.setattr(retriever, "_emb_failed_at", {})
    assert retriever._load_embedding_tier("m") is None
    assert retriever._load_embedding_tier("m") is None  # within the wait: not tried
    assert calls == ["m"]
    monkeypatch.setattr(retriever, "_EMB_RETRY_SECONDS", 0.0)
    assert retriever._load_embedding_tier("m") is None
    assert calls == ["m", "m"]


def test_a_bug_building_the_embedding_index_is_raised(monkeypatch):
    import retriever

    pytest.importorskip("sentence_transformers")

    def build(model_name, yaml_path):
        raise KeyError("terms")

    monkeypatch.setattr(retriever, "build_embedding_index", build)
    monkeypatch.setattr(retriever, "_emb_indices", {})
    monkeypatch.setattr(retriever, "_emb_failed_at", {})
    with pytest.raises(KeyError):
        retriever._load_embedding_tier("m")
