"""Vocabulary files that cannot be read are logged and the search goes on
without them (RAG/reference_vocabulary.py, RAG/cohort_count.py); a missing
optional file stays silent."""
from __future__ import annotations

import logging

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
