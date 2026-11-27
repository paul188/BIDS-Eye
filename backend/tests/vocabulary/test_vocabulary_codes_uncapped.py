"""A vocabulary term keeps every code it resolves to (backend/vocabulary/retriever.py).

The retriever used to cut each term at 20 codes. A broad term such as
"memory" resolves to more than a hundred tasks, mostly with one shared
score in file order, so the cut dropped datasets at random. The cap now
applies to the fuzzy name fields only."""
from __future__ import annotations

import json

from retriever import MetadataRetriever, _resolve_yaml_term, _resolve_yaml_term_scored


def _retriever(tmp_path, authors):
    path = tmp_path / "name_index.json"
    path.write_text(json.dumps({"authors": authors, "funding_sources": [], "institutions": []}))
    return MetadataRetriever(path)


def test_a_broad_task_term_keeps_all_its_codes(tmp_path) -> None:
    retriever = _retriever(tmp_path, [])
    every = _resolve_yaml_term_scored("task", "memory")
    assert len(every) > 20
    assert retriever.retrieve_for_field_multi_scored("task", ["memory"])["memory"] == every
    assert retriever.retrieve_for_field_multi("task", ["memory"])["memory"] == _resolve_yaml_term("task", "memory")


def test_name_fields_are_still_capped(tmp_path) -> None:
    retriever = _retriever(tmp_path, [f"Smith, A{i}" for i in range(40)])
    assert len(retriever.retrieve_for_field("author", "Smith", n_results=5)) <= 5
