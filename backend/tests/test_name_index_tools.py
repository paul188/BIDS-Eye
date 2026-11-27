"""The offline tool that writes the name lists the retriever falls back on
(RAG/tools/build_name_index.py) never replaces good values with an empty list
after a failed query."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_name_index = _load(ROOT / "RAG" / "tools" / "build_name_index.py", "build_name_index")


def _funding_query_fails(monkeypatch, module):
    import sqlalchemy

    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url: create_engine("sqlite://"))
    monkeypatch.setattr(module, "_get_column_type", lambda conn, table, column: "text[]")

    def distinct(conn, column, *rest):
        if column == "funding":
            raise ProgrammingError("SELECT", {}, Exception('column "funding" does not exist'))
        return [f"{column} value"]
    return distinct


def test_a_failed_name_query_leaves_the_name_index_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(build_name_index, "_distinct_array_values",
                        _funding_query_fails(monkeypatch, build_name_index))
    out = tmp_path / "name_index.json"
    out.write_text(json.dumps({"funding_sources": ["NIH"]}))
    monkeypatch.setattr("sys.argv", ["build_name_index", "--db-url", "x", "--out", str(out)])
    with pytest.raises(SystemExit, match="funding_sources"):
        build_name_index.main()
    assert json.loads(out.read_text()) == {"funding_sources": ["NIH"]}
