"""Each folder's pyproject.toml lists every module that other code imports by name.

The editable installs (Dockerfile, CI, README.md) make a top-level module
importable only when it is listed, so one left out would fail at import in
the image."""
from __future__ import annotations

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "folder, not_imported",
    [("RAG", set()), ("LLM_preprocessor", set()), ("backend", {"main"})],
)
def test_every_top_level_module_is_listed(folder: str, not_imported: set) -> None:
    config = tomllib.loads((ROOT / folder / "pyproject.toml").read_text())
    listed = set(config["tool"]["setuptools"]["py-modules"])
    on_disk = {path.stem for path in (ROOT / folder).glob("*.py")}
    assert listed == on_disk - not_imported
