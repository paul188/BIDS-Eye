"""Construct expansion follows KINDOF links, plus only the PARTOF links listed in
backend/vocabulary/construct_parts.yaml, one level deep (yaml_to_llamaindex._parse_flat).

Both sides are guarded: a domain such as language keeps its parts, and a
process such as set shifting or decision making is not widened by its
sub-steps again (they reached 153 and 147 tasks when every PARTOF link was
followed)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import yaml_to_llamaindex as y

VOCABULARY = Path(__file__).resolve().parents[3] / "backend" / "vocabulary"


@pytest.fixture(scope="module")
def constructs() -> dict:
    text = (VOCABULARY / "cognitive_atlas_constructs.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)["cognitive_construct"]


def _groups(concepts: dict, parts) -> dict:
    """group_db of the construct category, built with *parts*."""
    dbs = [{"c": {}} for _ in range(4)]
    y._parse_flat("c", concepts, *dbs, {}, {"c": {}}, parts)
    return dbs[2]["c"]


def _tasks(codes) -> set:
    return {task for code in codes for task in y.measures_db.get(code, [])}


def test_every_listed_link_is_a_cognitive_atlas_partof_link(constructs):
    missing = [f"{part} PARTOF {whole}"
               for whole, parts in y.load_construct_parts().items()
               for part in parts
               if whole not in constructs or part not in constructs
               or whole not in (constructs[part].get("part_of") or [])]
    assert missing == []


def test_partof_is_followed_one_level_only():
    concepts = {
        "whole": {"label": "whole", "standard_code": "whole"},
        "part": {"label": "part", "standard_code": "part", "part_of": ["whole"]},
        "kind_of_part": {"label": "kind of part", "standard_code": "kind_of_part", "broader": ["part"]},
        "part_of_part": {"label": "part of part", "standard_code": "part_of_part", "part_of": ["part"]},
        "unlisted": {"label": "unlisted", "standard_code": "unlisted", "part_of": ["whole"]},
    }
    groups = _groups(concepts, {"whole": ["part"], "part": ["part_of_part"]})
    assert set(groups["whole"]) == {"whole", "part", "kind_of_part"}
    assert "whole" not in _groups(concepts, None)  # no KINDOF children and no listed parts: no group


def test_language_keeps_its_parts(constructs):
    assert len(_tasks(y.group_db["cognitive_construct"]["language"])) >= 50


@pytest.mark.parametrize("label", ["set shifting", "decision making", "task switching", "cognitive control"])
def test_process_constructs_are_not_widened_by_their_sub_steps(constructs, label):
    """These expand exactly as far as their KINDOF links reach."""
    kind_only = _groups(constructs, None).get(label, [])
    assert _tasks(y.group_db["cognitive_construct"].get(label, [])) == _tasks(kind_only)
