"""What a task measures is written in its own value_mappings.yaml entry
(`ca_id`, `measures`), so the task vocabulary and the Cognitive Atlas links
cannot drift apart: these tests fail when a link sits on a group, names a
construct cognitive_atlas_constructs.yaml does not have, or comes back as a
separate file (backend/vocabulary/tools/apply_task_measures.py,
yaml_to_llamaindex.measures_db)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

VOCABULARY = Path(__file__).resolve().parents[3] / "backend" / "vocabulary"
sys.path.insert(0, str(VOCABULARY / "tools"))
import apply_task_measures  # noqa: E402
import fill_bridge_gaps  # noqa: E402
import generate_task_bridge  # noqa: E402


@pytest.fixture(scope="module")
def tasks() -> dict:
    return yaml.safe_load((VOCABULARY / "value_mappings.yaml").read_text(encoding="utf-8"))["task"]


@pytest.fixture(scope="module")
def constructs() -> set:
    return apply_task_measures.construct_codes(VOCABULARY / "cognitive_atlas_constructs.yaml")


def test_only_tasks_carry_links(tasks):
    """A group key is no task code: a link on it reached the search as a code
    no dataset has ("spatial ability" filtered on spatial_memory_group)."""
    on_groups = [key for key, entry in tasks.items()
                 if isinstance(entry, dict) and not entry.get("standard_code")
                 and ("measures" in entry or "ca_id" in entry)]
    assert on_groups == []


def test_every_measure_is_a_cognitive_atlas_construct(tasks, constructs):
    unknown = {
        entry["standard_code"]: missing
        for entry in tasks.values()
        if isinstance(entry, dict) and entry.get("standard_code")
        for missing in [[m for m in entry.get("measures") or [] if m not in constructs]]
        if missing
    }
    assert unknown == {}


def test_measures_are_sorted_without_repeats(tasks):
    for entry in tasks.values():
        if isinstance(entry, dict) and "measures" in entry:
            measures = entry["measures"]
            assert isinstance(measures, list) and all(isinstance(m, str) for m in measures), entry
            assert measures == sorted(set(measures)), entry["standard_code"]


def test_the_links_have_no_second_file():
    assert not (VOCABULARY / "bids_to_ca_map.json").exists()


def test_the_retriever_reads_the_measures_of_the_vocabulary(tasks):
    from yaml_to_llamaindex import build_measures_db, measures_db

    assert measures_db == build_measures_db(tasks)
    assert "sternberg_task" in measures_db["working_memory_retrieval"]


def test_the_proposal_tools_offer_tasks_only(tmp_path):
    vocabulary = tmp_path / "value_mappings.yaml"
    vocabulary.write_text(_SAMPLE)
    expected = {"sternberg_task": "Sternberg task", "n_back": "n-back"}
    assert generate_task_bridge.load_bids_tasks(str(vocabulary)) == expected
    assert fill_bridge_gaps.load_bids_tasks(str(vocabulary)) == expected


_SAMPLE = """\
diagnosis:
  x:
    label: x
    standard_code: x
task:
  memory:
    label: memory
    is_group: true
  sternberg_task:
    label: Sternberg task
    standard_code: sternberg_task
    broader:
    - memory
  n_back:
    label: n-back
    standard_code: n_back
    ca_id: tsk_1
    measures:
    - updating
    synonyms:
    - term: nback
      weight: 1.0
suffix:
  bold:
    label: bold
    standard_code: bold
"""
_CONSTRUCTS = {"maintenance", "updating", "working_memory"}


def test_apply_writes_the_links_after_the_standard_code():
    out, notes = apply_task_measures.apply(
        _SAMPLE, {"sternberg_task": {"ca_task_id": "tsk_9", "measures": ["working_memory", "maintenance"]}},
        _CONSTRUCTS)
    added = "    ca_id: tsk_9\n    measures:\n    - maintenance\n    - working_memory\n"
    assert notes == []
    assert "    standard_code: sternberg_task\n" + added + "    broader:\n" in out
    assert out.replace(added, "", 1) == _SAMPLE


def test_apply_only_adds():
    out, _ = apply_task_measures.apply(
        _SAMPLE, {"n_back": {"ca_task_id": "tsk_2", "measures": ["working_memory"]}}, _CONSTRUCTS)
    entry = yaml.safe_load(out)["task"]["n_back"]
    assert entry["ca_id"] == "tsk_1"
    assert entry["measures"] == ["updating", "working_memory"]
    assert entry["synonyms"] == [{"term": "nback", "weight": 1.0}]


def test_apply_leaves_out_groups_and_unknown_constructs():
    out, notes = apply_task_measures.apply(
        _SAMPLE, {"memory": {"ca_task_id": "tsk_3", "measures": ["working_memory"]},
                  "sternberg_task": {"measures": ["not_a_construct"]}},
        _CONSTRUCTS)
    assert out == _SAMPLE
    assert len(notes) == 2


def test_apply_twice_changes_nothing_more():
    proposals = {"sternberg_task": {"ca_task_id": "tsk_9", "measures": ["maintenance"]}}
    once, _ = apply_task_measures.apply(_SAMPLE, proposals, _CONSTRUCTS)
    twice, _ = apply_task_measures.apply(once, proposals, _CONSTRUCTS)
    assert twice == once
