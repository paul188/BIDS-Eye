"""Names of stored codes (backend/vocabulary/code_names.py) and how the labels are written."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from code_names import code_name

_LABEL_CASE = Path(__file__).resolve().parents[2] / "vocabulary" / "tools" / "label_case.py"


def test_short_label_comes_before_label():
    assert code_name("suffix", "fmri_bold") == "fMRI"
    assert code_name("datatype", "functional_mri") == "fMRI"
    assert code_name("diagnosis", "healthy_control") == "healthy control"
    assert code_name("source_type", "openneuro") == "OpenNeuro"


def test_one_code_is_named_per_field():
    # intracranial_eeg is a datatype and a suffix; both have a name.
    assert code_name("datatype", "intracranial_eeg") == code_name("suffix", "intracranial_eeg") == "iEEG"


def test_the_standard_code_is_looked_up_not_the_entry_key():
    # Entry amyotrophic_lateral_sclerosis stores als_general.
    assert code_name("diagnosis", "als_general").startswith("amyotrophic lateral sclerosis")


def test_unknown_code_keeps_its_words():
    assert code_name("task", "not_a_listed_task") == "not a listed task"


def test_labels_are_written_as_inside_a_sentence():
    # A new entry written "Unipolar Depression" would show next to "depression";
    # backend/vocabulary/tools/label_case.py --write rewrites it.
    spec = importlib.util.spec_from_file_location("label_case", _LABEL_CASE)
    label_case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_case)
    lines = label_case.VALUE_MAPPINGS.read_text(encoding="utf-8").split("\n")
    assert label_case.rewrite(lines)[1] == []
