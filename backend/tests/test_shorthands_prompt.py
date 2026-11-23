"""The planning model is told what a shorthand stands for; it does not guess.

"Dataset with epilepsy patients and APT": the summary said "Arterial Perfusion
Tagging" (live, 2026-11-23). The listings (value_mappings.yaml,
RAG/shorthands.yaml) go to the model with the question.
"""

from __future__ import annotations

import json

from preprocess import intent


def test_the_shorthands_of_a_question_with_their_listed_meanings() -> None:
    assert intent.listed_shorthands("Dataset with epilepsy patients and APT") == {
        "APT": ["amide proton transfer imaging (APT) (scan type)"]}
    # "XQZ" and "GB" are in no listing: left to the model as before.
    assert intent.listed_shorthands("ERP in XQZ patients under 10 GB") == {
        "ERP": ["event-related potential"]}
    # The plain scan abbreviations are left to the model.
    assert intent.listed_shorthands("resting-state fMRI and EEG, MRI at 3T") == {}
    # Capitals for emphasis are not a shorthand; neither is a plain question.
    assert intent.listed_shorthands("epilepsy ONLY postictal") == {}
    assert intent.listed_shorthands("datasets of people listening to music") == {}


def test_the_prompt_carries_them_only_when_there_are_some(monkeypatch) -> None:
    prompts = []

    def fake_call(prompt, key, json_schema=None):
        prompts.append(prompt)
        return json.dumps({"query_family": "concept_query", "natural_language_summary": "s"})

    monkeypatch.setattr(intent, "_call_gemini", fake_call)
    intent.preprocess_query("20 patients with TLE", api_key="k")
    intent.preprocess_query("datasets of people listening to music", api_key="k")
    assert '"shorthands": {"TLE": ["temporal lobe epilepsy (diagnosis)"]}' in prompts[0]
    assert "## Shorthands" in prompts[0] and "do not pick one meaning" in prompts[0]
    assert "horthands" not in prompts[1]
