from __future__ import annotations

import json

import pytest

import llm_client
from services.indexing import readme_profile as rp

README = (
    "# Bipolar disorder and reward\n"
    "We recorded EEG from 30 patients with **bipolar disorder** and 30 healthy controls "
    "during a monetary incentive delay task.\n"
    "Exclusion criteria: history of stroke or epilepsy.\n"
    "Depressive symptoms were measured with the Beck Depression Inventory.\n"
    "EOG was recorded with bipolar electrodes."
)
SOURCE = rp.ProfileSource(accession_id="ds999999", name="Reward in BD", readme=README,
                          study_domain="Reward processing")


def _answer(**overrides):
    data = {
        "summary": "EEG study of reward anticipation in adults with bipolar disorder and healthy controls.",
        "conditions": [
            {"term": "bipolar disorder", "role": "patients",
             "evidence": "30 patients with bipolar disorder and 30 healthy controls"},
            {"term": "stroke", "role": "excluded", "evidence": "Exclusion criteria: history of stroke"},
            {"term": "depression", "role": "measured",
             "evidence": "Depressive symptoms were measured with the Beck Depression Inventory"},
            {"term": "schizophrenia", "role": "patients", "evidence": "patients with schizophrenia"},
        ],
        "species": ["Human"],
        "age_groups": ["adults", "teenagers"],
        "topics": ["reward processing", "reward processing", "mood disorders"],
        "paradigms": [{"term": "monetary incentive delay task",
                       "evidence": "during a monetary incentive delay task"}],
    }
    data.update(overrides)
    return json.dumps(data)


def test_quotes_are_checked_against_the_text_the_model_read() -> None:
    text = SOURCE.text()
    assert rp.quote_in_source("30 patients with bipolar disorder", text)      # markdown ignored
    assert rp.quote_in_source("30 PATIENTS with  bipolar\ndisorder", text)    # case and spacing
    assert rp.quote_in_source("We recorded EEG ... during a monetary incentive delay task", text)
    assert not rp.quote_in_source("patients with schizophrenia", text)
    assert not rp.quote_in_source("EEG", text)                                # too short to mean anything


def test_parse_keeps_roles_and_uses_only_verified_studied_conditions() -> None:
    profile = rp.parse_response(_answer(), SOURCE)
    roles = {(f.term, f.role, f.verified) for f in profile.facts if f.kind == "condition"}
    assert ("bipolar disorder", "patients", True) in roles
    assert ("stroke", "excluded", True) in roles
    # Invented quote: kept for review, never used for search.
    assert ("schizophrenia", "patients", False) in roles
    assert profile.studied_conditions == ["bipolar disorder"]
    assert rp.studied_conditions_text(profile) == "bipolar disorder"
    assert profile.age_groups == ["adults"]          # unknown group dropped
    assert profile.species == ["human"]
    assert profile.topics == ["reward processing", "mood disorders"]

    text = profile.profile_text()
    assert text.startswith("EEG study of reward anticipation")
    assert "Paradigms: monetary incentive delay task" in text
    assert "Conditions studied: bipolar disorder" in text
    assert "stroke" not in text and "schizophrenia" not in text


def test_parse_rejects_unusable_answers_and_tolerates_fences() -> None:
    with pytest.raises(ValueError):
        rp.parse_response("Sure! Here is the profile.", SOURCE)
    with pytest.raises(ValueError):
        rp.parse_response("[]", SOURCE)
    fenced = "```json\n" + _answer(conditions=[{"term": "x", "role": "nonsense", "evidence": ""}]) + "\n```"
    profile = rp.parse_response(fenced, SOURCE)
    assert [f for f in profile.facts if f.kind == "condition"] == []


def test_codes_and_rows_for_the_facts_table() -> None:
    profile = rp.parse_response(_answer(), SOURCE)
    asked = []

    def lookup(term):
        asked.append(term)
        return [("bipolar_disorder", 0.97), ("bipolar_i_disorder", 0.9)]

    rp.attach_codes(profile, lookup)
    assert asked == ["bipolar disorder"]                 # only verified studied conditions
    rows = rp.fact_rows(profile)
    bipolar = [r for r in rows if r["term"] == "bipolar disorder"]
    assert [r["code"] for r in bipolar] == ["bipolar_disorder", "bipolar_i_disorder"]
    assert all(r["verified"] for r in bipolar)
    assert any(r["term"] == "stroke" and r["code"] is None and r["role"] == "excluded" for r in rows)
    assert any(r["kind"] == "paradigm" for r in rows)

    meta = rp.meta_record(profile, SOURCE, "gemini-test", "2026-11-13T00:00:00Z")
    assert meta["source_hash"] == SOURCE.source_hash()
    assert meta["unverified_facts"] == 1


def test_only_changed_or_missing_profiles_are_redone() -> None:
    row = {"id": "a", "accession_id": "ds999999", "name": "Reward in BD", "readme_text": README,
           "study_domain": "Reward processing", "llm_profile_meta": None}
    assert len(rp.sources_needing_work([row])) == 1
    row["llm_profile_meta"] = json.dumps({"source_hash": SOURCE.source_hash()})
    assert rp.sources_needing_work([row]) == []
    assert len(rp.sources_needing_work([row], force=True)) == 1
    row["readme_text"] = README + "\nUpdated."
    assert len(rp.sources_needing_work([row])) == 1


def test_long_readmes_are_cut_before_they_reach_the_model() -> None:
    source = rp.ProfileSource(accession_id="x", name="x", readme="word " * 10_000)
    assert len(source.text()) < rp.MAX_README_CHARS + 500
    assert "[README truncated]" in rp.build_prompt(source)


# ── llm_client options used by extraction and re-ranking ─────────────────────

def test_llm_generate_passes_model_schema_and_thinking(monkeypatch) -> None:
    calls = []

    def fake_gemini(model, prompt, system, temperature, api_key, **kwargs):
        calls.append((model, kwargs))
        if model == "gemini-down":
            raise RuntimeError("400 invalid model")
        return '{"ok": true}'

    monkeypatch.setattr(llm_client, "_call_gemini", fake_gemini)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    schema = {"type": "object"}
    out = llm_client.llm_generate(
        "p", api_key="k", models=["gemini-down", "gemini-ok"], json_schema=schema,
        thinking_level="low", call_timeout=5, use_fallback=False,
    )
    assert out == '{"ok": true}'
    assert [c[0] for c in calls] == ["gemini-down", "gemini-ok"]
    assert calls[1][1]["json_schema"] == schema and calls[1][1]["thinking_level"] == "low"

    # use_fallback=False never reaches Claude.
    calls.clear()
    monkeypatch.setattr(llm_client, "_call_gemini",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("400 bad")))
    with pytest.raises(llm_client.LLMAllFailedError) as info:
        llm_client.llm_generate("p", api_key="k", models=["gemini-x"], use_fallback=False)
    assert "anthropic" not in str(info.value)


def test_default_cascade_is_unchanged_without_options(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL_CASCADE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert [t.model for t in llm_client._build_cascade()] == llm_client._DEFAULT_GEMINI_CASCADE
