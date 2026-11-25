"""The fields a dataset may not record (services/uncertain/fields/): each is
one SoftField, and the tier handles all of them through it."""

from __future__ import annotations

import pytest
import sqlglot

import text_evidence
from services.evidence.probable import NamedInOwnText
from services.uncertain import classify as tier_classify, fields
from services.uncertain.fields.base import SoftField, participant_rows_record

_HEAD = "SELECT d.id FROM bids_datasets AS d WHERE "


def _classify(condition: str):
    where = sqlglot.parse_one(_HEAD + condition, read="postgres").args["where"].this
    return tier_classify.classify(where, "d")


def test_every_field_is_described_completely():
    assert [field.kind for field in fields.FIELDS] == ["age", "diagnosis", "sex"]
    for field in fields.FIELDS:
        assert field.label
        assert field.columns and "d." in field.recorded("d")
    # No column belongs to two fields.
    columns = [column for field in fields.FIELDS for column in field.columns]
    assert len(columns) == len(set(columns))


def test_a_new_field_needs_only_its_description(monkeypatch):
    handedness = "EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id " \
                 "AND p2.handedness = 'left_handed')"
    assert _classify(handedness) is None   # a hard filter today

    class LeftHandedNamed(NamedInOwnText):
        def sql(self, alias):
            return f"{alias}.readme_text ~* 'left-handed'"

        def reason(self):
            return "names it"

    field = SoftField(
        kind="handedness", label="handedness", columns=("handedness",),
        recorded=participant_rows_record("handedness"),
        named_in_text=lambda node: LeftHandedNamed(),
    )
    monkeypatch.setitem(fields.BY_KIND, "handedness", field)
    monkeypatch.setitem(fields.KIND_OF_COLUMN, "handedness", "handedness")

    soft = _classify(handedness)
    assert soft.kind == "handedness"
    assert soft.probable == "d.readme_text ~* 'left-handed'" and soft.weaker[-1].reason() == "names it"
    assert "uk.handedness IS NOT NULL" in soft.known
    assert soft.not_recorded.reason() == "handedness not recorded"
    # Recorded values that do not fit contradict, as for ages and sex.
    assert soft.state_sql().endswith("ELSE 3 END") and "WHEN NOT (EXISTS" in soft.state_sql()


def test_text_evidence_must_say_why():
    class NoReason(NamedInOwnText):
        def sql(self, alias):
            return "TRUE"

    with pytest.raises(TypeError, match="reason"):
        NoReason()


def test_text_beats_recorded_only_for_diagnosis():
    assert [field.kind for field in fields.FIELDS if field.text_beats_recorded] == ["diagnosis"]
    assert [field.kind for field in fields.FIELDS if not field.paper_evidence] == ["age"]


def test_the_words_come_from_the_vocabulary():
    assert set(text_evidence.text_words("sex")) == {"male", "female"}
    assert "women" in text_evidence.text_words("sex")["female"]
    assert "participants" in text_evidence.people_words()
    groups = text_evidence.age_groups()
    assert (0.0, 3.0) == groups[0][:2] and "infants" in groups[0][2]
    assert not any("aging" in words for _, _, words in groups)


def test_a_condition_lists_the_ways_that_confirm_it() -> None:
    """One per OR branch; the SQL of the condition still decides."""
    import sqlglot
    from services.evidence.base import Source, Strength
    from services.uncertain.classify import classify

    sql = ("(EXISTS(SELECT 1 FROM bids_participants AS p WHERE p.dataset_id = d.id "
           "AND p.diagnosis = 'x') OR d.name ILIKE '%epilepsy%')")
    node = sqlglot.parse_one(f"SELECT 1 FROM bids_datasets AS d WHERE {sql}",
                             read="postgres").args["where"].this
    condition = classify(node, "d")
    assert condition is not None
    assert [e.source for e in condition.confirming] == [Source.PARTICIPANTS, Source.SHORT_TEXT]
    assert all(e.strength == Strength.CONFIRMED for e in condition.confirming)
    assert Strength.CONFIRMED not in [e.strength for e in condition.evidence]
