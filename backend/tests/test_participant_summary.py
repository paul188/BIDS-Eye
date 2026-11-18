from __future__ import annotations

from services import dataset_cards, search
from schemas import ParticipantSchema


def _p(i: int, **kw) -> ParticipantSchema:
    return ParticipantSchema(participant_id=f"sub-{i:03d}", **kw)


def test_summary_counts_every_participant_past_the_cap() -> None:
    # 69 mice: the card used to see only the first 50 and drop all counts.
    people = ([_p(i, diagnosis="wild_type_animal_model", sex="F", age=0.3) for i in range(30)]
              + [_p(30 + i, diagnosis="genetic_knockout_animal_model", sex="male") for i in range(39)])
    summary = dataset_cards.summarize_participants(people)
    assert summary.total == 69
    assert summary.diagnoses == {"wild_type_animal_model": 30, "genetic_knockout_animal_model": 39}
    assert (summary.female, summary.male) == (30, 39)
    assert summary.age_min == summary.age_max == 0.3


def test_summary_skips_blank_values_and_empty_lists() -> None:
    assert dataset_cards.summarize_participants([]) is None
    summary = dataset_cards.summarize_participants([_p(1, diagnosis="  ", sex="n/a", age=0.0)])
    assert summary.diagnoses == {} and summary.female == summary.male == 0
    assert summary.age_min is None


def test_found_message_uses_the_right_plural() -> None:
    assert search._found_message(1) == "Found 1 dataset matching your query."
    assert search._found_message(2) == "Found 2 datasets matching your query."
