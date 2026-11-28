"""What a condition compares (services/sql_tree.py): values and number bounds."""

from __future__ import annotations

from services import sql_tree
from services.schema import tables


def _where(conditions: str):
    return sql_tree.parse_select(f"SELECT 1 FROM bids_datasets AS d WHERE {conditions}").args["where"].this


def _age(side) -> bool:
    return sql_tree.compares(side, ["age"])


def test_values_of_a_column_or_json_key():
    node = _where("p.sex IN ('F', 'female') AND 'x' = p.diagnosis "
                  "AND o.other_entities ->> 'Manufacturer' ILIKE '%GE%'")
    assert sql_tree.compared_values(node, ["sex"]) == ["F", "female"]
    assert sql_tree.compared_values(node, ["diagnosis"]) == ["x"]
    assert sql_tree.compared_values(node, json_key="Manufacturer") == ["GE"]


def test_bounds_either_way_round_and_tightest():
    bounds = sql_tree.compared_bounds(_where("p.age BETWEEN 10 AND 19 AND 12 <= p.age"), _age)
    assert (bounds.low, bounds.high) == (12, 19)
    strength = sql_tree.compared_bounds(
        _where("(o.other_entities ->> 'MagneticFieldStrength')::float > 6.5 "
               "AND (o.other_entities ->> 'MagneticFieldStrength')::float < 7.5"),
        lambda side: sql_tree.compares(side, json_key="MagneticFieldStrength"))
    assert strength == sql_tree.Bounds(6.5, 7.5, False, False)
    assert strength.allows(7.0) and not strength.allows(7.5)
    assert sql_tree.compared_bounds(_where("p.age = 30"), _age).exact == 30


def test_a_count_a_default_or_a_negation_sets_no_bound():
    # The count of participants aged over 9 is a count, not an age of 20 or more.
    count = _where("(SELECT COUNT(*) FROM bids_participants AS p WHERE p.age > 9) >= 20")
    assert not _age(count.this)
    assert sql_tree.compared_bounds(count, _age) == sql_tree.Bounds(low=9, low_inclusive=False)
    missing = _where("COALESCE(d.age_min, 0) <= 0 AND NOT EXISTS(SELECT 1 FROM p WHERE p.age > 0)")
    assert sql_tree.compared_bounds(missing, lambda side: sql_tree.compares(side, ["age", "age_min"])) is None


def test_every_column_name_is_shown_as_one_word():
    assert tables.COLUMN_SHOWN_AS["doi"] == "DOI"
    assert tables.COLUMN_SHOWN_AS["country"] == "Location"
    assert tables.COLUMN_VOCABULARY["source_type"] == "source_type"
