"""The evidence read from conditions recorded in the benchmark runs.

data/recorded_conditions.json holds one condition per shape (its SQL without
the values), with what the code read from it on 2026-11-25. It shows a reading
that changes, in this code or in a new sqlglot; it does not say the reading
was right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.evidence.branches import confirming_evidence
from services.results import query_filters

RECORDED = json.loads((Path(__file__).parent / "data" / "recorded_conditions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("recorded", RECORDED, ids=lambda r: r["condition"][:60])
def test_the_evidence_is_read_as_recorded(recorded):
    _, found = query_filters.where_conditions(
        f"SELECT d.id FROM bids_datasets AS d WHERE {recorded['condition']}")
    assert len(found) == 1
    chip, node = found[0]
    ways = [f"{type(w).__name__} {w.source.value} {','.join(w.read_columns())}".strip()
            for w in confirming_evidence(node, "d")]
    assert (chip.negated, ways) == (recorded["negated"], recorded["evidence"])
