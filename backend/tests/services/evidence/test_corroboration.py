"""Which source groups confirm a clause (ClauseEvidence.confirming) orders the
confirmed results (services/evidence/SCORING.md)."""

from __future__ import annotations

from services.fulltext.ranking import RankQuery
from services.results.rankers import DEFAULT_RANKER, EvidenceFirst
from services.translate.sql_rewriter import build_page_sql
from services.uncertain import tier

HEAD = "SELECT d.id, d.name FROM bids_datasets AS d WHERE "
DIAGNOSIS = ("(EXISTS(SELECT 1 FROM bids_participants AS p WHERE p.dataset_id = d.id "
             "AND p.diagnosis = 'x') OR d.search_tsv @@ to_tsquery('english', 'epilepsy'))")


def test_a_clause_two_groups_can_confirm_gets_the_column() -> None:
    tiered = tier.apply(HEAD + DIAGNOSIS, papers=False)
    assert f" AS {tier.CORROBORATION_COLUMN}" in tiered
    assert "GREATEST(evidence.by_records_0::int + evidence.by_own_text_0::int - 1, 0)" in tiered


def test_a_clause_one_group_can_confirm_has_no_column() -> None:
    tiered = tier.apply(HEAD + "d.search_tsv @@ to_tsquery('english', 'young & children')", papers=False)
    assert tier.LEVEL_COLUMN in tiered and tier.CORROBORATION_COLUMN not in tiered


def test_the_default_ranker_puts_it_after_the_strength_and_before_the_score() -> None:
    tiered = tier.apply(HEAD + DIAGNOSIS, papers=False)
    page, _ = build_page_sql(tiered, [], True, 20, 0, rank_query=RankQuery("epilepsy"))
    order = page[page.rindex("ORDER BY"):]
    assert order.index("uncertain_level") < order.index("corroboration DESC") < order.index("text_rank")
    without, _ = build_page_sql(tiered, [], True, 20, 0, rank_query=RankQuery("epilepsy"),
                                ranker=EvidenceFirst(corroboration=False))
    assert "corroboration" not in without[without.rindex("ORDER BY"):]
    assert DEFAULT_RANKER.corroboration   # type: ignore[attr-defined]


def test_a_label_the_text_search_also_holds_is_not_a_second_witness() -> None:
    label = ("(d.dx_status ILIKE '%epilepsy%' OR d.search_tsv @@ to_tsquery('english', 'epilepsy'))")
    assert tier.CORROBORATION_COLUMN not in tier.apply(HEAD + label, papers=False)


def test_the_labels_counted_with_the_text_are_those_of_the_search_vector() -> None:
    """A label column added to search_tsv needs label_in_text in schema/tables.py."""
    import re
    import sys
    from pathlib import Path
    from services.schema.tables import LABELS_IN_TEXT

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "ingest" / "openneuro-crawler"))
    from remote import openneuro_metadata as meta

    columns = set(re.findall(r"coalesce\((\w+),", meta.SEARCH_VECTOR_EXPRESSION))
    own_text = {"name", "description_text", "llm_conditions", "llm_profile", "readme_text"}
    assert columns - own_text == LABELS_IN_TEXT


def test_the_default_order_is_unchanged_without_a_second_witness() -> None:
    sql = HEAD + "d.search_tsv @@ to_tsquery('english', 'young & children')"
    tiered = tier.apply(sql, papers=False)
    page, _ = build_page_sql(tiered, [], True, 20, 0, rank_query=RankQuery("young"))
    before, _ = build_page_sql(tiered, [], True, 20, 0, rank_query=RankQuery("young"),
                               ranker=EvidenceFirst(corroboration=False))
    assert page == before
