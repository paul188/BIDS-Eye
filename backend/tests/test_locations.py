"""Places in questions: "datasets from Bonn" (RAG/place_index.py, preprocess.py location lines).

"bonn" (2026-11-18) found nothing: no dataset text says Bonn, but two
datasets name the University of Bonn's ethics committee. The sites of
bids_dataset_sites are matched to ROR (remote/site_places.py), and a place
the user names becomes a join on them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sqlglot = pytest.importorskip("sqlglot")

import place_index  # noqa: E402
import preprocess as pp  # noqa: E402
from services.translate import participant_query_compiler as pqc  # noqa: E402
from services.results import query_filters  # noqa: E402
from services.fulltext import text_search  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crawlers" / "openneuro-crawler"))
from remote import site_places  # noqa: E402

PLACES = [("city", "Bonn"), ("city", "Cologne"), ("city", "Washington"), ("region", "Washington"),
          ("city", "Frankfurt am Main"), ("city", "St Louis"), ("region", "North Rhine-Westphalia"), ("country", "Germany"), ("country", "United States")]


@pytest.fixture(autouse=True)
def _places(monkeypatch):
    monkeypatch.setattr(text_search, "_lookup_counts", lambda queries: None)
    place_index.set_places(PLACES)
    yield
    place_index.set_places(())


def _plan(**fields) -> pp.QueryPlan:
    return pp.QueryPlan(query_family=list(pp.QueryFamily)[0], natural_language_summary="demo", **fields)


def test_place_names_resolve_to_the_stored_value() -> None:
    index = place_index.get_index()
    assert index.resolve("bonn") == ("city", "Bonn")
    assert index.resolve("Köln") == ("city", "Cologne")
    assert index.resolve("Koeln") == ("city", "Cologne")
    assert index.resolve("USA") == ("country", "United States")
    assert index.resolve("NRW") == ("region", "North Rhine-Westphalia")
    assert index.resolve("Washington") == ("city", "Washington")  # the city before the state
    assert index.resolve("Heidelberg") is None
    assert index.resolve("Frankfurt") == ("city", "Frankfurt am Main")
    assert index.resolve("Saint Louis") == ("city", "St Louis")


def test_a_stored_place_joins_the_sites_and_an_unknown_one_is_text() -> None:
    question = pp.augment_with_rag(_plan(location_terms=["Köln", "Heidelberg"]), {}).augmented_question
    assert ('  location "Köln": (EXISTS (SELECT 1 FROM bids_dataset_sites AS s JOIN bids_site_places AS sp '
            "ON sp.name_raw = s.name_raw WHERE s.dataset_id = d.id AND sp.city = 'Cologne') "
            "OR text_match('koln | cologne', 'short'))") in question
    assert "  location \"Heidelberg\": text_match('heidelberg')" in question


def test_a_place_alone_compiles_and_reads_as_a_location_chip() -> None:
    plan = _plan(location_terms=["Bonn"])
    augmented = pp.augment_with_rag(plan, {}).augmented_question
    compiled = pqc.compile_plan(plan, [], augmented)
    assert compiled.sql and "sp.city = 'Bonn'" in compiled.sql, compiled.reason

    expanded = text_search.expand_text_search(compiled.sql, fts=True, papers=False).sql
    chip = query_filters.describe(expanded)[0]
    assert (chip.field, chip.label) == ("Location", "Bonn")


def test_ror_matches_become_place_rows() -> None:
    item = {"chosen": True, "organization": {
        "id": "https://ror.org/041nas322",
        "names": [{"value": "University of Bonn", "types": ["ror_display", "label"]}],
        "locations": [{"geonames_details": {
            "name": "Bonn", "country_subdivision_name": "North Rhine-Westphalia",
            "country_name": "Germany", "country_code": "DE", "lat": 50.73, "lng": 7.1}}]}}
    [row] = site_places.place_rows("Ethics Committee of the University of Bonn", [item])
    assert (row["ror_id"], row["institution"], row["city"], row["region"], row["country_code"]) == (
        "https://ror.org/041nas322", "University of Bonn", "Bonn", "North Rhine-Westphalia", "DE")
    # No confident match: one empty row, so the name is not asked again.
    [empty] = site_places.place_rows("Yes", [])
    assert empty["ror_id"] == "" and empty["city"] is None


def test_an_institution_is_also_found_through_its_ror_name() -> None:
    # "datasets from the University of Bonn": no scan names it, its ethics committee does.
    place_index.set_places(PLACES + [("institution", "University of Bonn")])
    question = pp.augment_with_rag(_plan(institution_terms=["University of Bonn"]), {}).augmented_question
    line = next(l for l in question.splitlines() if l.startswith('  institution "University of Bonn"'))
    assert "sp.institution = ANY(ARRAY['University of Bonn']::text[])" in line
    sql = ("SELECT d.id FROM bids_datasets AS d WHERE " + line.split(": ", 1)[1] + " GROUP BY d.id")
    expanded = text_search.expand_text_search(sql, fts=True, papers=False).sql
    chip = query_filters.describe(expanded)[0]
    assert (chip.field, chip.label) == ("Institution", "University of Bonn")


def test_lead_authors_institutions_become_sites() -> None:
    from remote import paper_affiliations as pa

    def author(position, *institutions):
        return {"author_position": position, "institutions": [
            {"display_name": name, "ror": ror} for name, ror in institutions]}

    works = {
        "10.1/a": [author("first", ("University of Bonn", "https://ror.org/041nas322")),
                   author("middle", ("Harvard University", "https://ror.org/03vek6s52")),
                   author("last", ("University of Bonn", "https://ror.org/041nas322"),
                          ("No ROR Lab", None))],
        "10.1/b": None,   # OpenAlex unreachable
    }
    found = pa.collect([("d1", "ds1", "10.1/a"), ("d2", "ds2", "10.1/b"), ("d2", "ds2", "10.1/a")],
                       lookup=works.get, pause=0)
    # Middle authors and institutions without a ROR id are left out.
    assert found["d1"] == [("University of Bonn", "https://ror.org/041nas322", "first author of doi:10.1/a")]
    # One failed lookup keeps the dataset's old rows.
    assert found["d2"] is None
