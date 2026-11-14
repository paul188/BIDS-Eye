from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "crawlers" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

from remote import paper_links as links  # noqa: E402
from remote import paper_lookup as lookup  # noqa: E402
from remote import paper_store as store  # noqa: E402
from remote.paper_links import PaperId  # noqa: E402


# ── Identifiers ───────────────────────────────────────────────────────────────

def test_clean_doi_strips_what_uploaders_leave_attached() -> None:
    cases = {
        "10.1016/j.neuroimage.2020.116945}": "10.1016/j.neuroimage.2020.116945",
        "10.1007/s00406-022-01462-0#citeas": "10.1007/s00406-022-01462-0",
        "10.1088/1741-2552/ac69bc/meta": "10.1088/1741-2552/ac69bc",
        "10.1101/2020.12.17.423254v3\\": "10.1101/2020.12.17.423254",
        "10.1101/2024.02.13.580092v2.full": "10.1101/2024.02.13.580092",
        "10.64898/2025.12.03.692160v1": "10.64898/2025.12.03.692160",
        "10.1038/s41467-024-47495-y](": "10.1038/s41467-024-47495-y",
        "10.1101/2022.10.10.511419]()": "10.1101/2022.10.10.511419",
        "10.1016/S1053-8119(03)00123-4).": "10.1016/s1053-8119(03)00123-4",
        "10.1002/hbm.26128**": "10.1002/hbm.26128",
        "10.21105/joss.01896file:///media/x/dataset_description.json": "10.21105/joss.01896",
    }
    for raw, expected in cases.items():
        assert links.clean_doi(raw) == expected, raw


def test_data_repository_dois_are_not_papers() -> None:
    assert links.clean_doi("10.18112/openneuro.ds004746.v1.0.1") is None
    assert links.clean_doi("10.5281/zenodo.123456") is None
    assert links.clean_doi("10.17605/OSF.IO/ABCDE") is None


def test_extract_ids_reads_every_kind_in_order() -> None:
    text = (
        "Preprint: https://doi.org/10.1101/2023.09.21.558825. See also "
        "https://www.ncbi.nlm.nih.gov/pubmed/25977808, PMC8261645, arXiv:2305.09057, "
        "https://www.nature.com/articles/sdata20143 and https://www.nature.com/articles/s41597-019-0104-8 "
        "and 10.48550/arXiv.2305.13226. Data: doi:10.18112/openneuro.ds000117.v1.0.5"
    )
    assert [str(i) for i in links.extract_ids(text)] == [
        "doi:10.1101/2023.09.21.558825", "pmid:25977808", "pmcid:PMC8261645", "arxiv:2305.09057",
        "doi:10.1038/sdata.2014.3", "doi:10.1038/s41597-019-0104-8", "arxiv:2305.13226",
    ]


def test_doi_fallbacks_drop_landing_page_paths() -> None:
    assert links.doi_fallbacks("10.1093/cercor/bhx202/4080827/how-we-transmit") == [
        "10.1093/cercor/bhx202/4080827", "10.1093/cercor/bhx202",
    ]
    assert links.doi_fallbacks("10.7554/elife.36493.001") == ["10.7554/elife.36493"]
    assert links.doi_fallbacks("10.1038/sdata.2018.110") == []


def test_collect_links_keeps_every_source_most_trusted_first() -> None:
    found = links.collect_links(
        associated="https://doi.org/10.1038/s41467-024-50103-8",
        references=["Preprint: https://doi.org/10.1101/2023.09.21.558825"],
        acknowledge="Please cite https://doi.org/10.1101/2023.09.21.558825",
        readme="MNE-BIDS (10.21105/joss.01896) was used.",
    )
    by_id = {str(link.paper_id): link for link in found}
    assert list(by_id) == [
        "doi:10.1038/s41467-024-50103-8", "doi:10.1101/2023.09.21.558825", "doi:10.21105/joss.01896",
    ]
    assert by_id["doi:10.1101/2023.09.21.558825"].sources == ["references", "acknowledge"]
    assert by_id["doi:10.1101/2023.09.21.558825"].raw.startswith("Preprint:")
    assert by_id["doi:10.21105/joss.01896"].raw is None


def test_tool_list_holds_clean_paper_dois() -> None:
    for doi in links.TOOL_PAPERS:
        assert doi == doi.lower() and links.clean_doi(doi) == doi, doi


# ── Authors and roles ─────────────────────────────────────────────────────────

def test_surnames_from_the_shapes_in_the_catalogue() -> None:
    assert links.surnames(["Poldrack, R.A.", "Gorgolewski, K."]) == {"poldrack", "gorgolewski"}
    assert links.surnames(["Russell A. Poldrack"]) == {"poldrack"}
    assert links.surnames(["Poldrack RA", "van der Meer JN"]) == {"poldrack", "meer"}
    assert links.surnames(["Klaus Gramann, Thomas Töllner, Hermann J. Müller"]) == {
        "gramann", "tollner", "muller"}
    assert links.surnames(["Chen, J., Leong, Y.C., Honey, C.J."]) == {"chen", "leong", "honey"}
    assert links.surnames(["Jonathan Touryan (data and curation)", "Brent Lance (data)"]) == {
        "touryan", "lance"}
    assert links.surnames(["James F Cavanagh  jcavanagh@unm.edu"]) == {"cavanagh"}


def _paper(key: str, title: str, authors: list, **extra) -> dict:
    return {"key": key, "doi": key.split(":", 1)[1], "title": title, "authors": authors, **extra}


def _link(value: str, *sources: str) -> links.Link:
    return links.Link(PaperId("doi", value), list(sources))


def test_roles_tool_primary_related() -> None:
    pairs = [
        (_link("10.21105/joss.01896", "readme"),
         _paper("doi:10.21105/joss.01896", "MNE-BIDS: Organizing electrophysiological data", ["Appelhoff S"])),
        (_link("10.1/own", "references"),
         _paper("doi:10.1/own", "Neural Overlap in Item Representations Across Episodes Impairs Context Memory",
                ["Kim G", "Norman KA", "Turk-Browne NB"])),
        (_link("10.1/other", "references"),
         _paper("doi:10.1/other", "A systematic review of something else", ["Smith J", "Jones K"])),
    ]
    roles = links.assign_roles(
        accession="ds001430",
        dataset_name="Neural Overlap in Item Representations Across Episodes Impairs Context Memory",
        dataset_authors=["Kim, G., Norman, K.A., Turk-Browne, N.B."],
        links=pairs,
    )
    assert [role for role, _ in roles] == ["tool", "primary", "related"]
    assert "MNE-BIDS" in roles[0][1]


def test_only_paper_needs_a_shared_author() -> None:
    # A stimulus paper cited in the README is not the dataset's paper.
    inscapes = _paper("doi:10.1/inscapes", "Inscapes: A movie paradigm to improve compliance",
                      ["Vanderwal T", "Kelly C"])
    roles = links.assign_roles(
        accession="ds004697", dataset_name="Food and Brain Study",
        dataset_authors=["Bari Fuchs", "Alaina Pearce"], links=[(_link("10.1/inscapes", "readme"), inscapes)])
    assert roles[0][0] == "related"

    own = _paper("doi:10.1/own", "Children's portion size and brain response", ["Fuchs BA", "Pearce AL"])
    roles = links.assign_roles(
        accession="ds004697", dataset_name="Food and Brain Study",
        dataset_authors=["Bari Fuchs", "Alaina Pearce"], links=[(_link("10.1/own", "readme"), own)])
    assert roles[0][0] == "primary"


def test_one_shared_lab_head_is_not_enough_among_several_papers() -> None:
    pairs = [
        (_link("10.1/a", "references"),
         _paper("doi:10.1/a", "A longitudinal dataset on lexical processing in school-aged children",
                ["Lytle MN", "McNorgan C", "Booth JR"])),
        (_link("10.1/b", "references"),
         _paper("doi:10.1/b", "Reward modulates working memory in adults",
                ["Cooke GE", "Gayda J", "Booth JR"])),
    ]
    roles = links.assign_roles(
        accession="ds002687", dataset_name="Working Memory and Reward in Adults",
        dataset_authors=["James R. Booth", "Gillian E. Cooke", "Jessica Gayda", "Other Person", "Fifth Person"],
        links=pairs)
    assert [role for role, _ in roles] == ["related", "primary"]


def test_associated_field_wins_and_reuse_field_alone_is_reuse() -> None:
    pairs = [
        (_link("10.1/assoc", "associated"), _paper("doi:10.1/assoc", "Anything", ["Nobody X"])),
        (_link("10.1/reuse", "reuse_field"), _paper("doi:10.1/reuse", "A reanalysis", ["Stranger Y"])),
    ]
    roles = links.assign_roles(accession="ds1", dataset_name="Some data", dataset_authors=["Owner Z"], links=pairs)
    assert [role for role, _ in roles] == ["primary", "reuse"]


# ── Lookup ────────────────────────────────────────────────────────────────────

_PREPRINT = {
    "id": "PPR727000", "source": "PPR", "doi": "10.1101/2023.09.21.558825",
    "title": "Placebo treatment affects brain systems", "abstractText": "<p>Preprint abstract.</p>",
    "pubYear": "2023", "authorList": {"author": [{"fullName": "Botvinik-Nezer R"}]},
    "commentCorrectionList": {"commentCorrection": [
        {"source": "MED", "id": "39019888", "type": "Preprint of"}]},
}
_PUBLISHED = {
    "id": "39019888", "source": "MED", "pmid": "39019888", "pmcid": "PMC11255344",
    "doi": "10.1038/s41467-024-50103-8",
    "title": "Placebo treatment affects brain systems related to affective and cognitive processes.",
    "abstractText": "We analyzed placebo analgesia in a large fMRI study (N = 392).",
    "pubYear": "2024", "journalInfo": {"journal": {"title": "Nature communications"}},
    "authorList": {"author": [{"fullName": "Botvinik-Nezer R"}, {"fullName": "Wager TD"}]},
    "meshHeadingList": {"meshHeading": [
        {"descriptorName": "Humans", "majorTopic_YN": "N"},
        {"descriptorName": "Placebo Effect", "majorTopic_YN": "Y"}]},
    "keywordList": {"keyword": ["placebo"]}, "isOpenAccess": "Y", "license": "cc by",
}


def _fake_fetch(url: str):
    if "europepmc" in url:
        results = []
        if "558825" in url or "PPR" in url:
            results.append(_PREPRINT)
        if "39019888" in url or "s41467-024-50103-8" in url:
            results.append(_PUBLISHED)
        return {"resultList": {"result": results}}
    return None


def test_preprint_and_published_version_become_one_paper() -> None:
    preprint = PaperId("doi", "10.1101/2023.09.21.558825")
    published = PaperId("doi", "10.1038/s41467-024-50103-8")
    unknown = PaperId("doi", "10.9999/nothing")
    key_of, papers = lookup.resolve([preprint, published, unknown], _fake_fetch, use_openalex=False)

    assert key_of[preprint] == key_of[published] == "doi:10.1038/s41467-024-50103-8"
    assert unknown not in key_of
    paper = papers["doi:10.1038/s41467-024-50103-8"]
    assert paper["preprint_doi"] == "10.1101/2023.09.21.558825"
    assert paper["is_preprint"] is False
    assert paper["mesh_major"] == ["Placebo Effect"]
    assert paper["abstract"].startswith("We analyzed")
    assert paper["journal"] == "Nature communications"


def test_clean_text_handles_markup_and_double_escaping() -> None:
    assert lookup.clean_text("First-in-human evaluation of [&amp;lt;sup&amp;gt;11&amp;lt;/sup&amp;gt;C]PS13") \
        == "First-in-human evaluation of [ 11 C]PS13"
    assert lookup.clean_text("<h4>Background</h4>Text\n here") == "Background Text here"
    assert lookup.clean_text("   ") is None


def test_openalex_abstract_is_rebuilt_in_order() -> None:
    assert lookup.abstract_from_inverted_index({"brain": [1, 3], "The": [0], "is": [2]}) == "The brain is brain"


# ── Stored text ───────────────────────────────────────────────────────────────

def test_paper_text_parts_and_cap() -> None:
    paper = {"title": "A title", "abstract": "An abstract.", "mesh_terms": ["Humans", "Child", "Chronic Pain"],
             "keywords": ["fMRI"]}
    assert store.paper_text([paper]) == "A title. An abstract. Child; Chronic Pain. fMRI."
    assert store.paper_text([paper], parts=("title",)) == "A title."
    assert store.paper_text([]) is None
    many = [dict(paper, title=f"Paper {i}") for i in range(9)]
    assert len(re.findall(r"Paper \d", store.paper_text(many))) == store.MAX_INDEXED_PAPERS


def test_dataset_entries_merge_identifiers_of_one_paper() -> None:
    preprint = PaperId("doi", "10.1101/2023.09.21.558825")
    published = PaperId("doi", "10.1038/s41467-024-50103-8")
    key = "doi:10.1038/s41467-024-50103-8"
    dataset = {
        "accession": "ds004746", "name": "Paingen_placebo", "authors": ["Rotem Botvinik-Nezer", "Tor Wager"],
        "links": links.collect_links(
            associated="https://doi.org/10.1038/s41467-024-50103-8",
            references=["Preprint: https://doi.org/10.1101/2023.09.21.558825"]),
    }
    papers = {key: {"key": key, "doi": published.value, "title": "Placebo treatment",
                    "authors": ["Botvinik-Nezer R", "Wager TD"]}}
    entries = store.dataset_entries(dataset, {preprint: key, published: key}, papers, {key: 1})
    assert len(entries) == 1
    assert entries[0]["role"] == "primary"
    assert entries[0]["sources"] == ["associated", "references"]


# ── Search column and switch ──────────────────────────────────────────────────

def test_migration_matches_the_paper_search_expression() -> None:
    def squash(value: str) -> str:
        return re.sub(r"\s+", "", value)

    migration = (ROOT / "crawlers" / "migrate_add_paper_search.sql").read_text()
    assert squash(store.PAPER_SEARCH_VECTOR_EXPRESSION) in squash(migration)
    for statement in store.DDL:
        assert squash(statement) in squash(migration)
    # The paper text is the only addition to search_tsv, at the lowest weight.
    from remote.openneuro_metadata import SEARCH_VECTOR_EXPRESSION
    assert store.PAPER_SEARCH_VECTOR_EXPRESSION.startswith(SEARCH_VECTOR_EXPRESSION + " || ")
    assert store.PAPER_SEARCH_VECTOR_EXPRESSION.endswith("'D')")


def _text_search():
    backend = ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from services import schema_capabilities, text_search
    return schema_capabilities, text_search


_SQL = (
    "SELECT d.id FROM bids_datasets d WHERE text_match('chronic pain') "
    "AND NOT text_match('migraine') "
    "AND NOT (d.age_min > 60 OR text_match('dementia')) "
    "AND (text_match('placebo') OR d.name ILIKE '%pain%')"
)


def test_papers_switch_widens_positive_matches_only(monkeypatch) -> None:
    capabilities, text_search = _text_search()
    off = text_search.expand_text_search(_SQL, fts=True, papers=False)
    on = text_search.expand_text_search(_SQL, fts=True, papers=True)

    assert "search_tsv_papers" not in off.sql
    assert on.sql.count("d.search_tsv_papers @@") == 2          # chronic pain, placebo
    assert "NOT d.search_tsv @@" in on.sql                      # migraine
    assert "OR d.search_tsv @@" in on.sql                       # dementia, inside NOT ( … )
    # Ranking never reads paper text.
    assert on.rank_tsquery == off.rank_tsquery
    # The benchmark's rewrite of the off-SQL gives the same as expanding with papers on.
    assert text_search.with_paper_text(off.sql) == on.sql


def test_papers_switch_needs_the_flag_and_the_column(monkeypatch) -> None:
    capabilities, text_search = _text_search()
    monkeypatch.setattr(capabilities, "_has_paper_fts", True)
    monkeypatch.setenv("TEXT_SEARCH_PAPERS", "true")
    assert text_search.papers_enabled()
    monkeypatch.setenv("TEXT_SEARCH_PAPERS", "false")
    assert not text_search.papers_enabled()
    monkeypatch.setenv("TEXT_SEARCH_PAPERS", "true")
    monkeypatch.setattr(capabilities, "_has_paper_fts", False)
    assert not text_search.papers_enabled()
    # Without search_tsv at all the switch changes nothing.
    sql = text_search.expand_text_search(_SQL, fts=False, papers=True).sql
    assert "search_tsv" not in sql


def test_negated_spans_ignore_parentheses_in_literals() -> None:
    _, text_search = _text_search()
    sql = "SELECT 1 WHERE NOT (a = '(x)' OR b) AND (c OR NOT(d))"
    spans = text_search.negated_spans(sql)
    assert [sql[start:end + 1] for start, end in sorted(spans)] == ["(a = '(x)' OR b)", "(d)"]


def test_overrides_survive_a_fresh_plan() -> None:
    entries = {"ds1": [{"key": "doi:10.1/a", "role": "primary", "reason": "only paper", "sources": [], "raw": None}]}
    overrides = {("uuid-1", "doi:10.1/a"): ("related", store.OVERRIDE_PREFIX + "paper-relation check: tool")}
    assert store.apply_overrides(entries, overrides, {"ds1": "uuid-1"}) == 1
    assert entries["ds1"][0]["role"] == "related"
    assert entries["ds1"][0]["reason"].startswith(store.OVERRIDE_PREFIX)
