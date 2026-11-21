"""Executed SQL reads the per-dataset summaries instead of the file table (sql_rewriter._use_summaries)."""
import pytest

from services.catalogue.catalogue import Summaries, catalogue
from services.translate.sql_rewriter import SqlRewriteService

BASE = ("SELECT d.id, d.name, COUNT(DISTINCT o.subject) AS subject_count FROM bids_datasets AS d "
        "LEFT JOIN bids_objects AS o ON o.dataset_id = d.id AND NOT o.subject IS NULL ")
FMRI = "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id AND (o2.datatype = 'functional_mri' OR o2.suffix = 'fmri_bold'))"


@pytest.fixture
def svc():
    catalogue.summaries = Summaries(facets=True, subject_counts=True)
    yield SqlRewriteService.__new__(SqlRewriteService)
    catalogue.summaries = Summaries(facets=False, subject_counts=False)


def test_nothing_changes_without_the_summaries():
    catalogue.summaries = Summaries(facets=False, subject_counts=False)
    sql = BASE + "WHERE " + FMRI + " GROUP BY d.id"
    assert SqlRewriteService.__new__(SqlRewriteService)._use_summaries(sql) == sql


def test_filter_and_subject_count_leave_the_file_table(svc):
    out = svc._use_summaries(BASE + "WHERE " + FMRI + " GROUP BY d.id HAVING COUNT(DISTINCT o.subject) >= 20")
    assert "FROM bids_dataset_facets AS o2" in out
    assert "LEFT JOIN" not in out and "GROUP BY" not in out and "HAVING" not in out
    # the stored number, with the live count only for a dataset not summarised yet
    assert out.count("COALESCE(d.n_subjects, (SELECT COUNT(DISTINCT sc.subject) FROM bids_objects AS sc") == 2
    assert ">= 20" in out


def test_the_join_stays_when_the_files_are_used_for_more(svc):
    used = BASE + "WHERE o.datatype = 'functional_mri' GROUP BY d.id"
    assert "LEFT JOIN bids_objects AS o" in svc._use_summaries(used)
    other_aggregate = ("SELECT d.id, COUNT(DISTINCT o.subject) AS subject_count, COUNT(DISTINCT o.session) AS n "
                       "FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id GROUP BY d.id")
    assert svc._use_summaries(other_aggregate) == other_aggregate


def test_grouping_stays_next_to_another_join(svc):
    sql = ("SELECT d.id, COUNT(DISTINCT o.subject) AS subject_count FROM bids_datasets AS d "
           "LEFT JOIN bids_objects AS o ON o.dataset_id = d.id AND NOT o.subject IS NULL "
           "JOIN bids_dataset_papers AS dp ON dp.dataset_id = d.id GROUP BY d.id")
    out = svc._use_summaries(sql)
    assert "bids_objects AS o " not in out and "GROUP BY d.id" in out and "bids_dataset_papers" in out


def test_only_conditions_on_datatype_suffix_task_use_the_facets(svc):
    sidecar = "EXISTS(SELECT 1 FROM bids_objects AS o3 WHERE o3.dataset_id = d.id AND o3.other_entities ->> 'Manufacturer' = 'GE')"
    per_subject = ("d.id NOT IN (SELECT s.dataset_id FROM bids_objects AS s WHERE NOT s.subject IS NULL "
                   "GROUP BY s.dataset_id, s.subject HAVING COUNT(DISTINCT s.session) < 3)")
    listed = "d.id IN (SELECT x.dataset_id FROM bids_objects AS x WHERE x.task = 'rest')"
    out = svc._use_summaries(BASE + f"WHERE {FMRI} AND {sidecar} AND {per_subject} AND {listed} GROUP BY d.id")
    assert "FROM bids_dataset_facets AS o2" in out and "FROM bids_dataset_facets AS x" in out
    assert "FROM bids_objects AS o3" in out and "FROM bids_objects AS s" in out


def test_ranking_score_reads_the_facets(svc):
    base = BASE + "WHERE " + FMRI + " GROUP BY d.id"
    page, _ = svc.build_page_sql(base, [{"field": "task", "code": "rest", "score": 1.0}], True, 20, 0)
    assert "bids_dataset_facets AS mt" in page and "bids_objects AS mt" not in page
    assert "FROM bids_dataset_facets" in svc.build_count_sql(base)
