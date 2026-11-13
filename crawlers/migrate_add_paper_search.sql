-- Migration: the papers datasets name, and a search vector that includes them.
--
-- Adds
--   bids_papers           one row per paper: title, abstract, MeSH, keywords, …
--   bids_dataset_papers   which dataset names which paper, its role (primary,
--                         related, tool, reuse) and why
--   bids_datasets.paper_text         title, abstract and keywords of the
--                                    dataset's primary papers
--   bids_datasets.papers_checked_at  when the papers were last looked up
--   bids_datasets.search_tsv_papers  search_tsv plus paper_text at weight D
--
-- Filled by  python scripts/backfill_papers.py  (see crawlers/paper_enrichment_plan.md).
-- The backend uses search_tsv_papers only when TEXT_SEARCH_PAPERS=true, and
-- only to match: ranking and NOT terms stay on search_tsv
-- (backend/services/text_search.py).
--
-- Run AFTER migrate_add_search_tsv.sql. Every statement is idempotent except
-- the search column, which is dropped and recreated so a changed definition
-- takes effect (a table rewrite: about two seconds for 1,700 datasets).
-- Restart the backend afterwards; it detects the column at startup.
--
--   psql -h localhost -p 5429 -U user -d bids_sql -f crawlers/migrate_add_paper_search.sql
--
-- Keep the expression in sync with PAPER_SEARCH_VECTOR_EXPRESSION in
-- crawlers/openneuro-crawler/remote/paper_store.py.

BEGIN;

CREATE TABLE IF NOT EXISTS bids_papers (
    id             BIGSERIAL PRIMARY KEY,
    paper_key      TEXT NOT NULL UNIQUE,   -- doi:… / pmid:… / pmcid:… / arxiv:…
    doi            TEXT,                   -- published version
    preprint_doi   TEXT,
    pmid           TEXT,
    pmcid          TEXT,
    arxiv_id       TEXT,
    title          TEXT NOT NULL,
    abstract       TEXT,
    year           INTEGER,
    journal        TEXT,
    is_preprint    BOOLEAN NOT NULL DEFAULT FALSE,  -- no published version known
    authors        TEXT[],
    mesh_terms     TEXT[],
    mesh_major     TEXT[],
    keywords       TEXT[],
    is_open_access BOOLEAN,
    license        TEXT,
    source         TEXT NOT NULL,          -- europepmc, biorxiv, medrxiv, arxiv, openalex
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bids_dataset_papers (
    dataset_id  UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
    paper_id    BIGINT NOT NULL REFERENCES bids_papers(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('primary', 'related', 'tool', 'reuse')),
    role_reason TEXT,
    sources     TEXT[] NOT NULL,           -- associated, references, acknowledge, readme, reuse_field
    raw_ref     TEXT,                      -- the reference string the link was read from
    position    INTEGER NOT NULL DEFAULT 0, -- order within the dataset, most trusted first
    PRIMARY KEY (dataset_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_bids_dataset_papers_paper ON bids_dataset_papers (paper_id);

ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS paper_text TEXT;
ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS papers_checked_at TIMESTAMPTZ;

ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv_papers;

ALTER TABLE bids_datasets
    ADD COLUMN search_tsv_papers tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig, translate(coalesce(name, '')::text, '/', ' ')), 'A') ||
        setweight(to_tsvector('english'::regconfig, translate(
            coalesce(dx_status, '') || ' ' || coalesce(study_domain, '') || ' ' ||
            coalesce(study_design, '') || ' ' || coalesce(description_text, '') || ' ' ||
            coalesce(llm_conditions, ''), '/', ' ')), 'B') ||
        setweight(to_tsvector('english'::regconfig, translate(coalesce(llm_profile, ''), '/', ' ')), 'C') ||
        setweight(to_tsvector('english'::regconfig, translate(coalesce(readme_text, ''), '/', ' ')), 'D') ||
        setweight(to_tsvector('english'::regconfig, translate(coalesce(paper_text, ''), '/', ' ')), 'D')
    ) STORED;

CREATE INDEX idx_bids_datasets_search_tsv_papers
    ON bids_datasets USING GIN (search_tsv_papers);

COMMIT;

-- Roll back (the backend falls back to search_tsv by itself):
--   ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv_papers;
--   ALTER TABLE bids_datasets DROP COLUMN IF EXISTS paper_text, DROP COLUMN IF EXISTS papers_checked_at;
--   DROP TABLE IF EXISTS bids_dataset_papers, bids_papers;
