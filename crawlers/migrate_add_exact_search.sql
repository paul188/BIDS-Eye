-- Migration: unstemmed search columns and the catalogue's word forms.
--
-- search_tsv is built with the 'english' stemmer, so a searched word matches
-- every word with the same stem: "ageing" finds "age" and "aged", "weight"
-- finds "T1-weighted", "severe" finds "several". Where that changes the
-- meaning, the backend also requires one of the word forms that were asked
-- for (backend/services/text_search.py, TEXT_SEARCH_EXACT=true):
--   (d.search_tsv @@ to_tsquery('english', 'ageing')
--    AND d.search_tsv_exact @@ to_tsquery('simple', 'ageing | aging'))
--
-- Adds
--   bids_datasets.search_tsv_exact         search_tsv without stemming
--   bids_datasets.search_tsv_exact_papers  search_tsv_papers without stemming
--   bids_word_forms                        every word form in the catalogue
--                                          with its stem and how many datasets
--                                          contain it
-- Both columns keep the fields and weights (A, B, C, D) of their stemmed
-- twins and differ only in the text search configuration, 'simple'. They are
-- GENERATED, so PostgreSQL keeps them current.
--
-- bids_word_forms is filled by  python scripts/refresh_word_forms.py  and
-- after each crawl (crawlers/openneuro-crawler/post_crawl.py). It shows which
-- forms a stem merges, which is what decides whether a searched word needs
-- its exact forms (RAG/word_forms.py).
--
-- Run AFTER migrate_add_search_tsv.sql and, for the second column,
-- migrate_add_paper_search.sql. The three parts are separate transactions:
-- on a database without paper_text the second one fails and the others stand.
--
--   psql -h localhost -p 5429 -U user -d bids_sql -f crawlers/migrate_add_exact_search.sql
--
-- The columns are dropped and recreated on every run, so a changed definition
-- takes effect. That rewrites the table under an exclusive lock: about two
-- seconds for 1,700 datasets. Restart the backend afterwards; it detects the
-- columns and the table at startup.
--
-- Keep the expressions in sync with EXACT_SEARCH_VECTOR_EXPRESSION in
-- crawlers/openneuro-crawler/remote/openneuro_metadata.py,
-- PAPER_EXACT_SEARCH_VECTOR_EXPRESSION in remote/paper_store.py and
-- TABLES_SQL in backend/services/fulltext/word_form_counts.py
-- (backend/tests/test_word_forms.py checks this).

BEGIN;

ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv_exact;

ALTER TABLE bids_datasets
    ADD COLUMN search_tsv_exact tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('simple'::regconfig,
            translate(coalesce(name, '')::text, '/', ' ')), 'A') ||
        setweight(to_tsvector('simple'::regconfig, translate(
            coalesce(dx_status, '') || ' ' || coalesce(study_domain, '') || ' ' ||
            coalesce(study_design, '') || ' ' || coalesce(description_text, '') || ' ' ||
            coalesce(llm_conditions, ''), '/', ' ')), 'B') ||
        setweight(to_tsvector('simple'::regconfig,
            translate(coalesce(llm_profile, ''), '/', ' ')), 'C') ||
        setweight(to_tsvector('simple'::regconfig,
            translate(coalesce(readme_text, ''), '/', ' ')), 'D')
    ) STORED;

CREATE INDEX idx_bids_datasets_search_tsv_exact
    ON bids_datasets USING GIN (search_tsv_exact);

COMMIT;

BEGIN;

ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv_exact_papers;

ALTER TABLE bids_datasets
    ADD COLUMN search_tsv_exact_papers tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('simple'::regconfig, translate(coalesce(name, '')::text, '/', ' ')), 'A') ||
        setweight(to_tsvector('simple'::regconfig, translate(
            coalesce(dx_status, '') || ' ' || coalesce(study_domain, '') || ' ' ||
            coalesce(study_design, '') || ' ' || coalesce(description_text, '') || ' ' ||
            coalesce(llm_conditions, ''), '/', ' ')), 'B') ||
        setweight(to_tsvector('simple'::regconfig, translate(coalesce(llm_profile, ''), '/', ' ')), 'C') ||
        setweight(to_tsvector('simple'::regconfig, translate(coalesce(readme_text, ''), '/', ' ')), 'D') ||
        setweight(to_tsvector('simple'::regconfig, translate(coalesce(paper_text, ''), '/', ' ')), 'D')
    ) STORED;

CREATE INDEX idx_bids_datasets_search_tsv_exact_papers
    ON bids_datasets USING GIN (search_tsv_exact_papers);

COMMIT;

BEGIN;

CREATE TABLE IF NOT EXISTS bids_word_forms (
    form  TEXT PRIMARY KEY,
    stem  TEXT NOT NULL,
    ndoc  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bids_word_forms_stem ON bids_word_forms (stem);

COMMIT;

-- Roll back (the backend falls back to stemmed search by itself):
--   ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv_exact_papers;
--   ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv_exact;
--   DROP TABLE IF EXISTS bids_word_forms;
