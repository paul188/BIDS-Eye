-- Migration: full-text search column for bids_datasets.
--
-- search_tsv is a stored tsvector over the dataset's text, weighted so that
-- one column serves three search scopes and the ranking:
--   A  name
--   B  dx_status, study_domain, study_design, description_text,
--      llm_conditions (conditions the README says were studied)
--   C  llm_profile (LLM-written profile of the README)
--   D  readme_text
-- The backend expands text_match('...') into
--   d.search_tsv @@ to_tsquery('english', '...')
-- and ranks with ts_rank_cd (backend/services/text_search.py).
--
-- The column is GENERATED, so PostgreSQL keeps it current on every INSERT and
-- UPDATE. The crawler and the backfill scripts need no change.
--
-- "/" is indexed as a space: PostgreSQL reads "ADD/ADHD" as one file-path
-- token, so a search for adhd would miss it.
--
-- Run AFTER migrate_add_openneuro_metadata.sql and
-- migrate_add_readme_profiles.sql (the expression reads the columns those
-- add). Filling llm_profile later needs no rerun: the column is generated.
--
--   psql -h localhost -p 5429 -U user -d bids_sql -f crawlers/migrate_add_search_tsv.sql
--
-- The column is dropped and recreated on every run, so a changed definition
-- takes effect. That rewrites the table under an exclusive lock: about two
-- seconds for 1,700 datasets. Restart the backend afterwards; it detects the
-- column at startup.
--
-- Keep the expression in sync with SEARCH_VECTOR_EXPRESSION in
-- crawlers/openneuro-crawler/remote/openneuro_metadata.py
-- (backend/tests/test_text_search.py checks this).

BEGIN;

-- Recreated on every run, so a changed definition takes effect; the index
-- goes with the column.
ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv;

ALTER TABLE bids_datasets
    ADD COLUMN search_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig,
            translate(coalesce(name, '')::text, '/', ' ')), 'A') ||
        setweight(to_tsvector('english'::regconfig, translate(
            coalesce(dx_status, '') || ' ' || coalesce(study_domain, '') || ' ' ||
            coalesce(study_design, '') || ' ' || coalesce(description_text, '') || ' ' ||
            coalesce(llm_conditions, ''), '/', ' ')), 'B') ||
        setweight(to_tsvector('english'::regconfig,
            translate(coalesce(llm_profile, ''), '/', ' ')), 'C') ||
        setweight(to_tsvector('english'::regconfig,
            translate(coalesce(readme_text, ''), '/', ' ')), 'D')
    ) STORED;

CREATE INDEX idx_bids_datasets_search_tsv
    ON bids_datasets USING GIN (search_tsv);

COMMIT;

-- Roll back:
--   DROP INDEX IF EXISTS idx_bids_datasets_search_tsv;
--   ALTER TABLE bids_datasets DROP COLUMN IF EXISTS search_tsv;
--
-- Optional cleanup. These two expression indexes from
-- migrate_add_description_text.sql were never used (the generated SQL searched
-- with ILIKE) and search_tsv supersedes them:
--   DROP INDEX IF EXISTS idx_bids_datasets_description_text_fts;
--   DROP INDEX IF EXISTS idx_bids_datasets_name_fts;
