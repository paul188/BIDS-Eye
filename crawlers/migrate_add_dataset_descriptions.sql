-- Migration: how each dataset's primary paper relates to it, and a generated
-- summary for datasets whose README is too thin.
--
-- Adds to bids_datasets
--   paper_relation        {paper_key, relation, shown, mismatches, evidence,
--                          input_sha, prompt_version, model, checked_at}
--                         relation: reports | larger_sample | related_study |
--                         tool | unclear; shown when reports / larger_sample
--                         with a verified quote
--   dataset_summary       2-3 generated sentences, only where README +
--                         description are under 500 characters
--   dataset_summary_meta  {input_sha, paper_used, prompt_version, model,
--                          generated_at, rejected}
-- and drops paper_summary / paper_summary_meta, an earlier design that was
-- never filled.
--
-- Filled by  python scripts/describe_datasets.py  and by the crawler
-- (post_crawl.py); see backend/services/dataset_description.py. Needs the
-- paper tables from migrate_add_paper_search.sql.
--
-- The ADD statements equal DESCRIPTION_DDL in
-- crawlers/openneuro-crawler/remote/paper_store.py, which the crawler applies
-- itself. Idempotent; adding nullable columns does not rewrite the table.
-- Restart the backend afterwards; it detects the columns at startup.

BEGIN;

ALTER TABLE bids_datasets DROP COLUMN IF EXISTS paper_summary;
ALTER TABLE bids_datasets DROP COLUMN IF EXISTS paper_summary_meta;
ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS paper_relation JSONB;
ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS dataset_summary TEXT;
ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS dataset_summary_meta JSONB;

COMMIT;
