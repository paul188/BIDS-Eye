-- Migration: a one-sentence description of each dataset, written by an LLM
-- from the abstract of its primary paper.
--
-- Adds
--   bids_datasets.paper_summary       the sentence shown on the result card
--   bids_datasets.paper_summary_meta  {paper_key, abstract_sha, prompt_version,
--                                      model, generated_at}
--
-- Filled by  python scripts/generate_paper_summaries.py  (needs the paper
-- tables from migrate_add_paper_search.sql). The summary is keyed on the
-- paper's paper_key, not on paper_text, so re-running backfill_papers.py does
-- not orphan it; the script clears summaries whose paper is no longer the
-- dataset's primary paper.
--
-- Same statements as SUMMARY_DDL in crawlers/openneuro-crawler/remote/paper_store.py,
-- which the crawler applies itself. Idempotent; adding nullable columns does
-- not rewrite the table. Restart the backend afterwards; it detects the
-- column at startup.

BEGIN;

ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS paper_summary TEXT;
ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS paper_summary_meta JSONB;

COMMIT;
