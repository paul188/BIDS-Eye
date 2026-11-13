-- Migration: what an LLM read from each dataset's README.
--
-- Filled by  python scripts/extract_readme_profiles.py
-- (backend/services/readme_profile.py has the prompt and the evidence check).
--
--   llm_profile       short profile in standard terminology; full-text
--                     indexed (weight C) and read by the search re-ranker
--   llm_conditions    conditions the dataset is about (patients or animal
--                     model, quote verified); full-text indexed with the
--                     uploader's diagnosis field (weight B)
--   llm_profile_meta  model, prompt version, hash of the input, raw answer
--   bids_dataset_facts  every condition and paradigm with its role, the
--                     verbatim evidence quote, whether the quote was found in
--                     the input, and canonical diagnosis codes
--
-- Run BEFORE migrate_add_search_tsv.sql: the search vector reads
-- llm_profile and llm_conditions. Every statement is idempotent.
--
--   psql -h localhost -p 5429 -U user -d bids_sql -f crawlers/migrate_add_readme_profiles.sql

BEGIN;

ALTER TABLE bids_datasets
    ADD COLUMN IF NOT EXISTS llm_profile      TEXT,
    ADD COLUMN IF NOT EXISTS llm_conditions   TEXT,
    ADD COLUMN IF NOT EXISTS llm_profile_meta JSONB;

CREATE TABLE IF NOT EXISTS bids_dataset_facts (
    id          BIGSERIAL PRIMARY KEY,
    dataset_id  UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('condition', 'paradigm')),
    term        TEXT NOT NULL,
    role        TEXT,            -- conditions: patients, model, controls, excluded, measured, mentioned
    code        TEXT,            -- canonical diagnosis code (value_mappings.yaml), NULL if none
    score       REAL,            -- resolver match score of code
    evidence    TEXT,            -- verbatim quote from the dataset text
    verified    BOOLEAN NOT NULL -- evidence was found in the text the model read
);
CREATE INDEX IF NOT EXISTS idx_bids_dataset_facts_dataset ON bids_dataset_facts (dataset_id);
CREATE INDEX IF NOT EXISTS idx_bids_dataset_facts_code ON bids_dataset_facts (kind, code)
    WHERE verified;

COMMIT;
