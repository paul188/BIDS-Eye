-- migrate_add_description_text.sql
-- Adds description_text as a first-class column and strips now-redundant
-- fields from the description JSON blob.
--
-- Run on the compute node (psql $DATABASE_URL -f this_file)

-- 1. Add the new column
ALTER TABLE bids_datasets
    ADD COLUMN IF NOT EXISTS description_text TEXT;

-- 2. Backfill from the JSON blob
--    (->> works on JSON; ? and - are JSONB-only operators)
UPDATE bids_datasets
SET description_text = NULLIF(TRIM(description->>'Description'), '')
WHERE description_text IS NULL
  AND description->>'Description' IS NOT NULL;

-- 3. FTS index on description_text
CREATE INDEX IF NOT EXISTS idx_bids_datasets_description_text_fts
    ON bids_datasets USING GIN (to_tsvector('english', COALESCE(description_text, '')));

-- 4. FTS index on name (if not already present)
CREATE INDEX IF NOT EXISTS idx_bids_datasets_name_fts
    ON bids_datasets USING GIN (to_tsvector('english', COALESCE(name, '')));

-- 5. Strip promoted fields from the JSON blob (cast to jsonb for key removal, cast back)
UPDATE bids_datasets
SET description = (
    (description::jsonb)
    - 'Authors'
    - 'License'
    - 'DatasetDOI'
    - 'ReferencesAndLinks'
    - 'Funding'
    - 'Description'
)::json
WHERE description IS NOT NULL;
