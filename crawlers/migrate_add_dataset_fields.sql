-- Migration: promote key dataset_description.json fields to first-class columns
-- Run once against the live DB:
--   psql -h localhost -p 5429 -U user -d bids_sql -f migrate_add_dataset_fields.sql

BEGIN;

-- 1. Add new columns (all nullable so existing rows are unaffected)
ALTER TABLE bids_datasets
    ADD COLUMN IF NOT EXISTS authors      TEXT[],   -- BIDS Authors array
    ADD COLUMN IF NOT EXISTS license      TEXT,     -- BIDS License string
    ADD COLUMN IF NOT EXISTS doi          TEXT,     -- BIDS DatasetDOI
    ADD COLUMN IF NOT EXISTS paper_references   TEXT[],   -- BIDS ReferencesAndLinks array
    ADD COLUMN IF NOT EXISTS funding      TEXT[];   -- BIDS Funding array

-- 2. Backfill from existing description JSON
--    Authors / ReferencesAndLinks / Funding are JSON arrays → TEXT[]
--    License / DatasetDOI are plain strings → TEXT

UPDATE bids_datasets SET
    authors = ARRAY(
        SELECT json_array_elements_text(description->'Authors')
    )
    WHERE description->'Authors' IS NOT NULL
      AND json_typeof(description->'Authors') = 'array';

UPDATE bids_datasets SET
    license = description->>'License'
    WHERE description->>'License' IS NOT NULL;

UPDATE bids_datasets SET
    doi = description->>'DatasetDOI'
    WHERE description->>'DatasetDOI' IS NOT NULL;

UPDATE bids_datasets SET
    paper_references = ARRAY(
        SELECT json_array_elements_text(description->'ReferencesAndLinks')
    )
    WHERE description->'ReferencesAndLinks' IS NOT NULL
      AND json_typeof(description->'ReferencesAndLinks') = 'array';

UPDATE bids_datasets SET
    funding = ARRAY(
        SELECT json_array_elements_text(description->'Funding')
    )
    WHERE description->'Funding' IS NOT NULL
      AND json_typeof(description->'Funding') = 'array';

-- 3. Indexes
--    GIN on arrays for fast ANY/overlap queries (e.g. 'Poldrack' = ANY(authors))
--    btree on license and doi for equality/range filtering
CREATE INDEX IF NOT EXISTS ix_bids_datasets_authors    ON bids_datasets USING GIN (authors);
CREATE INDEX IF NOT EXISTS ix_bids_datasets_license    ON bids_datasets (license);
CREATE INDEX IF NOT EXISTS ix_bids_datasets_doi        ON bids_datasets (doi);
CREATE INDEX IF NOT EXISTS ix_bids_datasets_references ON bids_datasets USING GIN (paper_references);
CREATE INDEX IF NOT EXISTS ix_bids_datasets_funding    ON bids_datasets USING GIN (funding);

COMMIT;
