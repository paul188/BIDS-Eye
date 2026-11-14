-- Migration: dataset-level metadata pulled from the OpenNeuro GraphQL API.
--
-- Adds the README text, the uploader-entered study fields, the reported age
-- range, the dataset size and an upstream-status mark to bids_datasets. Every
-- statement is idempotent,
-- so the file is safe to run more than once.
--
-- Run once against the live DB, then fill the columns with
--   python scripts/backfill_openneuro_metadata.py
--
--   psql -h localhost -p 5429 -U user -d bids_sql -f crawlers/migrate_add_openneuro_metadata.sql
--
-- Keep the column list in sync with METADATA_COLUMNS and STATUS_COLUMNS in
-- crawlers/openneuro-crawler/remote/openneuro_metadata.py
-- (backend/tests/test_openneuro_metadata.py checks this).

BEGIN;

ALTER TABLE bids_datasets
    ADD COLUMN IF NOT EXISTS readme_text  TEXT,              -- sanitised README of the latest snapshot
    ADD COLUMN IF NOT EXISTS dx_status    TEXT,              -- uploader-entered diagnosis status, e.g. 'Healthy / Control'
    ADD COLUMN IF NOT EXISTS study_domain TEXT,              -- uploader-entered research topic
    ADD COLUMN IF NOT EXISTS study_design TEXT,              -- uploader-entered design, e.g. 'Longitudinal'
    ADD COLUMN IF NOT EXISTS species      TEXT,              -- uploader-entered species, e.g. 'Human', 'Mouse'
    ADD COLUMN IF NOT EXISTS study_longitudinal TEXT,        -- uploader-entered 'Cross-Sectional' / 'Longitudinal' / …
    ADD COLUMN IF NOT EXISTS senior_author TEXT,             -- uploader-entered senior author
    ADD COLUMN IF NOT EXISTS publish_date TIMESTAMPTZ,       -- when the dataset went public
    ADD COLUMN IF NOT EXISTS on_brainlife BOOLEAN,           -- processed on brainlife.io
    ADD COLUMN IF NOT EXISTS downloads    INTEGER,           -- OpenNeuro download count (latest snapshot)
    ADD COLUMN IF NOT EXISTS views        INTEGER,           -- OpenNeuro view count (latest snapshot)
    ADD COLUMN IF NOT EXISTS age_min      DOUBLE PRECISION,  -- youngest reported participant age
    ADD COLUMN IF NOT EXISTS age_max      DOUBLE PRECISION,  -- oldest reported participant age
    ADD COLUMN IF NOT EXISTS size_bytes   BIGINT,            -- total size of the latest snapshot
    ADD COLUMN IF NOT EXISTS deprecated_reason TEXT;         -- NULL = fine upstream; else why it is deprecated

COMMIT;
