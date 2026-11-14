-- Migration: linked tables for derivatives and collection sites.
--
--   bids_derivatives    one row per processed version of a dataset (pipeline,
--                       version, subject count, where it lives)
--   bids_dataset_sites  one row per site name and the evidence it came from
--                       ('dicom' scanner headers, 'ethics_board' approvals)
--
-- Additive and idempotent. Keep in sync with TABLES_SQL in
-- crawlers/openneuro-crawler/remote/dataset_links.py
-- (backend/tests/test_dataset_links.py checks this).
-- Fill with: python scripts/backfill_dataset_links.py
--
--   psql -h localhost -p 5429 -U user -d bids_sql -f crawlers/migrate_add_dataset_links.sql

BEGIN;

CREATE TABLE IF NOT EXISTS bids_derivatives (
    id               BIGSERIAL PRIMARY KEY,
    dataset_id       UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
    pipeline         TEXT NOT NULL,          -- normalised: 'fmriprep', 'mriqc', 'freesurfer', … 'unspecified'
    pipeline_name    TEXT,                   -- as the source names it (GeneratedBy Name or folder)
    pipeline_version TEXT,                   -- GeneratedBy Version
    source           TEXT NOT NULL,          -- 'openneuro_derivatives' | 'in_dataset'
    location         TEXT NOT NULL,          -- s3://openneuro-derivatives/… or derivatives/<folder>
    n_subjects       INTEGER,                -- sub-* folders in the derivative
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, source, location)
);
CREATE INDEX IF NOT EXISTS idx_bids_derivatives_dataset ON bids_derivatives (dataset_id);
CREATE INDEX IF NOT EXISTS idx_bids_derivatives_pipeline ON bids_derivatives (pipeline);

CREATE TABLE IF NOT EXISTS bids_dataset_sites (
    id          BIGSERIAL PRIMARY KEY,
    dataset_id  UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
    name_raw    TEXT NOT NULL,               -- the name or statement as found
    source      TEXT NOT NULL,               -- 'dicom' | 'ethics_board' (later 'paper_affiliation')
    ror_id      TEXT,                        -- Research Organization Registry id, once resolved
    evidence    TEXT,                        -- quote, where the source is prose
    UNIQUE (dataset_id, source, name_raw)
);
CREATE INDEX IF NOT EXISTS idx_bids_dataset_sites_dataset ON bids_dataset_sites (dataset_id);
CREATE INDEX IF NOT EXISTS idx_bids_dataset_sites_source ON bids_dataset_sites (source);

COMMIT;
