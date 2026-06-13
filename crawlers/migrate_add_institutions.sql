-- Migration: add institutions column to bids_datasets
-- Run once against the live DB, then run scripts/backfill_institutions.py.
--   psql -h localhost -p 5429 -U user -d bids_sql -f crawlers/migrate_add_institutions.sql

BEGIN;

ALTER TABLE bids_datasets
    ADD COLUMN IF NOT EXISTS institutions TEXT[];

CREATE INDEX IF NOT EXISTS ix_bids_datasets_institutions ON bids_datasets USING GIN (institutions);

COMMIT;
