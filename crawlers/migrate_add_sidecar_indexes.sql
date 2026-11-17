-- Indexes for searches on scanner and acquisition fields of the file sidecars
-- ("7 Tesla", "TR under 0.5 s", "Philips", "Prisma").
--
-- bids_objects.other_entities is JSON text of about 1.3 kB per file. Without
-- these indexes such a search parses it for every file of every dataset:
-- 3 to 28 seconds on a copy sized like the server (2026-11-17). With them:
--
--   field strength 7 T              27.7 s -> 0.03 s
--   field strength 9.4 T or more    14.6 s -> 0.03 s
--   diffusion at 1.5 T               3.2 s -> 0.03 s
--   repetition time > 2 s            7.1 s -> 0.2 s
--   BOLD with TR < 0.5 s            18.5 s -> 0.2 s
--   sampling frequency >= 2000 Hz   21.8 s -> 0.2 s
--   Manufacturer = 'Bruker'         21.4 s -> 0.2 s
--   Manufacturer ILIKE '%philips%'  15.2 s -> 1.3 s
--   model ILIKE '%prisma%'          13.9 s -> 0.001 s
--
-- Numbers: the index is on the exact expression the SQL rewriter writes for a
-- number cast of a sidecar value (backend/services/sql_rewriter.py,
-- _guard_json_number_casts: the number pattern, DOUBLE PRECISION). PostgreSQL
-- uses an expression index only for the same expression, so
-- backend/tests/test_sidecar_indexes.py fails when the two drift apart.
-- A search for another field, or written another way, works as before, slowly.
--
-- Text: a trigram index serves ILIKE '%...%' and ~*; the plain one serves =.
--
-- Additive: an extension and indexes, kept current by PostgreSQL itself.
-- Undo with DROP INDEX CONCURRENTLY. About 65 MB together. Each build reads
-- the whole file table (15 to 40 seconds each on the copy; slower on the
-- server). Run each statement on its own, with `SET statement_timeout = 0`.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_field_strength
    ON bids_objects (dataset_id, (CASE WHEN (other_entities ->> 'MagneticFieldStrength') ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$' THEN CAST((other_entities ->> 'MagneticFieldStrength') AS DOUBLE PRECISION) END))
    WHERE (CASE WHEN (other_entities ->> 'MagneticFieldStrength') ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$' THEN CAST((other_entities ->> 'MagneticFieldStrength') AS DOUBLE PRECISION) END) IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_repetition_time
    ON bids_objects (dataset_id, (CASE WHEN (other_entities ->> 'RepetitionTime') ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$' THEN CAST((other_entities ->> 'RepetitionTime') AS DOUBLE PRECISION) END))
    WHERE (CASE WHEN (other_entities ->> 'RepetitionTime') ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$' THEN CAST((other_entities ->> 'RepetitionTime') AS DOUBLE PRECISION) END) IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_sampling_frequency
    ON bids_objects (dataset_id, (CASE WHEN (other_entities ->> 'SamplingFrequency') ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$' THEN CAST((other_entities ->> 'SamplingFrequency') AS DOUBLE PRECISION) END))
    WHERE (CASE WHEN (other_entities ->> 'SamplingFrequency') ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$' THEN CAST((other_entities ->> 'SamplingFrequency') AS DOUBLE PRECISION) END) IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_manufacturer
    ON bids_objects (dataset_id, (other_entities ->> 'Manufacturer'))
    WHERE (other_entities ->> 'Manufacturer') IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_manufacturer_trgm
    ON bids_objects USING gin ((other_entities ->> 'Manufacturer') gin_trgm_ops)
    WHERE (other_entities ->> 'Manufacturer') IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_scanner_model_trgm
    ON bids_objects USING gin ((other_entities ->> 'ManufacturerModelName') gin_trgm_ops)
    WHERE (other_entities ->> 'ManufacturerModelName') IS NOT NULL;
