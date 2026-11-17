-- Two combined indexes on the file table, so a search no longer reads all of it.
--
-- Every search counts the subjects of the datasets it returns
-- (COUNT(DISTINCT o.subject) over the join to bids_objects) and many check
-- "has files of type X" per dataset. With only single-column indexes the
-- count scanned the whole table (2.7 million rows, 2.9 GB) even for 35
-- matching datasets. Measured on 167 real search statements, on a copy sized
-- like the server (2026-11-17): median 797 ms -> 53 ms, 90th percentile
-- 1,940 ms -> 422 ms; same row counts.
--
-- Additive: nothing else changes. Undo with DROP INDEX CONCURRENTLY.
-- Applied on production 2026-11-17 (21 MB each, 3 and 4 seconds).
--
-- Run each statement on its own (CONCURRENTLY cannot run in a transaction
-- block), with `SET statement_timeout = 0` on a database whose role has a
-- timeout. CONCURRENTLY does not block searches or the crawler.
--
-- The count is an index-only read, which needs the table vacuumed. After a
-- large crawl: VACUUM (PARALLEL 0, ANALYZE) bids_objects;
-- (PARALLEL 0: the postgres container's 64 MB /dev/shm is too small for
-- parallel vacuum workers.)

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_dataset_subject
    ON bids_objects (dataset_id, subject);

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_bids_objects_dataset_filetype
    ON bids_objects (dataset_id, datatype, suffix, task);
