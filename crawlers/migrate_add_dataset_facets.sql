-- Per-dataset summaries of bids_objects, so a search does not read the file
-- table (2.7 million rows) to answer "does this dataset have fMRI files" or
-- "how many subjects does it have".
--
--   bids_dataset_facets     one row per distinct (dataset, datatype, suffix, task)
--                           of bids_objects: about 21,000 rows.
--   bids_datasets.n_subjects  COUNT(DISTINCT subject) of the dataset's files.
--
-- Additive: nothing existing is changed. Both are copies and go stale when a
-- crawl changes bids_objects; crawlers/openneuro-crawler/remote/dataset_facets.py
-- refreshes them after each crawl (post_crawl.py) and can check them.
-- The backend needs them at startup and uses them while they cover every
-- dataset (backend/services/catalogue/catalogue.py, checked every 10 minutes).
--
-- On a busy database run the ALTER TABLE as its own short transaction with
-- SET LOCAL lock_timeout = '3s' and retry: searches hold locks on bids_datasets.

CREATE TABLE IF NOT EXISTS bids_dataset_facets (
    dataset_id UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
    datatype   TEXT,
    suffix     VARCHAR,
    task       VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_bids_dataset_facets_dataset  ON bids_dataset_facets (dataset_id);
CREATE INDEX IF NOT EXISTS ix_bids_dataset_facets_task     ON bids_dataset_facets (task);
CREATE INDEX IF NOT EXISTS ix_bids_dataset_facets_suffix   ON bids_dataset_facets (suffix);
CREATE INDEX IF NOT EXISTS ix_bids_dataset_facets_datatype ON bids_dataset_facets (datatype);

ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS n_subjects INTEGER;

-- Backfill (same statements as dataset_facets.refresh for every dataset).
BEGIN;
DELETE FROM bids_dataset_facets;
INSERT INTO bids_dataset_facets (dataset_id, datatype, suffix, task)
SELECT DISTINCT dataset_id, datatype, suffix, task FROM bids_objects;
COMMIT;

UPDATE bids_datasets d
SET n_subjects = c.n
FROM (SELECT d2.id, COUNT(DISTINCT o.subject) AS n
      FROM bids_datasets d2
      LEFT JOIN bids_objects o ON o.dataset_id = d2.id AND o.subject IS NOT NULL
      GROUP BY d2.id) c
WHERE c.id = d.id AND d.n_subjects IS DISTINCT FROM c.n;

ANALYZE bids_dataset_facets;
