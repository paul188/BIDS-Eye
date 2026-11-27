-- Read-only: the values the ordering gold requests sort by, one JSON line per dataset.
-- subjects is computed as the search computes subject_count (distinct subjects in bids_objects).
SELECT json_build_object(
  'acc', d.accession_id, 'name', d.name,
  'subjects', (SELECT count(DISTINCT o.subject) FROM bids_objects o WHERE o.dataset_id = d.id AND o.subject IS NOT NULL),
  'n_subjects_stored', d.n_subjects,
  'size', d.size_bytes, 'published', d.publish_date, 'downloads', d.downloads, 'views', d.views)
FROM bids_datasets d ORDER BY d.accession_id;
