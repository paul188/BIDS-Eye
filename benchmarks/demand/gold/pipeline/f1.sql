WITH ss AS (
  SELECT dataset_id, subject, coalesce(session,'') ses,
    bool_or(suffix='t1_weighted_mri' AND class_='image_file') t1,
    bool_or(suffix='fmri_bold' AND class_='image_file') bold
  FROM bids_objects WHERE subject IS NOT NULL GROUP BY 1,2,3),
sub AS (
  SELECT dataset_id, subject, count(*) FILTER (WHERE ses<>'') n_ses, bool_or(t1) t1, bool_or(bold) bold, bool_or(t1 AND bold) same_ses
  FROM ss GROUP BY 1,2),
agg AS (
  SELECT dataset_id, count(*) n_sub, min(n_ses) ses_min, max(n_ses) ses_max,
    count(*) FILTER (WHERE n_ses>=2) n_sub_ge2ses, count(*) FILTER (WHERE n_ses>=3) n_sub_ge3ses,
    count(*) FILTER (WHERE t1) n_sub_t1, count(*) FILTER (WHERE bold) n_sub_bold,
    count(*) FILTER (WHERE t1 AND bold) n_sub_t1_bold, count(*) FILTER (WHERE same_ses) n_sub_t1_bold_same_ses
  FROM sub GROUP BY 1),
sx AS (
  SELECT dataset_id, json_object_agg(suffix, n) per_suffix FROM (
    SELECT dataset_id, suffix, count(DISTINCT subject) n FROM bids_objects
    WHERE subject IS NOT NULL AND class_ IN ('image_file','data_file') AND suffix NOT IN ('','entities_and_processing_keys','task_events')
    GROUP BY 1,2) t GROUP BY 1)
SELECT json_build_object('acc', d.accession_id, 'n_sub', a.n_sub, 'ses_min', a.ses_min, 'ses_max', a.ses_max,
  'n_sub_ge2ses', a.n_sub_ge2ses, 'n_sub_ge3ses', a.n_sub_ge3ses, 'n_sub_t1', a.n_sub_t1, 'n_sub_bold', a.n_sub_bold,
  'n_sub_t1_bold', a.n_sub_t1_bold, 'n_sub_t1_bold_same_ses', a.n_sub_t1_bold_same_ses, 'sub_per_suffix', sx.per_suffix)
FROM bids_datasets d LEFT JOIN agg a ON a.dataset_id=d.id LEFT JOIN sx ON sx.dataset_id=d.id;
