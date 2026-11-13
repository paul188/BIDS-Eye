\pset pager off
\pset tuples_only on
\pset format unaligned
WITH obj AS (
  SELECT o.dataset_id,
         array_agg(DISTINCT o.datatype) FILTER (WHERE o.datatype IS NOT NULL AND o.datatype <> '') AS datatypes,
         array_agg(DISTINCT o.task) FILTER (WHERE o.task IS NOT NULL) AS tasks,
         array_agg(DISTINCT o.suffix) FILTER (WHERE o.suffix IS NOT NULL AND o.suffix <> '') AS suffixes,
         count(DISTINCT o.subject) FILTER (WHERE o.subject IS NOT NULL) AS n_subjects,
         array_agg(DISTINCT (o.other_entities->>'MagneticFieldStrength')) FILTER (WHERE o.extension = '.json' AND o.other_entities->>'MagneticFieldStrength' IS NOT NULL) AS field_strengths
  FROM bids_objects o GROUP BY o.dataset_id
), par AS (
  SELECT p.dataset_id,
         array_agg(DISTINCT p.diagnosis) FILTER (WHERE p.diagnosis IS NOT NULL) AS diagnoses,
         min(p.age) AS p_age_min, max(p.age) AS p_age_max, count(*) AS n_participant_rows
  FROM bids_participants p GROUP BY p.dataset_id
)
SELECT json_build_object(
  'acc', d.accession_id, 'id', d.id, 'name', d.name, 'description', d.description_text,
  'readme', d.readme_text, 'dx_status', d.dx_status, 'study_domain', d.study_domain,
  'study_design', d.study_design, 'age_min', d.age_min, 'age_max', d.age_max,
  'size_bytes', d.size_bytes, 'deprecated', d.deprecated_reason,
  'datatypes', obj.datatypes, 'tasks', obj.tasks, 'suffixes', obj.suffixes,
  'n_subjects', obj.n_subjects, 'field_strengths', obj.field_strengths,
  'diagnoses', par.diagnoses, 'p_age_min', par.p_age_min, 'p_age_max', par.p_age_max,
  'n_participant_rows', par.n_participant_rows)
FROM bids_datasets d LEFT JOIN obj ON obj.dataset_id = d.id LEFT JOIN par ON par.dataset_id = d.id
ORDER BY d.accession_id;
