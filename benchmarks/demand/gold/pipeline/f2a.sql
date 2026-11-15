WITH a AS (
  SELECT dataset_id,
    array_agg(DISTINCT task) FILTER (WHERE task IS NOT NULL) bold_tasks,
    array_agg(DISTINCT lower(substring(filename from 'task-([A-Za-z0-9]+)'))) FILTER (WHERE filename ~ 'task-') bold_raw_tasks,
    array_agg(DISTINCT round((oe->>'RepetitionTime')::numeric,3)) FILTER (WHERE oe->>'RepetitionTime' ~ '^[0-9.]+$') bold_tr,
    array_agg(DISTINCT round((oe->>'RepetitionTime')::numeric,3)) FILTER (WHERE oe->>'RepetitionTime' ~ '^[0-9.]+$' AND filename !~* 'task-[a-z0-9]*rest') bold_tr_task,
    array_agg(DISTINCT round((oe->>'RepetitionTime')::numeric,3)) FILTER (WHERE oe->>'RepetitionTime' ~ '^[0-9.]+$' AND filename ~* 'task-[a-z0-9]*rest') bold_tr_rest,
    array_agg(DISTINCT round((oe->>'MagneticFieldStrength')::numeric,1)) FILTER (WHERE oe->>'MagneticFieldStrength' ~ '^[0-9.]+$') field_bold,
    array_agg(DISTINCT round((oe->>'MagneticFieldStrength')::numeric,1)) FILTER (WHERE oe->>'MagneticFieldStrength' ~ '^[0-9.]+$' AND filename ~* 'task-[a-z0-9]*rest') field_rest,
    array_agg(DISTINCT oe->>'Manufacturer') FILTER (WHERE oe->>'Manufacturer' IS NOT NULL) manufacturer_bold,
    array_agg(DISTINCT oe->>'direction') FILTER (WHERE oe->>'direction' IS NOT NULL) bold_dir,
    array_agg(DISTINCT oe->>'direction') FILTER (WHERE oe->>'direction' IS NOT NULL AND filename ~* 'task-[a-z0-9]*rest') rest_dir,
    array_agg(DISTINCT oe->>'PhaseEncodingDirection') FILTER (WHERE oe->>'PhaseEncodingDirection' IS NOT NULL) bold_ped,
    array_agg(DISTINCT oe->>'PhaseEncodingDirection') FILTER (WHERE filename ~* 'task-[a-z0-9]*rest' AND oe->>'PhaseEncodingDirection' IS NOT NULL) rest_ped,
    count(*) FILTER (WHERE oe->>'EffectiveEchoSpacing' IS NOT NULL) n_bold_ees,
    array_agg(DISTINCT round((oe->>'SliceThickness')::numeric,2)) FILTER (WHERE oe->>'SliceThickness' ~ '^[0-9.]+$') bold_slice_thk,
    count(*) n_bold_files, count(*) FILTER (WHERE filename ~* 'task-[a-z0-9]*rest') n_rest_files
  FROM (SELECT dataset_id, task, filename, other_entities oe FROM bids_objects WHERE suffix='fmri_bold' AND class_='image_file') i GROUP BY 1)
SELECT json_build_object('acc', d.accession_id, 'bold_tasks', a.bold_tasks, 'bold_raw_tasks', a.bold_raw_tasks, 'bold_tr', a.bold_tr, 'bold_tr_task', a.bold_tr_task,
  'bold_tr_rest', a.bold_tr_rest, 'field_bold', a.field_bold, 'field_rest', a.field_rest, 'manufacturer_bold', a.manufacturer_bold, 'bold_dir', a.bold_dir, 'rest_dir', a.rest_dir,
  'bold_ped', a.bold_ped, 'rest_ped', a.rest_ped, 'n_bold_ees', a.n_bold_ees, 'bold_slice_thk', a.bold_slice_thk, 'n_bold_files', a.n_bold_files, 'n_rest_files', a.n_rest_files)
FROM bids_datasets d JOIN a ON a.dataset_id=d.id;
