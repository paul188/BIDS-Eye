WITH img AS (
  SELECT dataset_id, suffix, datatype, task, filename, other_entities oe FROM bids_objects
  WHERE class_='image_file'),
a AS (
  SELECT dataset_id,
    array_agg(DISTINCT datatype) FILTER (WHERE datatype IS NOT NULL) datatypes,
    array_agg(DISTINCT suffix) suffixes,
    array_agg(DISTINCT task) FILTER (WHERE task IS NOT NULL AND suffix='fmri_bold') bold_tasks,
    array_agg(DISTINCT lower(substring(filename from 'task-([A-Za-z0-9]+)'))) FILTER (WHERE suffix='fmri_bold' AND filename ~ 'task-') bold_raw_tasks,
    array_agg(DISTINCT round((oe->>'RepetitionTime')::numeric,3)) FILTER (WHERE suffix='fmri_bold' AND oe->>'RepetitionTime' ~ '^[0-9.]+$') bold_tr,
    array_agg(DISTINCT round((oe->>'RepetitionTime')::numeric,3)) FILTER (WHERE suffix='fmri_bold' AND oe->>'RepetitionTime' ~ '^[0-9.]+$' AND filename !~* 'task-[a-z0-9]*rest') bold_tr_task,
    array_agg(DISTINCT round((oe->>'RepetitionTime')::numeric,3)) FILTER (WHERE suffix='fmri_bold' AND oe->>'RepetitionTime' ~ '^[0-9.]+$' AND filename ~* 'task-[a-z0-9]*rest') bold_tr_rest,
    array_agg(DISTINCT round((oe->>'MagneticFieldStrength')::numeric,1)) FILTER (WHERE oe->>'MagneticFieldStrength' ~ '^[0-9.]+$') field,
    array_agg(DISTINCT round((oe->>'MagneticFieldStrength')::numeric,1)) FILTER (WHERE suffix='fmri_bold' AND oe->>'MagneticFieldStrength' ~ '^[0-9.]+$') field_bold,
    array_agg(DISTINCT oe->>'Manufacturer') FILTER (WHERE oe->>'Manufacturer' IS NOT NULL) manufacturer,
    array_agg(DISTINCT oe->>'Manufacturer') FILTER (WHERE suffix='fmri_bold' AND oe->>'Manufacturer' IS NOT NULL) manufacturer_bold,
    array_agg(DISTINCT oe->>'ManufacturersModelName') FILTER (WHERE oe->>'ManufacturersModelName' IS NOT NULL) model,
    array_agg(DISTINCT oe->>'direction') FILTER (WHERE suffix='fmri_bold' AND oe->>'direction' IS NOT NULL) bold_dir,
    array_agg(DISTINCT oe->>'direction') FILTER (WHERE datatype='field_maps' AND oe->>'direction' IS NOT NULL) fmap_dir,
    array_agg(DISTINCT oe->>'PhaseEncodingDirection') FILTER (WHERE suffix='fmri_bold' AND oe->>'PhaseEncodingDirection' IS NOT NULL) bold_ped,
    array_agg(DISTINCT oe->>'PhaseEncodingDirection') FILTER (WHERE suffix='fmri_bold' AND filename ~* 'task-[a-z0-9]*rest' AND oe->>'PhaseEncodingDirection' IS NOT NULL) rest_ped,
    count(*) FILTER (WHERE datatype='field_maps') n_fmap,
    count(*) FILTER (WHERE datatype='field_maps' AND oe->>'EffectiveEchoSpacing' IS NOT NULL) n_fmap_ees,
    count(*) FILTER (WHERE suffix='fmri_bold' AND oe->>'EffectiveEchoSpacing' IS NOT NULL) n_bold_ees,
    array_agg(DISTINCT left(concat_ws(' | ', oe->>'acquisition', oe->>'SequenceName', oe->>'PulseSequenceDetails', oe->>'SeriesDescription', oe->>'ProtocolName', oe->>'ScanningSequence', oe->>'SequenceVariant'), 160)) FILTER (WHERE suffix='t1_weighted_mri') t1_seq,
    array_agg(DISTINCT oe->>'ArterialSpinLabelingType') FILTER (WHERE oe->>'ArterialSpinLabelingType' IS NOT NULL) asl_type,
    array_agg(DISTINCT oe->>'acquisition') FILTER (WHERE suffix IN ('arterial_spin_labelling','mr_angiography','swi') AND oe->>'acquisition' IS NOT NULL) asl_angio_acq,
    array_agg(DISTINCT round((oe->>'SliceThickness')::numeric,2)) FILTER (WHERE suffix='fmri_bold' AND oe->>'SliceThickness' ~ '^[0-9.]+$') bold_slice_thk,
    array_agg(DISTINCT round((oe->>'MultibandAccelerationFactor')::numeric,0)) FILTER (WHERE suffix='fmri_bold' AND oe->>'MultibandAccelerationFactor' ~ '^[0-9.]+$') multiband
  FROM img GROUP BY 1)
SELECT json_build_object('acc', d.accession_id, 'datatypes', a.datatypes, 'suffixes', a.suffixes, 'bold_tasks', a.bold_tasks, 'bold_raw_tasks', a.bold_raw_tasks,
  'bold_tr', a.bold_tr, 'bold_tr_task', a.bold_tr_task, 'bold_tr_rest', a.bold_tr_rest, 'field', a.field, 'field_bold', a.field_bold,
  'manufacturer', a.manufacturer, 'manufacturer_bold', a.manufacturer_bold, 'model', a.model, 'bold_dir', a.bold_dir, 'fmap_dir', a.fmap_dir,
  'bold_ped', a.bold_ped, 'rest_ped', a.rest_ped, 'n_fmap', a.n_fmap, 'n_fmap_ees', a.n_fmap_ees, 'n_bold_ees', a.n_bold_ees,
  't1_seq', a.t1_seq[1:25], 'asl_type', a.asl_type, 'asl_angio_acq', a.asl_angio_acq, 'bold_slice_thk', a.bold_slice_thk, 'multiband', a.multiband)
FROM bids_datasets d LEFT JOIN a ON a.dataset_id=d.id;
