WITH a AS (
  SELECT dataset_id,
    array_agg(DISTINCT round((oe->>'MagneticFieldStrength')::numeric,1)) FILTER (WHERE oe->>'MagneticFieldStrength' ~ '^[0-9.]+$') field,
    array_agg(DISTINCT round((oe->>'MagneticFieldStrength')::numeric,1)) FILTER (WHERE suffix='t1_weighted_mri' AND oe->>'MagneticFieldStrength' ~ '^[0-9.]+$') field_t1,
    array_agg(DISTINCT oe->>'Manufacturer') FILTER (WHERE oe->>'Manufacturer' IS NOT NULL) manufacturer,
    array_agg(DISTINCT oe->>'ManufacturersModelName') FILTER (WHERE oe->>'ManufacturersModelName' IS NOT NULL) model,
    array_agg(DISTINCT oe->>'direction') FILTER (WHERE datatype='field_maps' AND oe->>'direction' IS NOT NULL) fmap_dir,
    count(*) FILTER (WHERE datatype='field_maps') n_fmap,
    count(*) FILTER (WHERE datatype='field_maps' AND oe->>'EffectiveEchoSpacing' IS NOT NULL) n_fmap_ees,
    array_agg(DISTINCT suffix) FILTER (WHERE datatype='field_maps') fmap_suffixes,
    array_agg(DISTINCT left(concat_ws(' | ', oe->>'acquisition', oe->>'SequenceName', oe->>'PulseSequenceDetails', oe->>'SeriesDescription', oe->>'ProtocolName', oe->>'ScanningSequence', oe->>'SequenceVariant'), 160)) FILTER (WHERE suffix='t1_weighted_mri') t1_seq,
    array_agg(DISTINCT oe->>'ArterialSpinLabelingType') FILTER (WHERE oe->>'ArterialSpinLabelingType' IS NOT NULL) asl_type,
    array_agg(DISTINCT left(concat_ws(' | ', suffix, oe->>'acquisition', oe->>'SeriesDescription', oe->>'ProtocolName'),120)) FILTER (WHERE suffix IN ('arterial_spin_labelling','mr_angiography','swi','ct_scan','flair_mri','t2_star_weighted_mri','gradient_echo_mri') OR datatype='perfusion_asl') special_seq
  FROM (SELECT dataset_id, suffix, datatype, other_entities oe FROM bids_objects WHERE class_='image_file' AND suffix NOT IN ('fmri_bold','single_band_reference','diffusion_mri_dwi')) i GROUP BY 1)
SELECT json_build_object('acc', d.accession_id, 'field', a.field, 'field_t1', a.field_t1, 'manufacturer', a.manufacturer, 'model', a.model, 'fmap_dir', a.fmap_dir,
  'n_fmap', a.n_fmap, 'n_fmap_ees', a.n_fmap_ees, 'fmap_suffixes', a.fmap_suffixes, 't1_seq', a.t1_seq[1:25], 'asl_type', a.asl_type, 'special_seq', a.special_seq[1:25])
FROM bids_datasets d JOIN a ON a.dataset_id=d.id;
