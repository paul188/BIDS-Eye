WITH subj AS (
  SELECT dataset_id, subject,
    bool_or(suffix='t1_weighted_mri' AND class_='image_file') t1,
    bool_or(datatype='field_maps') fmap,
    bool_or(suffix='eeg') eeg
  FROM bids_objects WHERE subject IS NOT NULL GROUP BY 1,2),
j AS (
  SELECT p.dataset_id, p.age, lower(p.sex) sex, p.diagnosis, s.t1, s.fmap, s.eeg
  FROM bids_participants p JOIN subj s ON s.dataset_id=p.dataset_id
   AND regexp_replace(s.subject,'^sub-','') = regexp_replace(p.participant_id,'^sub-','')),
a AS (
  SELECT dataset_id, count(*) n_joined,
    count(*) FILTER (WHERE t1 AND age >= 18) n_adult_t1,
    count(*) FILTER (WHERE t1 AND age >= 18 AND diagnosis ~* 'healthy|control') n_adult_hc_t1,
    count(*) FILTER (WHERE t1 AND age >= 18 AND diagnosis IS NOT NULL AND diagnosis !~* 'healthy|control') n_adult_patient_t1,
    count(*) FILTER (WHERE t1 AND age BETWEEN 2 AND 12.999) n_child_t1,
    count(*) FILTER (WHERE t1 AND age BETWEEN 2 AND 12.999 AND diagnosis ~* 'healthy|control') n_child_hc_t1,
    count(*) FILTER (WHERE t1 AND age BETWEEN 2 AND 12.999 AND diagnosis IS NOT NULL AND diagnosis !~* 'healthy|control') n_child_patient_t1,
    count(*) FILTER (WHERE t1 AND age IS NULL) n_t1_no_age,
    count(*) FILTER (WHERE t1) n_t1_joined,
    count(*) FILTER (WHERE fmap AND sex IN ('m','male') AND age >= 40 AND age < 41) n_male40_fmap,
    count(*) FILTER (WHERE eeg AND sex IN ('m','male') AND age >= 18) n_adult_male_eeg,
    count(*) FILTER (WHERE eeg AND sex IN ('m','male')) n_male_eeg
  FROM j GROUP BY 1)
SELECT json_build_object('acc', d.accession_id, 'n_joined', a.n_joined, 'n_adult_t1', a.n_adult_t1, 'n_adult_hc_t1', a.n_adult_hc_t1, 'n_adult_patient_t1', a.n_adult_patient_t1,
  'n_child_t1', a.n_child_t1, 'n_child_hc_t1', a.n_child_hc_t1, 'n_child_patient_t1', a.n_child_patient_t1, 'n_t1_no_age', a.n_t1_no_age, 'n_t1_joined', a.n_t1_joined,
  'n_male40_fmap', a.n_male40_fmap, 'n_adult_male_eeg', a.n_adult_male_eeg, 'n_male_eeg', a.n_male_eeg)
FROM bids_datasets d JOIN a ON a.dataset_id=d.id;
