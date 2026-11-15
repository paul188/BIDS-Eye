WITH p AS (
  SELECT dataset_id, count(*) n_part,
    count(*) FILTER (WHERE lower(sex) IN ('m','male','1','man','boy')) n_male, count(*) FILTER (WHERE lower(sex) IN ('f','female','2','woman','girl')) n_female,
    array_agg(DISTINCT sex) FILTER (WHERE sex IS NOT NULL) sex_values,
    min(age) FILTER (WHERE age > 0) age_min_p, max(age) age_max_p, count(age) n_age,
    count(*) FILTER (WHERE age >= 18) n_adult, count(*) FILTER (WHERE age BETWEEN 2 AND 12) n_2_12, count(*) FILTER (WHERE age BETWEEN 5 AND 21) n_5_21,
    count(*) FILTER (WHERE lower(sex) IN ('m','male') AND age >= 40 AND age < 41) n_male_40,
    count(*) FILTER (WHERE lower(sex) IN ('m','male') AND age >= 18) n_adult_male,
    (SELECT json_object_agg(dx, n) FROM (SELECT coalesce(diagnosis,'(null)') dx, count(*) n FROM bids_participants q WHERE q.dataset_id=bp.dataset_id GROUP BY 1) t) dx_counts,
    (SELECT array_agg(DISTINCT k) FROM (SELECT json_object_keys(extra) k FROM bids_participants q WHERE q.dataset_id=bp.dataset_id AND json_typeof(extra)='object' LIMIT 4000) t) extra_keys,
    (SELECT json_object_agg(g, n) FROM (SELECT left(coalesce(CASE WHEN json_typeof(extra)='object' THEN coalesce(extra->>'group', extra->>'Group', extra->>'diagnosis', extra->>'Diagnosis', extra->>'dx') END, '(none)'),40) g, count(*) n FROM bids_participants q WHERE q.dataset_id=bp.dataset_id GROUP BY 1 ORDER BY 2 DESC LIMIT 12) t) raw_groups
  FROM bids_participants bp GROUP BY dataset_id),
dv AS (SELECT dataset_id, json_agg(json_build_object('pipeline', coalesce(pipeline_name, pipeline), 'version', pipeline_version, 'source', source, 'n', n_subjects)) derivs FROM bids_derivatives GROUP BY 1),
pp AS (SELECT dp.dataset_id, json_agg(json_build_object('role', dp.role, 'title', bp.title, 'year', bp.year, 'doi', bp.doi, 'abstract', left(bp.abstract, 1500), 'mesh', bp.mesh_major, 'keywords', bp.keywords) ORDER BY dp.role, dp.position) papers
       FROM bids_dataset_papers dp JOIN bids_papers bp ON bp.id=dp.paper_id GROUP BY 1)
SELECT json_build_object('acc', d.accession_id, 'name', d.name, 'license', left(d.license, 120), 'size_bytes', d.size_bytes, 'species', d.species, 'dx_status', d.dx_status,
  'study_domain', d.study_domain, 'study_design', d.study_design, 'study_longitudinal', d.study_longitudinal, 'age_min', d.age_min, 'age_max', d.age_max,
  'publish_date', d.publish_date, 'authors', d.authors, 'description', d.description_text, 'readme', d.readme_text, 'summary', d.dataset_summary,
  'deprecated', d.deprecated_reason, 'doi', d.doi, 'paper_references', d.paper_references, 'bids_version', d.bids_version, 'dataset_type', d.dataset_type,
  'participants', row_to_json(p), 'derivatives', dv.derivs, 'papers', pp.papers)
FROM bids_datasets d LEFT JOIN p ON p.dataset_id=d.id LEFT JOIN dv ON dv.dataset_id=d.id LEFT JOIN pp ON pp.dataset_id=d.id WHERE d.source_type='openneuro';
