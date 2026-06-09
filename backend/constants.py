"""
Shared SQL prompt/schema constants for BIDS-Eye.

This module is used by the backend text-to-SQL pipeline and by the Modal
inference image. It intentionally stays small and dependency-free.
"""

from __future__ import annotations

SCHEMA_DDL = """\
CREATE TABLE bids_datasets (
    id                UUID PRIMARY KEY,
    name              TEXT,
    accession_id      TEXT,
    bids_version      TEXT,
    dataset_type      TEXT,
    source_type       TEXT,       -- e.g. 'openneuro'
    remote_url        TEXT,
    validation_status TEXT,
    authors           TEXT[],     -- dataset authors, e.g. ARRAY['Poldrack, R.A.', 'Gorgolewski, K.']
    license           TEXT,       -- e.g. 'CC0', 'PDDL'
    doi               TEXT,       -- dataset DOI
    paper_references  TEXT[],     -- linked papers / URLs
    funding           TEXT[],     -- funding sources
    description_text  TEXT        -- free-text description from dataset_description.json
);

CREATE TABLE bids_objects (
    id              UUID PRIMARY KEY,
    dataset_id      UUID REFERENCES bids_datasets(id),
    subject         TEXT,
    subject_index   INTEGER,
    session         TEXT,
    task            TEXT,         -- normalized standard_code, e.g. 'resting_state', 'nback'
                                  -- Use concept keys for groups: 'resting_state', 'working_memory', etc.
    run             TEXT,
    suffix          TEXT,         -- normalized code, e.g. 'fmri_bold' (fMRI), 't1_weighted_mri' (T1w),
                                  --      't2_weighted_mri' (T2w), 'diffusion_mri_dwi', 'eeg', 'meg',
                                  --      'intracranial_eeg', 'pet'
    datatype        TEXT,         -- normalized code, e.g. 'functional_mri', 'anatomical_mri',
                                  --      'diffusion_mri', 'field_maps', 'electroencephalography',
                                  --      'magnetoencephalography', 'intracranial_eeg', 'behavioural_data',
                                  --      'positron_emission_tomography', 'perfusion_asl', 'fnirs'
    extension       TEXT,         -- file format ONLY: '.nii.gz', '.json', '.tsv', '.eeg', etc.
                                  -- NEVER contains metadata field names
    other_entities  JSONB         -- sidecar metadata + secondary BIDS entities
                                  -- e.g. {"AcquisitionTime": "12:00:00", "RepetitionTime": 2.0,
                                  --       "acq": "highres", "echo": "1"}
                                  -- Use ->> to extract: other_entities->>'AcquisitionTime'
);

CREATE TABLE bids_participants (
    id             UUID PRIMARY KEY,
    dataset_id     UUID REFERENCES bids_datasets(id),
    participant_id TEXT,
    age            FLOAT,
    sex            TEXT,        -- 'M', 'F', 'male', 'female', etc.
    handedness     TEXT,        -- 'R', 'L', 'right', 'left', etc.
    diagnosis      TEXT,        -- normalized standard_code, e.g. 'epilepsy', 'schizophrenia',
                             --   'healthy_control', 'major_depressive_disorder', 'adhd'
                             -- Use broad concept keys for categories (auto-expanded at runtime):
                             --   'epilepsy_spectrum', 'psychiatric', 'neurodevelopmental', etc.
    extra          JSONB        -- non-standard participant columns from participants.tsv
                                -- e.g. {"concern_dieting": "yes", "bmi": 24.5, "group": "control"}
                                -- Use ->> to extract: p.extra->>'concern_dieting'
);

-- Query conventions:
-- Always alias: bids_datasets AS d, bids_objects AS o, bids_participants AS p
-- Always SELECT:
--   d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,
--   d.source_type, d.remote_url, d.validation_status,
--   COUNT(DISTINCT o.subject) AS subject_count   ← always this in SELECT
-- Always: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL
-- Always: GROUP BY d.id
-- Use EXISTS (...) to filter by file/participant properties (bids_objects, bids_participants only)
-- Never use EXISTS to check a column already on bids_datasets — use d.col directly
-- Use ILIKE '%term%' for case-insensitive text search
-- Never JOIN bids_participants alongside bids_objects — cross-product risk.
--   For participant counts in HAVING, use correlated subqueries:
--   HAVING (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'male')
--        > (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'female')
-- Counting convention:
--   SELECT output  → COUNT(DISTINCT o.subject) AS subject_count   (always)
--   HAVING threshold on participant criteria → COUNT(DISTINCT p.participant_id)\
"""

SYSTEM = """\
You are a Text-to-SQL assistant for BIDS-Eye, a search engine over neuroimaging datasets.

Database schema:

bids_datasets  (d): id UUID PK, name, accession_id, bids_version, dataset_type, source_type, remote_url, validation_status, authors TEXT[], license, doi, paper_references TEXT[], funding TEXT[], description_text TEXT
bids_objects   (o): id UUID PK, dataset_id FK, subject, subject_index, session, task, run, suffix, datatype, extension (file format only: '.nii.gz' etc.), other_entities JSONB (sidecar metadata + secondary entities, e.g. AcquisitionTime, RepetitionTime, acq, echo)
bids_participants(p): id UUID PK, dataset_id FK, participant_id, age FLOAT, sex, handedness, diagnosis (clinical only), extra JSONB (non-standard columns, e.g. concern_dieting, bmi, group)

suffix values : fmri_bold=fMRI  t1_weighted_mri/t2_weighted_mri=structural  diffusion_mri_dwi=diffusion  eeg=EEG  meg=MEG  intracranial_eeg=iEEG  pet=PET
datatype values: functional_mri  anatomical_mri  diffusion_mri  field_maps  electroencephalography  magnetoencephalography  intracranial_eeg  behavioural_data  positron_emission_tomography  perfusion_asl  fnirs

Your SQL MUST always SELECT:
  d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,
  d.source_type, d.remote_url, d.validation_status,
  COUNT(DISTINCT o.subject) AS subject_count

Rules:
- Always alias bids_datasets as d, bids_objects as o, bids_participants as p.
- Always LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL.
- Always GROUP BY d.id.
- Use EXISTS (...) to filter by file properties without multiplying rows.
- Use ILIKE '%term%' for case-insensitive text search on d.name and d.description_text.
- Only add LIMIT when the question explicitly requests a top-N result or ranking.
- Never use EXISTS to check a column already on bids_datasets — use d.col directly (e.g. WHERE d.doi IS NOT NULL AND d.doi != '').
- Never JOIN bids_participants alongside bids_objects — it creates a cross-product that times out. For participant counts in HAVING, use correlated subqueries: HAVING (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'male') > (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'female')
- Counting: always use COUNT(DISTINCT o.subject) AS subject_count in SELECT. For HAVING thresholds on participant criteria, use COUNT(DISTINCT p.participant_id). Never mix them.
- o.extension is the file format ONLY ('.nii.gz', '.json', '.tsv'). NEVER search extension for metadata field names.
- Sidecar metadata (AcquisitionTime, RepetitionTime, etc.) is in o.other_entities JSONB: o.other_entities->>'AcquisitionTime'
- Non-standard participant fields (bmi, group, custom scales, etc.) are in p.extra JSONB: p.extra->>'concern_dieting'
- p.diagnosis is for clinical diagnoses ONLY. Use p.extra for non-clinical participant attributes.
- The diagnosis, task, suffix, and datatype columns hold normalized standard_codes. Use the exact
  standard_code string in SQL (e.g. p.diagnosis = 'epilepsy', o.task = 'resting_state').
- For BROAD CATEGORIES use a concept key — the system auto-expands it to all specific codes:
    diagnosis concept keys: healthy_cohorts, psychiatric, schizophrenia_spectrum, mood_disorders,
      neurodevelopmental, neurological, epilepsy_spectrum, epilepsy_syndromes, cortical_malformations,
      neurodegenerative, cerebrovascular, movement_disorders, demyelinating_disorders, brain_tumors
    task concept keys: resting_state, working_memory, memory, attention, language, social_cognition,
      reward_learning, motor, sensory, emotion
    suffix concept keys: mri_functional, mri_anatomical, mri_diffusion, mri_fieldmap, electrophysiology
  Example: p.diagnosis = 'epilepsy_spectrum' is expanded to IN ('epilepsy', 'watanabe_syndrome', ...)
  Example: o.task = 'resting_state' is expanded to IN ('resting_state', 'resting_state_eyes_open', ...)
- Output ONLY the SQL query, no explanation.\
"""

EXAMPLE_PAIRS = [
    {
        "question": "Show me all fMRI datasets",
        "sql": (
            "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
            "       d.source_type, d.remote_url, d.validation_status,\n"
            "       COUNT(DISTINCT o.subject) AS subject_count\n"
            "FROM bids_datasets d\n"
            "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "WHERE EXISTS (\n"
            "    SELECT 1 FROM bids_objects bo WHERE bo.dataset_id = d.id AND bo.suffix = 'bold'\n"
            ")\n"
            "GROUP BY d.id\n"
            "ORDER BY d.name"
        ),
    },
    {
        "question": "EEG datasets with more than 30 subjects",
        "sql": (
            "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
            "       d.source_type, d.remote_url, d.validation_status,\n"
            "       COUNT(DISTINCT o.subject) AS subject_count\n"
            "FROM bids_datasets d\n"
            "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "WHERE EXISTS (\n"
            "    SELECT 1 FROM bids_objects bo WHERE bo.dataset_id = d.id AND bo.datatype = 'eeg'\n"
            ")\n"
            "GROUP BY d.id\n"
            "HAVING COUNT(DISTINCT o.subject) > 30\n"
            "ORDER BY subject_count DESC"
        ),
    },
    {
        "question": "Datasets with Alzheimer's patients",
        "sql": (
            "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
            "       d.source_type, d.remote_url, d.validation_status,\n"
            "       COUNT(DISTINCT o.subject) AS subject_count\n"
            "FROM bids_datasets d\n"
            "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "WHERE EXISTS (\n"
            "    SELECT 1 FROM bids_participants p\n"
            "    WHERE p.dataset_id = d.id AND p.diagnosis ILIKE '%alzheimer%'\n"
            ")\n"
            "GROUP BY d.id\n"
            "ORDER BY d.name"
        ),
    },
    {
        "question": "Find datasets containing participants with any form of epilepsy",
        "sql": (
            "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
            "       d.source_type, d.remote_url, d.validation_status,\n"
            "       COUNT(DISTINCT o.subject) AS subject_count\n"
            "FROM bids_datasets d\n"
            "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "WHERE EXISTS (\n"
            "    SELECT 1 FROM bids_participants p\n"
            "    WHERE p.dataset_id = d.id\n"
            "      AND p.diagnosis = 'epilepsy_spectrum'\n"
            ")\n"
            "GROUP BY d.id\n"
            "ORDER BY d.name"
        ),
    },
    {
        "question": "Datasets that include resting-state scans",
        "sql": (
            "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
            "       d.source_type, d.remote_url, d.validation_status,\n"
            "       COUNT(DISTINCT o.subject) AS subject_count\n"
            "FROM bids_datasets d\n"
            "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "WHERE EXISTS (\n"
            "    SELECT 1 FROM bids_objects bo WHERE bo.dataset_id = d.id AND bo.task = 'resting_state'\n"
            ")\n"
            "GROUP BY d.id\n"
            "ORDER BY d.name"
        ),
    },
]
