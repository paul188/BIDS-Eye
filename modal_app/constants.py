"""
training_data_generation/constants.py
--------------------------------------
Single source of truth for the system prompt used across:
  - sample_diverse_prompts.py  (embedded in every Gemini prompt)
  - collect_with_gemini.py     (Gemini system instruction)
  - post_process_pipeline.py   (passed to collect_response.py)
  - Text-To-SQL/train.py       (training-time system prompt)

If you change the schema or rules here, re-run the full pipeline so that
generated SQL, training data, and inference all stay aligned.
"""

SYSTEM = """\
You are a Text-to-SQL assistant for BIDS-Eye, a search engine over neuroimaging datasets.

Database schema:

bids_datasets  (d): id UUID PK, name, accession_id, bids_version, dataset_type, source_type, remote_url, validation_status
bids_objects   (o): id UUID PK, dataset_id FK, subject, subject_index, session, task, run, suffix, datatype, extension
bids_participants(p): id UUID PK, dataset_id FK, participant_id, age FLOAT, sex, handedness, diagnosis

suffix values : bold=fMRI  T1w/T2w=structural  dwi=diffusion  eeg=EEG  meg=MEG  ieeg=iEEG  pet=PET
datatype values: func  anat  dwi  fmap  eeg  meg  ieeg  beh  pet

Your SQL MUST always SELECT:
  d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,
  d.source_type, d.remote_url, d.validation_status,
  COUNT(DISTINCT o.subject) AS subject_count

Rules:
- Always alias bids_datasets as d, bids_objects as o, bids_participants as p.
- Always LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL.
- Always GROUP BY d.id.
- Use EXISTS (...) to filter by file properties without multiplying rows.
- Use ILIKE '%term%' for case-insensitive text search.
- Default LIMIT 50 unless the question specifies otherwise.
- Output ONLY the SQL query, no explanation.\
"""

# Canonical example SQL pairs shown in prompts so Gemini learns the output format.
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
            "ORDER BY d.name\n"
            "LIMIT 50"
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
            "ORDER BY subject_count DESC\n"
            "LIMIT 50"
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
            "ORDER BY d.name\n"
            "LIMIT 50"
        ),
    },
    {
        "question": "Datasets that have no DWI data",
        "sql": (
            "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
            "       d.source_type, d.remote_url, d.validation_status,\n"
            "       COUNT(DISTINCT o.subject) AS subject_count\n"
            "FROM bids_datasets d\n"
            "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "WHERE NOT EXISTS (\n"
            "    SELECT 1 FROM bids_objects bo WHERE bo.dataset_id = d.id AND bo.datatype = 'dwi'\n"
            ")\n"
            "GROUP BY d.id\n"
            "ORDER BY d.name\n"
            "LIMIT 50"
        ),
    },
    {
        "question": "Resting-state fMRI datasets from OpenNeuro with longitudinal data",
        "sql": (
            "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
            "       d.source_type, d.remote_url, d.validation_status,\n"
            "       COUNT(DISTINCT o.subject) AS subject_count\n"
            "FROM bids_datasets d\n"
            "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
            "WHERE d.source_type = 'openneuro'\n"
            "  AND EXISTS (\n"
            "    SELECT 1 FROM bids_objects bo\n"
            "    WHERE bo.dataset_id = d.id AND bo.suffix = 'bold' AND bo.task = 'rest'\n"
            "  )\n"
            "GROUP BY d.id\n"
            "HAVING COUNT(DISTINCT (SELECT bo2.session FROM bids_objects bo2\n"
            "    WHERE bo2.dataset_id = d.id AND bo2.session IS NOT NULL)) >= 2\n"
            "ORDER BY subject_count DESC\n"
            "LIMIT 50"
        ),
    },
]
