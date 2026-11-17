"""
Shared SQL prompt/schema constants for BIDS-Eye.

This module is used by the backend text-to-SQL pipeline. It intentionally
stays small and dependency-free.

The schema shown to the SQL model is built per request by build_schema_ddl():
columns in OPTIONAL_DATASET_COLUMNS appear only when the database really has
them filled (see services/schema_capabilities.py). A column the model is told
about but that is missing or empty produces failing or empty queries.
"""

from __future__ import annotations

from typing import Collection, Dict, List, Tuple

# (name, type, comment lines) — always present on bids_datasets.
_CORE_DATASET_COLUMNS: List[Tuple[str, str, List[str]]] = [
    ("id",                "UUID PRIMARY KEY", []),
    ("name",              "TEXT", []),
    ("accession_id",      "TEXT", []),
    ("bids_version",      "TEXT", []),
    ("dataset_type",      "TEXT", []),
    ("source_type",       "TEXT", ["e.g. 'openneuro'"]),
    ("remote_url",        "TEXT", []),
    ("validation_status", "TEXT", []),
    ("authors",           "TEXT[]", ["dataset authors, e.g. ARRAY['Poldrack, R.A.', 'Gorgolewski, K.']"]),
    ("license",           "TEXT", ["e.g. 'CC0', 'PDDL'"]),
    ("doi",               "TEXT", ["dataset DOI"]),
    ("paper_references",  "TEXT[]", [
        "linked papers / URLs (from dataset_description.json ReferencesAndLinks,",
        "with README URLs as fallback when the field is absent)",
    ]),
    ("funding",           "TEXT[]", ["funding sources"]),
    ("description_text",  "TEXT", [
        "free-text description from dataset_description.json;",
        "search it with text_match('keyword'), not ILIKE",
    ]),
]

# Columns that exist only after a migration + backfill. Each one is shown to
# the SQL model only when the running database has it populated.
OPTIONAL_DATASET_COLUMNS: Dict[str, Tuple[str, List[str]]] = {
    "institutions": ("TEXT[]", [
        "scanning institutions from MRI sidecar JSONs, e.g. ARRAY['Child Mind Institute', 'NYU_Center_for_Brain_Imaging']",
        "Names come from DICOM headers — expect underscores, abbreviations, and",
        "slight variants (e.g. 'Princeton_University' alongside 'Princeton University').",
        "An institution named in the question arrives as an 'institution' line in",
        "[Resolved DB filters] (d.institutions && ARRAY[...] with the exact stored names):",
        "copy that line verbatim. Only without such a line, match unnested elements with",
        "a case-insensitive whole-word regex (plain ILIKE '%MIT%' also hits 'Brains Unlimited'):",
        "  EXISTS (SELECT 1 FROM unnest(d.institutions) AS inst WHERE inst ~* '\\mMIT\\M')",
    ]),
    "readme_text": ("TEXT", [
        "content of the dataset README (long free text): paper links, scanner",
        "details, task descriptions, lab names. Do not ILIKE it — text_match('keyword')",
        "searches it together with name and description_text.",
    ]),
    "dx_status": ("TEXT", [
        "diagnosis / health status of the cohort as typed by the uploader (free text,",
        "NULL for many datasets), e.g. 'Healthy / Control', 'Epilepsy', 'Parkinson''s Disease'.",
        "Covered by text_match('term'); never compare with canonical diagnosis codes.",
    ]),
    "study_domain": ("TEXT", [
        "research topic as typed by the uploader (free text, NULL for many datasets),",
        "e.g. 'Basal Ganglia in Parkinson', 'Language'. Covered by text_match('term').",
    ]),
    "study_design": ("TEXT", [
        "study design as typed by the uploader (free text), e.g. 'Longitudinal',",
        "'Cross-Sectional'. Search with ILIKE '%term%'.",
    ]),
    "species": ("TEXT", [
        "species as typed by the uploader on OpenNeuro (free text, NULL for about 40%",
        "of datasets, nearly all of them human), e.g. 'Human', 'Mouse', 'Rat', 'Macaque',",
        "'Monkey', 'Nonhunam primates (Macaca mulatta)', 'Sus scrofa domesticus', 'Phantom'.",
        "When the question context has a 'species' line, copy that condition exactly and",
        "add no other species condition. Without one (older planner), use these recipes.",
        "Use it for animal / non-human questions, and ALWAYS add the title/description",
        "fallback for datasets without a species (many animal datasets are unlabelled):",
        "  mice: (d.species ~* '\\m(mouse|mice|mus)\\M'",
        "         OR (d.species IS NULL AND text_match('mouse | mice | murine', 'short')))",
        "  macaques/monkeys: (d.species ~* '(macaque|macaca|monkey|primate|baboon|marmoset|rhesus|capuchin|cynomolgus)'",
        "         OR ((d.species IS NULL OR d.species ~* '^\\s*(other|n/?a|unknown)\\s*$')",
        "             AND text_match('macaque | macaca | monkey | rhesus | marmoset | baboon')))",
        "  any animal: (d.species !~* '(human|homo sapiens|phantom|synthetic|printed|^\\s*(other|n/?a|unknown)\\s*$)'",
        "         OR ((d.species IS NULL OR d.species ~* '^\\s*(other|n/?a|unknown)\\s*$')",
        "             AND text_match('mouse | mice | rat | rats | macaque | monkey | animal', 'short')))",
        "A label 'Other', 'n/a' or 'unknown' says nothing: treat it like NULL (text fallback).",
        "Mouse/rat/any-animal fallbacks stay on title and description ('short'): human READMEs",
        "say 'mouse click'. Primate words are safe in the README of an unlabelled dataset.",
        "Never AND an extra text_match of the species word: the species condition is complete.",
    ]),
    "study_longitudinal": ("TEXT", [
        "design as picked by the uploader: 'Cross-Sectional', 'Longitudinal', 'Other',",
        "'Experimental' (NULL for about half). Longitudinal: d.study_longitudinal = 'Longitudinal'",
        "(study_design is a different free-text field, mostly 'Event-related' or 'Observational').",
    ]),
    "senior_author": ("TEXT", [
        "senior / last author as typed by the uploader, e.g. 'Russell A. Poldrack'.",
        "For 'the X lab' compare with ILIKE '%surname%' (it is not in d.authors for every dataset).",
    ]),
    "publish_date": ("TIMESTAMPTZ", [
        "when the dataset went public on OpenNeuro, e.g. recent datasets:",
        "d.publish_date >= now() - interval '1 year'",
    ]),
    "on_brainlife": ("BOOLEAN", [
        "true when the dataset is also available for processing on brainlife.io",
    ]),
    "downloads": ("INTEGER", [
        "OpenNeuro download count of the latest snapshot (NULL for removed datasets).",
        "'most popular': ORDER BY d.downloads DESC NULLS LAST (without NULLS LAST, NULL rows come first)",
    ]),
    "views": ("INTEGER", [
        "OpenNeuro page views of the latest snapshot",
    ]),
    "age_min": ("FLOAT", [
        "youngest participant age reported for the dataset (NULL when not reported)",
    ]),
    "age_max": ("FLOAT", [
        "oldest participant age reported for the dataset (NULL when not reported)",
        "age_min / age_max COMPLEMENT bids_participants.age: some datasets have only one of",
        "the two. For an age filter ALWAYS OR both sources, e.g. participants under 12:",
        "  (EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id AND p.age < 12)",
        "   OR d.age_min < 12)",
        "and participants over 65: (EXISTS (... p.age > 65) OR d.age_max > 65).",
    ]),
    "size_bytes": ("BIGINT", [
        "total size of the dataset in bytes (1 GB = 1073741824);",
        "e.g. no larger than 5 GB: d.size_bytes <= 5 * 1073741824::bigint",
    ]),
}


def _render_dataset_table(available: Collection[str]) -> str:
    columns = list(_CORE_DATASET_COLUMNS) + [
        (name, sql_type, comments)
        for name, (sql_type, comments) in OPTIONAL_DATASET_COLUMNS.items()
        if name in available
    ]
    lines = ["CREATE TABLE bids_datasets ("]
    for index, (name, sql_type, comments) in enumerate(columns):
        comma = "," if index < len(columns) - 1 else ""
        definition = f"    {name:<17} {sql_type}{comma}"
        if comments:
            lines.append(f"{definition:<34}-- {comments[0]}")
            lines.extend(f"{'':<34}-- {comment}" for comment in comments[1:])
        else:
            lines.append(definition)
    lines.append(");")
    return "\n".join(lines)


def build_schema_ddl(available: Collection[str] = ()) -> str:
    """Schema text for the SQL-generation and SQL-correction prompts.

    *available* names the OPTIONAL_DATASET_COLUMNS the database has populated.
    """
    return _render_dataset_table(available) + "\n\n" + _OTHER_TABLES_DDL


_OTHER_TABLES_DDL = """\
CREATE TABLE bids_objects (
    id              UUID PRIMARY KEY,
    dataset_id      UUID REFERENCES bids_datasets(id),
    subject         TEXT,
    subject_index   INTEGER,
    session         TEXT,
    task            TEXT,         -- normalized standard_code, e.g. 'resting_state', 'nback'
                                  -- A broad category key is expanded to its task codes at runtime:
                                  --   'memory', 'sleep', 'language_speech', 'working_memory', etc.
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
    other_entities  JSON          -- sidecar metadata + secondary BIDS entities (JSON, not JSONB:
                                  --   json_typeof works, jsonb_* functions and ?/@> do not)
                                  -- Scanner hardware keys (use ILIKE for string fields):
                                  --   Manufacturer          TEXT  e.g. 'Siemens', 'Philips', 'GE', 'Bruker'
                                  --   ManufacturerModelName TEXT  e.g. 'Prisma', 'Skyra', 'Achieva', 'MAGNETOM'
                                  --   MagneticFieldStrength FLOAT e.g. 1.5, 3.0, 7.0  (Tesla). Vendors write
                                  --     approximate values (2.89 for 3 T, 6.98 for 7 T): compare with a
                                  --     range, e.g. 7 T → BETWEEN 6.5 AND 7.5, never = 7.0
                                  --   SoftwareVersions      TEXT  e.g. 'syngo MR E11'
                                  -- PET tracer keys (on PET files):
                                  --   TracerName          TEXT  e.g. 'FDG', '[11C]PS13', 'UCB-J', 'AV-45 florbetapir'
                                  --   TracerRadionuclide  TEXT  written 'C11', '11C', '[11C]' or 'C-11': match
                                  --     every form, e.g. carbon-11 → other_entities->>'TracerRadionuclide' ~* '(c-?11|11c)'
                                  --     (fluorine-18 → '(f-?18|18f)'). Never search tracers with text_match.
                                  -- Acquisition params:
                                  --   RepetitionTime, EchoTime, FlipAngle, AcquisitionTime
                                  -- Secondary BIDS entities: acq, echo, dir, space, res
                                  -- Extract as text : other_entities->>'Manufacturer'
                                  -- Cast to numeric : (other_entities->>'MagneticFieldStrength')::float
                                  --   (non-numeric values such as 'n/a' are skipped automatically)
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
                             --   'psychiatric_disorder', 'neurodevelopmental', 'neurodegenerative', etc.
    extra          JSON         -- non-standard participant columns from participants.tsv (JSON, not JSONB)
                                -- e.g. {"concern_dieting": "yes", "bmi": 24.5, "group": "control"}
                                -- Use ->> to extract: p.extra->>'concern_dieting'
);

-- Query conventions:
-- Always alias: bids_datasets AS d, bids_objects AS o, bids_participants AS p
-- Begin EXACTLY with: SELECT {{COLS}}   ← literal placeholder; never list columns.
--   The full projection (id, name, …, authors, description_text,
--   COUNT(DISTINCT o.subject) AS subject_count) is injected automatically.
-- Always: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL
-- Always: GROUP BY d.id
-- Use EXISTS (...) to filter by file/participant properties (bids_objects, bids_participants only)
-- Never use EXISTS to check a column already on bids_datasets — use d.col directly
-- Keyword search on dataset text: text_match('keyword'). This is a macro; the system
--   expands it into  d.search_tsv @@ to_tsquery('english', '...')  (search_tsv is a
--   full-text index over name, description, study fields and README). Keep such an
--   expanded predicate unchanged when you see one in a query.
-- Use ILIKE '%term%' only for short fields (license, references, scanner strings)
-- Never JOIN bids_participants alongside bids_objects — cross-product risk.
--   For participant counts in HAVING, use correlated subqueries:
--   HAVING (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'male')
--        > (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'female')
-- Counting convention:
--   subject_count alias (from the injected projection) → COUNT(DISTINCT o.subject)
--   HAVING threshold on participant criteria → COUNT(DISTINCT p.participant_id)\
"""
