"""
The database schema as the SQL model is shown it.

Used by the SQL-generation and SQL-correction prompts. Names and types come
from services/schema/tables.py; here are only the notes shown to the model.

The schema shown to the SQL model is built per request by build_schema_ddl():
columns in OPTIONAL_DATASET_COLUMNS appear only when the database really has
them filled (see services/translate/optional_columns.py). A column the model is told
about but that is missing or empty produces failing or empty queries.
"""

from __future__ import annotations

from typing import Any, Collection, Dict, List, Tuple

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, BigInteger, JSON, String, Table, Uuid
from sqlalchemy.dialects.postgresql import ARRAY

from services.schema import tables as _tables
from services.schema.tables import referred_column


def prompt_type(column: Column[Any]) -> str:
    """The type of *column* as the SQL model is shown it."""
    if isinstance(column.type, ARRAY):
        return "TEXT[]"
    for type_, shown in ((Uuid, "UUID"), (DateTime, "TIMESTAMPTZ"), (Boolean, "BOOLEAN"),
                         (BigInteger, "BIGINT"), (Integer, "INTEGER"), (Float, "FLOAT"),
                         (JSON, "JSON"), (String, "TEXT")):
        if isinstance(column.type, type_):
            return shown
    raise TypeError(f"no prompt type for {column}")


# Column → comment lines; always present on bids_datasets.
_CORE_NOTES: Dict[str, List[str]] = {
    "id": [],
    "name": [],
    "accession_id": [],
    "bids_version": [],
    "dataset_type": [],
    "source_type": ["e.g. 'openneuro'"],
    "remote_url": [],
    "validation_status": [],
    "authors": ["dataset authors, e.g. ARRAY['Poldrack, R.A.', 'Gorgolewski, K.']"],
    "license": ["e.g. 'CC0', 'PDDL'"],
    "doi": ["dataset DOI"],
    "paper_references": [
        "linked papers / URLs (from dataset_description.json ReferencesAndLinks,",
        "with README URLs as fallback when the field is absent)",
    ],
    "funding": ["funding sources"],
    "description_text": [
        "free-text description from dataset_description.json;",
        "search it with text_match('keyword'), not ILIKE",
    ],
}

# Columns that exist only after a migration + backfill. Each one is shown to
# the SQL model only when the running database has it populated.
_OPTIONAL_NOTES: Dict[str, List[str]] = {
    "institutions": [
        "scanning institutions from MRI sidecar JSONs, e.g. ARRAY['Child Mind Institute', 'NYU_Center_for_Brain_Imaging']",
        "Names come from DICOM headers — expect underscores, abbreviations, and",
        "slight variants (e.g. 'Princeton_University' alongside 'Princeton University').",
        "An institution named in the question arrives as an 'institution' line in",
        "[Resolved DB filters] (d.institutions && ARRAY[...] with the exact stored names):",
        "copy that line verbatim. Only without such a line, match unnested elements with",
        "a case-insensitive whole-word regex (plain ILIKE '%MIT%' also hits 'Brains Unlimited'):",
        "  EXISTS (SELECT 1 FROM unnest(d.institutions) AS inst WHERE inst ~* '\\mMIT\\M')",
    ],
    "readme_text": [
        "content of the dataset README (long free text): paper links, scanner",
        "details, task descriptions, lab names. Do not ILIKE it — text_match('keyword')",
        "searches it together with name and description_text.",
    ],
    "dx_status": [
        "diagnosis / health status of the cohort as typed by the uploader (free text,",
        "NULL for many datasets), e.g. 'Healthy / Control', 'Epilepsy', 'Parkinson''s Disease'.",
        "Covered by text_match('term'); never compare with canonical diagnosis codes.",
    ],
    "study_domain": [
        "research topic as typed by the uploader (free text, NULL for many datasets),",
        "e.g. 'Basal Ganglia in Parkinson', 'Language'. Covered by text_match('term').",
    ],
    "study_design": [
        "study design as typed by the uploader (free text), e.g. 'Longitudinal',",
        "'Cross-Sectional'. Search with ILIKE '%term%'.",
    ],
    "species": [
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
        "non-human primates / NHP: (d.species ~* '(macaque|macaca|monkey|primate|marmoset|baboon|chimpanzee|rhesus|capuchin|cynomolgus)'",
        "         OR ((d.species IS NULL OR d.species ~* '^\\s*(other|n/?a|unknown)\\s*$')",
        "             AND text_match('macaque | macaca | monkey | rhesus | marmoset | primate | baboon')))",
        "  Never search the abbreviation 'NHP' as text.",
    ],
    "study_longitudinal": [
        "design as picked by the uploader: 'Cross-Sectional', 'Longitudinal', 'Other',",
        "'Experimental' (NULL for about half). Longitudinal: d.study_longitudinal = 'Longitudinal'",
        "(study_design is a different free-text field, mostly 'Event-related' or 'Observational').",
    ],
    "senior_author": [
        "senior / last author as typed by the uploader, e.g. 'Russell A. Poldrack'.",
        "For 'the X lab' compare with ILIKE '%surname%' (it is not in d.authors for every dataset).",
    ],
    "publish_date": [
        "when the dataset went public on OpenNeuro, e.g. recent datasets:",
        "d.publish_date >= now() - interval '1 year'",
    ],
    "on_brainlife": [
        "true when the dataset is also available for processing on brainlife.io",
    ],
    "downloads": [
        "OpenNeuro download count of the latest snapshot (NULL for removed datasets).",
        "'most popular': ORDER BY d.downloads DESC NULLS LAST (without NULLS LAST, NULL rows come first)",
    ],
    "views": [
        "OpenNeuro page views of the latest snapshot",
    ],
    "age_min": [
        "youngest participant age reported for the dataset (NULL when not reported)",
    ],
    "age_max": [
        "oldest participant age reported for the dataset (NULL when not reported)",
        "age_min / age_max COMPLEMENT bids_participants.age: some datasets have only one of",
        "the two. For an age filter ALWAYS OR both sources, e.g. participants under 12:",
        "  (EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id AND p.age < 12)",
        "   OR d.age_min < 12)",
        "and participants over 65: (EXISTS (... p.age > 65) OR d.age_max > 65).",
    ],
    "size_bytes": [
        "total size of the dataset in bytes (1 GB = 1073741824);",
        "e.g. no larger than 5 GB: d.size_bytes <= 5 * 1073741824::bigint",
    ],
}

# The other tables the model is shown, column → comment lines.
_OBJECT_NOTES: Dict[str, List[str]] = {
    "id": [],
    "dataset_id": [],
    "subject": [],
    "subject_index": [],
    "session": [],
    "task": [
        "normalized standard_code, e.g. 'resting_state', 'nback'",
        "A broad category key is expanded to its task codes at runtime:",
        "  'memory', 'sleep', 'language_speech', 'working_memory', etc.",
    ],
    "run": [],
    "suffix": [
        "normalized code, e.g. 'fmri_bold' (fMRI), 't1_weighted_mri' (T1w),",
        "     't2_weighted_mri' (T2w), 'diffusion_mri_dwi', 'eeg', 'meg',",
        "     'intracranial_eeg', 'pet'",
    ],
    "datatype": [
        "normalized code, e.g. 'functional_mri', 'anatomical_mri',",
        "     'diffusion_mri', 'field_maps', 'electroencephalography',",
        "     'magnetoencephalography', 'intracranial_eeg', 'behavioural_data',",
        "     'positron_emission_tomography', 'perfusion_asl', 'fnirs'",
    ],
    "extension": [
        "file format ONLY: '.nii.gz', '.json', '.tsv', '.eeg', etc.",
        "NEVER contains metadata field names",
    ],
    "other_entities": [
        "sidecar metadata + secondary BIDS entities (JSON, not JSONB:",
        "  json_typeof works, jsonb_* functions and ?/@> do not)",
        "Scanner hardware keys (use ILIKE for string fields):",
        "  Manufacturer          TEXT  e.g. 'Siemens', 'Philips', 'GE', 'Bruker'",
        "  ManufacturersModelName TEXT e.g. 'Prisma', 'Skyra', 'Achieva', 'MAGNETOM'",
        "    (the BIDS key; the SQL layer also reads a few datasets'",
        "    DICOM spelling ManufacturerModelName)",
        "  MagneticFieldStrength FLOAT e.g. 1.5, 3.0, 7.0  (Tesla). Vendors write",
        "    approximate values (2.89 for 3 T, 6.98 for 7 T): compare with a",
        "    range, e.g. 7 T → BETWEEN 6.5 AND 7.5, never = 7.0",
        "  SoftwareVersions      TEXT  e.g. 'syngo MR E11'",
        "PET tracer keys (on PET files):",
        "  TracerName          TEXT  e.g. 'FDG', '[11C]PS13', 'UCB-J', 'AV-45 florbetapir'",
        "  TracerRadionuclide  TEXT  written 'C11', '11C', '[11C]' or 'C-11': match",
        "    every form, e.g. carbon-11 → other_entities->>'TracerRadionuclide' ~* '(c-?11|11c)'",
        "    (fluorine-18 → '(f-?18|18f)'). Never search tracers with text_match.",
        "Acquisition params:",
        "  RepetitionTime, EchoTime, FlipAngle, AcquisitionTime",
        "Secondary BIDS entities from the file name: acq, echo, dir, space, res, desc",
        "  phase-encoding direction in the name (dir-AP / dir-PA): other_entities->>'dir' = 'AP'",
        "  a file-name part such as _desc-preproc_bold: other_entities->>'desc' = 'preproc'",
        "    together with the suffix (never a text search for a file name)",
        "Extract as text : other_entities->>'Manufacturer'",
        "Cast to numeric : (other_entities->>'MagneticFieldStrength')::float",
        "  (non-numeric values such as 'n/a' are skipped automatically)",
    ],
}

_PARTICIPANT_NOTES: Dict[str, List[str]] = {
    "id": [],
    "dataset_id": [],
    "participant_id": [],
    "age": [],
    "sex": ["'male', 'female' or 'other_diverse'; NULL when not known"],
    "handedness": ["'right_handed', 'left_handed' or 'ambidextrous'; NULL when not known"],
    "diagnosis": [
        "normalized standard_code, e.g. 'epilepsy', 'schizophrenia',",
        "  'healthy_control', 'major_depressive_disorder', 'adhd'",
        "Use broad concept keys for categories (auto-expanded at runtime):",
        "  'psychiatric_disorder', 'neurodevelopmental', 'neurodegenerative', etc.",
    ],
    "extra": [
        "non-standard participant columns from participants.tsv (JSON, not JSONB)",
        'e.g. {"concern_dieting": "yes", "bmi": 24.5, "group": "control"}',
        "Use ->> to extract: p.extra->>'concern_dieting'",
    ],
}

_DERIVATIVE_NOTES: Dict[str, List[str]] = {
    "dataset_id": [],
    "pipeline": [
        "lower case: 'fmriprep', 'mriqc', 'xcpd', 'freesurfer', 'fitlins',",
        "  'qsiprep', 'aslprep', 'eeglab', 'preprocessed', 'unspecified', ...",
    ],
    "pipeline_version": [],
    "source": ["'in_dataset' (derivatives/ folder) or 'openneuro_derivatives'"],
}

_DATASET_PAPER_NOTES: Dict[str, List[str]] = {
    "dataset_id": [],
    "paper_id": [],
    "role": [
        "'primary' = the paper that describes or first used the dataset;",
        "  'related', 'tool', 'reuse'",
    ],
}


def _shown_type(column: Column[Any]) -> str:
    """prompt_type with the key the model is shown: the single primary key, and
    the column of another table it refers to."""
    shown = prompt_type(column)
    if column.primary_key and len(column.table.primary_key.columns) == 1:
        shown += " PRIMARY KEY"
    target = referred_column(column)
    if target is not None:
        shown += f" REFERENCES {target.table.name}({target.name})"
    return shown


def _columns(table: Table, notes: Dict[str, List[str]]) -> List[Tuple[str, str, List[str]]]:
    """(name, type, comment lines) of the columns of *table* that *notes* names.
    A note on a column tables.py lacks fails here."""
    return [(name, _shown_type(table.c[name]), lines) for name, lines in notes.items()]


_CORE_DATASET_COLUMNS = _columns(_tables.datasets, _CORE_NOTES)
OPTIONAL_DATASET_COLUMNS: Dict[str, Tuple[str, List[str]]] = {
    name: (sql_type, notes) for name, sql_type, notes in _columns(_tables.datasets, _OPTIONAL_NOTES)
}


def _render_table(name: str, columns: List[Tuple[str, str, List[str]]], about: str = "",
                  name_width: int = 15, note_column: int = 34) -> str:
    """CREATE TABLE text for the prompt: *about* after the table name, each
    column's comment lines from *note_column* on."""
    head = f"CREATE TABLE {name} ("
    lines = [f"{head:<32}  -- {about}" if about else head]
    for index, (column, sql_type, comments) in enumerate(columns):
        comma = "," if index < len(columns) - 1 else ""
        definition = f"    {column:<{name_width}} {sql_type}{comma}"
        if comments:
            lines.append(f"{definition:<{note_column}}-- {comments[0]}")
            lines.extend(f"{'':<{note_column}}-- {comment}" for comment in comments[1:])
        else:
            lines.append(definition)
    lines.append(");")
    return "\n".join(lines)


def _render_dataset_table(available: Collection[str]) -> str:
    columns = list(_CORE_DATASET_COLUMNS) + [
        (name, sql_type, comments)
        for name, (sql_type, comments) in OPTIONAL_DATASET_COLUMNS.items()
        if name in available
    ]
    return _render_table(_tables.datasets.name, columns, name_width=17)


def build_schema_ddl(available: Collection[str] = ()) -> str:
    """Schema text for the SQL-generation and SQL-correction prompts.

    *available* names the OPTIONAL_DATASET_COLUMNS the database has populated.
    """
    return _render_dataset_table(available) + "\n\n" + _OTHER_TABLES_DDL


# Rules that follow a table, and the conventions of every query.
_DERIVATIVE_RULES = """\
-- Processed / preprocessed data, fMRIPrep, MRIQC, FreeSurfer outputs: ALWAYS this table, never a
--   text search for 'preprocessed' or a pipeline name (READMEs rarely mention them):
--   fMRIPrep outputs:   EXISTS (SELECT 1 FROM bids_derivatives dv WHERE dv.dataset_id = d.id AND dv.pipeline = 'fmriprep')
--   MRIQC outputs:      ... AND dv.pipeline = 'mriqc'
--   preprocessed fMRI:  ... AND dv.pipeline IN ('fmriprep', 'xcpd', 'cpac', 'halfpipe', 'tedana', 'preprocessed',
--                       'preprocessed-data', 'preprocessing', 'preproc')
--   masks / segmentations: ... AND (dv.pipeline ILIKE '%mask%' OR dv.pipeline ILIKE '%segment%'
--                       OR dv.pipeline IN ('labels', 'label'))
--   bids_datasets.dataset_type is 'raw' for every dataset: never use it for processed data.
--   A request that only PREFERS preprocessed data ("ideally preprocessed") gets no such filter."""

_DATASET_PAPER_RULES = """\
-- "has a publication / paper / data descriptor":
--   EXISTS (SELECT 1 FROM bids_dataset_papers dp WHERE dp.dataset_id = d.id AND dp.role = 'primary')
--   (d.doi is the DOI of the dataset itself, not of a paper: never use it for this.)
-- What the publication reports or contains cannot be filtered: add no condition for it."""

_QUERY_CONVENTIONS = """\
-- Query conventions:
-- Always alias: bids_datasets AS d, bids_objects AS o, bids_participants AS p
-- Begin EXACTLY with: SELECT {{COLS}}   ← literal placeholder; never list columns.
--   The full projection (id, name, …, authors, description_text,
--   COUNT(DISTINCT o.subject) AS subject_count) is injected automatically.
-- Always: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL
-- Always: GROUP BY d.id
-- Use EXISTS (...) to filter by file/participant properties (bids_objects, bids_participants,
--   bids_derivatives, bids_dataset_papers)
-- CT images: o.suffix = 'ct_scan' only. PET ('positron_emission_tomography') is not CT.
-- "not sequence X" (e.g. MPRAGE but not MP2RAGE): NOT EXISTS (... AND o.suffix = '<code of X>'),
--   never NOT EXISTS over a whole datatype.
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
-- "Every subject has X and Y" (e.g. a T1w image and a BOLD run per subject): compare subjects in
--   bids_objects, never rows of bids_participants (a third of the datasets has no participants rows):
--   NOT EXISTS (SELECT 1 FROM bids_objects s WHERE s.dataset_id = d.id AND s.subject IS NOT NULL
--               AND NOT EXISTS (SELECT 1 FROM bids_objects x WHERE x.dataset_id = d.id
--                               AND x.subject = s.subject AND x.suffix = 't1_weighted_mri'))
-- "At least N sessions (or runs) per subject": one grouped subquery over the whole table, never an
--   EXISTS with GROUP BY per dataset (that runs for minutes):
--   d.id NOT IN (SELECT s.dataset_id FROM bids_objects s WHERE s.subject IS NOT NULL
--                GROUP BY s.dataset_id, s.subject HAVING COUNT(DISTINCT s.session) < 3)
-- Counting convention:
--   subject_count alias (from the injected projection) → COUNT(DISTINCT o.subject)
--   HAVING threshold on participant criteria → COUNT(DISTINCT p.participant_id)\
"""


_OTHER_TABLES_DDL = "\n\n".join((
    _render_table(_tables.objects.name, _columns(_tables.objects, _OBJECT_NOTES)),
    _render_table(_tables.participants.name, _columns(_tables.participants, _PARTICIPANT_NOTES),
                  name_width=14, note_column=32),
    _render_table(_tables.derivatives.name, _columns(_tables.derivatives, _DERIVATIVE_NOTES),
                  about="processed outputs that exist for a dataset") + "\n" + _DERIVATIVE_RULES,
    _render_table(_tables.dataset_papers.name, _columns(_tables.dataset_papers, _DATASET_PAPER_NOTES),
                  about="papers linked to a dataset") + "\n" + _DATASET_PAPER_RULES,
    _QUERY_CONVENTIONS,
))
