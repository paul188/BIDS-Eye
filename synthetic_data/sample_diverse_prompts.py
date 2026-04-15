#!/usr/bin/env python3
"""
sample_diverse_prompts.py — Build diverse Gemini prompts for Text-to-SQL data.

This script samples real values from the BIDS DB and emits prompt files that
ask Gemini for controlled coverage over:
  - query families
  - SQL structures
  - long-tail values
  - limited paraphrase bundles

The prompts are meant for direct use by `collect_with_gemini.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "BIDS-SQL"))
from db.models import BIDSDataset, BIDSObject, BIDSParticipant

from synthetic_data_and_train.constants import SYSTEM, EXAMPLE_PAIRS
from value_mappings import (
    clean_diagnosis, clean_sex, clean_handedness, clean_task, clean_suffix, clean_datatype,
    SIDECAR_FIELDS, PARTICIPANT_EXTRA_FIELDS,
    FIELD_CONCEPT_EXPANSION,
    FIELD_SYNONYMS, FIELD_LABEL_TO_CODE,
)


@dataclass
class IntentSpec:
    family: str
    sql_structure: str
    paraphrase_count: int
    focus_values: Dict[str, Any]
    notes: List[str]


def _scalars(session: Session, stmt) -> List[Any]:
    return [v for v in session.execute(stmt).scalars() if v not in (None, "")]


def fetch_stats(session: Session) -> Dict[str, Any]:
    tasks_head = _scalars(
        session,
        select(BIDSObject.task)
        .where(BIDSObject.task.isnot(None))
        .group_by(BIDSObject.task)
        .order_by(func.count().desc())
        .limit(20),
    )
    tasks_tail = _scalars(
        session,
        select(BIDSObject.task)
        .where(BIDSObject.task.isnot(None))
        .group_by(BIDSObject.task)
        .order_by(func.count().asc())
        .limit(20),
    )
    suffixes_head = _scalars(
        session,
        select(BIDSObject.suffix)
        .where(BIDSObject.suffix.isnot(None), BIDSObject.suffix != "")
        .group_by(BIDSObject.suffix)
        .order_by(func.count().desc())
        .limit(20),
    )
    suffixes_tail = _scalars(
        session,
        select(BIDSObject.suffix)
        .where(BIDSObject.suffix.isnot(None), BIDSObject.suffix != "")
        .group_by(BIDSObject.suffix)
        .order_by(func.count().asc())
        .limit(20),
    )
    datatypes = _scalars(
        session,
        select(BIDSObject.datatype)
        .where(BIDSObject.datatype.isnot(None))
        .group_by(BIDSObject.datatype)
        .order_by(func.count().desc()),
    )
    # Only keep diagnosis values that look like real clinical labels:
    # at least 3 characters, no pure numbers, no single-letter codes.
    diagnoses = [
        d for d in _scalars(
            session,
            select(BIDSParticipant.diagnosis)
            .where(BIDSParticipant.diagnosis.isnot(None))
            .group_by(BIDSParticipant.diagnosis)
            .order_by(func.count().desc())
            .limit(60),
        )
        if len(d) >= 3 and not d.strip().lstrip("-").replace(".", "").isdigit()
    ][:30]

    # sex must be one of the canonical BIDS values
    _VALID_SEX = {"m", "f", "male", "female", "o", "other", "n/a", "unknown"}
    sex_values = [
        s for s in _scalars(
            session,
            select(BIDSParticipant.sex)
            .where(BIDSParticipant.sex.isnot(None))
            .group_by(BIDSParticipant.sex)
            .order_by(func.count().desc()),
        )
        if s.strip().lower() in _VALID_SEX
    ]

    # handedness must be a recognisable label, not a number
    _VALID_HANDEDNESS = {"r", "l", "a", "right", "left", "ambidextrous", "mixed", "n/a", "unknown"}
    handedness_values = [
        h for h in _scalars(
            session,
            select(BIDSParticipant.handedness)
            .where(BIDSParticipant.handedness.isnot(None))
            .group_by(BIDSParticipant.handedness)
            .order_by(func.count().desc()),
        )
        if h.strip().lower() in _VALID_HANDEDNESS
    ]
    json_keys = [
        row[0]
        for row in session.execute(
            text(
                """
                SELECT key
                FROM (
                    SELECT jsonb_object_keys(other_entities::jsonb) AS key
                    FROM bids_objects
                    WHERE other_entities IS NOT NULL
                      AND jsonb_typeof(other_entities::jsonb) = 'object'
                    LIMIT 2000
                ) t
                GROUP BY key
                ORDER BY COUNT(*) DESC
                LIMIT 20
                """
            )
        ).fetchall()
        if row[0]
    ]
    extra_keys = [
        row[0]
        for row in session.execute(
            text(
                """
                SELECT key
                FROM (
                    SELECT jsonb_object_keys(extra::jsonb) AS key
                    FROM bids_participants
                    WHERE extra IS NOT NULL
                      AND jsonb_typeof(extra::jsonb) = 'object'
                    LIMIT 2000
                ) t
                GROUP BY key
                ORDER BY COUNT(*) DESC
                LIMIT 20
                """
            )
        ).fetchall()
        if row[0]
    ]

    # Sample one real value per JSON key so Gemini sees concrete examples.
    # Uses a single query per key; safe because keys come from jsonb_object_keys.
    json_samples: Dict[str, str] = {}
    for key in json_keys:
        val = session.execute(
            text(
                "SELECT other_entities->>:key FROM bids_objects "
                "WHERE other_entities IS NOT NULL "
                "  AND other_entities->>:key IS NOT NULL "
                "LIMIT 1"
            ),
            {"key": key},
        ).scalar()
        if val is not None:
            json_samples[key] = str(val)

    extra_samples: Dict[str, str] = {}
    for key in extra_keys:
        val = session.execute(
            text(
                "SELECT extra->>:key FROM bids_participants "
                "WHERE extra IS NOT NULL "
                "  AND extra->>:key IS NOT NULL "
                "LIMIT 1"
            ),
            {"key": key},
        ).scalar()
        if val is not None:
            extra_samples[key] = str(val)

    # Subset of json_keys whose sample values parse as a float — these can be
    # used in numeric cast comparisons: (o.other_entities->>'key')::float > x
    json_numeric_samples: Dict[str, float] = {}
    for key, val in json_samples.items():
        try:
            json_numeric_samples[key] = float(val)
        except (ValueError, TypeError):
            pass

    # Top authors (unnest the TEXT[] column)
    authors = [
        row[0]
        for row in session.execute(
            text(
                """
                SELECT author, COUNT(*) AS cnt
                FROM bids_datasets, unnest(authors) AS author
                WHERE authors IS NOT NULL
                GROUP BY author
                ORDER BY cnt DESC
                LIMIT 20
                """
            )
        ).fetchall()
        if row[0] and row[0].strip()
    ]

    # Top licenses
    licenses = _scalars(
        session,
        select(BIDSDataset.license)
        .where(BIDSDataset.license.isnot(None), BIDSDataset.license != "")
        .group_by(BIDSDataset.license)
        .order_by(func.count().desc())
        .limit(15),
    )

    # Top funding sources (unnest the TEXT[] column)
    funding_sources = [
        row[0]
        for row in session.execute(
            text(
                """
                SELECT source, COUNT(*) AS cnt
                FROM bids_datasets, unnest(funding) AS source
                WHERE funding IS NOT NULL
                GROUP BY source
                ORDER BY cnt DESC
                LIMIT 15
                """
            )
        ).fetchall()
        if row[0] and row[0].strip()
    ]

    # Count of datasets that have a DOI (useful for coverage stats)
    n_with_doi = session.execute(
        text("SELECT COUNT(*) FROM bids_datasets WHERE doi IS NOT NULL AND doi != ''")
    ).scalar() or 0

    # Age distribution — used to generate realistic age-filter and age-aggregate intents.
    # We pull min/max and the 10th/25th/75th/90th percentiles to anchor range queries.
    age_stats_row = session.execute(
        text(
            """
            SELECT
                MIN(age)                                            AS age_min,
                MAX(age)                                            AS age_max,
                ROUND(AVG(age)::numeric, 1)                         AS age_mean,
                ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY age)::numeric, 1) AS p10,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY age)::numeric, 1) AS p25,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY age)::numeric, 1) AS p75,
                ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY age)::numeric, 1) AS p90
            FROM bids_participants
            WHERE age IS NOT NULL AND age > 0 AND age < 120
            """
        )
    ).fetchone()
    age_stats = {
        "min":  float(age_stats_row[0]) if age_stats_row and age_stats_row[0] is not None else 0.0,
        "max":  float(age_stats_row[1]) if age_stats_row and age_stats_row[1] is not None else 90.0,
        "mean": float(age_stats_row[2]) if age_stats_row and age_stats_row[2] is not None else 30.0,
        "p10":  float(age_stats_row[3]) if age_stats_row and age_stats_row[3] is not None else 10.0,
        "p25":  float(age_stats_row[4]) if age_stats_row and age_stats_row[4] is not None else 20.0,
        "p75":  float(age_stats_row[5]) if age_stats_row and age_stats_row[5] is not None else 50.0,
        "p90":  float(age_stats_row[6]) if age_stats_row and age_stats_row[6] is not None else 70.0,
    }

    # Sample words that actually appear in description_text (for realistic search terms)
    description_terms = [
        row[0]
        for row in session.execute(
            text(
                """
                SELECT word
                FROM (
                    SELECT regexp_split_to_table(lower(description_text), '[^a-z]+') AS word
                    FROM bids_datasets
                    WHERE description_text IS NOT NULL
                    LIMIT 500
                ) t
                WHERE length(word) >= 4
                  AND word NOT IN (
                    'this','that','with','from','have','been','were','they',
                    'their','into','data','bids','dataset','subjects','study',
                    'using','used','were','also','each','both'
                  )
                GROUP BY word
                ORDER BY COUNT(*) DESC
                LIMIT 30
                """
            )
        ).fetchall()
        if row[0]
    ]

    multimodal = session.execute(
        text(
            """
            WITH mm AS (
                SELECT dataset_id,
                       ARRAY_AGG(DISTINCT datatype ORDER BY datatype) AS datatypes
                FROM bids_objects
                WHERE datatype IS NOT NULL
                GROUP BY dataset_id
                HAVING COUNT(DISTINCT datatype) >= 2
            )
            SELECT d.accession_id, mm.datatypes
            FROM mm
            JOIN bids_datasets d ON d.id = mm.dataset_id
            ORDER BY RANDOM()
            LIMIT 8
            """
        )
    ).fetchall()
    sessions = session.execute(
        text(
            """
            WITH sess AS (
                SELECT dataset_id, COUNT(DISTINCT session) AS n_sessions
                FROM bids_objects
                WHERE session IS NOT NULL
                GROUP BY dataset_id
                HAVING COUNT(DISTINCT session) >= 2
            )
            SELECT d.accession_id, sess.n_sessions
            FROM sess
            JOIN bids_datasets d ON d.id = sess.dataset_id
            ORDER BY RANDOM()
            LIMIT 8
            """
        )
    ).fetchall()
    # Pick random dataset IDs first (cheap), then aggregate only those rows.
    # Avoids a full three-way JOIN + GROUP BY + ORDER BY RANDOM() scan.
    dataset_samples = session.execute(
        text(
            """
            WITH sampled AS (
                SELECT id, accession_id, name
                FROM bids_datasets
                WHERE accession_id IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 8
            )
            SELECT s.accession_id,
                   s.name,
                   COUNT(DISTINCT p.id)                                                           AS n_participants,
                   ARRAY_AGG(DISTINCT o.datatype ORDER BY o.datatype) FILTER (WHERE o.datatype IS NOT NULL) AS datatypes,
                   ARRAY_AGG(DISTINCT o.task     ORDER BY o.task)     FILTER (WHERE o.task     IS NOT NULL) AS tasks
            FROM sampled s
            LEFT JOIN bids_participants p ON s.id = p.dataset_id
            LEFT JOIN bids_objects      o ON s.id = o.dataset_id
            GROUP BY s.accession_id, s.name
            """
        )
    ).fetchall()

    # Apply human-readable mappings — raw DB values become natural language
    # so Gemini generates questions that look like real researcher queries.
    clean_diagnoses = [
        label for label in (clean_diagnosis(d) for d in diagnoses) if label
    ]
    sex_labels = [label for label in (clean_sex(s) for s in sex_values) if label]
    seen: set = set()
    sex_labels = [x for x in sex_labels if not (x in seen or seen.add(x))]

    handedness_labels = [label for label in (clean_handedness(h) for h in handedness_values) if label]
    seen = set()
    handedness_labels = [x for x in handedness_labels if not (x in seen or seen.add(x))]

    return {
        "tasks_head": [x for x in (clean_task(t) for t in tasks_head) if x],
        "tasks_tail": [x for x in (clean_task(t) for t in tasks_tail) if x],
        "suffixes_head": [x for x in (clean_suffix(s) for s in suffixes_head) if x],
        "suffixes_tail": [x for x in (clean_suffix(s) for s in suffixes_tail) if x],
        "datatypes": [x for x in (clean_datatype(d) for d in datatypes) if x],
        "diagnoses": clean_diagnoses,
        "sex_values": sex_labels,
        "handedness_values": handedness_labels,
        "json_keys": json_keys,
        "json_samples": json_samples,
        "json_numeric_samples": json_numeric_samples,
        "extra_keys": extra_keys,
        "extra_samples": extra_samples,
        "authors": authors,
        "licenses": licenses,
        "funding_sources": funding_sources,
        "n_with_doi": n_with_doi,
        "description_terms": description_terms,
        # Static catalogs from value_mappings.yaml — used in prompt signals and intent planning
        "sidecar_fields": SIDECAR_FIELDS,
        "participant_extra_fields": PARTICIPANT_EXTRA_FIELDS,
        # Broad concept keys available for concept_query intents
        "concept_keys": FIELD_CONCEPT_EXPANSION,
        "age_stats": age_stats,
        "multimodal": [{"accession_id": r[0], "datatypes": [clean_datatype(d) for d in (r[1] or [])]} for r in multimodal],
        "sessions": [{"accession_id": r[0], "n_sessions": r[1]} for r in sessions],
        "dataset_samples": [
            {
                "accession_id": r[0],
                "name": r[1],
                "n_participants": r[2],
                "datatypes": [x for x in (clean_datatype(d) for d in (r[3] or [])) if x],
                "tasks": [x for x in (clean_task(t) for t in (r[4] or [])) if x],
            }
            for r in dataset_samples
        ],
    }


def weighted_choice(values: List[str], rng: random.Random, fallback: str) -> str:
    if not values:
        return fallback
    return rng.choice(values)


def _pick_synonym(field: str, code: str, rng: random.Random) -> str | None:
    """
    Return a random synonym for `code` (standard_code or concept key) in `field`.
    Returns None if no synonyms are defined, so callers can fall back gracefully.
    """
    syns = FIELD_SYNONYMS.get(field, {}).get(code, [])
    return rng.choice(syns) if syns else None


def _label_to_code(field: str, label: str) -> str | None:
    """Reverse-lookup: human label → standard_code for `field`. Returns None if not found."""
    return FIELD_LABEL_TO_CODE.get(field, {}).get(label.lower())


def build_intent_plan(stats: Dict[str, Any], n_pairs: int, seed: int) -> List[IntentSpec]:
    rng = random.Random(seed)

    # Each entry is (family_name, relative_weight).
    # Weight controls how often the family appears in the *overflow* slots after
    # the coverage pass.  All families appear at least once regardless of weight.
    # Families listed at weight 2 are twice as likely to fill extra slots.
    FAMILY_WEIGHTS = [
        ("scan_filter",        2),
        ("participant_filter",  2),
        ("combined_filter",    2),
        ("absence_query",      1),
        ("multimodal_query",   1),
        ("session_query",      1),
        ("ranking_query",      1),
        ("aggregate_query",    1),
        ("json_query",         1),
        ("comparison_query",   1),
        ("metadata_query",     1),
        ("author_query",       1),
        ("license_query",      1),
        ("funding_query",      1),
        ("doi_query",          1),
        ("description_search", 1),
        ("concept_query",           2),  # weight 2 = appears ~twice as often as weight-1 families
        ("age_query",               2),  # age filters are among the most natural researcher questions
        ("subject_multimodal_query",1),  # same subject has two modalities — double EXISTS pattern
        ("json_numeric_query",      1),  # numeric cast on other_entities JSON text values
    ]
    family_names   = [f for f, _ in FAMILY_WEIGHTS]
    family_weights = [w for _, w in FAMILY_WEIGHTS]

    # ── Coverage + overflow ────────────────────────────────────────────────────
    # Goal: every family appears at least once when n_pairs >= n_families.
    # When n_pairs < n_families we can't fit all families, so we use a weighted
    # sample *without* replacement (Efraimidis-Spirakis exponential-key trick):
    # each family gets a random key u^(1/w), and the top-n_pairs by key are
    # chosen. Higher-weight families get priority, but there is still randomness.
    n_families = len(family_names)

    if n_pairs >= n_families:
        # Enough slots — guarantee one slot per family, then fill the rest by
        # weighted draw (with replacement, so popular families appear more often).
        coverage = list(family_names)
        rng.shuffle(coverage)
        n_overflow = n_pairs - n_families
        overflow = rng.choices(family_names, weights=family_weights, k=n_overflow)
        family_sequence = coverage + overflow
    else:
        # Fewer slots than families — pick the n_pairs most-needed families via
        # weighted sampling without replacement, so high-weight families (scan,
        # participant, concept, age) are less likely to be cut than low-weight ones.
        keys = [
            (rng.random() ** (1.0 / max(w, 1)), f)
            for f, w in FAMILY_WEIGHTS
        ]
        family_sequence = [f for _, f in sorted(keys, reverse=True)[:n_pairs]]

    # Final shuffle so the ordered coverage slots aren't always first.
    rng.shuffle(family_sequence)

    sql_structures = [
        "join_distinct",
        "exists",
        "not_exists",
        "group_by_having",
        "filtered_aggregate",
        "ranking_limit",
    ]

    plans: List[IntentSpec] = []
    while len(plans) < n_pairs:
        family = family_sequence[len(plans)]
        # SQL structure: random pick, not cycled — avoids the same structure
        # always pairing with the same family.
        structure = rng.choice(sql_structures)
        paraphrase_count = rng.choices([1, 2, 3], weights=[5, 3, 1], k=1)[0]

        focus_values: Dict[str, Any] = {}
        notes: List[str] = []

        if family in {"scan_filter", "combined_filter", "comparison_query"}:
            use_tail = rng.random() < 0.5
            task_label = weighted_choice(
                stats["tasks_tail"] if use_tail else stats["tasks_head"],
                rng,
                "rest",
            )
            focus_values["task"] = task_label
            task_sc = _label_to_code("task", task_label)
            if task_sc:
                focus_values["task_standard_code"] = task_sc
                task_syn = _pick_synonym("task", task_sc, rng)
                if task_syn:
                    focus_values["task_synonym"] = task_syn
                    notes.append(
                        f"Task: use o.task = '{task_sc}' in SQL. "
                        f"In the question text, use the natural term '{task_syn}' "
                        f"instead of the raw code name."
                    )

            suffix_label = weighted_choice(
                stats["suffixes_tail"] if use_tail else stats["suffixes_head"],
                rng,
                "bold",
            )
            focus_values["suffix"] = suffix_label
            suffix_sc = _label_to_code("suffix", suffix_label)
            if suffix_sc:
                focus_values["suffix_standard_code"] = suffix_sc
                suffix_syn = _pick_synonym("suffix", suffix_sc, rng)
                if suffix_syn:
                    focus_values["suffix_synonym"] = suffix_syn
                    notes.append(
                        f"Suffix: use o.suffix = '{suffix_sc}' in SQL. "
                        f"In the question text, use '{suffix_syn}'."
                    )

            focus_values["datatype"] = weighted_choice(stats["datatypes"], rng, "func")
        if family in {"participant_filter", "combined_filter", "comparison_query"}:
            diag_label = weighted_choice(stats["diagnoses"], rng, "control")
            focus_values["diagnosis"] = diag_label
            diag_sc = _label_to_code("diagnosis", diag_label)
            if diag_sc:
                focus_values["diagnosis_standard_code"] = diag_sc
                diag_syn = _pick_synonym("diagnosis", diag_sc, rng)
                if diag_syn:
                    focus_values["diagnosis_synonym"] = diag_syn
                    notes.append(
                        f"Diagnosis: use p.diagnosis = '{diag_sc}' in SQL. "
                        f"In the question text, use the natural synonym '{diag_syn}' "
                        f"instead of the raw code name."
                    )
            focus_values["sex_value"] = weighted_choice(stats["sex_values"], rng, "female")
            focus_values["handedness_value"] = weighted_choice(stats["handedness_values"], rng, "right")
            focus_values["threshold"] = rng.choice([5, 10, 15, 20, 25, 30, 50])
        if family == "comparison_query":
            comp_type = rng.choice(["subject_count", "demographic_ratio"])
            focus_values["comparison_type"] = comp_type
            if comp_type == "demographic_ratio":
                notes.append(
                    "Compare counts of different sex or diagnosis groups within a single query. "
                    "Use COUNT(*) FILTER (WHERE p.sex = '...') or subquery counts in HAVING — "
                    "do NOT write two separate queries. Example intent: datasets where female "
                    "participants outnumber male participants."
                )
        if family in {"json_query", "metadata_query"}:
            focus_values["other_entities_key"] = weighted_choice(stats["json_keys"], rng, "acq")
            focus_values["extra_key"] = weighted_choice(stats["extra_keys"], rng, "group")
            # Sample a sidecar field and an extra participant field from the mapping catalog
            if stats["sidecar_fields"]:
                sf = rng.choice(list(stats["sidecar_fields"].values()))
                focus_values["sidecar_field_code"] = sf["codes"][0]
                focus_values["sidecar_field_synonym"] = rng.choice(sf["synonyms"]) if sf["synonyms"] else sf["label"]
            if stats["participant_extra_fields"]:
                ef = rng.choice(list(stats["participant_extra_fields"].values()))
                focus_values["extra_field_code"] = ef["codes"][0]
                focus_values["extra_field_synonym"] = rng.choice(ef["synonyms"]) if ef["synonyms"] else ef["label"]
            notes.append(
                "sidecar metadata lives in o.other_entities JSONB (use ->>), "
                "non-standard participant fields in p.extra JSONB (use ->>). "
                "NEVER search o.extension for metadata field names. "
                "Use the natural-language synonym (sidecar_field_synonym / extra_field_synonym) "
                "in the question — no apostrophes or technical key names in the question text."
            )
        if family in {"session_query"}:
            focus_values["min_sessions"] = rng.choice([2, 3, 4])
            notes.append("prefer COUNT(DISTINCT session) logic")
        if family in {"ranking_query"}:
            focus_values["limit"] = rng.choice([5, 10])
            notes.append("do not return extra columns beyond what the question asks")
        if family in {"absence_query"}:
            notes.append("use NOT EXISTS naturally")
        if family in {"multimodal_query"}:
            mm = rng.choice(stats["multimodal"] or [{"accession_id": "example", "datatypes": ["func", "eeg"]}])
            focus_values["modalities"] = mm["datatypes"][:2]
            notes.append("use explicit modality presence checks")
        if family in {"metadata_query"}:
            notes.append(
                "use o.other_entities->>'FieldName' for sidecar metadata fields. "
                "o.extension is file format only ('.nii.gz') — never use it for metadata lookups."
            )
        if family in {"author_query"}:
            focus_values["author"] = weighted_choice(
                stats["authors"], rng, "Poldrack, R.A."
            )
            notes.append(
                "use '= ANY(d.authors)' or ILIKE ANY(d.authors) for author lookup. "
                "For NOT queries (datasets WITHOUT this author) always guard against NULL: "
                "NOT ('X' = ANY(d.authors)) OR d.authors IS NULL"
            )
        if family in {"license_query"}:
            focus_values["license"] = weighted_choice(stats["licenses"], rng, "CC0")
            notes.append("filter on d.license column (exact match or ILIKE)")
        if family in {"funding_query"}:
            focus_values["funding_source"] = weighted_choice(
                stats["funding_sources"], rng, "NIH"
            )
            notes.append(
                "use '= ANY(d.funding)' for funding source lookup. "
                "For NOT queries (datasets WITHOUT this funding) always guard against NULL: "
                "NOT ('X' = ANY(d.funding)) OR d.funding IS NULL"
            )
        if family in {"description_search"}:
            term = weighted_choice(stats["description_terms"], rng, "aging")
            focus_values["search_term"] = term
            # Sometimes search name too, sometimes description_text only, sometimes both
            scope = rng.choice(["description_text", "name", "both"])
            focus_values["search_scope"] = scope
            if scope == "description_text":
                notes.append("use d.description_text ILIKE '%term%'")
            elif scope == "name":
                notes.append("use d.name ILIKE '%term%'")
            else:
                notes.append(
                    "use d.name ILIKE '%term%' OR d.description_text ILIKE '%term%'"
                )
        if family in {"doi_query"}:
            notes.append(
                "d.doi is a plain column on bids_datasets — use WHERE d.doi IS NOT NULL AND d.doi != '' directly"
            )
            # paper_references: occasionally generate a 'datasets citing paper X' query
            if rng.random() < 0.4:
                notes.append(
                    "optionally combine with paper_references: use EXISTS with unnest or ILIKE ANY(d.paper_references)"
                )
        if family == "concept_query":
            # Build candidate list: (field, alias, concept_key, n_children)
            # Only include concepts with >1 child so the expansion is meaningful.
            _FIELD_ALIAS = {"diagnosis": "p", "task": "o", "suffix": "o", "datatype": "o"}
            candidates = [
                (field, _FIELD_ALIAS[field], key, len(children))
                for field, exp in FIELD_CONCEPT_EXPANSION.items()
                for key, children in exp.items()
                if field in _FIELD_ALIAS and len(children) > 1
            ]
            if candidates:
                field, alias, concept_key, n_children = rng.choice(candidates)
                focus_values["concept_field"] = field
                focus_values["concept_key"] = concept_key
                focus_values["concept_alias_col"] = f"{alias}.{field}"
                focus_values["concept_children_count"] = n_children
                # Provide a natural-language synonym so Gemini uses it in the question
                # rather than inventing a term or exposing the internal key name.
                concept_natural_term = (
                    _pick_synonym(field, concept_key, rng)
                    or concept_key.replace("_", " ")  # fallback: at least remove underscores
                )
                focus_values["concept_natural_term"] = concept_natural_term
                notes.append(
                    f"Use concept key in SQL: {alias}.{field} = '{concept_key}'. "
                    f"The system auto-expands this to {n_children} specific standard_codes at runtime. "
                    "Do NOT list individual sub-codes — just use the single concept key. "
                    f"In the question text, use the natural language term '{concept_natural_term}' "
                    f"(or a close paraphrase of it) — never expose the internal key '{concept_key}'."
                )
                structure = "exists"  # concept queries always use EXISTS to stay idiomatic

        if family == "subject_multimodal_query":
            # "Find datasets where the same subject has BOTH <condition A> and <condition B>."
            # Requires a double EXISTS on bids_objects correlated on (dataset_id, subject).
            # Conditions can mix any bids_objects columns: datatype, task, suffix.
            # This teaches the LLM the subject-level AND/OR cross-condition pattern.

            # Build a pool of concrete (column, value) conditions from real DB values
            cond_pool = []
            for dt in (stats["datatypes"] or []):
                cond_pool.append(("datatype", dt))
            for t in (stats["tasks_head"] or []):
                cond_pool.append(("task", t))
            for s in (stats["suffixes_head"] or []):
                cond_pool.append(("suffix", s))
            if not cond_pool:
                cond_pool = [("datatype", "func"), ("datatype", "eeg")]

            # Pick two distinct conditions (can be same column or different columns)
            cond_a, cond_b = rng.sample(cond_pool, min(2, len(cond_pool)))
            # If cond_pool has only 1 entry, duplicate with a fallback
            if cond_a == cond_b:
                cond_b = ("datatype", "eeg") if cond_a != ("datatype", "eeg") else ("datatype", "func")

            # Logical connective — AND is most natural and teaches the double-EXISTS idiom
            # OR is less common but valid (subject has EITHER modality)
            connective = rng.choice(["AND", "AND", "AND", "OR"])  # AND 3× more likely
            focus_values["condition_a_col"] = cond_a[0]
            focus_values["condition_a_val"] = cond_a[1]
            focus_values["condition_b_col"] = cond_b[0]
            focus_values["condition_b_val"] = cond_b[1]
            focus_values["connective"] = connective

            col_a, val_a = cond_a
            col_b, val_b = cond_b

            if connective == "AND":
                notes.append(
                    f"Same subject satisfies BOTH: {col_a}='{val_a}' AND {col_b}='{val_b}'.\n"
                    "SQL — double EXISTS correlated on (dataset_id, subject):\n"
                    "  WHERE EXISTS (\n"
                    "      SELECT 1 FROM bids_objects o1\n"
                    f"     WHERE o1.dataset_id = d.id AND o1.{col_a} = '{val_a}'\n"
                    "        AND EXISTS (\n"
                    "            SELECT 1 FROM bids_objects o2\n"
                    "            WHERE o2.dataset_id = d.id AND o2.subject = o1.subject\n"
                    f"             AND o2.{col_b} = '{val_b}'\n"
                    "        )\n"
                    "  )\n"
                    "This is a subject-level AND — the same physical subject must appear in both. "
                    "A dataset-level check (just HAVING two datatypes) is WRONG here."
                )
            else:  # OR
                notes.append(
                    f"Subject has EITHER {col_a}='{val_a}' OR {col_b}='{val_b}' (or both).\n"
                    "SQL — two EXISTS joined with OR:\n"
                    "  WHERE (\n"
                    "      EXISTS (SELECT 1 FROM bids_objects o1\n"
                    f"             WHERE o1.dataset_id = d.id AND o1.{col_a} = '{val_a}'\n"
                    f"               AND o1.subject = <subject_from_outer>)\n"
                    "      OR\n"
                    "      EXISTS (SELECT 1 FROM bids_objects o2\n"
                    f"             WHERE o2.dataset_id = d.id AND o2.{col_b} = '{val_b}'\n"
                    f"               AND o2.subject = <subject_from_outer>)\n"
                    "  )\n"
                    "Alternatively: check at the dataset level with two separate EXISTS "
                    "if the question is about dataset coverage rather than individual subject."
                )
            structure = "exists"

        if family == "json_numeric_query":
            # Numeric comparisons on other_entities JSONB values, which are stored as TEXT.
            # The cast pattern is: (o.other_entities->>'key')::float > threshold
            numeric = stats["json_numeric_samples"]
            # Known MRI/EEG acquisition parameters that are almost always numeric
            KNOWN_NUMERIC = {
                "RepetitionTime":   (0.5,  4.0,  "seconds"),
                "EchoTime":         (0.01, 0.1,  "seconds"),
                "FlipAngle":        (5,    90,   "degrees"),
                "TotalReadoutTime": (0.01, 0.1,  "seconds"),
                "NumberOfVolumes":  (50,   500,  "volumes"),
                "SamplingFrequency":(128,  2048, "Hz"),
            }
            # Prefer a key we have real data for; fall back to known numeric keys
            candidates = list(numeric.keys()) or list(KNOWN_NUMERIC.keys())
            key = rng.choice(candidates) if candidates else "RepetitionTime"
            # Determine a plausible threshold
            if key in KNOWN_NUMERIC:
                lo, hi, unit = KNOWN_NUMERIC[key]
                threshold = round(rng.uniform(lo, hi), 2)
            else:
                # Use the sampled value as a reference point, vary by ±50 %
                ref = numeric.get(key, 2.0)
                threshold = round(ref * rng.uniform(0.5, 1.5), 2)
                unit = "units"
            operator = rng.choice([">", "<", ">=", "<="])
            focus_values["json_key"] = key
            focus_values["json_threshold"] = threshold
            focus_values["json_operator"] = operator
            notes.append(
                f"Numeric filter on JSONB sidecar field '{key}': "
                f"(o.other_entities->>'{key}')::float {operator} {threshold}.\n"
                "JSONB values are stored as TEXT — always cast with ::float (or ::int) "
                "before any numeric comparison.  Never compare as a string.\n"
                f"Use a natural description of '{key}' in the question text, not the raw key name."
            )
            structure = rng.choice(["exists", "join_distinct", "group_by_having"])

        if family == "age_query":
            # Pick a query style:
            #   range_filter   — WHERE p.age BETWEEN x AND y   (most natural)
            #   threshold      — WHERE p.age > x  or  < x
            #   having_mean    — HAVING AVG(...) > threshold
            age_style = rng.choice(["range_filter", "range_filter", "threshold", "having_mean"])
            focus_values["age_style"] = age_style
            a = stats["age_stats"]

            if age_style == "range_filter":
                # Pick one of several clinically meaningful windows
                windows = [
                    (0,  2,   "neonates / infants"),
                    (0,  12,  "children"),
                    (0,  18,  "pediatric participants"),
                    (18, 30,  "young adults"),
                    (18, 65,  "working-age adults"),
                    (60, 120, "elderly participants"),
                    (65, 120, "older adults (65+)"),
                    # data-driven windows anchored to actual DB percentiles
                    (a["p10"], a["p25"], "younger quartile"),
                    (a["p75"], a["p90"], "older quartile"),
                ]
                lo, hi, label = rng.choice(windows)
                # Clamp to actual data range so the window isn't empty
                lo = max(lo, a["min"])
                hi = min(hi, a["max"])
                focus_values["age_lo"] = round(lo)
                focus_values["age_hi"] = round(hi)
                focus_values["age_label"] = label
                notes.append(
                    f"Filter: p.age BETWEEN {round(lo)} AND {round(hi)}  "
                    f"('{label}').  "
                    "Use natural phrasing for the age group — do not put the numbers in the question verbatim unless the intent is very specific."
                )

            elif age_style == "threshold":
                direction = rng.choice(["above", "below"])
                if direction == "above":
                    threshold = rng.choice([40, 50, 60, 65, 70, round(a["p75"])])
                    focus_values["age_threshold"] = threshold
                    focus_values["age_direction"] = "above"
                    notes.append(f"Filter: WHERE p.age > {threshold}  (use HAVING or EXISTS as appropriate)")
                else:
                    threshold = rng.choice([5, 10, 12, 18, 25, round(a["p25"])])
                    focus_values["age_threshold"] = threshold
                    focus_values["age_direction"] = "below"
                    notes.append(f"Filter: WHERE p.age < {threshold}  (use HAVING or EXISTS as appropriate)")

            elif age_style == "having_mean":
                threshold = rng.choice([20, 25, 30, 40, 50, round(a["mean"])])
                direction = rng.choice(["above", "below"])
                focus_values["mean_age_threshold"] = threshold
                focus_values["age_direction"] = direction
                op = ">" if direction == "above" else "<"
                notes.append(
                    f"HAVING AVG(p.age) {op} {threshold}  — filter datasets whose mean participant age "
                    f"is {'above' if direction == 'above' else 'below'} {threshold}.  "
                    "Never use subject_count in HAVING for age comparisons."
                )
                structure = "group_by_having"

        plans.append(
            IntentSpec(
                family=family,
                sql_structure=structure,
                paraphrase_count=paraphrase_count,
                focus_values=focus_values,
                notes=notes,
            )
        )

    return plans


def format_prompt(stats: Dict[str, Any], intent_plan: List[IntentSpec], pairs_per_prompt: int, prompt_index: int) -> str:
    bundle_count = sum(1 for item in intent_plan if item.paraphrase_count > 1)
    dataset_samples = "\n".join(
        [
            f"  - {d['accession_id']} | {d['name']!r} | n_participants={d['n_participants']} | "
            f"datatypes={d['datatypes']} | tasks={d['tasks']}"
            for d in stats["dataset_samples"]
        ]
    )
    sessions = "\n".join([f"  - {s['accession_id']} | {s['n_sessions']} sessions" for s in stats["sessions"][:5]])
    multimodal = "\n".join([f"  - {m['accession_id']} | datatypes={m['datatypes']}" for m in stats["multimodal"][:5]])
    plan_json = json.dumps([asdict(item) for item in intent_plan], indent=2)

    examples_text = "\n\n".join(
        f'Q: {ex["question"]}\n```sql\n{ex["sql"]}\n```'
        for ex in EXAMPLE_PAIRS
    )

    return f"""
You are generating high-quality Text-to-SQL training data for a BIDS neuroimaging database.

Your job is to generate EXACTLY {pairs_per_prompt} records. Some records may share the same SQL intent,
but only in controlled paraphrase bundles. Diversity matters more than raw volume.

═══════════════════════════════════════════════════════════
SYSTEM PROMPT (this is what the model sees at inference time — match it exactly)
═══════════════════════════════════════════════════════════
{SYSTEM}

═══════════════════════════════════════════════════════════
CANONICAL OUTPUT FORMAT (every generated SQL must follow this SELECT structure)
═══════════════════════════════════════════════════════════
{examples_text}

═══════════════════════════════════════════════════════════
REAL DATABASE SIGNALS
═══════════════════════════════════════════════════════════
Common tasks: {stats['tasks_head']}
Long-tail tasks: {stats['tasks_tail']}
Common suffixes: {stats['suffixes_head']}
Long-tail suffixes: {stats['suffixes_tail']}
Datatypes: {stats['datatypes']}
Diagnoses: {stats['diagnoses']}
Sex values: {stats['sex_values']}
Handedness values: {stats['handedness_values']}
Common object JSON keys: {stats['json_keys']}
Common participant extra keys: {stats['extra_keys']}
Authors (top contributors): {stats['authors']}
Licenses in use: {stats['licenses']}
Funding sources: {stats['funding_sources']}
Datasets with a DOI: {stats['n_with_doi']} (use d.doi IS NOT NULL or ILIKE to match)
Description search terms (real words from description_text): {stats['description_terms']}
  → use d.description_text ILIKE '%term%' or d.name ILIKE '%term%' for topic searches
Participant age (p.age FLOAT, NULL if not recorded):
  range {stats['age_stats']['min']:.0f}–{stats['age_stats']['max']:.0f} yrs | mean {stats['age_stats']['mean']:.1f} | p25={stats['age_stats']['p25']:.0f} p75={stats['age_stats']['p75']:.0f}
  → age filters: WHERE p.age BETWEEN lo AND hi  |  WHERE p.age > threshold  |  HAVING AVG(p.age) > threshold

Broad concept keys (auto-expanded by post-processor — use in SQL instead of listing all sub-codes):
(In question text, use one of the listed natural-language synonyms — never the raw key name.)
{chr(10).join(
    "  - {alias}.{field} = '{key}'  ({n} sub-codes){syn_hint}".format(
        alias={"diagnosis": "p", "task": "o", "suffix": "o", "datatype": "o"}.get(field, "o"),
        field=field, key=key, n=n,
        syn_hint=(
            "  — natural terms: " + ", ".join(
                repr(s) for s in FIELD_SYNONYMS.get(field, {}).get(key, [])[:4]
            )
            if FIELD_SYNONYMS.get(field, {}).get(key) else ""
        ),
    )
    for field, exp in stats['concept_keys'].items()
    for key, children in sorted(exp.items(), key=lambda x: -len(x[1]))
    for n in [len(children)]
    if n > 1
)}

Numeric sidecar keys (cast required — these values are TEXT in JSONB):
{chr(10).join(
    f"  - o.other_entities->>'{k}' → e.g. {v}  (use ::float for numeric comparisons)"
    for k, v in list(stats['json_numeric_samples'].items())[:10]
) or "  (none detected — use RepetitionTime, EchoTime, FlipAngle as typical examples)"}

Sidecar metadata fields (o.other_entities JSONB) — use the natural-language synonym in the question, the code in SQL:
{chr(10).join(
    f"  - synonym: {', '.join(v['synonyms'][:3]) or v['label']}  →  o.other_entities->>{repr(v['codes'][0])}"
    + (f"  (e.g. {repr(stats['json_samples'].get(v['codes'][0], ''))})" if stats['json_samples'].get(v['codes'][0]) else "")
    for v in stats['sidecar_fields'].values()
)}

Participant extra fields (p.extra JSONB) — use the natural-language synonym in the question, the code in SQL:
{chr(10).join(
    f"  - synonym: {', '.join(v['synonyms'][:3]) or v['label']}  →  p.extra->>{repr(v['codes'][0])}"
    + (f"  (e.g. {repr(stats['extra_samples'].get(v['codes'][0], ''))})" if stats['extra_samples'].get(v['codes'][0]) else "")
    for v in stats['participant_extra_fields'].values()
)}

Sample datasets:
{dataset_samples}

Longitudinal examples:
{sessions}

Multimodal examples:
{multimodal}

═══════════════════════════════════════════════════════════
GENERATION RULES
═══════════════════════════════════════════════════════════
1. Every SQL query must be valid PostgreSQL for this schema.

2. Column values — SQL vs. question text:
   - diagnosis / task / suffix / datatype: SQL uses the exact lowercase standard_code or concept key
     (from the intent plan's *_standard_code field). Question text uses the natural synonym or
     description (intent plan's *_synonym / concept_natural_term) — never the raw code name.
     p.diagnosis is for clinical/group labels only — no drug names, procedures, or non-clinical terms.
   - Concept keys (e.g. p.diagnosis = 'epilepsy_spectrum', o.task = 'resting_state'): write the key
     as-is. The post-processor expands it — do NOT expand into IN (...) yourself.
   - sex / handedness: use only the values from the provided lists.
   - o.extension is file format only ('.nii.gz', '.json', '.tsv'). Never search it for metadata field
     names ('bold', 'AcquisitionTime', etc.). Sidecar metadata lives in o.other_entities JSONB:
     o.other_entities->>'AcquisitionTime'. Non-standard participant attributes (BMI, questionnaires)
     live in p.extra JSONB: p.extra->>'bmi'. Do NOT put these in p.diagnosis.

3. SQL diversity: use a genuine mix of EXISTS, NOT EXISTS, JOIN+DISTINCT, GROUP BY+HAVING, and
   ranking (ORDER BY / LIMIT). ≥40% of the batch must use a structure other than plain JOIN+WHERE.
   ≥25% must draw on long-tail values from the database signals. Do not repeat the same skeleton.

4. Mandatory SELECT — every query, no exceptions, no extra columns unless the question demands them:
     d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,
     d.source_type, d.remote_url, d.validation_status,
     COUNT(DISTINCT o.subject) AS subject_count
   Counting: COUNT(DISTINCT o.subject) always in SELECT. For HAVING thresholds on participant
   attributes (sex, diagnosis, age, etc.) use COUNT(DISTINCT p.participant_id) — never subject_count.

5. Question text: no raw column names, code names, or technical keys in apostrophes. Use natural
   language throughout (e.g. "studies with elderly participants", not "datasets with p.age > 60" or
   "datasets with 'AcquisitionTime'"). Use synonyms from the sidecar/extra field lists above.

6. Family-specific SQL patterns:
   - age_query: p.age is FLOAT (NULL if not recorded). Patterns: WHERE p.age BETWEEN lo AND hi,
     WHERE p.age > threshold, HAVING AVG(p.age) > threshold. Return standard 9 columns only —
     never add AVG(p.age) or any other aggregate to SELECT.
   - subject_multimodal_query: "same subject has both X and Y" → double EXISTS correlated on
     (dataset_id, subject). A dataset-level HAVING check only proves the dataset has both types,
     not that one person does. Always use the double EXISTS pattern from the intent note.
   - json_numeric_query: o.other_entities values are TEXT. Always cast for numeric comparisons:
     (o.other_entities->>'RepetitionTime')::float > 2.0   (never compare as string).

7. JOIN hygiene — two hard anti-patterns:
   a) No self-join on bids_datasets: EXISTS (SELECT 1 FROM bids_datasets bd WHERE bd.id = d.id …)
      is pointless — check d.* columns directly.
        WRONG:   WHERE EXISTS (SELECT 1 FROM bids_datasets bd WHERE bd.id = d.id AND bd.doi IS NOT NULL)
        CORRECT: WHERE d.doi IS NOT NULL AND d.doi != ''
   b) No cross-product: never JOIN bids_participants alongside the mandatory bids_objects join.
      For participant counts in HAVING, use correlated subqueries:
        WRONG:   JOIN bids_participants p ON p.dataset_id = d.id … HAVING COUNT(DISTINCT p.participant_id) FILTER (WHERE p.sex = 'male') > …
        CORRECT: HAVING (SELECT COUNT(*) FROM bids_participants p WHERE p.dataset_id = d.id AND p.sex = 'male') > …
      Joining bids_participants alone (no bids_objects in the query) is safe.

8. NULL-safe negation on nullable d.* columns (TEXT[]: funding, authors, paper_references;
   TEXT: license, doi, description_text): NOT (NULL) = NULL which WHERE treats as FALSE,
   silently excluding rows with no data. Always guard with OR col IS NULL:
     WRONG:   WHERE NOT ('NIH' = ANY(d.funding))   /   WHERE d.license != 'CC0'
     CORRECT: WHERE NOT ('NIH' = ANY(d.funding)) OR d.funding IS NULL
              WHERE d.license != 'CC0' OR d.license IS NULL
   NOT EXISTS is already NULL-safe — do NOT add OR IS NULL to NOT EXISTS.

═══════════════════════════════════════════════════════════
PARAPHRASE POLICY
═══════════════════════════════════════════════════════════
- This prompt includes {bundle_count} planned paraphrase bundles.
- For a paraphrase bundle, keep one SQL intent and produce 2-3 stylistically distinct questions.
- Good paraphrase styles: concise, formal/scientific, natural/casual, indirect.
- Never change thresholds, negation, requested output, or comparison meaning.

═══════════════════════════════════════════════════════════
INTENT PLAN FOR THIS PROMPT
═══════════════════════════════════════════════════════════
Use this plan as coverage guidance. You may refine wording, but you must preserve the
family, SQL-structure intent, and paraphrase counts.

```json
{plan_json}
```

═══════════════════════════════════════════════════════════
SELF-REVIEW (mandatory before returning output)
═══════════════════════════════════════════════════════════
Before emitting the final JSON array, silently review every generated pair
against this checklist.  Fix any pair that fails — do not include it broken.

For each (question, sql) pair verify:
  [ ] SELECT contains all 9 mandatory columns: d.id, d.name, d.accession_id,
      d.bids_version, d.dataset_type, d.source_type, d.remote_url,
      d.validation_status, COUNT(DISTINCT o.subject) AS subject_count
  [ ] No extra SELECT columns unless the question explicitly requires them
  [ ] o.extension is NOT used for metadata lookups (file format only)
  [ ] JSONB numeric comparisons use ::float or ::int cast, not string compare
  [ ] sex / handedness values are from the provided lists only
  [ ] diagnosis uses standard clinical terms or concept keys — not drug names
  [ ] Concept keys are used as-is (e.g. p.diagnosis = 'epilepsy_spectrum'),
      NOT expanded into IN (...) lists
  [ ] Subject-level AND queries use double EXISTS correlated on subject,
      not a dataset-level HAVING check
  [ ] HAVING clauses for participant criteria use COUNT(DISTINCT p.participant_id),
      never subject_count
  [ ] EXISTS is NOT used to check a bids_datasets column (self-join anti-pattern);
      d.doi, d.license, d.validation_status etc. are checked directly on d
  [ ] bids_participants is NOT joined alongside bids_objects (cross-product risk);
      participant counts in HAVING use correlated subqueries instead
  [ ] Negations on nullable columns (d.funding, d.authors, d.license, etc.) use
      OR <col> IS NULL so that rows with no data are not silently excluded
  [ ] The question uses natural language — no raw column names, key names,
      or internal codes in apostrophes

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════
Return ONLY a JSON array.
Each element must have:
  - "question"
  - "sql"
  - "pattern"
  - "family"
  - "sql_structure"
  - "paraphrase_bundle_id"  (null if standalone)

Use short pattern labels if you want, but keep them consistent.
Prompt index: {prompt_index}
""".strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--pairs-per-prompt", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = create_engine(args.db_url)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        stats = fetch_stats(session)

    for i in range(1, args.n_prompts + 1):
        plan = build_intent_plan(stats, args.pairs_per_prompt, seed=args.seed + i)
        prompt_text = format_prompt(stats, plan, args.pairs_per_prompt, i)
        out_path = args.out_dir / f"prompt_{i:03d}.txt"
        out_path.write_text(prompt_text, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
