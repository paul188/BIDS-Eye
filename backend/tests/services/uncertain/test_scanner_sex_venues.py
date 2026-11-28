"""Text venues for participant sex and scanner metadata in the uncertain tier
(services/uncertain/tier.py): a dataset that does not record the
value counts as probably relevant when its own text names it."""

from __future__ import annotations

import re

import pytest
import sqlglot
from sqlglot import exp

from config.settings import settings_obj
from services.uncertain import classify as tier_classify, scanner as tier_scanner, tier as ut
from services import asked
from services.results.display import highlights
from services.results.display.chips import describe
from services.fulltext.text_search import Clause

_HEAD = "SELECT d.id, d.name FROM bids_datasets AS d WHERE "
# The guarded cast the rewriter writes around sidecar numbers.
_CAST = ("CASE WHEN (o2.other_entities ->> 'MagneticFieldStrength') ~ "
         "'^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$' "
         "THEN CAST((o2.other_entities ->> 'MagneticFieldStrength') AS DOUBLE PRECISION) END")
STRENGTH_7T = (f"EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
               f"AND {_CAST} > 6.5 AND {_CAST} < 7.5)")
STRENGTH_7T_FMRI = (f"EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
                    f"AND (o2.datatype IN ('functional_mri') OR o2.suffix IN ('fmri_bold')) "
                    f"AND {_CAST} BETWEEN 6.5 AND 7.5)")
GE_OR_PHILIPS = (
    "(EXISTS(SELECT 1 FROM bids_objects AS bo WHERE bo.dataset_id = d.id "
    "AND bo.other_entities ->> 'Manufacturer' ILIKE '%GE%') OR EXISTS(SELECT 1 FROM bids_objects "
    "AS bo WHERE bo.dataset_id = d.id AND bo.other_entities ->> 'Manufacturer' ILIKE '%Philips%'))")
PRISMA = ("EXISTS(SELECT 1 FROM bids_objects AS bo WHERE bo.dataset_id = d.id "
          "AND bo.other_entities ->> 'ManufacturerModelName' ILIKE '%Prisma_fit%')")
FEMALE = ("EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
          "AND LOWER(p2.sex) IN ('female', 'f'))")
ONLY_FEMALE = (FEMALE + " AND NOT EXISTS (SELECT 1 FROM bids_participants p3 WHERE "
               "p3.dataset_id = d.id AND (p3.sex IS NULL OR LOWER(p3.sex) NOT IN ('female', 'f')))")


def _node(condition: str):
    return sqlglot.parse_one(_HEAD + condition, read="postgres").args["where"].this


def _classify(condition: str):
    return tier_classify.classify(Clause.read(_node(condition)), "d")


# ── Field strength: what is asked, and the pattern ────────────────────────────

@pytest.mark.parametrize(("condition", "expected"), [
    (f"{_CAST} > 6.5 AND {_CAST} < 7.5", [7.0]),
    (f"{_CAST} BETWEEN 2.5 AND 3.5", [3.0]),
    (f"{_CAST} = 1.5", [1.5]),
    (f"{_CAST} >= 7", [7.0, 9.4, 10.5, 11.7, 14.0]),
    (f"{_CAST} > 3", [4.0, 4.7, 7.0, 9.4, 10.5, 11.7, 14.0]),
    (f"{_CAST} < 1.0", [0.5]),
    ("o2.datatype = 'anat'", []),
])
def test_asked_field_strengths(condition, expected):
    node = _node(f"EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id AND {condition})")
    assert asked.field_strengths(node) == expected


@pytest.mark.parametrize(("text", "hit"), [
    ("Data were acquired on a 7T Siemens scanner.", True),
    ("Scanned at 7 T.", True),
    ("a 7-Tesla system", True),
    ("7.0T MAGNETOM", True),
    ("at 7 T with a 32-channel coil", True),
    ("7 tesla", True),
    ("A 17T magnet", False),
    ("0.7T", False),
    ("sub-01_ses-7T_bold.nii", False),
    ("7 T1w images were collected", False),
    ("gradients of 200 T/m/s", False),
    ("3T Prisma", False),
])
def test_field_strength_regex_in_python(text, hit):
    assert bool(re.search(tier_scanner.field_strength_regex([7.0]), text)) is hit


def test_postgres_form_has_no_backslashes():
    pattern = tier_scanner.field_strength_regex([1.5, 7.0], postgres=True)
    assert "\\" not in pattern and "[[:alnum:]_" in pattern and "1[.,]50*" in pattern
    # SQLAlchemy's text() would read ":7" / ":T" after "(?" as bind parameters.
    assert "(?:" not in pattern
    # The whole tier query compiles unchanged, as the search runs it.
    for condition in (STRENGTH_7T, STRENGTH_7T_FMRI, GE_OR_PHILIPS, PRISMA, FEMALE, ONLY_FEMALE):
        sql = ut.apply(_HEAD + condition, papers=False)
        assert _as_run(sql) == sql, condition


def _as_run(sql: str) -> str:
    """*sql* as SQLAlchemy's text() sends it to asyncpg ("(?:7" became "(?$1")."""
    from sqlalchemy import text
    from sqlalchemy.dialects.postgresql import asyncpg
    return str(text(sql).compile(dialect=asyncpg.dialect()))
    assert tier_scanner.field_strength_regex([]) is None
    # Display and translation share one pattern.
    assert highlights.field_strength_pattern([7.0]).pattern == tier_scanner.field_strength_regex([7.0])


# ── Sidecar conditions in the tier ────────────────────────────────────────────

def test_field_strength_condition_gets_a_text_venue():
    soft = _classify(STRENGTH_7T)
    assert soft.kind == "sidecar"
    # The matching datasets once, as a WITH query, not a correlated EXISTS
    # that the tier's OR would probe once per dataset.
    (name, select), = soft.ctes
    assert soft.sql == f"d.id IN (SELECT dataset_id FROM {name})"
    assert select.startswith("SELECT DISTINCT o2.dataset_id FROM bids_objects AS o2 WHERE")
    assert "dataset_id = d.id" not in select and "MagneticFieldStrength" in select
    # Indexed words first, then the regex over the dataset's own text.
    assert soft.probable.startswith("d.search_tsv @@ TO_TSQUERY('english', '7 | 7t | 7tesla | 7.0")
    assert "readme_text" in soft.probable and "(7([.]0+)?)" in soft.probable
    assert "bids_objects" not in soft.probable   # no file part to keep
    # Recorded or not is asked only where the text names the value.
    assert soft.known.startswith(f"(({soft.probable}) AND EXISTS(SELECT 1 FROM bids_objects AS uk")
    assert soft.weaker[-1].reason() == "field strength not in the file metadata; the title, description or README names 7 T"


def test_field_strength_with_a_file_part_keeps_it():
    # "7T fMRI" as one condition: a README naming 7T counts only with fMRI files.
    soft = _classify(STRENGTH_7T_FMRI)
    files = "(o2.datatype IN ('functional_mri') OR o2.suffix IN ('fmri_bold'))"
    confirmed, file_cte = soft.ctes
    assert "MagneticFieldStrength" in confirmed[1] and files in confirmed[1]
    assert file_cte[1] == f"SELECT DISTINCT o2.dataset_id FROM bids_objects AS o2 WHERE {files}"
    assert soft.probable.endswith(f"AND (d.id IN (SELECT dataset_id FROM {file_cte[0]}))")


def test_file_part_reads_the_facets_summary_when_it_covers_every_dataset(monkeypatch):
    from services.database_stats.database_stats import Summaries, database_stats
    monkeypatch.setattr(database_stats, "summaries", Summaries(facets=True))
    soft = _classify(STRENGTH_7T_FMRI)
    assert len(soft.ctes) == 1   # only the sidecar lookup
    assert soft.probable.endswith(
        "AND (d.id IN (SELECT o2.dataset_id FROM bids_dataset_facets AS o2 WHERE "
        "(o2.datatype IN ('functional_mri') OR o2.suffix IN ('fmri_bold'))))")


def test_known_is_every_candidate_with_a_possible_level(monkeypatch):
    monkeypatch.setattr(settings_obj, "UNCERTAIN_POSSIBLE", True)
    assert _classify(STRENGTH_7T).known.startswith("EXISTS(SELECT 1 FROM bids_objects AS uk")


def test_manufacturer_and_model_conditions():
    soft = _classify(GE_OR_PHILIPS)
    assert soft.kind == "sidecar"
    assert soft.sql.count("d.id IN (SELECT dataset_id FROM _qf_sidecar_") == 2 and len(soft.ctes) == 2
    assert "general <-> electric" in soft.probable and "(philips)" in soft.probable
    assert soft.weaker[-1].reason().endswith("names GE or Philips")
    model = _classify(PRISMA)
    assert "'(prisma <-> fit)'" in model.probable
    # A short fragment of an unknown make is no evidence.
    unknown = PRISMA.replace("ManufacturerModelName", "Manufacturer").replace("Prisma_fit", "AB")
    assert _classify(unknown) is None


@pytest.mark.parametrize("condition", [
    # Two files of one subject (a session at each strength): no text can say that.
    (f"EXISTS(SELECT 1 FROM bids_objects AS o_low WHERE o_low.dataset_id = d.id AND "
     f"{_CAST.replace('o2.', 'o_low.')} < 1.0 AND EXISTS(SELECT 1 FROM bids_objects AS o_high "
     f"WHERE o_high.dataset_id = d.id AND o_high.subject = o_low.subject AND "
     f"{_CAST.replace('o2.', 'o_high.')} > 2.0))"),
    # Repetition time: READMEs state it in too many forms.
    ("EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id AND "
     "CAST(o2.other_entities ->> 'RepetitionTime' AS DOUBLE PRECISION) < 1.0)"),
    # Negated: the dataset must not have it.
    f"NOT {STRENGTH_7T}",
    # A sidecar test mixed with something that is not a file subquery.
    f"({STRENGTH_7T} OR d.search_tsv @@ to_tsquery('english', '7t'))",
])
def test_other_sidecar_shapes_stay_hard(condition):
    assert _classify(condition) is None


# ── Participant sex in the tier ───────────────────────────────────────────────

def _python_pattern(probable: str) -> str:
    return re.search(r"~\* '(.*)'\)$", probable).group(1).replace("''", "'").replace(
        "[^[:alnum:]_'’]", r"[^\w'’]").replace("[^[:alnum:]_]", r"[^\w]")


def test_one_sex_is_read_where_the_text_speaks_of_participants():
    soft = _classify(FEMALE)
    assert soft.kind == "sex" and "~*" in soft.probable
    assert soft.weaker[-1].reason() == "participant sex not recorded; the title, description or README names female participants"
    pattern = _python_pattern(soft.probable)
    # The words around a mention decide, the same for every word.
    for text, hit in [("22 women took part.", True), ("12 healthy young girls", True),
                      ("Females: 14, males: 12", True), ("women (n = 14)", True),
                      ("Female participants only.", True), ("All subjects were female.", True),
                      ("An all-female sample of 30 adults", True),
                      ("One woman, 34 years old", False), ("1 woman, 34 years old", True),
                      ("a woman's voice read the story", False),
                      ("a 34-year-old woman", True), ("voices of 2 women's choirs", False),
                      ("one female patient", True), ("faces of women", False),
                      ("women were excluded", False), ("22 men took part.", False)]:
        assert bool(re.search(pattern, text, re.IGNORECASE)) is hit, text
    male = _python_pattern(_classify(FEMALE.replace("'female', 'f'", "'male', 'm'")).probable)
    for text, hit in [("20 men", True), ("14 boys aged 8 to 10", True), ("male volunteers", True),
                      ("recorded on the Isle of Man", False),
                      ("the brain of man", False), ("12 right-handed men", True), ("12 women", False),
                      ("10 female participants", False)]:
        assert bool(re.search(male, text, re.IGNORECASE)) is hit, text


def test_only_one_sex_needs_the_text_to_say_only():
    soft = _classify(ONLY_FEMALE)
    assert "~*" in soft.probable and "all[- ]?(female|females|woman|women|girl|girls)" in soft.probable
    assert soft.weaker[-1].reason().endswith("all participants are female")
    pattern = _python_pattern(soft.probable)
    for text, hit in [("An all-female sample", True), ("Only women took part.", True),
                      ("female-only cohort", True), ("All participants were female.", True),
                      ("22 women and 20 men", False), ("female and male", False)]:
        assert bool(re.search(pattern, text, re.IGNORECASE)) is hit, text


def test_only_one_sex_stays_one_condition_in_the_tier():
    # Flattened, "only" was lost and the NOT EXISTS alone held for datasets
    # without participant rows.
    sql = ut.apply(_HEAD + ONLY_FEMALE, papers=False, hoist_terms=False)
    assert "all participants are female" in sql
    assert "names female participants" not in sql
    assert sql.count("evidence.u") and "evidence.u1" not in sql   # one soft condition
    # Without the NOT EXISTS, nothing is paired.
    assert "names female participants" in ut.apply(_HEAD + FEMALE, papers=False)


def test_both_sexes_or_sex_with_age_add_no_text():
    both = FEMALE.replace("('female', 'f')", "('female', 'f', 'male', 'm')")
    assert _classify(both).probable is None
    with_age = FEMALE.replace("IN ('female', 'f')", "IN ('female', 'f') AND p2.age >= 40")
    assert _classify(with_age) is None   # two recorded fields at once stay hard


# ── The whole tier query ──────────────────────────────────────────────────────

def test_tier_query_carries_the_new_reasons():
    sql = ut.apply(_HEAD + STRENGTH_7T + " AND " + FEMALE, papers=False, hoist_terms=True)
    assert "field strength not in the file metadata" in sql
    assert "participant sex not recorded; the title, description or README names female participants" in sql
    assert isinstance(sqlglot.parse_one(sql, read="postgres"), exp.Select)


# ── Marking ───────────────────────────────────────────────────────────────────

def test_chips_mark_the_words_of_their_text_venue():
    female = describe(_HEAD + FEMALE)[0]
    assert female.kind == "sex" and {"women", "female"} <= set(female.text_words)
    assert ("women",) in highlights.highlight_terms(_HEAD + FEMALE)
    vendor = describe(_HEAD + GE_OR_PHILIPS)[0]
    assert {"philips", "ge healthcare", "general electric"} <= set(vendor.text_words)
    terms = highlights.highlight_terms(_HEAD + GE_OR_PHILIPS)
    assert ("philips",) in terms and ("ge", "healthcare") in terms and ("signa",) in terms
    assert ("prisma", "fit") in highlights.highlight_terms(_HEAD + PRISMA)
    assert highlights.highlight_field_strengths(_HEAD + STRENGTH_7T) == [7.0]


def test_two_bounds_are_one_range():
    # "> 6.5 AND < 7.5" read as "> 6.5" and marked 9.4T and 14T as asked.
    chip = describe(_HEAD + STRENGTH_7T)[0]
    assert chip.field == "Magnetic field strength" and chip.label == "6.5–7.5"
    one_sided = STRENGTH_7T.replace(f" AND {_CAST} < 7.5", "")
    assert describe(_HEAD + one_sided)[0].label == "> 6.5"


def test_field_strength_inside_a_modality_chip_is_still_marked():
    # "7T fMRI" as one condition is labelled by the modality; the strength
    # travels as the phrase "7T", which page and excerpt read as a strength.
    chip = describe(_HEAD + STRENGTH_7T_FMRI)[0]
    assert chip.kind == "modality" and "7T" in chip.text_words
    assert highlights.highlight_field_strengths(_HEAD + STRENGTH_7T_FMRI) == [7.0]


@pytest.mark.parametrize("value, soft", [
    ("o2.other_entities ->> 'Manufacturer' ILIKE '%GE%'", True),
    ("o2.other_entities ->> 'Manufacturer' ILIKE '%GE%' AND NOT o2.other_entities ->> 'Manufacturer' IS NULL", True),
    ("o2.other_entities ->> 'Manufacturer' ILIKE '%GE%' AND NOT CASE WHEN (o2.other_entities ->> "
     "'Manufacturer') ~ 'a' THEN 1 END IS NULL", True),
    ("o2.other_entities ->> 'Manufacturer' NOT ILIKE '%GE%'", False),
    ("NOT o2.other_entities ->> 'Manufacturer' ILIKE '%GE%'", False),
    ("NOT o2.other_entities ->> 'Manufacturer' IN ('GE')", False),
    ("o2.other_entities ->> 'Manufacturer' <> 'GE'", False),
])
def test_a_denied_scanner_stays_a_hard_filter(value, soft):
    from services import sql_tree
    from services.uncertain.classify import classify

    tree = sql_tree.parse_select(
        "SELECT d.id FROM bids_datasets AS d WHERE EXISTS(SELECT 1 FROM bids_objects AS o2 "
        f"WHERE o2.dataset_id = d.id AND {value})")
    assert (classify(Clause.read(tree.args["where"].this), "d") is not None) is soft
