"""Filter chips read from generated SQL, and removing them (services/results/query_filters.py)."""

from __future__ import annotations

import asyncio
from uuid import UUID

import sqlglot

from services.results import query_filters as qf

_HEAD = (
    "SELECT d.id, d.name, COUNT(DISTINCT o.subject) AS subject_count "
    "FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
    "AND NOT o.subject IS NULL "
)

# Shapes taken from the recall benchmark (benchmarks/recall/results).
DIAGNOSIS = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.diagnosis IN ('parkinsons_disease')) OR d.search_tsv @@ "
    "to_tsquery('english', 'parkinson:AB | parkinsons:AB | (paralysis:AB & agitans:AB)'))"
)
MODALITY = (
    "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
    "AND (o2.datatype = 'functional_mri' OR o2.suffix = 'fmri_bold'))"
)
NO_EEG = (
    "NOT EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
    "AND (o2.datatype = 'electroencephalography' OR o2.suffix = 'eeg'))"
)
TASK = (
    "(EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id AND o2.task IN "
    "('auditory_oddball', 'deviant_detection_task', 'oddball_task')) OR d.search_tsv @@ "
    "to_tsquery('english', 'auditory:AB & oddball:AB'))"
)
AGE = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.age BETWEEN 10 AND 19) OR (d.age_min <= 19 AND d.age_max >= 10))"
)
ADULTS = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.age >= 18) OR d.age_max >= 18)"
)
TEXT = "d.search_tsv @@ to_tsquery('english', 'sleep & deprivation')"


def _sql(*conditions: str, having: str = "") -> str:
    sql = _HEAD + "WHERE " + " AND ".join(conditions) + " GROUP BY d.id"
    return sql + (f" HAVING {having}" if having else "")


def _by_id(sql: str) -> dict:
    return {chip.id: chip for chip in qf.describe(sql)}


def test_each_top_level_condition_becomes_one_chip():
    chips = _by_id(_sql(DIAGNOSIS, MODALITY, TASK, AGE, TEXT,
                        having="COUNT(DISTINCT o.subject) >= 40"))

    assert [c.kind for c in chips.values()] == [
        "diagnosis", "modality", "task", "age", "text", "subjects"]

    assert chips["w0"].field == "Diagnosis"
    assert chips["w0"].label == "Parkinson's disease"
    assert chips["w0"].text_words == ["parkinson", "parkinsons", "paralysis agitans"]
    assert "Or the dataset text mentions" in chips["w0"].detail

    assert chips["w1"].label == "fMRI"
    # Task codes expand to many; the words the user typed read better.
    assert chips["w2"].label == "auditory oddball"
    assert "Any of 3 coded tasks" in chips["w2"].detail
    assert chips["w3"].label == "10–19"
    assert chips["w4"].label == "“sleep deprivation”"
    assert chips["h0"].field == "Subjects"
    assert chips["h0"].label == "≥ 40"


def test_count_of_participants_in_an_age_range_keeps_its_bound():
    # Live SQL for "identify datasets containing at least 20 adolescents"
    # (2026-11-14): the chip read "Age · 10–19" and dropped the "≥ 20".
    sql = (_HEAD + "GROUP BY d.id HAVING (SELECT COUNT(DISTINCT p.participant_id) "
           "FROM bids_participants AS p WHERE p.dataset_id = d.id "
           "AND p.age >= 10 AND p.age <= 19) >= 20")
    chips = _by_id(sql)
    assert list(chips) == ["h0"]
    assert chips["h0"].kind == "subjects"
    assert chips["h0"].field == "Subjects"
    assert chips["h0"].label == "≥ 20 aged 10–19"


def test_count_from_both_sources_keeps_its_bound():
    sql = (_HEAD + "GROUP BY d.id HAVING GREATEST(COUNT(DISTINCT o.subject), "
           "(SELECT COUNT(DISTINCT p.participant_id) FROM bids_participants AS p "
           "WHERE p.dataset_id = d.id)) >= 100")
    chips = _by_id(sql)
    assert chips["h0"].field == "Subjects"
    assert chips["h0"].label == "≥ 100"


def test_negated_condition_and_one_sided_age():
    chips = _by_id(_sql(MODALITY, NO_EEG, ADULTS))
    assert chips["w1"].negated is True
    assert chips["w1"].label == "EEG"
    assert chips["w2"].label == "18+"


def test_query_without_conditions_has_no_chips():
    assert qf.describe(_HEAD + "GROUP BY d.id ORDER BY d.name LIMIT 200") == []
    assert qf.describe("not sql at all") == []
    assert qf.describe(None) == []


def test_remove_drops_only_the_named_conditions():
    sql = _sql(DIAGNOSIS, MODALITY, TEXT, having="COUNT(DISTINCT o.subject) >= 40")

    refined = qf.remove(sql, ["w0", "h0"])
    chips = qf.describe(refined)
    assert [c.kind for c in chips] == ["modality", "text"]
    assert "HAVING" not in refined.upper()
    assert "parkinsons_disease" not in refined
    sqlglot.parse_one(refined, read="postgres")


def test_remove_keeps_or_conditions_grouped():
    refined = qf.remove(_sql(MODALITY, DIAGNOSIS, TEXT), ["w0"])
    # The diagnosis OR text-match stays one condition, ANDed with the rest.
    assert [c.kind for c in qf.describe(refined)] == ["diagnosis", "text"]


def test_removing_everything_returns_every_dataset():
    refined = qf.remove(_sql(MODALITY, TEXT), ["w0", "w1"])
    assert "WHERE" not in refined.upper()


def test_remove_with_nothing_or_unknown_ids_keeps_the_query():
    sql = _sql(MODALITY, TEXT)
    assert qf.remove(sql, []) == sql
    assert [c.kind for c in qf.describe(qf.remove(sql, ["w9"]))] == ["modality", "text"]


def test_tier_still_applies_after_a_chip_is_removed(monkeypatch):
    from services.fulltext import uncertain_tier

    refined = qf.remove(_sql(MODALITY, ADULTS, TEXT), ["w2"])
    tiered = uncertain_tier.apply(refined, papers=False)
    assert uncertain_tier.LEVEL_COLUMN in tiered
    sqlglot.parse_one(tiered, read="postgres")


# ── Chips are display only ────────────────────────────────────────────────────

def test_filters_cannot_be_changed_through_the_api():
    """The page shows the chips without remove buttons; the API has no
    endpoint that re-runs a search without some of them either."""
    import routers.query as query_router

    paths = {route.path for route in query_router.router.routes}
    assert "/query" in paths and "/query/page" in paths
    assert "/query/refine" not in paths
    assert not hasattr(query_router, "query_refine")


# ── README excerpts ───────────────────────────────────────────────────────────

def test_highlight_query_takes_words_of_text_and_coded_chips():
    terms = qf.highlight_terms(_sql(DIAGNOSIS, MODALITY, NO_EEG, ADULTS, TEXT))
    assert {("parkinson",), ("parkinsons",), ("paralysis", "agitans"), ("parkinson", "disease"),
            ("fMRI",), ("sleep", "deprivation")} <= set(terms)
    # An age chip marks the words of its age groups (what the uncertain tier
    # accepts from a README), never its numbers.
    assert {("adults",), ("elderly",), ("older", "adults")} <= set(terms)
    # Negated chips have nothing to mark.
    assert ("EEG",) not in terms
    words = qf.highlight_tsquery(_sql(DIAGNOSIS, MODALITY, NO_EEG, ADULTS, TEXT))
    assert "eeg" not in words and "18" not in words
    assert qf.highlight_tsquery(_HEAD + "GROUP BY d.id") is None
    assert qf.highlight_tsquery(None) is None


_INFANT_WORDS = {("infant",), ("infants",), ("newborn",), ("neonate",), ("neonatal",),
                 ("baby",), ("babies",), ("toddler",), ("toddlers",)}


def _expanded(*conditions: str) -> str:
    from services.fulltext import text_search
    return text_search.expand_text_search(_sql(*conditions), papers=False).sql


def test_age_group_chip_marks_its_words():
    # "infant datasets" matches recorded ages under 3; a README excerpt still
    # marks the infant words where it has them, and the chip carries them for
    # the page's own marking of descriptions (frontend/src/highlight.ts).
    from criteria import age_condition
    sql = _expanded(age_condition(["infants"]))
    age = next(chip for chip in qf.describe(sql) if chip.kind == "age")
    assert {"infant", "newborn", "baby"} <= set(age.text_words)
    assert _INFANT_WORDS <= set(qf.highlight_terms(sql))
    children = set(qf.highlight_terms(_expanded(age_condition(["children"]))))
    assert {("children",), ("child",), ("adolescents",), ("pediatric",)} <= children
    older = set(qf.highlight_terms(_expanded(age_condition(["older_adults"]))))
    assert {("elderly",), ("older", "adults")} <= older and ("infant",) not in older


def test_numeric_age_chip_marks_the_words_of_its_age_group():
    # "under 2" has no words in its SQL; the README evidence of the uncertain
    # tier for that range is the infant words.
    under_two = ("EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
                 "AND p2.age < 2)")
    chip = _by_id(_sql(under_two))["w0"]
    assert chip.kind == "age" and chip.label == "under 2"
    assert _INFANT_WORDS <= set(qf.highlight_terms(_sql(under_two)))
    # A range no age group covers a good part of marks nothing.
    narrow = ("EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
              "AND p2.age BETWEEN 30 AND 31)")
    assert ("adults",) in qf.highlight_terms(_sql(narrow))   # inside the adult group
    assert qf.highlight_terms(_sql("EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE "
                                   "p2.dataset_id = d.id AND p2.age > 0)")) == []


def test_every_chip_kind_marks_its_words_but_a_negated_one_does_not():
    from criteria import species_condition
    species = _expanded(species_condition(["mouse"]))
    assert {("mouse",), ("mice",)} <= set(qf.highlight_terms(species))
    negated = f"NOT ({TEXT})"
    assert qf.highlight_terms(_sql(negated)) == []
    negated_age = f"NOT ({ADULTS})"
    assert qf.highlight_terms(_sql(negated_age)) == []


def test_highlight_terms_pair_the_words_of_a_phrase():
    # "structural" alone stems like "structure" and would mark every
    # "Brain Imaging Data Structure"; it only counts next to "MRI".
    mri = ("EXISTS (SELECT 1 FROM bids_files f WHERE f.dataset_id = d.id "
           "AND f.datatype IN ('anatomical_mri', 'functional_mri', 'diffusion_mri'))")
    terms = qf.highlight_terms(_sql(mri))
    assert ("structural", "mri") in terms and ("MRI",) in terms and ("fMRI",) in terms
    assert ("structural",) not in terms and ("or",) not in terms
    # Text phrases: pairs without stopwords, single words alone.
    text = "d.search_tsv @@ to_tsquery('english', '(scanning & of & one & individual) | rest')"
    terms = qf.highlight_terms(_sql(text))
    assert terms == [("scanning", "one"), ("one", "individual"), ("rest",)]
    assert qf.terms_tsquery(terms[:1]) == "(scanning <-> one | scanning <2> one)"


FIELD_7T = ("EXISTS (SELECT 1 FROM bids_objects bo WHERE bo.dataset_id = d.id AND "
            "CAST((bo.other_entities ->> 'MagneticFieldStrength') AS DOUBLE PRECISION) "
            "BETWEEN 6.5 AND 7.5)")


def test_field_strengths_come_from_the_chip_and_from_searched_phrases():
    assert qf.highlight_field_strengths(_sql(FIELD_7T)) == [7.0]
    at_least_3 = FIELD_7T.replace("BETWEEN 6.5 AND 7.5", ">= 3")
    assert qf.highlight_field_strengths(_sql(at_least_3)) == [3.0, 4.0, 4.7, 7.0, 9.4, 10.5, 11.7, 14.0]
    assert qf.highlight_field_strengths(_sql(f"NOT {FIELD_7T}")) == []
    # A typed "7tesla" that stayed a text search names a strength too.
    text = "d.search_tsv @@ to_tsquery('english', '7tesla | (7 & tesla)')"
    assert qf.highlight_field_strengths(_sql(text)) == [7.0]
    assert qf.highlight_field_strengths(_sql(TEXT)) == []


def test_field_strength_pattern_finds_the_ways_readmes_write_it():
    pattern = qf.field_strength_pattern([7.0, 1.5])
    text = ("at 7T, a 7 Tesla Siemens, 7-Tesla, 7.0T, 7 T scanner, 7TESLA, 1.5T, 1,5 T, "
            "1.50 Tesla. Not 17T, 0.7T, 7 T1w images, 7 tests, 1.5 mm, 3T, ses-7T, 7 T/m")
    assert pattern.findall(text) == ["7T", "7 Tesla", "7-Tesla", "7.0T", "7 T", "7TESLA",
                                     "1.5T", "1,5 T", "1.50 Tesla"]
    assert pattern.findall("a 7\u2009T and a 7 - T scanner") == ["7\u2009T", "7 - T"]
    assert qf.field_strength_pattern([]) is None
    # Postgres keeps "7T" whole; "1.5T" is the word "1.5" there ("T" is a stopword).
    assert qf.field_strength_tsquery([7.0, 1.5]) == "7t | 7tesla | 7 <-> tesla | 1.5"


def test_phrase_digits_pair_but_never_stand_alone():
    text = "d.search_tsv @@ to_tsquery('english', '(7 & tesla) | 7T | (2 & back) | 7')"
    assert qf.highlight_terms(_sql(text)) == [("7", "tesla"), ("7T",), ("2", "back")]


def test_attach_snippets_marks_field_strengths():
    from services.results import readme_snippets
    from services.catalogue import schema_capabilities as caps

    # ts_headline marked the Postgres words; the pattern decides: "1.5 mm" is no strength.
    session = _SnippetSession([
        (UUID(int=1), "scanned on a ⟦1.5⟧T Siemens with ⟦1.5⟧ mm voxels; ⟦BOLD⟧ runs",
         ["1.5", "1.5", "bold"], ["bold"]),
    ])
    datasets = [_dataset(1)]
    caps.set_display_columns({"readme_text"})
    try:
        asyncio.run(readme_snippets.attach_snippets(session, datasets, [("BOLD",)], (), [1.5]))
    finally:
        caps.set_display_columns(())
    assert session.calls[0][1]["q"] == "bold | 1.5"
    assert datasets[0].snippet == "… scanned on a ⟦1.5T⟧ Siemens with 1.5 mm voxels; ⟦BOLD⟧ runs …"


def test_keep_justified_marks_drops_pair_words_alone():
    from services.results.readme_snippets import keep_justified_marks
    terms = [("structural", "mri"), ("MRI",), ("MS",)]
    lexemes = {"structural": {"structur"}, "mri": {"mri"}, "ms": {"ms"}}
    headline = ("Brain Imaging Data ⟦Structure⟧ format; ⟦structural⟧ ⟦MRI⟧ "
                "of ⟦MS⟧ patients, 500 ⟦ms⟧ trials, ⟦MRI⟧ again")
    marks = ["structur", "structur", "mri", "ms", "ms", "mri"]
    assert keep_justified_marks(headline, marks, terms, lexemes) == (
        "Brain Imaging Data Structure format; ⟦structural⟧ ⟦MRI⟧ "
        "of ⟦MS⟧ patients, 500 ms trials, ⟦MRI⟧ again")
    # One word between still counts: "Parkinson's disease".
    assert keep_justified_marks("⟦Parkinson⟧'s ⟦disease⟧", ["parkinson", "diseas"],
                                [("parkinson", "disease")],
                                {"parkinson": {"parkinson"}, "disease": {"diseas"}}) \
        == "⟦Parkinson⟧'s ⟦disease⟧"


def test_clean_snippet_strips_markdown():
    from services.results import readme_snippets

    raw = ("# Dataset ⟦Parkinson⟧ study\n![logo](img.png) See [the paper](https://doi.org/x) "
           "and https://openneuro.org.\n| col | col |\n**Bold** text with sub_01 and __init__")
    assert readme_snippets.clean_snippet(raw) == (
        "Dataset ⟦Parkinson⟧ study See the paper and col col Bold text with sub_01 and init")


class _SnippetSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def begin_nested(self):
        class _Nested:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False
        return _Nested()

    async def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        return self.rows


def _dataset(n: int):
    from schemas import DatasetSchema
    return DatasetSchema(id=UUID(int=n), name=f"ds{n}", dataset_type="raw",
                         source_type="openneuro", participants=[])


def test_attach_snippets_marks_matches_and_falls_back_to_the_start():
    from services.results import readme_snippets
    from services.catalogue import schema_capabilities as caps

    long_start = "Plain README start. " * 30
    session = _SnippetSession([
        (UUID(int=1), "patients with ⟦Parkinson's⟧ ⟦disease⟧ … scanned with ⟦fMRI⟧ and ⟦disease⟧",
         ["parkinson", "diseas", "fmri", "diseas"], ["parkinson", "diseas", "fmri"]),
        (UUID(int=2), long_start, [], ["parkinson", "diseas", "fmri"]),
    ])
    datasets = [_dataset(1), _dataset(2), _dataset(3)]
    caps.set_display_columns({"readme_text"})
    try:
        asyncio.run(readme_snippets.attach_snippets(
            session, datasets, [("parkinson", "disease"), ("fMRI",)]))
    finally:
        caps.set_display_columns(())

    sql, params = session.calls[0]
    assert "ts_headline" in sql
    assert params["q"] == "(parkinson <-> disease | parkinson <2> disease) | fmri"
    assert params["words"] == ["parkinson", "disease", "fmri"]
    # A passage from mid-sentence is marked as an excerpt at both ends; the
    # pair's second word away from its partner is not marked.
    assert datasets[0].snippet == (
        "… patients with ⟦Parkinson's⟧ ⟦disease⟧ … scanned with ⟦fMRI⟧ and disease …")
    # No match in the README: ts_headline returns its start, cut at a word.
    assert datasets[1].snippet.endswith(" …") and len(datasets[1].snippet) <= 330
    assert "⟦" not in datasets[1].snippet
    assert datasets[2].snippet is None


def test_attach_snippets_without_the_column_does_nothing():
    from services.results import readme_snippets

    session = _SnippetSession([])
    datasets = [_dataset(1)]
    asyncio.run(readme_snippets.attach_snippets(session, datasets, [("x",)]))
    assert session.calls == [] and datasets[0].snippet is None


def test_source_chip_reads_openneuro():
    chips = _by_id(_sql("d.source_type = 'openneuro'"))
    assert (chips["w0"].field, chips["w0"].label) == ("Source", "OpenNeuro")


def test_order_and_limit_become_one_removable_chip():
    largest = ("GROUP BY d.id ORDER BY GREATEST(COUNT(DISTINCT o.subject), (SELECT COUNT(DISTINCT "
               "p.participant_id) FROM bids_participants AS p WHERE p.dataset_id = d.id)) DESC LIMIT 1")
    sql = _HEAD + "WHERE d.source_type = 'openneuro' " + largest
    chips = _by_id(sql)
    assert list(chips) == ["w0", "o0"]
    assert chips["o0"].kind == "sort"
    assert (chips["o0"].field, chips["o0"].label) == ("Sorted", "most participants first · top 1")

    by_size = _by_id(_HEAD + "GROUP BY d.id ORDER BY d.size_bytes DESC NULLS LAST LIMIT 3")
    assert by_size["o0"].label == "largest size first · top 3"
    assert _by_id(_HEAD + "GROUP BY d.id LIMIT 5")["o0"].label == "first 5"
    assert "o0" not in _by_id(_sql(TEXT))

    widened = qf.remove(sql, ["o0"])
    assert "ORDER BY" not in widened and "LIMIT" not in widened
    assert "source_type" in widened
    assert qf.remove(sql, ["w0"]).rstrip().endswith("LIMIT 1")


def test_count_of_patients_with_a_diagnosis_says_patients():
    # "datasets with at least 25 frontotemporal dementia patients" (2026-11-17):
    # the chip read "Subjects: ≥ 25", but the count is of the 23 FTD patients,
    # not of the 88 participants.
    import cohort_count

    count = cohort_count.confirmed_sql(["frontotemporal_dementia"], 25)
    chips = _by_id(_HEAD + f"WHERE {count} GROUP BY d.id")
    chip = chips["w0"]
    assert (chip.kind, chip.field, chip.label) == ("count", "Patients", "≥ 25 with Frontotemporal dementia")
    count = cohort_count.confirmed_sql(["parkinsons_disease", "de_novo_parkinsons"], 40)
    assert _by_id(_HEAD + f"WHERE {count} GROUP BY d.id")["w0"].label == "≥ 40 with Parkinson's disease +1"


def test_topic_chip_text_words_stay_a_list():
    # "datasets of twin studies" (2026-11-18) answered 500: the topic branch
    # reused the name `searched` for its detail string, so text_words became a
    # string and the response schema rejected it after the search had run.
    from schemas import QueryFilterSchema

    near = "(monozygotic <1> twins | twins <1> monozygotic) | (genetic <1> studies | studies <1> genetic)"
    condition = (f"(d.search_tsv @@ TO_TSQUERY('english', 'twin & studies') "
                 f"OR d.search_tsv @@ TO_TSQUERY('english', '{near}'))")
    chip = _by_id(_HEAD + f"WHERE {condition} GROUP BY d.id")["w0"]
    assert chip.kind == "topic"
    assert isinstance(chip.text_words, list) and "twin studies" in chip.text_words
    assert "Searched as: twin studies" in chip.detail
    QueryFilterSchema(**chip.to_dict())


def test_a_shortened_label_keeps_the_full_list():
    # "Diagnosis · Epilepsy +12" (2026-11-18): the 12 were only in a tooltip.
    codes = ", ".join(f"'{c}'" for c in ("epilepsy", "focal_epilepsy", "absence_epilepsy"))
    condition = (f"EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
                 f"AND p2.diagnosis IN ({codes}))")
    chip = _by_id(_HEAD + f"WHERE {condition} GROUP BY d.id")["w0"]
    assert chip.label == "Epilepsy +2"
    assert chip.values == ["Epilepsy", "Focal epilepsy", "Absence epilepsy"]
    assert _by_id(_sql(DIAGNOSIS))["w0"].values == []  # one diagnosis: nothing hidden
