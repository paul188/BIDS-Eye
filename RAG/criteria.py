"""Search criteria: how each kind of filter a user asks for is decided.

One entry per criterion. Each records how people phrase it, which evidence
answers it (strongest first), what does not count as evidence, its
vocabulary, and what happens when a dataset does not record the value. The
planner (LLM_preprocessor/preprocess/) fills a QueryPlan field per
criterion; the condition is built here, so the SQL model copies one fixed
line instead of choosing a filter itself.

Species is the first entry. Before it, "animal studies done at Harvard"
became the text search 'animal & studies' in some runs and a d.species filter
in others: the README/paper tier then returned two human studies whose papers
mention "animal studies".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from text_terms import MODE_ALL, MODE_SHORT, format_text_match


@dataclass(frozen=True)
class Criterion:
    name: str
    phrasings: Tuple[str, ...]      # how users ask for it
    evidence: Tuple[str, ...]       # what answers it, strongest first
    not_evidence: Tuple[str, ...]   # what must not count
    when_missing: str               # behaviour when the dataset does not record it
    default: Optional[str] = None   # the value assumed when nothing says otherwise


# ── Species ──────────────────────────────────────────────────────────────────

SPECIES = Criterion(
    name="species",
    phrasings=("mouse datasets", "rat fMRI", "animal studies", "non-human primates",
               "monkeys", "rodents", "data from rats", "3xTg Alzheimer mice"),
    evidence=(
        "d.species: the species the uploader picked on OpenNeuro",
        "the dataset name and description, when the label says nothing",
        "the README, for primate words only, when the label says nothing",
    ),
    not_evidence=(
        "README or paper text for mouse/rat/animal words: human studies say 'mouse click', "
        "'compared with animal studies'",
        "a label naming another species: a 'Human' dataset is not a mouse dataset because its "
        "README mentions mice",
    ),
    when_missing="NULL, empty, 'Other', 'n/a' and 'unknown' say nothing: the dataset is "
                 "assumed human unless its name or description names an animal species; "
                 "for an animal question that text decides",
    default="human",
)

# A label that says nothing about the species.
_UNKNOWN_LABEL = r"^\s*(other|n/?a|unknown)?\s*$"
SPECIES_UNKNOWN = f"(d.species IS NULL OR d.species ~* '{_UNKNOWN_LABEL}')"

# Anchored: "Non-Human Primates" and "Phantom Human" contain "human".
_HUMAN_LABEL = r"^\s*(humans?|homo sapiens)\s*$"
HUMAN = "human"


@dataclass(frozen=True)
class SpeciesGroup:
    code: str
    label: str                  # Postgres regex over d.species (case-insensitive)
    words: Tuple[str, ...]      # text fallback for unlabelled datasets
    mode: str = MODE_SHORT      # where the fallback looks
    ncbi_taxon: Optional[int] = None   # for the harmonised code stored later


# Labels seen on OpenNeuro (2026-11-14): Human, Mouse, mouse, 'Mice B6/129P', Rat,
# rat, Macaque, Monkey, Non-Human Primates, 'Nonhunam primates (Macaca mulatta)',
# baboon, 'Sus scrofa domesticus', Sheep, canine, Bat, Pigeon, Other, phantoms.
SPECIES_GROUPS: Dict[str, SpeciesGroup] = {g.code: g for g in (
    SpeciesGroup("mouse", r"\m(mouse|mice|mus)\M", ("mouse", "mice", "murine"), ncbi_taxon=10090),
    SpeciesGroup("rat", r"\m(rats?|rattus)\M", ("rat", "rats", "rattus"), ncbi_taxon=10116),
    # Primate words are rare in human READMEs, so the fallback reads the whole text.
    SpeciesGroup("primate", r"(macaque|macaca|monkey|primate|baboon|marmoset|rhesus|capuchin|cynomolgus)",
                 ("macaque", "macaca", "monkey", "monkeys", "rhesus", "marmoset", "baboon"),
                 mode=MODE_ALL, ncbi_taxon=9443),
    SpeciesGroup("pig", r"(\mpigs?\M|piglet|swine|sus scrofa|porcine)", ("pig", "piglet", "swine", "porcine"),
                 ncbi_taxon=9823),
    SpeciesGroup("sheep", r"(sheep|\movis\M|ovine)", ("sheep", "ovine"), ncbi_taxon=9940),
    SpeciesGroup("dog", r"(\mdogs?\M|canine|canis)", ("dog", "dogs", "canine"), ncbi_taxon=9615),
    SpeciesGroup("cat", r"(\mcats?\M|feline|felis)", ("cat", "cats", "feline"), ncbi_taxon=9685),
    SpeciesGroup("ferret", r"(ferret|mustela)", ("ferret", "ferrets"), ncbi_taxon=9669),
    SpeciesGroup("bat", r"\mbats?\M", ("bat", "bats"), ncbi_taxon=9397),
    SpeciesGroup("bird", r"(bird|pigeon|columba|finch|songbird|taeniopygia)",
                 ("bird", "birds", "pigeon", "songbird", "finch"), ncbi_taxon=8782),
    SpeciesGroup("zebrafish", r"(zebrafish|danio)", ("zebrafish",), ncbi_taxon=7955),
)}

# Any non-human animal: a label that is neither human nor a phantom, or an
# unlabelled dataset whose name or description names an animal.
_NOT_ANIMAL_LABEL = r"(human|homo sapiens|phantom|synthetic|printed)"
_ANY_ANIMAL_WORDS = ("animal", "animals", "mouse", "mice", "rat", "rats", "rodent",
                     "macaque", "monkey", "primate", "nonhuman")
ANY_ANIMAL = "any_animal"

SPECIES_CODES = (HUMAN, ANY_ANIMAL) + tuple(SPECIES_GROUPS)

# The default: a dataset without a species label is human, unless its name or
# description names an animal species. Of 670 unlabelled datasets on the live
# database (2026-11-18), these words named 13, all of them animal studies
# (Mouse_rest_*, Rat_rest_STZ, TheVirtualBrain Macaque, the pig ischemia
# model). Not "animal": it shares the stem 'anim' with "animated", and
# ds000228 (children watching an animated film) is human. Not cat, bat, bird
# or dog: "CAT snacks", "CAT Faces" are human studies.
_ANIMAL_NAME_WORDS = tuple(word for code in ("mouse", "rat", "primate", "pig")
                           for word in SPECIES_GROUPS[code].words)
SPECIES_ASSUMED_HUMAN = (f"({SPECIES_UNKNOWN} AND NOT "
                         f"{format_text_match(list(_ANIMAL_NAME_WORDS), MODE_SHORT)})")
# Human by label, or by default.
HUMAN_SPECIES = f"(d.species ~* '{_HUMAN_LABEL}' OR {SPECIES_ASSUMED_HUMAN})"

# What the planner may write instead of a code.
_SYNONYMS = {
    "mice": "mouse", "murine": "mouse", "rats": "rat", "rodent": None, "rodents": None,
    "monkey": "primate", "monkeys": "primate", "macaque": "primate", "macaques": "primate",
    "nonhuman_primate": "primate", "non-human primate": "primate", "nhp": "primate",
    "primates": "primate", "marmoset": "primate", "baboon": "primate", "rhesus": "primate",
    "pigs": "pig", "swine": "pig", "dogs": "dog", "canine": "dog", "cats": "cat",
    "bats": "bat", "birds": "bird", "pigeon": "bird", "songbird": "bird",
    "animal": ANY_ANIMAL, "animals": ANY_ANIMAL, "non-human": ANY_ANIMAL,
    "nonhuman": ANY_ANIMAL, "any animal": ANY_ANIMAL,
    "humans": HUMAN, "homo sapiens": HUMAN,
}


def normalise_species(values: Iterable[str]) -> List[str]:
    """The planner's species values as codes, unknown ones dropped, order kept."""
    out: List[str] = []
    for value in values or ():
        key = str(value).strip().lower().replace("_", " ") if value else ""
        key = key.replace(" ", "_") if key.replace(" ", "_") in SPECIES_CODES else key
        if key in ("rodent", "rodents"):
            codes = ["mouse", "rat"]
        elif key in SPECIES_CODES:
            codes = [key]
        elif _SYNONYMS.get(key):
            codes = [_SYNONYMS[key]]
        else:
            codes = []
        out.extend(c for c in codes if c not in out)
    if ANY_ANIMAL in out:
        # Covers every other animal, not humans.
        return [c for c in out if c in (HUMAN, ANY_ANIMAL)]
    return out


def _text(words: Sequence[str], mode: str) -> str:
    match = format_text_match(list(words), mode)
    assert match, words
    return match


def species_condition(codes: Sequence[str], has_column: bool = True) -> Optional[str]:
    """One SQL condition for the species asked for (OR between them).

    With the column: the uploader's label decides; a dataset whose label says
    nothing counts when its text names the species. Without it: text only.

    Human is the default (HUMAN_SPECIES): a dataset without a label counts
    unless its name or description names an animal. Without the column
    every dataset counts as human, so asking for humans is no condition.
    """
    codes = normalise_species(codes)
    if not codes or (HUMAN in codes and not has_column):
        return None
    parts: List[str] = []
    for code in codes:
        if code == HUMAN:
            parts.append(HUMAN_SPECIES)
            continue
        if code == ANY_ANIMAL:
            text = _text(_ANY_ANIMAL_WORDS, MODE_SHORT)
            if not has_column:
                parts.append(text)
                continue
            parts.append(f"(d.species !~* '{_NOT_ANIMAL_LABEL}' AND d.species !~* '{_UNKNOWN_LABEL}')"
                         f" OR ({SPECIES_UNKNOWN} AND {text})")
            continue
        group = SPECIES_GROUPS[code]
        text = _text(group.words, group.mode)
        if not has_column:
            parts.append(text)
            continue
        parts.append(f"d.species ~* '{group.label}' OR ({SPECIES_UNKNOWN} AND {text})")
    return "(" + " OR ".join(parts) + ")"


# ── Age group ────────────────────────────────────────────────────────────────

AGE_GROUP = Criterion(
    name="age_group",
    phrasings=("children", "kids", "pediatric", "youth", "adolescents", "teenagers",
               "infants", "toddlers", "preschoolers", "elderly", "older adults"),
    evidence=(
        "recorded ages: bids_participants.age, then d.age_min / d.age_max; a dataset counts "
        "when any of its ages fall in the group",
        "the short fields (name, description, study fields) naming the group, when no age is recorded",
        "the README naming the group, when no age is recorded: probably relevant only "
        "(services/fulltext/uncertain_tier.py)",
    ),
    not_evidence=(
        "a group word in a dataset whose recorded ages lie outside the group: adult datasets "
        "say 'child' in example phenotype files, 'Child Mind Institute', 'parent-child'",
        "'juvenile': used for animals ('juvenile rats')",
    ),
    when_missing="no recorded age: the group's words in the short fields decide; "
                 "in the README only, the dataset is probably relevant",
)

# Before this, "impulse control in children" searched the word "children"
# (2026-11-15): Postgres stems it to "children", not "child", so "youth",
# "adolescents" and "pediatric" were missed and recorded ages were never read.


@dataclass(frozen=True)
class AgeGroup:
    code: str
    low: float                  # inclusive
    high: Optional[float]       # exclusive; None = no upper bound
    words: Tuple[str, ...]      # short-field fallback when no age is recorded


_ADOLESCENT_WORDS = ("adolescent", "adolescents", "adolescence", "teen", "teens",
                     "teenager", "teenagers", "youth", "youths")

# "children" includes adolescents (Paul, 2026-11-17): a request for children
# wants every dataset with participants under 18.
AGE_GROUPS: Dict[str, AgeGroup] = {g.code: g for g in (
    AgeGroup("infants", 0, 3, ("infant", "infants", "newborn", "newborns", "neonate", "neonates",
                               "neonatal", "toddler", "toddlers", "baby", "babies")),
    AgeGroup("children", 0, 18, ("child", "children", "kids", "pediatric", "paediatric",
                                 "preschool", "preschoolers", "school-age", "schoolchildren",
                                 "preadolescent", "preadolescents", "minors") + _ADOLESCENT_WORDS),
    # Below 18: with 19, every adult study of 18-year-old students counted (401 datasets).
    AgeGroup("adolescents", 12, 18, _ADOLESCENT_WORDS),
    AgeGroup("older_adults", 60, None, ("elderly", "older-adults", "older-adult", "older-participants",
                                        "older-people", "geriatric", "seniors", "old-age")),
)}

AGE_GROUP_CODES = tuple(AGE_GROUPS)

# A group that contains another: asking for both means the wider one.
_AGE_COVERS = {"children": ("infants", "adolescents")}

_AGE_SYNONYMS = {
    "child": "children", "kids": "children", "kid": "children", "pediatric": "children",
    "paediatric": "children", "youth": "children", "youths": "children", "minors": "children",
    "young people": "children", "school age": "children", "school-age": "children",
    "schoolchildren": "children", "preschoolers": "children", "preschool": "children",
    "preadolescents": "children",
    "adolescent": "adolescents", "adolescence": "adolescents", "teens": "adolescents",
    "teen": "adolescents", "teenagers": "adolescents", "teenager": "adolescents",
    "infant": "infants", "babies": "infants", "baby": "infants", "newborns": "infants",
    "newborn": "infants", "neonates": "infants", "neonatal": "infants", "toddlers": "infants",
    "toddler": "infants",
    "elderly": "older_adults", "older adults": "older_adults", "older adult": "older_adults",
    "seniors": "older_adults", "geriatric": "older_adults", "older people": "older_adults",
    "aged": "older_adults", "old age": "older_adults",
}


def normalise_age_groups(values: Iterable[str]) -> List[str]:
    """The planner's age-group values as codes, unknown ones dropped, order kept."""
    out: List[str] = []
    for value in values or ():
        key = str(value).strip().lower().replace("_", " ") if value else ""
        code = key.replace(" ", "_") if key.replace(" ", "_") in AGE_GROUPS else _AGE_SYNONYMS.get(key)
        if code and code not in out:
            out.append(code)
    covered = {c for code in out for c in _AGE_COVERS.get(code, ())}
    return [code for code in out if code not in covered]


def _num(value: float) -> str:
    return f"{value:g}"


def _recorded_age_in(low: float, high: Optional[float]) -> str:
    """Recorded ages that overlap [low, high): a participant row, or the dataset's range.

    d.age_min / d.age_max complement the participant rows; some datasets have
    only one of the two bounds. An age of 0 is a placeholder (live database,
    2026-11-17: 33 of 35 participants of ds006559, every participant of the
    multiband test datasets), so it never counts as a recorded age.
    """
    bounds = [f"pa.age >= {_num(low)}" if low > 0 else "pa.age > 0"]
    if high is not None:
        bounds.append(f"pa.age < {_num(high)}")
    rows = ("EXISTS (SELECT 1 FROM bids_participants pa WHERE pa.dataset_id = d.id AND "
            + " AND ".join(bounds) + ")")
    if low <= 0:
        hi = _num(high)
        dataset = (f"((d.age_min > 0 AND d.age_min < {hi})"
                   f" OR (d.age_max > 0 AND d.age_max < {hi}))")
    elif high is None:
        dataset = f"(d.age_max >= {_num(low)} OR d.age_min >= {_num(low)})"
    else:
        lo, hi = _num(low), _num(high)
        dataset = (f"((d.age_min > 0 AND d.age_min < {hi} AND d.age_max >= {lo})"
                   f" OR (d.age_max IS NULL AND d.age_min >= {lo} AND d.age_min < {hi})"
                   f" OR ((d.age_min IS NULL OR d.age_min <= 0)"
                   f" AND d.age_max >= {lo} AND d.age_max < {hi}))")
    return f"{rows} OR {dataset}"


# No age above 0 anywhere (see _recorded_age_in).
AGE_UNKNOWN = ("(COALESCE(d.age_min, 0) <= 0 AND COALESCE(d.age_max, 0) <= 0 AND NOT EXISTS "
               "(SELECT 1 FROM bids_participants pa WHERE pa.dataset_id = d.id AND pa.age > 0))")

# Animal ages are in days or weeks: a non-human dataset is never a human age
# group (ds004819, pigs aged 0-14). The species default decides for datasets
# without a label (HUMAN_SPECIES).


def age_condition(codes: Sequence[str], has_species_column: bool = True) -> Optional[str]:
    """One SQL condition for the age groups asked for (OR between them).

    Recorded ages decide. A dataset without any recorded age counts when its
    short fields name the group; the uncertain tier adds those whose README
    names it as probably relevant. The species guard is a separate AND
    term, so the uncertain tier still sees the age part as an age condition.
    """
    codes = normalise_age_groups(codes)
    if not codes:
        return None
    parts = []
    for code in codes:
        group = AGE_GROUPS[code]
        text = _text(group.words, MODE_SHORT)
        parts.append(f"{_recorded_age_in(group.low, group.high)} OR ({AGE_UNKNOWN} AND {text})")
    ages = "(" + " OR ".join(parts) + ")"
    return f"({ages} AND {HUMAN_SPECIES})" if has_species_column else ages


def age_group_label(codes: Sequence[str]) -> str:
    """"children", "children or older adults": for the filter line."""
    return " or ".join(code.replace("_", " ") for code in normalise_age_groups(codes))
