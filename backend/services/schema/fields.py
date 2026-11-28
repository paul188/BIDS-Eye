"""
services/schema/fields.py
-------------------------
The catalogue's searchable fields, each with the column that stores it.
Plain strings and no imports from the database side, so backend/vocabulary/ and
backend/llm/ can use them too; tests/services/schema/test_schema_tables.py
checks the tables and columns named here against services/schema/tables.py.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Optional

DATASETS = "bids_datasets"
OBJECTS = "bids_objects"
PARTICIPANTS = "bids_participants"

# The outer query's alias for bids_datasets.
DATASETS_ALIAS = "d"


class Stored(NamedTuple):
    """Where a field's values are stored, for writing SQL about it."""
    table: str
    alias: str    # 'd' on bids_datasets; else the alias of the correlated EXISTS
    column: str

    @property
    def ref(self) -> str:
        """The column as the SQL names it, e.g. 'o2.task' or 'd.institutions'."""
        return f"{self.alias}.{self.column}"

    @property
    def needs_exists(self) -> bool:
        """True for rows of their own (files, participants): one EXISTS per dataset."""
        return self.alias != DATASETS_ALIAS


class CatalogueField(str, Enum):
    """A searchable field; the value is its name, .stored its column."""
    # Files: one row per file, alias o2.
    DATATYPE    = "datatype",    Stored(OBJECTS,      "o2", "datatype")
    SUFFIX      = "suffix",      Stored(OBJECTS,      "o2", "suffix")
    TASK        = "task",        Stored(OBJECTS,      "o2", "task")
    # Participants: one row per participant, alias p2.
    DIAGNOSIS   = "diagnosis",   Stored(PARTICIPANTS, "p2", "diagnosis")
    SEX         = "sex",         Stored(PARTICIPANTS, "p2", "sex")
    HANDEDNESS  = "handedness",  Stored(PARTICIPANTS, "p2", "handedness")
    AGE         = "age",         Stored(PARTICIPANTS, "p2", "age")
    # Datasets: TEXT[] columns of bids_datasets itself, no EXISTS.
    AUTHOR      = "author",      Stored(DATASETS,     DATASETS_ALIAS, "authors")
    FUNDING     = "funding",     Stored(DATASETS,     DATASETS_ALIAS, "funding")
    INSTITUTION = "institution", Stored(DATASETS,     DATASETS_ALIAS, "institutions")

    stored: Stored

    # *stored* defaults to None only so that a lookup, CatalogueField("task"),
    # type-checks; Enum never calls __new__ for a lookup. A member defined
    # without a Stored fails here, when the module is imported.
    def __new__(cls, value: str, stored: Optional[Stored] = None) -> "CatalogueField":
        if stored is None:
            raise TypeError(f"CatalogueField {value!r} needs a Stored column")
        member = str.__new__(cls, value)
        member._value_ = value
        member.stored = stored
        return member

    def __str__(self) -> str:
        # The value, on every Python version (format() of a str enum changed in 3.11).
        return self.value

    @classmethod
    def named(cls, name: str) -> Optional["CatalogueField"]:
        """The field called *name*, or None."""
        try:
            return cls(name)
        except ValueError:
            return None


# The fields whose values are vocabulary codes (backend/vocabulary/value_mappings.yaml).
# Fields of the files: columns of bids_objects and bids_dataset_facets.
FILE_FIELDS = (CatalogueField.DATATYPE.value, CatalogueField.SUFFIX.value, CatalogueField.TASK.value)
# Fields of the participants: columns of bids_participants.
PARTICIPANT_FIELDS = (CatalogueField.DIAGNOSIS.value, CatalogueField.SEX.value,
                      CatalogueField.HANDEDNESS.value)
CODED_FIELDS = FILE_FIELDS + PARTICIPANT_FIELDS
# Fields found from the words of a question, and scored; sex and handedness are not.
CONCEPT_FIELDS = (CatalogueField.DATATYPE.value, CatalogueField.TASK.value,
                  CatalogueField.DIAGNOSIS.value, CatalogueField.SUFFIX.value)
