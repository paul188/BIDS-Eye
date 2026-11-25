"""
services/schema/coded_fields.py
-------------------------------
The fields whose values are vocabulary codes (RAG/value_mappings.yaml), named
once. Plain strings and no imports from the database side, so RAG/ and
LLM_preprocessor/ can use them too.

Where each field is stored: services/schema/tables.py. tests/test_schema_tables.py
checks that these columns and the entries of RAG/join_paths.yaml agree.
"""

from __future__ import annotations

from enum import Enum


class CodedField(str, Enum):
    """A field with vocabulary codes; the value is its column name."""
    DATATYPE = "datatype"
    SUFFIX = "suffix"
    TASK = "task"
    DIAGNOSIS = "diagnosis"
    SEX = "sex"
    HANDEDNESS = "handedness"

    def __str__(self) -> str:
        # The value, on every Python version (format() of a str enum changed in 3.11).
        return self.value


# Fields of the files: columns of bids_objects and bids_dataset_facets.
FILE_FIELDS = (CodedField.DATATYPE.value, CodedField.SUFFIX.value, CodedField.TASK.value)
# Fields of the participants: columns of bids_participants.
PARTICIPANT_FIELDS = (CodedField.DIAGNOSIS.value, CodedField.SEX.value, CodedField.HANDEDNESS.value)
CODED_FIELDS = FILE_FIELDS + PARTICIPANT_FIELDS
# Fields found from the words of a question, and scored; sex and handedness are not.
CONCEPT_FIELDS = (CodedField.DATATYPE.value, CodedField.TASK.value,
                  CodedField.DIAGNOSIS.value, CodedField.SUFFIX.value)
