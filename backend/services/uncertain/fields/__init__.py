"""The fields a dataset may not record, one file each.

  base.py        SoftField: what a field says about itself
  age.py         participant ages; the text names the age group
  diagnosis.py   diagnosis; no text form of its own
  sex.py         participant sex; the text names participants of that sex

A condition on one of them is uncertain for a dataset that does not record
the field (classify.soft_kind decides, classify.classify builds the states).

To add a field: write its file with a FIELD, list it in FIELDS below, and add
its file to stores/interpretations._VERSION_FILES. The words its text form looks for
belong in backend/vocabulary/value_mappings.yaml (backend/vocabulary/text_evidence.py reads them).

Scanner values are not here: they are recorded with the files, not as a
column, and need their own lookup (scanner.py).
"""

from __future__ import annotations

from typing import Dict, Tuple

from services.uncertain.fields import age, diagnosis, sex
from services.uncertain.fields.base import SoftField

FIELDS: Tuple[SoftField, ...] = (age.FIELD, diagnosis.FIELD, sex.FIELD)

BY_KIND: Dict[str, SoftField] = {field.kind: field for field in FIELDS}
# column → the kind of the field it belongs to.
KIND_OF_COLUMN: Dict[str, str] = {column: field.kind for field in FIELDS for column in field.columns}
