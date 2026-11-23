"""
services/uncertain/fields/diagnosis.py
------------------------------------------
Diagnosis as a field a dataset may not record. Most datasets code none
(tier.py).

It has no text form of its own: a diagnosis condition already holds a text
search for the diagnosis, and the weaker forms of that search (text.py) are
its text evidence.
"""

from __future__ import annotations

from services.uncertain.fields.base import SoftField, participant_rows_record


FIELD = SoftField(
    kind="diagnosis",
    columns=("diagnosis", "dx_status"),
    recorded=participant_rows_record("diagnosis"),
    # Coded diagnoses are often incomplete: a dataset that codes some other
    # diagnosis may still study the asked one.
    text_beats_recorded=True,
)
