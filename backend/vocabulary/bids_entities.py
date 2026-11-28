"""BIDS filename entities a user may type as a scan term ("dir-AP", "acq-highres").

Such a term names a file property, not a word a README uses: "resting-state
fMRI with dir-AP and dir-PA scans" was searched as the text 'dir-ap' and
'dir-pa' and found 1 of 58 datasets (gold request qf23). The crawler
(BIDS-SQL/input_pipeline.py) stores each file's entities in
bids_objects.other_entities under pybids' long names; entity_condition()
turns the term into a condition on that.
"""
from __future__ import annotations

import re
from typing import Optional

# BIDS key in the filename -> pybids name in other_entities. Only names that
# occur in the catalogue (checked on the live database, 2026-11-27).
# sub, ses, task and run have columns of their own and are not listed.
ENTITY_NAMES = {
    "acq": "acquisition",
    "trc": "tracer",
    "stain": "staining",
    "rec": "reconstruction",
    "dir": "direction",
    "echo": "echo",
    "flip": "flip",
    "inv": "inv",
    "mt": "mt",
    "part": "part",
    "recording": "recording",
    "space": "space",
    "atlas": "atlas",
    "hemi": "hemisphere",
    "res": "resolution",
    "label": "label",
    "desc": "desc",
    "sample": "sample",
    "tracksys": "tracksys",
}

# Not ce (contrast agent): "CE-MRI" is the usual name of contrast-enhanced MRI.

# The labels an entity can have where the specification fixes them; a term
# like "echo-planar" or "part-time" is a word, not an entity.
_LABELS = {
    "echo": r"[0-9]+", "flip": r"[0-9]+", "inv": r"[0-9]+",
    "mt": r"on|off", "part": r"mag|phase|real|imag", "hemi": r"L|R",
}

# The whole term is one key-label pair, as in a filename: labels are
# alphanumeric (BIDS specification, "Entities").
_ENTITY_RE = re.compile(r"_?(?P<key>[a-z]+)-(?P<label>[A-Za-z0-9]+)", re.IGNORECASE)


def entity_condition(term: str, objects_table: str = "bids_objects") -> Optional[str]:
    """An EXISTS on the files that carry the entity *term* names, or None.

    The label is compared without case: users write "dir-ap" for files named
    dir-AP.
    """
    match = _ENTITY_RE.fullmatch(term.strip())
    if not match:
        return None
    key = match.group("key").lower()
    name = ENTITY_NAMES.get(key)
    if name is None:
        return None
    if key in _LABELS and not re.fullmatch(_LABELS[key], match.group("label"), re.IGNORECASE):
        return None
    label = match.group("label").lower()
    return (f"EXISTS (SELECT 1 FROM {objects_table} o2 WHERE o2.dataset_id = d.id "
            f"AND LOWER(o2.other_entities->>'{name}') = '{label}')")
