"""Evidence: the ways a dataset can meet one condition of a search.

A condition ("ages under 3", "epilepsy", "7 T") can be met in several ways:
the participant list records it, a column of the dataset holds it, the files
show it, the title, the README or a linked paper names it, or nothing is
recorded and a default is assumed. Each way is one Evidence: where it was
found (its source), what it proves (its strength), the SQL that is true for a
dataset it holds for, and the text a result card shows.

  base.py          Evidence, the sources and the strengths
  confirmed.py     evidence with numbers or values of its own: participant
                   rows, a dataset column, an assumed default
  weaker.py        evidence for a dataset the condition does not hold for:
                   only a paper, a weaker form of the search, the value named
                   in the text, the field not recorded, other values recorded
  reason_texts.py  every text a result card gives, in one list

Used by results/match_reasons.py (why a confirmed result matched: one
Evidence per OR branch of a condition) and by services/uncertain/ (which
state a condition has for a dataset: an ordered list of Evidence, the first
that holds decides).

Nothing is imported here: the translation reads base.py, weaker.py and
reason_texts.py (stores/interpretations._VERSION_FILES), not confirmed.py.
"""
