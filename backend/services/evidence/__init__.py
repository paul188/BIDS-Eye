"""Evidence: the ways a dataset can meet one condition of a search.

A condition ("ages under 3", "epilepsy", "7 T") can be met in several ways:
the participant list records it, a column of the dataset holds it, the files
show it, the title, the README or a linked paper names it, or nothing is
recorded and a default is assumed. Each way is one Evidence: where it was
found (its source), what it proves (its strength), the SQL that is true for a
dataset it holds for, and the text a result card shows.

  base.py          Evidence, where it is found (Source) and what it proves (Strength)

One file per strength; a kind holds what it is about and writes its SQL and
its card text from that:
  confirmed.py     one way the condition holds as generated (an OR branch of it)
  probable.py      only in a paper, a weaker form of the search, the value
                   named in the dataset's own text
  possible.py      the field is not recorded
  contradicted.py  the field is recorded with other values

Evidence is about one condition of a query. The query combines its
conditions: services/uncertain/conditions.QueryEvidence (the weakest
condition decides; it writes what a result card says).

Used by results/match_reasons.py (why a confirmed result matched: one
ConfirmedMatch per OR branch of a condition) and by services/uncertain/
(which state a condition has for a dataset it does not hold for: an ordered
list of Evidence, the first that holds decides).
"""
