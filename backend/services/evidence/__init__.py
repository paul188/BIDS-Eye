"""Evidence: the ways a dataset can meet one condition of a search.

A condition ("ages under 3", "epilepsy", "7 T") can be met in several ways:
the participant list records it, a column of the dataset holds it, the files
show it, the title, the README or a linked paper names it, or nothing is
recorded and a default is assumed. Each way is one PieceOfEvidence: where it was
found (its source), what it proves (its strength), the SQL that is true for a
dataset it holds for, and the text a result card shows.

  base.py          PieceOfEvidence, where it is found (Source) and what it proves (Strength),
                   UncertainEvidence (card text per level; confirmed: per dataset)

One file per strength; a kind holds what it is about and writes its SQL and
its card text from that:
  confirmed.py     one way the condition holds as generated (an OR branch of it)
  probable.py      only in a paper, a weaker form of the search, the value
                   named in the dataset's own text
  possible.py      the field is not recorded
  contradicted.py  the field is recorded with other values
  branches.py      reads the confirming evidence from a condition's SQL

Three sizes, each made of the one before:
  PieceOfEvidence  one way one condition is met (this package)
  ClauseEvidence   one condition (AND clause) of a query and its pieces, in
                   the order they count (services/uncertain/conditions.py)
  QueryEvidence    the whole query: its clauses combined (same file)

Used by results/match_reasons.py (why a confirmed result matched: one
ConfirmedEvidence per OR branch of a condition) and by services/uncertain/
(which state a condition has for a dataset it does not hold for: an ordered
list of pieces, the first that holds decides).
"""
