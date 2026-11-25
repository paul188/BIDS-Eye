"""The uncertain tier: datasets a filter cannot check, shown after the confirmed results.

  tier.py        the entry: gives each condition of the generated SQL a state per dataset
  classify.py    the evidence a condition can have, by kind: text, age, diagnosis,
                 sex, scanner, patient count
  conditions.py  ClauseEvidence (one condition, its evidence in the order it
                 counts) and what the parts share
  text.py        weaker forms of a text search
  fields/        the fields a dataset may not record, one file each (age,
                 diagnosis, sex), all described the same way (fields/base.py)
  scanner.py     field strength, vendor and model in the text

The kinds of evidence are in services/evidence/, one file per strength,
shared with the reasons of confirmed results (results/match_reasons.py). A
kind holds what it is about (the missing words, the sex a text names) and
writes its SQL and card text from that; the kinds of one field are in that
field's file here.

Applied by translate/sql_rewriter.py. results/query_filters.py reads the asked
ages, sex and scanner values with the same functions for its chips.
"""
