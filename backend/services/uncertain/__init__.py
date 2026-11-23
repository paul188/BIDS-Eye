"""The uncertain tier: datasets a filter cannot check, shown after the confirmed results.

  tier.py        the entry: gives each condition of the generated SQL a state per dataset
  classify.py    the evidence a condition can have, by kind: text, age, diagnosis,
                 sex, scanner, patient count
  conditions.py  SoftCondition (one condition, its evidence in the order it
                 counts) and what the parts share
  text.py        weaker forms of a text search
  fields/        the fields a dataset may not record, one file each (age,
                 diagnosis, sex), all described the same way (fields/base.py)
  scanner.py     field strength, vendor and model in the text

The kinds of evidence and every text a result card gives are in
services/evidence/ (weaker.py, reason_texts.py), shared with the reasons of
confirmed results (results/match_reasons.py).

Applied by translate/sql_rewriter.py. results/query_filters.py reads the asked
ages, sex and scanner values with the same functions for its chips.
"""
