"""The uncertain tier: datasets a filter cannot check, shown after the confirmed results.

  tier.py        the entry: gives each condition of the generated SQL a state per dataset
  classify.py    which states a condition can take, by kind: text, age, diagnosis,
                 sex, scanner, patient count
  conditions.py  SoftCondition and what the parts share
  reason_texts.py     every reason a result card gives, in one list
  text.py        weaker forms of a text search
  sex.py         participant sex in the dataset's own text
  scanner.py     field strength, vendor and model in the text

Applied by translate/sql_rewriter.py. results/query_filters.py reads the asked
ages, sex and scanner values with the same functions for its chips.
"""
