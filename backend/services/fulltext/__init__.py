"""Full-text search inside the generated SQL.

  text_search.py       text_match() placeholders become conditions on the vectors
  tsquery.py           tsquery strings from search terms
  ranking.py           the text rank and the RankQuery that carries it
  word_frequencies.py  how many datasets match a tsquery
  vectors.py           the stored vectors and the switches that choose them
  uncertain_tier.py    the tier of datasets a filter cannot check, and its parts:
    tier_classify.py   which states a condition can take
    tier_text.py       weaker forms of a text search
    tier_sex.py        participant sex in the text
    tier_scanner.py    field strength, vendor and model in the text
    tier_conditions.py SoftCondition and what the parts share
"""
