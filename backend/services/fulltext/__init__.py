"""Full-text search inside the generated SQL.

  text_search.py       text_match() placeholders become conditions on the vectors
  tsquery.py           tsquery strings from search terms
  ranking.py           the text rank and the rank string
  word_frequencies.py  how many datasets match a tsquery
  vectors.py           the stored vectors and the switches that choose them
  uncertain_tier.py    the tier of datasets a filter cannot check
"""
