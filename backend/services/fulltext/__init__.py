"""Full-text search inside the generated SQL.

  text_search.py       text_match() placeholders become conditions on the vectors
  tsquery.py           tsquery strings from search terms
  ranking.py           the text rank and the RankQuery that carries it
  word_frequencies.py  how many datasets match a tsquery
  vectors.py           the stored vectors and the switches that choose them

The tier of datasets a filter cannot check builds on these: services/uncertain/.
"""
