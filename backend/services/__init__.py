"""The search pipeline, grouped by stage.

The request flow stays at this level:
  search.py         a question from translation to result page
  search_runner.py  runs a translated search against the catalogue, one page at a time
  query_guard.py    guards that keep one generated query from taking the server down

The stages it calls:
  translate/   question to SQL: model calls, the deterministic compiler, SQL rewrites
  fulltext/    full-text search inside the SQL, and the tier of unconfirmed datasets
  catalogue/   what the database contains: optional columns, known values, word forms
  results/     what a result page shows: filter chips, highlighting, re-ranking, cards
  stores/      records of translated questions (durable and in memory)
  indexing/    offline LLM jobs over datasets (scripts/ and the crawler, not requests)
"""
