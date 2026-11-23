"""The search pipeline, grouped by stage.

The request flow stays at this level:
  search.py         a question from translation to result page
  search_runner.py  runs a translated search against the catalogue, one page at a time
  query_guard.py    guards that keep one generated query from taking the server down

The stages it calls:
  translate/   question to SQL: model calls, the deterministic compiler, SQL rewrites
  fulltext/    full-text search inside the SQL and word form counts
  uncertain/   the tier of datasets a filter cannot check (ages, diagnosis, sex,
               scanner or text that a dataset does not record)
  database_stats/  what the database contains, kept in memory: column
               coverage, known values, institution and place names
  results/     what a result page shows: filter chips, highlighting, re-ranking, cards
  stores/      records of translated questions (durable and in memory)
  indexing/    offline LLM jobs over datasets (scripts/ and the crawler, not requests)

A module is named for what it holds, a noun phrase with the main noun last
(word_form_counts, interpretations, translated_queries). The package already
says the role, so no _store, _service, _manager or _utils suffix.

The groups are by subject, not by order, because one search calls the
language model up to three times and several groups serve more than one step
(fulltext/ word form counts feed the intent prompt and the SQL rewrites). In order:

  1. stored translation   stores/interpretations: the same question gets
                          the same translation; a hit skips steps 2-4
  2. preprocessing        LLM_preprocessor/ (outside backend/): vocabulary,
                          word forms and topic counts, then the INTENT MODEL
                          writes a QueryPlan
  3. SQL                  translate/participant_query_compiler compiles the
                          plan, or translate/text_to_sql asks the SQL MODEL
  4. SQL rewrites         translate/sql_rewriter: structural fixes, the
                          value check (translate/value_check, against the
                          values database_stats/ loaded), full-text conditions and
                          (fulltext/) and the uncertain tier (uncertain/)
  5. run                  search_runner + query_guard: one page; a relaxed
                          re-run when too few datasets match
  6. result page          results/: the RERANK MODEL orders the first
                          candidates, then filter chips, cards, snippets
"""
