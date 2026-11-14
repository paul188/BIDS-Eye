# Openneuro-Crawler

Indexes OpenNeuro datasets into the BIDS-Eye database. `crawler_service.py`
runs one cycle per `CRAWL_INTERVAL_HOURS` over the accessions in
`OPENNEURO_ACCESSIONS` or `accessions.yaml`, one dataset at a time.

A cycle:

1. **Index** each dataset (`remote/openneuro.py`): files, participants,
   README and the OpenNeuro metadata fields.
2. **Papers** for the datasets just indexed (`post_crawl.py` →
   `remote/paper_store.enrich`): finds the papers each dataset names and
   looks them up in Europe PMC, bioRxiv/medRxiv, arXiv and OpenAlex. Only
   datasets whose papers were never checked or were checked more than
   `PAPER_RECHECK_DAYS` (30) days ago.
3. **Paper summaries** (`post_crawl.py` →
   `backend/services/paper_summary.py`): a one-sentence description of what
   data each dataset holds, written by Gemini from its primary paper's
   abstract and checked (no invented numbers, no copied phrases). The search
   page shows it labelled as generated, with a link to the paper. Only where
   it is missing or stale. Needs `GEMINI_API_KEY`.

Steps 2 and 3 create the tables and columns they write, skip what is done,
and never fail the cycle. Switch them off with `CRAWL_PAPERS=false` or
`CRAWL_PAPER_SUMMARIES=false`.

Run it as a service (`docker compose --profile crawl up crawler`), as one
cycle on the HPC (`crawl_job.sh`), or for all datasets at once with the
scripts `scripts/backfill_papers.py` and `scripts/generate_paper_summaries.py`.
