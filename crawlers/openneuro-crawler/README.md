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
3. **Descriptions** (`post_crawl.py` →
   `backend/services/dataset_description.py`): Gemini checks how each
   primary paper relates to its dataset (reports it, a larger sample, a
   related study, a tool paper, unclear), with evidence quotes verified
   against README, catalogue and abstract; the search page shows
   "Linked paper: …" only for the first two. Datasets whose README is under
   500 characters also get a 2–3 sentence generated summary, from README and
   catalogue first and from the paper only as "According to the linked
   paper, …". Only what is missing or stale. Needs `GEMINI_API_KEY`.

Steps 2 and 3 create the tables and columns they write, skip what is done,
and never fail the cycle. Switch them off with `CRAWL_PAPERS=false` or
`CRAWL_DESCRIPTIONS=false`.

Run it as a service (`docker compose --profile crawl up crawler`), as one
cycle on the HPC (`crawl_job.sh`), or for all datasets at once with the
scripts `scripts/backfill_papers.py` and `scripts/describe_datasets.py`.
