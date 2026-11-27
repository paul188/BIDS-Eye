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
3. **README profiles** (`post_crawl.py` →
   `backend/services/indexing/readme_profile_runner.py`): Gemini reads each
   README once and returns a short profile, the conditions with their role
   (patients, controls, excluded, …) and the paradigms, each with a quote
   that is checked against the text. The search indexes the profile and the
   studied conditions, and the re-ranker reads the profile. Only datasets
   whose README or study fields changed. Needs `GEMINI_API_KEY`.
4. **Descriptions** (`post_crawl.py` →
   `backend/services/indexing/dataset_description.py`): Gemini checks how each
   primary paper relates to its dataset (reports it, a larger sample, a
   related study, a tool paper, unclear), with evidence quotes verified
   against README, catalogue and abstract; the search page shows
   "Linked paper: …" only for the first two. Datasets whose README is under
   500 characters also get a 2–3 sentence generated summary, from README and
   catalogue first and from the paper only as "According to the linked
   paper, …". Only what is missing or stale. Needs `GEMINI_API_KEY`.

Steps 2 to 4 skip what is done and never fail the cycle. Switch them off
with `CRAWL_PAPERS=false`, `CRAWL_README_PROFILES=false` or
`CRAWL_DESCRIPTIONS=false`; `post_crawl.py` lists the other steps and their
switches.

Run it as a service (`docker compose --profile crawl up crawler`) or as one
cycle on the HPC (`crawl_job.sh`). To run the steps after the crawl over
datasets that are already indexed, without crawling them again:

    python post_crawl.py --all                           # every step, every dataset
    python post_crawl.py --all --only profiles,descriptions
    python post_crawl.py --accessions ds000171,ds004627

The step names are in `post_crawl.STEPS`. To look at a step's results before
writing them, the scripts in `scripts/` run the same code with `--dry-run`
and a review file: `backfill_papers.py`, `backfill_dataset_links.py`,
`backfill_openneuro_metadata.py`, `extract_readme_profiles.py` and
`describe_datasets.py`.
