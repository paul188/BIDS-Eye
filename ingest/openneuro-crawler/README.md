# Openneuro-Crawler

Indexes OpenNeuro datasets into the BIDS-Eye database. `crawler_service.py`
runs one cycle per `CRAWL_INTERVAL_HOURS` (`--once`: one cycle, then exit).
On production it runs as the `crawler` service of `docker-compose-prod.yml`.

A cycle:

1. **Plan** (`remote/snapshot_plan.py`). The catalogue is every dataset in
   OpenNeuro's public S3 bucket plus every dataset already in the database
   (`OPENNEURO_ACCESSIONS` or `CRAWL_CATALOGUE=file` for `accessions.yaml`
   instead). One API request per dozen datasets returns their metadata and
   latest snapshot. A dataset's files are indexed again only when it is new,
   its snapshot commit (`snapshot_hexsha`) changed, `INDEXER_VERSION` was
   raised, its last indexing is older than `CRAWL_MAX_AGE_DAYS` (30), or it
   was indexed within 48 hours of its snapshot (S3 may still have had the old
   files). `CRAWL_FORCE=true` indexes all again. For every other dataset the
   cycle writes the API metadata from the same answers: README, uploader
   fields, downloads and views, deprecation.
2. **Index** the planned datasets (`remote/openneuro.py`), one at a time, at
   most `CRAWL_MAX_PER_CYCLE` (0 = all): files, participants, README and the
   OpenNeuro metadata fields. Then the snapshot is recorded, unless a
   metadata file failed to download (the next cycle tries again). The local
   mirror is deleted afterwards (`CRAWL_KEEP_MIRRORS=true` keeps it): all
   mirrors together need more inodes than the server has. Datasets with more
   than `CRAWL_MAX_FILES` (50,000) files are skipped and logged as `[DEFER]`,
   because pybids holds the whole tree in memory.
3. **Papers** for the datasets of the cycle (`post_crawl.py` →
   `remote/paper_store.enrich`): finds the papers each dataset names and
   looks them up in Europe PMC, bioRxiv/medRxiv, arXiv and OpenAlex. Only
   datasets whose papers were never checked or were checked more than
   `PAPER_RECHECK_DAYS` (30) days ago.
4. **README profiles** (`post_crawl.py` →
   `backend/services/indexing/readme_profile_runner.py`): Gemini reads each
   README once and returns a short profile, the conditions with their role
   (patients, controls, excluded, …) and the paradigms, each with a quote
   that is checked against the text. The search indexes the profile and the
   studied conditions, and the re-ranker reads the profile. Only datasets
   whose README or study fields changed. Needs `GEMINI_API_KEY`.
5. **Descriptions** (`post_crawl.py` →
   `backend/services/indexing/dataset_description.py`): Gemini checks how each
   primary paper relates to its dataset (reports it, a larger sample, a
   related study, a tool paper, unclear), with evidence quotes verified
   against README, catalogue and abstract; the search page shows
   "Linked paper: …" only for the first two. Datasets whose README is under
   500 characters also get a 2–3 sentence generated summary, from README and
   catalogue first and from the paper only as "According to the linked
   paper, …". Only what is missing or stale. Needs `GEMINI_API_KEY`.

Steps 3 to 5 skip what is done and never fail the cycle. The steps that read
the files (summaries, participant groups and codes, derivatives, vacuum) run
only for the datasets indexed in the cycle; profiles, descriptions and
affiliations also for those whose README or uploader fields changed. Switch them off
with `CRAWL_PAPERS=false`, `CRAWL_README_PROFILES=false` or
`CRAWL_DESCRIPTIONS=false`; `post_crawl.py` lists the other steps and their
switches.

When the indexing code changes in a way the existing datasets should get,
raise `INDEXER_VERSION` in `remote/snapshot_plan.py`.

Run it as a service (`docker compose --profile crawl up crawler`) or as one
cycle (`python crawler_service.py --once`). To run the steps after the crawl over
datasets that are already indexed, without crawling them again:

    python post_crawl.py --all                           # every step, every dataset
    python post_crawl.py --all --only profiles,descriptions
    python post_crawl.py --accessions ds000171,ds004627

The step names are in `post_crawl.STEPS`. To look at a step's results before
writing them, the scripts in `ingest/backfill/` run the same code with `--dry-run`
and a review file: `backfill_papers.py`, `backfill_dataset_links.py`,
`backfill_openneuro_metadata.py`, `extract_readme_profiles.py` and
`describe_datasets.py`.
