-- Where the institutions and ethics boards of bids_dataset_sites are:
-- each raw name matched to the Research Organization Registry (ROR), with the
-- city, region and country ROR records for it. The search reads it for
-- questions that name a place ("datasets from Bonn", "data collected in
-- Germany"; LLM_preprocessor/preprocess.py, location lines).
--
-- One raw name can hold several institutions ("IRBs of Wayne State University
-- and Tohoku University"), so a name has one row per matched organisation.
-- A name ROR could not match gets one row with ror_id '' and no place, so it
-- is not asked again on every crawl.
--
-- Filled and kept current by crawlers/openneuro-crawler/remote/site_places.py
-- (post_crawl step "places"). Additive: nothing else reads or changes it.

BEGIN;

CREATE TABLE IF NOT EXISTS bids_site_places (
    name_raw      TEXT NOT NULL,             -- bids_dataset_sites.name_raw, as stored
    ror_id        TEXT NOT NULL DEFAULT '',  -- https://ror.org/...; '' = checked, no match
    institution   TEXT,                      -- ROR display name
    city          TEXT,
    region        TEXT,                      -- state, province, Land ('North Rhine-Westphalia')
    country       TEXT,
    country_code  TEXT,                      -- ISO 3166-1 alpha-2
    lat           DOUBLE PRECISION,
    lng           DOUBLE PRECISION,
    matched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (name_raw, ror_id)
);
CREATE INDEX IF NOT EXISTS idx_bids_site_places_city ON bids_site_places (city);
CREATE INDEX IF NOT EXISTS idx_bids_site_places_region ON bids_site_places (region);
CREATE INDEX IF NOT EXISTS idx_bids_site_places_country ON bids_site_places (country);
-- The search joins sites to places on the raw name.
CREATE INDEX IF NOT EXISTS idx_bids_dataset_sites_name_raw ON bids_dataset_sites (name_raw);

COMMIT;
