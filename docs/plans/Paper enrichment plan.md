# Plan: read the papers a dataset names, and use them in search

Status on 2026-11-13: research and measurements done; phases 1 and 2 built
and run against a local test database (see "Built so far" at the end).
Nothing on the server has changed. Every number below was measured today on
the production catalogue (1,721 datasets) unless a source is given.

## Why

A dataset's own metadata often does not say who the participants were. The
paper it links usually does.

- ds004746 (Paingen placebo): the README never says whether the participants
  are pain patients. Its paper does: 392 community volunteers given heat and
  pressure pain. One lookup settled the benchmark label.
- ds003500 (ADHD, "children"): no ages in the database, none in the README.

| Gap in the catalogue | Datasets | of which name a paper (reference field or README) |
|---|---|---|
| No recorded ages at all | 814 | 246, and 127 of those papers carry MeSH age tags |
| No diagnosis information | 972 | 314 |

Of the relevant datasets the old live search missed on the 167-question
benchmark (run A), 41% name a paper that has an abstract. That is the ceiling
for what paper text can recover, not a forecast.

## What the data looks like

### Where paper links are

| Source | Datasets | Notes |
|---|---|---|
| `paper_references` (BIDS `ReferencesAndLinks`) or README | 711 | 943 distinct ids: 797 DOIs, 68 PubMed ids, 65 PMC ids, 13 arXiv ids |
| OpenNeuro `metadata.associatedPaperDOI` | 380 filled, 358 with a DOI | "Papers published from the lab that collected this dataset". **Our crawler never read this field.** 146 of these datasets name no paper anywhere else |
| OpenNeuro `metadata.openneuroPaperDOI` | 264 filled | Meant for papers by people who downloaded the dataset (reuse). Uploaders often fill it with the same paper |
| **Any source** | **856 (50%)** | |
| Free-text citation only, no id | about 72 | e.g. "Aron, Gluck, Poldrack (2006). Neuroimage 29(3):1000-6" |
| Reference field holds no paper | about 540 | GitHub and OSF links, "None", "Pending", "unpublished" |

Among datasets with several papers, 82 have an `associatedPaperDOI` that
tells us which one is the dataset's own, and 192 do not.

### What a lookup returns (Europe PMC, all 943 ids, no key, 4 minutes)

| | Papers (of 943) | Datasets (of 711) |
|---|---|---|
| Found | 833 | 653 |
| Abstract | 826 | 651 |
| MeSH keywords | 605 | 484 |
| Author keywords | 417 | |
| Open-access full text | 345 | 356 |
| Preprint | 147, of which 104 link to the published version | |

MeSH (Medical Subject Headings) are the controlled keywords PubMed assigns to
each paper, such as "Chronic Pain", "Child" or "Parkinson Disease".

The 110 misses: about 40 are malformed DOIs that better cleaning fixes
(`…116945}`, `…#citeas`, `…/meta`, `…v3`, unbalanced brackets in old Elsevier
DOIs), 13 are arXiv papers, the rest are conference papers and book chapters.
OpenAlex found 64 of the 110 (54 with an abstract).

### Citations that are not about the dataset

| Paper | Cited by |
|---|---|
| MNE-BIDS (a file-format tool) | 63 datasets |
| EEG-BIDS | 34 |
| MEG-BIDS | 23 |
| iEEG-BIDS | 9 |
| MRIQC, fMRIPrep | 6, 5 |

"Cited by many datasets" is not a safe filter on its own: the Healthy Brain
Network paper is cited by 12 datasets that really are releases of that study.
Of the 47 papers cited by three or more datasets, about 8 are tools.

Datasets also cite background work: ds004552 lists a systematic review,
ds005605 lists radiochemistry papers.

### Finding papers when the dataset names none

Searching Europe PMC full text for the accession number ("ds004283" AND
openneuro) found something for 9 of 30 datasets without a paper. Several hits
were reuse papers by other groups, and one was unrelated (a dataset called
"test"). Usable only with an author check. Other routes exist (DataCite
citations per version DOI, OpenCitations, Crossref's data-citations beta).

## What the research says

Full source lists are in the two research reports of this session; the points
that shape the design:

1. **No study shows that paper text improves dataset search.** The nearest
   evidence is mixed. In the bioCADDIE 2016 challenge, adding extra text
   fields alone scored below the baseline for one top team (0.387 against
   0.402 infNDCG; Ji et al. 2018), MeSH query expansion gave +11% on average
   but from −21% to +96% per question (Wright et al. 2017), and two expanded
   runs of another team lost quality (Teodoro et al. 2017). DataFinder
   (Viswanathan et al. 2023) found that adding sentences from citing papers
   changed nothing. So this must be measured here, per question, with the
   paper text switchable.
2. **The bioCADDIE judges read the linked PubMed articles to decide
   relevance** (Cohen et al. 2017). Metadata alone was not enough even to
   judge. That matches our review-page experience and means our ground truth
   needs the papers too.
3. **Most of the gain from document expansion comes from re-weighting terms
   the document already has, and unsupported additions hurt** (doc2query
   ablation, Lin et al. 2020; Doc2Query--, Gospodinov et al. 2023: filtering
   bad expansions gained up to 16%). Weight is how search engines control
   this (BM25F: per-field weights tuned on judgments).
4. **Extracting participant data from papers works on full text, not on
   abstracts.** GPT-4o on text up to the Methods section got groups,
   diagnoses and counts right for 26 of 30 papers (Turner et al. 2025); GPT-4
   on full text had under 15% error on counts (Neurosynth Compose 2025); exact
   sample size was found in only 36–50% of papers by older methods (Dockès et
   al. 2025). Nobody evaluated abstracts alone.
5. **MeSH age tags are sensitive, not precise.** "Aged" had 94% sensitivity
   and 28% precision as a filter, because the tag applies when a single
   participant qualifies (Kastner et al. 2006). Since 2022 all MeSH indexing
   is automated from title and abstract; check tags score F1 87% (NLM 2024).
   Preprints and non-MEDLINE journals have no MeSH.
6. **Nobody has solved "which citation is the dataset's own paper".** BIDS
   has no field for it. OpenNeuro's `associatedPaperDOI` is the best signal.
   DataCite records of OpenNeuro DOIs carry no paper links.

### Services (verified today)

| Service | Key or email | Limit | Use |
|---|---|---|---|
| Europe PMC REST | none | about 10 requests/s | **Primary**: abstract, MeSH, author keywords, licence, open-access flag, PMC id, preprint → published link, in one call. Batches by OR-joining ids |
| NCBI efetch + ID converter | none | 3 requests/s | Fallback for PubMed ids |
| bioRxiv API, Crossref `relation` | none | Crossref 5/s single lookups | Preprint → published version |
| arXiv API | none | | arXiv abstracts |
| OpenAlex | none for DOI lookups (a free key needs a verified email) | keyless: shared daily budget per IP | Fallback for DOIs outside biomedicine. Abstract comes as a word index and must be rebuilt |
| Crossref `query.bibliographic` | none | 1 request/s | Later: citations without an id. Needs our own check of title, year and first author; a bare score cut-off has 0.77–0.87 precision |
| NCBI BioC, Europe PMC `fullTextXML` | none | | Later: Methods sections of open-access papers (BioC labels passages `METHODS`) |
| Unpaywall, Semantic Scholar | email / shared pool | | Not needed |

**No email address is sent to any service.** Requests carry a generic client
name only.

Terms: no service grants a licence to abstracts; copyright stays with the
publisher or author. Nothing forbids storing and indexing them. Position
taken here: store and index abstracts, show only title, journal, year and a
link, and show abstract text only where the record carries a CC licence. This
is a design choice, not legal advice.

## Design

### 1. Collect links, per dataset

In priority order, each link remembers where it came from:

1. `metadata.associatedPaperDOI` (new: add to the OpenNeuro query)
2. `paper_references`
3. `HowToAcknowledge` (new) and the README
4. `metadata.openneuroPaperDOI` (new), kept apart as possible reuse papers

Identifier extraction: DOIs with clean-up (trailing punctuation and brackets,
`#fragment`, `/meta`, `/full`, version suffixes, balanced parentheses),
PubMed and PMC ids, arXiv ids, and DOIs derived from `nature.com/articles/…`
links. Dropped: the dataset's own DOI and data repositories (OpenNeuro,
Zenodo, OSF, figshare, Dryad, Dataverse). An id counts only after a service
resolves it.

### 2. Resolve each paper once

Europe PMC first; then by id type bioRxiv, arXiv, NCBI, OpenAlex. A preprint
with a published version is merged into one paper that keeps both DOIs and
takes abstract and MeSH from the published version. The same paper named by
DOI, PubMed id and PMC id becomes one row. Results are cached; a paper is
fetched again only after 90 days if it had no MeSH or was a preprint.

### 3. Decide each link's role

| Role | Rule | Indexed |
|---|---|---|
| `tool` | On a hand-kept list (`TOOL_PAPERS` in `remote/paper_links.py`: BIDS extensions, MNE-BIDS, fMRIPrep, MRIQC, …), published in the Journal of Open Source Software, or cited by 5 or more datasets with a title that reads like software | no |
| `primary` | Named in `associatedPaperDOI`; or shares an author with the dataset and (the title overlaps the dataset name, or the text mentions the accession number, or the dataset asks to cite it); or the only non-tool paper and shares an author; or at least half of the shorter author list is on both | yes |
| `related` | Everything else a dataset cites, including a sole paper with no shared author and papers that share only one author among several | no |
| `reuse` | Only in `openneuroPaperDOI`, no shared author | no |

The role and the reason are stored, so a wrong rule can be found and fixed.
Check before using: I label 60 links by hand (stratified: single paper,
several papers, shared papers) and the rules must agree on at least 90% of
the `primary` calls; otherwise the rules change before anything is indexed.

### 4. Store

```
papers            one row per paper: doi, preprint_doi, pmid, pmcid, title,
                  abstract, year, journal, is_preprint, mesh_terms[],
                  mesh_major[], keywords[], is_open_access, license, source,
                  fetched_at
dataset_papers    dataset_id, paper_id, role, role_reason, sources[], raw_ref
bids_datasets.paper_text    text of the dataset's primary papers, written by
                            the enrichment step (a generated column cannot
                            read another table)
bids_datasets.search_tsv_papers   generated: the search_tsv expression plus
                                  paper_text at weight D, with its own GIN index
```

`search_tsv` stays exactly as it is.

### 5. Search

- **Matching.** With the switch on, `text_match()` expands to
  `search_tsv_papers` instead of `search_tsv`. One combined column is needed,
  not two separate ones: a question such as "children AND ADHD" must match
  when "ADHD" is in the README and "children" only in the paper.
- **Ranking: paper text adds datasets, it never promotes them.** Order by the
  rank on the dataset's own text first (`ts_rank_cd` over `search_tsv`), then
  by the rank including the paper. A dataset found only through its paper
  sorts after every dataset matched on its own text. This follows finding 3.
- **Switch.** `TEXT_SEARCH_PAPERS=true|false` in the backend environment,
  and the backend only honours it when the column exists
  (`schema_capabilities`). Off is the default until measured.
- **Why a result matched.** The API response marks datasets matched only
  through a paper; the card shows "matched in linked paper: <title>" with a
  link. Later phase, with the frontend.

### 6. Structured use of MeSH, without an LLM (after the text index is measured)

- **Age groups.** For datasets with *no* recorded ages, MeSH age tags of the
  primary paper become `paper_age_groups` (Infant, Child, Adolescent, Adult,
  Aged, …). An age filter then also accepts a dataset without ages whose
  paper tags overlap the asked range. Never used when ages are recorded, and
  never used to exclude. Finding 5 says to expect false hits for "older
  adults"; the benchmark has 35 age questions to measure this.
- **Species.** "Animals" without "Humans" marks animal datasets.
- **Conditions.** MeSH disease headings run through the existing diagnosis
  resolver (`backend/vocabulary/`), stored as paper-derived conditions with their source,
  apart from uploader-provided fields.

### 7. Ingestion

The enrichment runs as a step of the crawler cycle for new or changed
datasets, and as `ingest/backfill/backfill_papers.py` for the existing ones
(`--dry-run`, `--limit`, `--accessions`, `--refresh-older-than`,
`--save-jsonl` / `--from-jsonl`, like the OpenNeuro metadata backfill). A full
backfill is about 1,300 papers: under ten minutes.

## How it is measured

1. Baseline: the best of runs A, B, C on the 167 questions, on the database
   copy.
2. Same code and same database, switch off and on, re-ranker off, with the
   paired runner (`paired_rank_run.py`) so the SQL translation is identical
   and only the text matching differs.
3. Variants of `paper_text`: title only; title + abstract; title + abstract +
   MeSH and author keywords. Ranking: "adds only" against mixed.
4. **Judge before comparing.** Datasets found only through papers were never
   judged, so they would count as wrong. Every newly returned unjudged
   dataset is judged with the paper on the card. Existing "unsure" and "not
   relevant" labels of datasets whose paper matches the question's pool
   keywords are re-judged with the paper too (finding 2). A sample goes to
   the review page.
5. Reported per question group and per question: recall, lenient recall,
   precision, result-set size, questions better / worse. A measure that
   raises recall but drops precision by more than a few points is not
   switched on as it is.

## Phases

| Phase | What | Touches | Needs your OK |
|---|---|---|---|
| 1 | Link collection, resolver, `papers` / `dataset_papers`, backfill script, tests. Run against fetched data locally | new files only | no |
| 2 | Roles, tool list, 60-link hand check | new files only | no |
| 3 | `paper_text`, `search_tsv_papers`, backend switch. Migrate the **database copy** on the server, run the benchmark variants, judge new hits | `text_search.py` (another session has uncommitted changes there; agreed with it first), one migration file | no (copy only) |
| 4 | MeSH age groups, species, conditions; benchmark again | preprocessing of age filters | no (copy only) |
| 5 | Production: migration, backfill, deploy with the switch set by the results; crawler step; card shows the paper | production | **yes** |
| 6 | Optional: papers for datasets that name none (accession search with author check); citations without id; Methods sections of open-access papers; LLM extraction of groups, diagnoses and ages from Methods | sends public paper text to Gemini | **yes** |

## Risks

- The wrong paper is linked, or one paper is attached to many datasets
  (roles, hand check).
- The dataset is a subset of the paper's sample, or the paper covers several
  datasets.
- An abstract mentions a condition as motivation or as an exclusion ("has
  implications for chronic pain"). Paper matches therefore never outrank
  dataset-text matches, and precision is tracked.
- MeSH age tags over-apply (finding 5): fallback only, never a filter.
- Long abstracts are favoured by `ts_rank_cd` without length normalisation.
- Benchmark labels predate the papers (step 4 of the measurement).

## Decisions for you

1. Showing abstract text on the site: the plan shows title and link only.
2. Phase 6 LLM extraction sends paper text (public) to Gemini. Not started
   without your OK.
3. A free OpenAlex key would need your email. Not needed for this plan.

## Built so far (2026-11-13)

| File | What |
|---|---|
| `ingest/openneuro-crawler/remote/paper_links.py` | identifier extraction and clean-up, link collection per dataset, author-name parsing, roles |
| `ingest/openneuro-crawler/remote/paper_lookup.py` | Europe PMC, bioRxiv/medRxiv, arXiv, OpenAlex; preprint and published version merged into one paper |
| `ingest/openneuro-crawler/remote/paper_store.py` | tables `bids_papers`, `bids_dataset_papers`; `bids_datasets.paper_text`; `enrich()` |
| `ingest/backfill/backfill_papers.py` | `--dry-run`, `--limit`, `--accessions`, `--parts`, `--save-json`, `--from-json`, `--show` |
| `backend/tests/test_paper_links.py` | 16 tests |

Full run against a local test database holding the real link fields of 1,718
datasets (4 min 42 s, then repeated from the saved lookups to check that a
second run changes nothing):

| | |
|---|---|
| Paper ids found / resolved | 1,191 / 1,180 (99%) |
| Datasets that name a paper | 916 |
| Datasets with a primary paper | 839 (831 with an abstract, 676 with MeSH) |
| Links: primary / related / tool / reuse | 1,205 / 154 / 149 / 1 |
| Papers stored | 1,127 (1,069 from Europe PMC, 46 OpenAlex, 12 arXiv) |

### Role check

60 links, stratified by rule, labelled by hand (by Claude, from dataset name
and authors against paper title, authors and abstract; 4 could not be
decided). Labels: `benchmarks/recall/work/papers/role_sample_labels.json`.

| | Primary calls right | Primary papers kept |
|---|---|---|
| First rules | 48 of 56 decisions agreed | |
| After tightening | 41 of 47 (87%) | 41 of 41 |

Two rule changes came out of it: a dataset's only paper is primary only with
a shared author (a README that cites just its stimulus set, "Inscapes", was
taken for the dataset's paper), and one shared author among several papers is
not enough (a lab head is on every paper of the lab; an adult dataset had been
given the lab's paper on children). Four tool-list DOIs written from memory
were other papers and were replaced after looking each one up.

The six remaining errors: one wrong entry in OpenNeuro's own associated-paper
field (a lab listed six papers), and five cases of a second experiment by the
same authors (derivative datasets, a stimulus-database paper by the same
group). Author metadata cannot separate those. Weighted by how often each
rule fires, about 90% of primary calls are right. What limits the damage: at
most five papers per dataset are indexed, and a dataset found only through its
paper never outranks one matched on its own text.

Not built yet: the search column and backend switch (phase 3), everything
after it, and the crawler hook.
