# Search criteria, facets and metadata models of neuroimaging and neuroscience data repositories (state as of Nov 2026)

How the evidence was gathered. "Verified" means I read it directly in source code, a schema or a live API response on 2026-11-14:
- OpenNeuro, from the GitHub source of the search UI and the GraphQL/Elasticsearch query builder.
- Neurobagel, from the docs, the live OpenAPI spec of the OpenNeuro node, and the SPARQL builder source.
- DANDI, from dandischema `models.py` and the archive's search operator code.
- EBRAINS KG Search, from the live `/api/settings` facet config.
- openMINDS, from its schema templates.
- CONP, from the portal's GitHub source (last commit 2026-11-10).
- BIDS, from the participants.tsv spec.
- NeuroVault, from its Django model.

The other systems (NDA, NITRC-IR, LONI IDA, COINS, XNAT, OpenfMRI, INDI, brainlife, SciCrunch/dkNET, DataMed, Google Dataset Search) come from papers, manuals or search snippets. Some of those sources are old and are flagged inline.

## Q1. Which filters/facets does each repository/tool expose, which metadata field answers each, and which controlled vocabularies are used?

### Takeaway
Real facet sets are small and mostly flat. OpenNeuro has about 20 search inputs, but several are answered from a free-text, curator-entered `metadata` record (diagnosis, species, study design and domain) rather than from BIDS files. Others are answered from a BIDS-derived snapshot summary (modality, age, sex, tasks, subject count, PET fields).

Only some systems bind criteria to ontologies:
- Neurobagel: SNOMED CT for sex, diagnosis and assessment; NCIt for healthy control; NIDM for imaging modality; the Nipoppy pipeline catalog for pipelines.
- openMINDS / EBRAINS: controlled terms for species, technique, approach and brain region.
- DANDI: typed identifiers for species, approach and technique, and UBERON/SNOMED for `about`. The schema does not enforce these.

### Cited Findings

#### Comparison table: criterion by system, with the evidence field each is answered from
(V = verified in code, schema or live API this session; D = from docs or papers; I = inferred; — = no such filter found)

| Criterion | OpenNeuro (V) | Neurobagel (V) | DANDI (V) | EBRAINS KG Search / openMINDS (V) | CONP portal (V) | NeuroVault (V model; UI facets not verified) | NDA / NITRC-IR / LONI IDA / COINS / XNAT (D) |
|---|---|---|---|---|---|---|---|
| Free text | `keywords` searches dataset id^20, README, `description.Name`^6, `description.Authors`^3 and `contributors.name`^2, with fuzzy matching | — (structured only) | Free text plus operators; free text uses the existing full-text filter | Free text plus facets | `search`/`q` over title, keywords, modalities and formats | Text search (not verified) | NDA: keyword search of the data dictionary and studies; IDA: per-project |
| Modality | `modality` ∈ {mri, pet, eeg, ieeg, meg, nirs} matched against `latestSnapshot.summary.modalities`. The backend also supports secondary modalities (Diffusion, Structural, Functional, Perfusion; PET Static, Dynamic) via `summary.secondaryModalities`, but the UI child options are commented out | `image_modal` (NIDM contrast-type terms) matched against `nb:hasAcquisition/nb:hasContrastType` per imaging session | `technique:` → `assetsSummary.measurementTechnique[*].name`; `approach:` → `assetsSummary.approach[*].name` | Facets "Techniques" (`techniques`), "Experimental approach" (`experimentalApproach`) and "Content types"; openMINDS `DatasetVersion.technique` and `experimentalApproach` | "Modality" facet from DATS `types`/modalities, with acronym normalisation (DWI = Diffusion-weighted Imaging) | `modality` choice list: fMRI-BOLD, fMRI-CBF, fMRI-CBV, Diffusion MRI, Structural MRI, PET FDG, PET [15O]-water, PET other, MEG, EEG, Other | IDA: modality, weighting, pulse sequence, acquisition type; NITRC-IR: field strength, TR |
| Age | `ageRange` [min, max]: ES range on `summary.subjectMetadata.age` from participants.tsv, with relation INTERSECTS. A dataset matches if any subject falls in range (I) | `min_age`/`max_age` (float years) on `nb:hasAge` per phenotypic session. Raw formats are normalised at annotation time | — (no age operator) | Age is in the KG but not shown as a Search UI facet | — | Collection fields `subject_age_mean`, `subject_age_min`, `subject_age_max` | NITRC-IR: subject age across projects; COINS: min/max age, inclusive, rounded down to whole years; IDA: subject age |
| Sex | `sex` ∈ {Male, Female} matched against `subjectMetadata.sex` with the BIDS spellings (male, m, M, MALE, Male, …). No "other" option | `sex` = SNOMED CT term on `nb:hasSex` | — | Subject biological sex in openMINDS (no Search UI facet) | — | `proportion_male_subjects` | XNAT: subject gender; IDA, NITRC-IR: sex/gender |
| Diagnosis / clinical group | `diagnosis` ∈ {Healthy / Control, Schizophrenia, ADD/ADHD, Alzheimers, Other} matched against `metadata.dxStatus`, a single free-text string per dataset | `diagnosis` = list of SNOMED CT terms (healthy control = NCIt term) on `nb:hasDiagnosis`. The separate `is_control` parameter is gone from the current API | Dandiset `about`: `Disorder` ("Biolink, SNOMED, or other identifier"). No search operator | openMINDS `studyTarget` (no disease facet in Search UI) | — | `group_comparison` boolean | IDA/ADNI: diagnosis group; NDA: phenotype data structures and concepts |
| Task / paradigm | `tasks`: simple_query_string on `summary.tasks` (BIDS task labels) | — | — | openMINDS `protocol` / BehavioralProtocol (no facet) | — | `cognitive_paradigm_cogatlas`, `cognitive_contrast_cogatlas` (Cognitive Atlas FKs) | OpenfMRI (historical): Cognitive Atlas annotation |
| Assessment / instrument | — | `assessment` = SNOMED CT instrument terms on `nb:hasAssessment` | `variable:` → `assetsSummary.variableMeasured[*]` (bare strings) | — | — | — | NDA: data structures (data dictionary); COINS: assessments; IDA: MMSE/CDR/NPI/FAQ (ADNI) |
| Species | `species` ∈ {Human, Rat, Mouse, Other} on `metadata.species` (free text, fuzzy). "Other" = must_not Human/Rat/Mouse | — (human cohorts implied) | `species:` → `assetsSummary.species[*].name` | "Species" facet (`species`); openMINDS controlled term Species | — | — | — |
| Number of subjects | `subjectCountRange`: painless script on the length of `summary.subjects` | Result column `num_matching_subjects` (not a filter) | `assetsSummary.numberOfSubjects` (no operator) | — | Sort "Most subjects" | `number_of_subjects` per map | — |
| Sessions / longitudinal | `section` ∈ {Cross-Sectional, Longitudinal, Other} on `metadata.studyLongitudinal` (free text) | `min_num_imaging_sessions`, `min_num_phenotypic_sessions` | `assetsSummary.numberOfSessions` derived from `ses-*` tokens (no operator) | — | — | — | IDA, ADNI: visit (baseline, Month 6) |
| Study domain / topic | `studyDomains` on `metadata.studyDomain` (free text) | — | `keywords`, `studyTarget` (free strings) | "Keywords" facet | — | — | Google DS: "topic" |
| Brain region / anatomy | PET `bodyParts` on `summary.pet.BodyPart` | — | `about`: `Anatomy` ("UBERON or other identifier") | "Studied brain region" (`studiedBrainRegion`) | — | `target_template_image` (template space) | — |
| Scanner / acquisition | PET only: `scannerManufacturers`, `scannerManufacturersModelNames`, `tracerNames`, `tracerRadionuclides` from `summary.pet.*` (BIDS sidecars) | — | — | — | — | `scanner_make`, `field_strength`, `smoothing_fwhm`, `software_package` | NITRC-IR: field strength, TR; IDA: weighting, pulse sequence |
| Processing / derivatives | `bidsDatasetType` ∈ {raw, derivative} on `description.DatasetType` | `pipeline` (name from Nipoppy catalog + version) on `nb:hasCompletedPipeline` | `standard:` → `assetsSummary.dataStandard` | `dataType` (raw/derived/simulated) in openMINDS | — | `analysis_level`, `map_type` | IDA: processed images (Advanced Search only) |
| Authors / people | `authors`: fuzzy match on `contributors.name` | — | `owner:` (GitHub login, email, name) | "Contributor" result type | — | — | — |
| Institution / affiliation | — (no facet) | — | Affiliation/Organization `identifier` = ROR (no operator) | openMINDS Organization with `digitalIdentifier`, `hasParent` (no facet) | "Origin" facet: Canadian vs International from `conpStatus`. DATS `origin_institution`/`city`/`province`/`country` shown, not faceted | — | — |
| Date | `dateRange` ∈ {Last 30 d, 180 d, 12 mo} on `created` | — | `created_/modified_/published_before/after:` | — | Sort by added/updated | — | Google DS: "last updated" |
| Access / licence | `datasetType` (All Public, Following, My Datasets, Bookmarks); BRAIN Initiative flag; synthetic flag | Node-level: `records_protected`, access_type | — | "Accessibility", "License" | "Access" (Open / Account needed / Not stated), "How to get it", "Format", "Linked publication", "Used in research" | — | Google DS: usage rights, free-to-use, format |

Sources for the OpenNeuro column:
- [initial-search-params.tsx](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-app/src/scripts/search/initial-search-params.tsx) (facet value lists)
- [build-search-query.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/resolvers/build-search-query.ts) (field paths)
- [inputs.ts DatasetSearchInput](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/schema/inputs.ts) (GraphQL `advancedSearch` input)
- [use-search-results.tsx](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-app/src/scripts/search/use-search-results.tsx)

Sources for the other columns:
- Neurobagel: [query tool docs](https://neurobagel.org/user_guide/query_tool/), [OpenAPI of OpenNeuro node](https://openneuro.neurobagel.org/node/docs), [api utility.py](https://github.com/neurobagel/api/blob/main/app/api/utility.py)
- DANDI: [dandischema models.py](https://github.com/dandi/dandi-schema/blob/master/dandischema/models.py), [search filters.py](https://github.com/dandi/dandi-archive/blob/master/dandiapi/api/services/search/filters.py), [parser.py](https://github.com/dandi/dandi-archive/blob/master/dandiapi/api/services/search/parser.py)
- EBRAINS: [KG Search live settings](https://search.kg.ebrains.eu/api/settings), [openMINDS DatasetVersion](https://github.com/openMetadataInitiative/openMINDS_core/blob/main/schemas/products/datasetVersion.schema.tpl.json)
- CONP: [dataset_list.py](https://github.com/CONP-PCNO/conp-portal/blob/master/app/search/dataset_list.py)
- NeuroVault: [statmaps/models.py](https://github.com/NeuroVault/NeuroVault/blob/master/neurovault/apps/statmaps/models.py)
- The NDA, NITRC-IR, IDA, COINS and XNAT sources are listed below.

#### OpenNeuro (verified from source)
- The GraphQL `advancedSearch(query: DatasetSearchInput!, allDatasets, datasetType, datasetStatus, first, after)` takes these input fields: keywords, modality, ageRange, subjectCountRange, diagnosis, tasks, authors, sex, dateRange, species, studyStructure, studyDomains, bidsDatasetType, brainInitiative, bodyParts, scannerManufacturers, scannerManufacturersModelNames, tracerNames, tracerRadionuclides, sortBy, userId — [inputs.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/schema/inputs.ts). An older plain `search(q: String!)` field also exists — [dataset-search.ts schema](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/schema/dataset-search.ts).
- Two evidence stores answer these filters — [build-search-query.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/resolvers/build-search-query.ts):
  - The BIDS-derived snapshot summary: `latestSnapshot.summary.{modalities, secondaryModalities, subjects, subjectMetadata.age/sex, tasks, pet.*}` and `latestSnapshot.description.DatasetType`.
  - A separate Mongo "Metadata" record: `metadata.{dxStatus, species, studyLongitudinal, studyDomain, syntheticDataset}`.
- The Metadata model fields include ages, modalities, dxStatus, tasksCompleted, trialCount, studyDesign, studyDomain, studyLongitudinal, species, associatedPaperDOI, seniorAuthor, grantFunderName, grantIdentifier, affirmedDefaced, affirmedConsent and syntheticDataset. All are plain `String` fields; code comments say values can be "'true' | 'false' | 'user input string'" — [metadata.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/models/metadata.ts).
- Per-dataset search results carry `summary.subjectMetadata {participantId, age, sex, group}`. That is participant-level data, but the search only returns datasets — [use-search-results.tsx](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-app/src/scripts/search/use-search-results.tsx).
- In 2019 the Stanford/OpenNeuro team kept an external metadata sheet (subject counts, modalities, diagnosis status, species) because it "supports complex queries that are not currently supported by the site's search functionality". This is historical context and is dated — [Stanford reproducibility blog](https://reproducibility.stanford.edu/index.html@p=921.html).

#### Neurobagel (verified)
- The query tool has 9 criteria: age (min/max years), sex, diagnosis, minimum number of imaging sessions, minimum number of phenotypic sessions, assessment tool, imaging modality, pipeline name and pipeline version — [Neurobagel query tool docs](https://neurobagel.org/user_guide/query_tool/).
- The live node API (`openneuro.neurobagel.org/node`) has a `DatasetsQueryModel` with min_age, max_age, sex, diagnosis[], min_num_imaging_sessions, min_num_phenotypic_sessions, assessment[], image_modal[] and pipeline[{name, version}]. Values must be CURIEs (`^[a-zA-Z]+[:]\S+$`). Endpoints: POST /datasets, POST /subjects, and GET /assessments, /diagnoses, /imaging-modalities, /pipelines plus their `/vocab` — [Neurobagel OpenNeuro node OpenAPI](https://openneuro.neurobagel.org/node/docs). api.neurobagel.org/docs now 301-redirects there.
- Namespaces, per the [term naming standards](https://neurobagel.org/data_models/term_naming_standards/):
  - `snomed` (SNOMED CT) for "diagnoses, sex, and assessments or instruments"
  - `ncit` (NCI Thesaurus) for group designation, e.g. healthy control
  - `nidm` for imaging modalities
  - `np` (Nipoppy pipeline-catalog) for pipelines
  - `nb` for Neurobagel-owned terms
- Age annotation formats are `nb:FromFloat`, `nb:FromEuro` ("31,5"), `nb:FromBounded` ("30+"), `nb:FromRange` ("30-35") and `nb:FromISO8061` ("31Y6M"). They exist "to ensure age values are represented as floats" — [Neurobagel data dictionaries](https://neurobagel.org/data_models/dictionaries/).
- Assessment columns get `IsAbout: nb:Assessment` plus `IsPartOf` naming the instrument (SNOMED CT). Group columns use `nb:Diagnosis` with `Levels`, e.g. `CTRL` mapped to the NCIt "Healthy Control" term — [dictionaries](https://neurobagel.org/data_models/dictionaries/).
- A dataset-level catalog maps `nb:Assessment`, `nb:Diagnosis` and `nb:Sex` to the catalog fields `available_assessments`, `available_diagnoses` and `available_sex` — [utility.py](https://github.com/neurobagel/api/blob/main/app/api/utility.py).

#### DANDI (verified)
- `AssetsSummary` has these fields: numberOfBytes, numberOfFiles, numberOfSubjects, numberOfSamples, numberOfCells, numberOfSessions ("derived from BIDS or DANDI `ses-*` tokens"), dataStandard, approach, measurementTechnique, variableMeasured (List[str]) and species (List[SpeciesType]) — [dandischema models.py](https://github.com/dandi/dandi-schema/blob/master/dandischema/models.py).
- Its controlled-term types (SpeciesType, Anatomy "UBERON or other identifier", ApproachType, MeasurementTechniqueType, Disorder "Biolink, SNOMED, or other identifier", with `dxdate`) all share `identifier` (URL or CURIE) and `name` (1–150 chars). The schema has no NCBITaxon pattern; species is an untyped identifier — [models.py](https://github.com/dandi/dandi-schema/blob/master/dandischema/models.py).
- Dandiset `about` takes Disorder, Anatomy or GenericType. `keywords` and `studyTarget` are free strings; `ethicsApproval` is a list — [models.py](https://github.com/dandi/dandi-schema/blob/master/dandischema/models.py).
- New Gmail-style `key:value` operators (commits June–Aug 2026):
  - created_before/after, modified_before/after, published_before/after
  - species, approach, technique, standard, variable, owner
  - `species`, `approach`, `technique`, `standard` and `variable` are case-insensitive regex matches on `$.assetsSummary.species[*].name` etc. Multiple clauses are ANDed on one Version row.
  - Unknown operators return an error with "did you mean?" suggestions.
  - Sources: [parser.py](https://github.com/dandi/dandi-archive/blob/master/dandiapi/api/services/search/parser.py), [filters.py](https://github.com/dandi/dandi-archive/blob/master/dandiapi/api/services/search/filters.py), [commit history](https://github.com/dandi/dandi-archive/commits/master/dandiapi/api/services/search/parser.py). A DANDI blog post "Find Dandisets Faster with Advanced Search" (2026-06-11) exists in search results, but its preview URL returned 404 when fetched — [preview URL](https://deploy-preview-118--dandi-about.netlify.app/blog/2026/06/11/find-dandisets-faster-with-advanced-search/).

#### EBRAINS KG Search / openMINDS (verified)
- The live Search UI facets for the Dataset type are Bookmarks, Types, Accessibility, Services, Species, Experimental approach, Studied brain region, Techniques, Keywords and Content types. All are `type: list`, most `isFilterable: true`, all `isHierarchical: false`. Other result types are Contributor, (Meta)Data Model, Model, Project, Software, Web service and Others — [search.kg.ebrains.eu/api/settings](https://search.kg.ebrains.eu/api/settings).
- Filters can be set as URL parameters, e.g. `facet_Dataset_speciesFilter[0]=Mus musculus`. The KG holds more metadata (e.g. age) than the Search UI shows — [EBRAINS FENS 2022 wiki](https://wiki.ebrains.eu/bin/view/Collabs/fens-2022-poster-examples/). This source is dated 2022.
- openMINDS `DatasetVersion` has dataType (raw/derived/simulated), experimentalApproach, preparationType, protocol (BehavioralProtocol/Protocol), studiedSpecimen (Subject, SubjectGroup, TissueSample and their States), studyTarget and technique. studyTarget and technique link via `_linkedCategories` rather than fixed types — [openMINDS datasetVersion schema](https://github.com/openMetadataInitiative/openMINDS_core/blob/main/schemas/products/datasetVersion.schema.tpl.json).
- SubjectState has ageCategory, attribute and handedness (controlled terms), so age category is per subject state — [openMINDS subjectState](https://github.com/openMetadataInitiative/openMINDS_core/blob/main/schemas/research/subjectState.schema.tpl.json).

#### CONP portal (verified, code from Nov 2026)
- Sidebar facets: Origin, Access, How to get it, Modality, Format, Linked publication and Used in research. Sorts: relevance, added, updated, name, subjects, size, has Evidence publication, most used in research — [conp-portal dataset_list.py](https://github.com/CONP-PCNO/conp-portal/blob/master/app/search/dataset_list.py).
- Metadata is DATS JSON. The portal also exposes SPARQL examples over DATS converted to schema.org, filtering on license (`sdo:license/sdo:name`), encoding format (MINC) and access mode — [conp-portal queries.py](https://github.com/CONP-PCNO/conp-portal/blob/master/app/search/queries.py). "Uniform search queries based on DATS" are also described in [Harding et al. 2023 Sci Data](https://www.nature.com/articles/s41597-023-01946-1).

#### NeuroVault (model verified; UI facets not verified)
- StatisticMap has `modality` (choice list in the table), `map_type`, `analysis_level`, `number_of_subjects`, `cognitive_paradigm_cogatlas` and `cognitive_contrast_cogatlas` (foreign keys to Cognitive Atlas) and `target_template_image`. Collection has subject_age_mean/min/max, handedness, proportion_male_subjects, inclusion_exclusion_criteria, group_comparison, scanner_make, field_strength, software_package and smoothing_fwhm — [NeuroVault models.py](https://github.com/NeuroVault/NeuroVault/blob/master/neurovault/apps/statmaps/models.py).
- Programmatic filtering (nilearn `fetch_neurovault`) works on these metadata keys, e.g. `map_type='Z map'`, `modality='fMRI-BOLD'` — [nilearn docs](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_neurovault.html).

#### NDA, NITRC-IR, LONI IDA, COINS, XNAT (documentation-based, older sources)
- **NDA.** The Query Tool builds filters from "data from labs, data from papers, data dictionary" and can query by concepts or by GUIDs. Filters of the same type are ORed; different filter types in the cart are ANDed — [NIMH NDA data-access webinar 2022](https://www.nimh.nih.gov/news/media/2022/nimh-data-archive-nda-data-access-webinar). "Query by GUID" finds all data for a subject across collections — [NDA query by GUID](https://nda.nih.gov/query_by_guid.html).
- **NITRC-IR.**
  - Uses "a standard XNAT data model for subjects and sessions" and searches across all projects the user may access.
  - Example: a search for "all subjects who are 10 years old" returned 270 matches from several projects.
  - Curators encode scan parameters, units and demographics identically across projects, which is what makes cross-project search work.
  - Source: [Kennedy et al., NITRC Image Repository, NeuroImage 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4651733).
  - A NITRC R client notes that age sits in subject data for some projects and in `xnat:mrSessionData/age` for others — [nitrcbot](https://rdrr.io/github/neuroconductor/nitrcbot/src/R/nitrc_demographics.R).
- **LONI IDA.**
  - Simple Search covers subjects and raw images and shows age, sex, modality and series description.
  - Advanced Search adds processed images and image criteria (weighting, pulse sequence, acquisition type) — [IDA User Manual 2024](https://ida.loni.usc.edu/services/Menu/PDF/IDA_User_Manual.pdf).
  - ADNI's Advanced Search also covered sex, diagnosis group, clinical scores (MMSE, CDR, NPI, FAQ), visit and study date — [ADNI IDA instructions (2012)](https://adni-lde.loni.usc.edu/wp-content/uploads/2012/08/instruction-how-to-use-IDA.pdf).
- **COINS.** The Query Builder supports "ad-hoc querying of single and cross-site studies for assessments, scans and demographics" — [COINS help](https://coins.zendesk.com/hc/en-us/categories/200130467-Query-Builder). Data Exchange filters are Studies, Subjects, Assessments and MR; ages are rounded down to whole integers and bounds are inclusive — [COINS Browse Available Data](https://coins.zendesk.com/hc/en-us/articles/204266439-Browse-Available-Data). The NKI-Rockland COINS page says it has not been maintained since July 2025 — [INDI COINS page](https://fcon_1000.projects.nitrc.org/indi/enhanced/COINS.html).
- **XNAT.** REST functions let a client discover XML-Schema data types, e.g. that a subject has a gender and that age is defined on `xnat:mrSessionData` — [PyXNAT, Frontiers 2013](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2013.00025/full).

#### Historical and aggregator systems
- **OpenfMRI (2013).** A "minimalist approach to metadata"; the Cognitive Atlas was used to annotate datasets with psychological constructs — [Poldrack et al. 2013](https://www.frontiersin.org/articles/10.3389/fninf.2013.00012/text).
- **SciCrunch/dkNET/NIF.** Search terms are matched to community ontologies (NIFSTD) to add synonyms, acronyms and related terms, and users can inspect or remove expansions. Faceted drill-down works within a source — [dkNET tutorial](https://www.slideshare.net/slideshow/dk-net-webinar-tutorial-pen/63160043), [SciCrunch ICBO 2014](https://ceur-ws.org/Vol-1327/icbo2014_paper_36.pdf).
- **DataMed (bioCADDIE).** An ingestion pipeline maps repository metadata to DATS (about 90% accuracy). Search uses Elasticsearch plus NLP/terminology query expansion (inferred AP 0.2033, P@10 0.6022) — [DataMed paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7378878), [JAMIA 2018](https://dbpia.nl.go.kr/jamia/article/25/3/300/4807508). Lexicon-based expansion is cheaper than Rocchio feedback — [Bouadjenek & Verspoor 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5737205/).
- **Google Dataset Search.** Indexes schema.org metadata, not file contents. Filters: download format, usage rights, last updated, topic, free-to-use. Values come from provider markup — [Google News Initiative quickstart](https://newsinitiative.withgoogle.com/resources/lessons/dataset-search-quickstart-guide) (2020, possibly outdated).
- **brainlife.io.** The docs only say "Datasets mapped by DataLad can [be] searched and imported directly"; no facets are documented — [brainlife docs](https://brainlife.io/docs/user/datasets/).

### Inferences
- OpenNeuro's diagnosis, species, longitudinal and domain facets depend on a sparsely filled, free-text curator record. Its modality, age, sex, task and PET facets come from BIDS files, so they are better populated.
  - This argues for a criteria registry that records the evidence source and its reliability per criterion, e.g. "participants.tsv column" vs "curated dataset-level string" vs "README text".
- Vocabulary depth varies widely:
  - OpenNeuro diagnosis is a 5-item list with no hierarchy.
  - Neurobagel uses SNOMED CT, which has a hierarchy, but the API matches exact CURIEs, and the code joins multiple diagnosis values as required triples (AND). I found no evidence of subsumption reasoning in the query.
  - EBRAINS facets are declared `isHierarchical: false`.
  - None of the verified systems expands a parent term to its children at query time. Only dkNET/NIF and DataMed document ontology-based query expansion.
- DANDI's `variableMeasured` is a bare string list (e.g. NWB neurodata types), so a search for "spike times" depends on string matching rather than vocabulary.

### Gaps
- I did not verify the OpenNeuro UI copy beyond the source code (for example, which facets are hidden behind "Show more").
- I could not fetch the DANDI web UI facet component (`SearchParameters.vue` was not in the expected path).
- EBRAINS species, technique and approach facet values were not enumerated: the search API POST endpoints returned 400/404 without an auth token.
- No current primary documentation for NDA Query Tool fields, NITRC-IR advanced search field list, INDI/FCP (it is a static listing of sites with phenotypic CSVs as far as I know; unverified), or XNAT's core subject schema fields.
- No primary source found on DataMed facet design or the current state of DataMed (it appears inactive; unverified).
- NeuroVault website search facets were not verified.

## Q2. How do they handle institutions/affiliations and species?

### Takeaway
Only openMINDS/EBRAINS models organisation hierarchy explicitly (`Organization.hasParent` plus persistent identifiers). DANDI stores ROR IDs on affiliations but offers no institution search. OpenNeuro, Neurobagel and CONP have no institution facet; CONP shows a free-text DATS "origin" and facets only Canadian vs International. Species is usually a short list or free text matched against NCBI Taxonomy names. BIDS defaults a missing species to *Homo sapiens*.

### Cited Findings
- **openMINDS Organization.** Has `acronym`, `alternateName`, `countryOfFormation` (controlled SovereignState), `digitalIdentifier` ("all globally unique and persistent digital identifiers"), `hasParent` ("Add all parent organizations"), `type` (legal entity or organizational unit), `membership` and `location` — [openMINDS organization schema](https://github.com/openMetadataInitiative/openMINDS_core/blob/main/schemas/actors/organization.schema.tpl.json).
- **DANDI.** `Affiliation` and `Organization` have `identifier: Optional[RORID]` with pattern `^https://ror.org/[a-z0-9]+$` and a required `name` — [dandischema models.py](https://github.com/dandi/dandi-schema/blob/master/dandischema/models.py). Search operators do not include affiliation; `owner:` matches the archive user (GitHub login, email or name) — [filters.py](https://github.com/dandi/dandi-archive/blob/master/dandiapi/api/services/search/filters.py).
- **OpenNeuro.** Search exposes `authors` (fuzzy match on `latestSnapshot.contributors.name`). Contributors have `name, givenName, familyName, orcid, contributorType`, with no affiliation field in the search payload. The Metadata record has `seniorAuthor` and `grantFunderName` strings — [build-search-query.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/resolvers/build-search-query.ts), [use-search-results.tsx](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-app/src/scripts/search/use-search-results.tsx), [metadata.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/models/metadata.ts).
- **CONP.** `origin` is built from DATS `extraProperties` categories starting `origin_` (e.g. origin_institution) and joined as text. The facet only distinguishes "Canadian" (`conpStatus` conp/canadian) and "International" (external) — [conp-portal dataset_list.py](https://github.com/CONP-PCNO/conp-portal/blob/master/app/search/dataset_list.py), [models.py](https://github.com/CONP-PCNO/conp-portal/blob/master/app/search/models.py).
- **BIDS species.** `species` "SHOULD be a binomial species name from the NCBI Taxonomy (for example, `homo sapiens`, `mus musculus`, `rattus norvegicus`)". If absent, "the participant is assumed to be `homo sapiens`". Non-human participants get `strain` and `strain_rrid` (RRID, e.g. `RRID:IMSR_JAX:000664`) — [BIDS participants.tsv spec](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/data-summary-files.html).
- **OpenNeuro species.** Options are Human, Rat, Mouse, Other, matched fuzzily against free-text `metadata.species`. "Other" = must_not Human/Rat/Mouse. "Human" is added as a `should` clause, not a `filter` — [build-search-query.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/resolvers/build-search-query.ts).
- **DANDI species.** `SpeciesType` has `identifier` (URL or CURIE) and `name`, with no NCBITaxon enforcement in the schema. The `species:` operator regex-matches the `name` — [models.py](https://github.com/dandi/dandi-schema/blob/master/dandischema/models.py), [filters.py](https://github.com/dandi/dandi-archive/blob/master/dandiapi/api/services/search/filters.py).
- **EBRAINS species.** "Species" is a non-hierarchical list facet; URL example `facet_Dataset_speciesFilter[0]=Mus musculus` — [KG settings](https://search.kg.ebrains.eu/api/settings), [EBRAINS wiki](https://wiki.ebrains.eu/bin/view/Collabs/fens-2022-poster-examples/).

### Inferences
- OpenNeuro's "Human" species option is a `should` clause in a bool query that always also has a `filter` (synthetic flag), so it is effectively optional. The OpenNeuro "Human" filter therefore appears to boost rather than restrict. This is consistent with BIDS's "assume homo sapiens when absent" default, but it is my reading of Elasticsearch semantics, not documented behaviour.
- For an institution criterion, the closest prior art is openMINDS (`hasParent`) plus ROR (which itself publishes parent/child relationships). Searching "datasets from Stanford" would need affiliations from README/paper text resolved to ROR IDs, then roll-up via parent relations. No surveyed repository does this at query time.

### Gaps
- I did not verify whether DANDI's species values in practice are NCBITaxon URLs (I believe so, but did not check live data).
- I did not check whether the EBRAINS KG resolves organisations to ROR, or whether ROR parent relations are used anywhere in KG Search.
- No information found on Neurobagel institution or affiliation metadata beyond dataset-level `authors`, `homepage` and `repository_url` fields — [OpenAPI DatasetQueryResponse](https://openneuro.neurobagel.org/node/docs).

## Q3. How do they handle missing metadata?

### Takeaway
Most systems silently exclude records that lack a value when a filter on that field is applied. Two exceptions:
- CONP explicitly buckets missing values as "Not stated".
- BIDS defines `n/a` and a species default.

Neurobagel annotates missing values at data-dictionary time (`MissingValues`), returns all available properties via OPTIONAL patterns, and excludes subjects lacking a filtered property.

### Cited Findings
- **Neurobagel annotation.** "Missing values are allowed for any phenotypic variable (column)" except identifiers, and are declared under `MissingValues`. A participant has an assessment if at least one item column is not a missing value — [Neurobagel dictionaries](https://neurobagel.org/data_models/dictionaries/).
- **Neurobagel query semantics.** Code comment: "We need OPTIONAL {} wrappers around subject variable triples … because not all properties/edges will exist for every subject-session". Filters, however, are required triple patterns inside `FILTER EXISTS { … }`, e.g. `?phenotypic_session nb:hasAge ?age. FILTER (?age >= x)` — [neurobagel api utility.py](https://github.com/neurobagel/api/blob/main/app/api/utility.py).
- **Neurobagel output.** Results show empty values as "nan". Aggregate-only nodes show "protected" in subject- and session-level columns — [query tool docs](https://neurobagel.org/user_guide/query_tool/).
- **CONP.** The placeholder set `{'', 'n/a', 'na', 'none', 'unknown', 'not specified', 'other', 'dataset', 'null', '-'}` is dropped from modality/format labels. The Access facet falls back to "Not stated" — [conp-portal dataset_list.py](https://github.com/CONP-PCNO/conp-portal/blob/master/app/search/dataset_list.py).
- **OpenNeuro subject count.** The script filter returns false when `summary.subjects` is empty. Age and sex filters are ES range/match queries on `subjectMetadata`, so datasets without participants.tsv values cannot match — [build-search-query.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/resolvers/build-search-query.ts).
- **BIDS.** "Missing values are written as `n/a`". Species absent means homo sapiens — [BIDS spec](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/data-summary-files.html).
- **NeuroVault.** Most fields are optional and "only exist if entered at upload". A Neurostars user reports that per-contrast Cognitive Atlas fields "are not filled by most users" — [Sci Rep 2022 / NeuroVault metadata](https://www.nature.com/articles/s41598-022-10710-1.pdf), [Neurostars thread](https://neurostars.org/t/accessing-collection-metadata-via-neurovault-website/5736). The Neurostars report is user-reported, not official.
- **XNAT/NITRC-IR.** Age lives in different places in different projects; NITRC-IR relies on curators to harmonise demographics — [nitrcbot](https://rdrr.io/github/neuroconductor/nitrcbot/src/R/nitrc_demographics.R), [Kennedy et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4651733).
- **COINS.** Data Exchange filters "return results for any study for which the attribute is available" — [COINS help](https://coins.zendesk.com/hc/en-us/articles/204266439-Browse-Available-Data).

### Inferences
- There are three distinct missing-data states worth separating in a criteria registry:
  - not recorded anywhere (unknown)
  - explicitly recorded as missing (`n/a`, Neurobagel `MissingValues`)
  - withheld for privacy (Neurobagel "protected")
- Of the verified systems, only CONP surfaces "not stated" to users as a selectable bucket. None offer an "include unknowns" toggle.
- An NL search engine that labels datasets as "unknown" for a criterion, rather than excluding them, would go beyond what any surveyed portal does.

### Gaps
- I did not find documentation of how EBRAINS KG Search or DANDI treat entities lacking a facet value. Inferred: they are simply absent from that facet bucket.

## Q4. Which repositories support participant-level queries (find subjects across datasets) vs dataset-level?

### Takeaway
Participant-level, cross-dataset search exists in:
- Neurobagel (POST /subjects; returns per-session rows grouped by dataset)
- NITRC-IR/XNAT (subject/session search across projects)
- NDA (GUID-based subject queries across collections)
- COINS (cross-study subject filters)
- LONI IDA (subject and image search, within a chosen study)

OpenNeuro, DANDI, EBRAINS KG Search, CONP, NeuroVault and Google Dataset Search are dataset-level. OpenNeuro's age and sex filters use participant rows but still return datasets.

### Cited Findings
- **Neurobagel.** Criteria apply to participants. "Each row corresponds to a single matching subject session". Dataset-level columns include `NumMatchingSubjects` — [Neurobagel query tool](https://neurobagel.org/user_guide/query_tool/). Separate `/datasets` (returns `num_matching_subjects`, `dataset_total_subjects`, `records_protected`) and `/subjects` endpoints exist — [OpenNeuro node OpenAPI](https://openneuro.neurobagel.org/node/docs).
- **Neurobagel on OpenNeuro.** api.neurobagel.org now redirects to an "openneuro" node, i.e. Neurobagel hosts a participant-level index of (annotated) OpenNeuro datasets — [openneuro.neurobagel.org/node/docs](https://openneuro.neurobagel.org/node/docs) (observed redirect from https://api.neurobagel.org/docs).
- **NITRC-IR.** Supports work "at a level 'above' the individual projects", e.g. all 10-year-old subjects across projects — [Kennedy et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4651733).
- **NDA.** "Query by GUID" / "Find All Subject Data" retrieves a subject's data across NDA collections (authenticated users only) — [NDA query by GUID](https://nda.nih.gov/query_by_guid.html), [NIMH NDA webinar](https://www.nimh.nih.gov/news/media/2022/nimh-data-archive-nda-data-access-webinar).
- **COINS.** "Querying of single and cross-site studies for assessments, scans and demographics" with an anonymised export — [COINS Query Builder](https://coins.zendesk.com/hc/en-us/categories/200130467-Query-Builder).
- **OpenNeuro.** ES range on `summary.subjectMetadata.age` and match on `subjectMetadata.sex` (participant rows), but the query returns datasets — [build-search-query.ts](https://github.com/OpenNeuroOrg/openneuro/blob/master/packages/openneuro-server/src/graphql/resolvers/build-search-query.ts).

### Inferences
- OpenNeuro's age filter has no joint constraint: "female AND age 60–70" matches a dataset that has any female and any 60–70-year-old, not necessarily the same person. The range query also uses relation INTERSECTS on each subject's age. Neurobagel avoids this because all conditions bind to the same `?phenotypic_session`.
- This joint-constraint distinction is worth encoding in a registry, as a criterion property such as "participant-scoped vs dataset-scoped".

### Gaps
- I did not verify how LONI IDA handles cross-study subject search (the manual implies per-study selection first).
- INDI/FCP has no search engine to my knowledge; I found no source.

## Q5. Documented natural-language or LLM-based search over neuroimaging datasets (2023–2026)

### Takeaway
Only a few LLM search interfaces exist, and most are prototypes or agent tools rather than production portal search:
- Neurobagel's GSoC 2024 "query-tool-ai" (NL to Neurobagel API URL, via LangChain + Ollama; last pushed Aug 2024).
- The Neurosift AI agent for DANDI (lexical plus semantic search, from Oct 2024).
- INCF special-interest projects.
- Several community MCP servers (2025–2026) wrapping OpenNeuro, DANDI and NeuroVault APIs.

No official OpenNeuro LLM search was found. DataMed's NLP query expansion is the main pre-LLM precedent.

### Cited Findings
- **Neurobagel query-tool-ai.** README: aims to make search "more user-friendly by adding an LLM style chatbot interface … that will be able to interpret the user prompts and initiate the API calls". Requires `NB_API_QUERY_URL`; runs on an Ollama base image. Repo last pushed 2024-08-26 — [github.com/neurobagel/query-tool-ai](https://github.com/neurobagel/query-tool-ai).
  - The developer write-up: `extract_information` uses LangChain plus LLMs (ChatOllama open models) to parse the NL query; FastAPI backend; future work is reducing hallucinations — [GSoC 2024 write-up](https://raya679.github.io/gsoc), [codebase notes](https://raya679.github.io/gsoc/codebase).
- **INCF projects.**
  - "Natural language interface for querying federated research data" (the Neurobagel chatbot) — [INCF SIG page](https://incf.org/sig/natural-language-interface-querying-federated-research-data).
  - A companion LLM-assisted tool for annotating data dictionaries (to reduce manual Neurobagel annotation) — [INCF LLM annotation project](https://incf.org/sig/llm-assisted-tool-annotate-research-data-machine-understandable-semantic-data-dictionaries).
- **Neurosift AI agent (DANDI).** Preliminary public version (Oct 2024) with "Lexical search, semantic search, search for neurodata types, search for column names in Units tables, the data structure of individual NWB files"; early testing used gpt-4o. Neurosift also offers semantic search by pasting an abstract — [Neurosift blog Oct 2024](https://magland.github.io/neurosift-blog/posts/2024-10.html). A Scientific Data 2025 citation for this was reported by a third party and not verified.
- **LLMs for neuroimaging-publication metadata.** LLMs annotate neuroimaging publications with metadata at accuracy 0.91–0.97 (zero-shot), with errors comparable to human errors. The authors argue keyword search works at the "syntactic" level and loses semantic context — [Frontiers in Neuroinformatics 2025 (PMC12405296)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12405296/).
- **Community MCP servers** (descriptions from GitHub metadata only; not evaluated):
  - `ykshah1309/neuro-research-discovery-mcp`: "bridging OpenNeuro, NeuroVault, and PubMed for AI-agent-driven neuroimaging research discovery", pushed 2026-06
  - `QuentinCody/openneuro-mcp-server`, pushed 2025-05
  - `stanczykwyatt/brainhound`: "Search and read public neuroscience data from your coding agent: DANDI, OpenNeuro, NeuroMorpho, Allen and BrainGlobe", pushed 2026-11
  - Sources: [neuro-research-discovery-mcp](https://github.com/ykshah1309/neuro-research-discovery-mcp), [openneuro-mcp-server](https://github.com/QuentinCody/openneuro-mcp-server), [brainhound](https://github.com/stanczykwyatt/brainhound)
- **Third-party LLM tool wrappers** (ToolUniverse) expose OpenNeuro modality search with options MRI/EEG/iEEG/MEG/PET — [ToolUniverse OpenNeuro tools](https://zitniklab.hms.harvard.edu/ToolUniverse/tools/openneuro_tools.html).
- **Pre-LLM precedent.** DataMed used NLP/terminology query expansion over DATS-normalised metadata — [DataMed](https://pmc.ncbi.nlm.nih.gov/articles/PMC7378878). dkNET/SciCrunch use ontology-based synonym expansion — [ICBO 2014](https://ceur-ws.org/Vol-1327/icbo2014_paper_36.pdf).

### Inferences
- The documented NL approaches translate text into the existing structured API: Neurobagel query-tool-ai generates an API URL, and MCP servers call repository APIs. So they inherit each backend's vocabulary and missing-data limits.
- No surveyed system documents an NL layer that reasons about where evidence lives (participants.tsv vs README vs paper) or reports "not recorded" per criterion. That is the gap a criteria registry would fill.

### Gaps
- No peer-reviewed evaluation of LLM-based dataset search over OpenNeuro/BIDS was found.
- I did not check whether the EBRAINS KG has added an LLM assistant (2025–2026) or whether OpenNeuro has a roadmap for one.
- Neurosift's current status and model were not verified beyond the 2024 blog.
