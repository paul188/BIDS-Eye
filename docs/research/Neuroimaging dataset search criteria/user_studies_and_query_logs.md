# How scientists search for and choose research datasets: user studies and query logs (with emphasis on neuroimaging)

Scope note: about 25 search, fetch and full-text extraction steps. Full texts were read for Gregory et al. 2020 (arXiv preprint PDF), the bioCADDIE reference-standard paper (Cohen et al. 2017), the SIB/Teodoro bioCADDIE participant paper, the DATS paper (Sansone et al. 2017), the OpenNeuro eLife paper (Markiewicz et al. 2021), Borghi & Van Gulick 2018, the European Data Portal Analytical Report 18 (Kacprzak, Simperl et al. 2020) and the Google Dataset Search corpus paper (Benjelloun et al. 2020). The Kacprzak et al. 2019 Journal of Web Semantics paper itself could not be fetched (HTTP 403 and connection refused). Its findings are taken from its abstract (via search results) and from the authors' own summary in the 2020 EDP report.

---

## 1. Neuroimaging and neuroscience surveys on data sharing and reuse: what drives dataset choice, what information is needed, what blocks discovery

### Takeaway
I found **no published survey that asks neuroimaging researchers how they search for or choose a dataset to reuse**. The neuroimaging surveys that exist (Borghi & Van Gulick 2018, N=144; Paret et al. 2022, N=283) are about data management, sharing and open-science practice. Evidence on reuse comes from OpenNeuro's analysis of published reuses. For selection criteria, the closest strong evidence is the cross-disciplinary Gregory et al. 2020 survey (n=1,677, with a medicine subset of 158 single-discipline respondents).

### Cited Findings
- **Borghi & Van Gulick 2018 (PLOS ONE 13(7): e0200562, doi:10.1371/journal.pone.0200562).** Online survey of active MRI researchers, run in summer 2017, N=144. Main findings:
  - "neuroimaging data is acquired in multifarious forms, transformed and analyzed using a wide variety of software tools."
  - Research data management practices "vary considerably both within and between research groups, with trainees reporting less consistency than faculty."
  - Maturity ratings were "relatively high during the data collection and analysis phases ... and significantly lower during the data sharing phase."
  - The survey defined "data sharing" broadly, from "coordinate tables published in the neuroimaging literature to raw and preprocessed datasets deposited in repositories alongside relevant code, stimuli, and documentation."
  - It did not ask how respondents search for or select datasets.
  - Sources: [Europe PMC abstract / PMC6047789](https://europepmc.org/article/PMC/PMC6047789); [figshare survey data](https://figshare.com/articles/Survey_data_on_research_data_management_practices_and_perceptions_of_MRI_researchers/5845656)
- **Paret, Unverhau, Feingold, Poldrack, Stirner, Schmahl & Sicorello 2022, "Survey on Open Science Practices in Functional Neuroimaging" (NeuroImage, doi:10.1016/j.neuroimage.2022.119306).**
  - Invitations went to authors of human fMRI studies published 2010–2020 (found via PubMed). 283 people completed the questionnaire.
  - "The majority of participants had experience with the sharing of primary neuroimaging data."
  - "Most of the participants were interested in implementing a standardized data structure such as BIDS in their labs."
  - "researchers with residence in the EU had a higher need for data governance."
  - It focuses on sharing and preregistration barriers, not on discovery or selection.
  - Source: [Europe PMC record](https://europepmc.org/search?query=DOI%3A%2210.1016%2Fj.neuroimage.2022.119306%22); [bioRxiv preprint 10.1101/2021.11.26.470115](https://doi.org/10.1101/2021.11.26.470115)
- **Markiewicz et al. 2021, "The OpenNeuro resource for sharing of neuroscience data" (eLife 10:e71774).**
  - Reuse is measured through published reuses. The authors "identified 165 publications that reused OpenNeuro datasets." "A total of 111 OpenNeuro datasets were reused at least once," and the most popular dataset (Poldrack et al. 2016) appeared in 28 reuses.
  - 31 of the 165 papers reused at least two datasets, up to a maximum of 40.
  - From May 2020 to April 2021, 406 TB were distributed.
  - Reuses were found with Google Scholar and CrossRef searches for "OpenNeuro". The authors call this an underestimate because it misses OpenfMRI-era papers.
  - Source: [eLife / PMC8550750](https://elifesciences.org/articles/71774)
  - Note: a search-engine summary said "99 datasets reused". The full text says 111, and I use 111.
- **OpenNeuro's own description of the corpus and its search index (Markiewicz et al. 2021).**
  - Datasets by modality: anatomical MRI 501, functional MRI 445, EEG 81, diffusion MRI 53, MEG 23, PET 10, iEEG 8, ASL 3.
  - "a machine-readable summary of BIDS metadata is collected by the BIDS validator and indexed with an ElasticSearch mapping." Metadata is also exposed via schema.org "which allows indexing by external resources such as Google Dataset Search."
  - The authors characterised dataset content by matching Cognitive Atlas concept and task terms against "titles and README files." This implies that task and construct are key descriptors, and that they often exist only in free text.
  - Source: [eLife 71774](https://elifesciences.org/articles/71774)
- **Access barriers named by OpenNeuro's authors.**
  - Restricted databases such as ADNI require "researchers to submit their scientific question for review."
  - Openness trades off against privacy. "data that are more easily or openly available will be more easily and readily reused."
  - Source: [eLife 71774](https://elifesciences.org/articles/71774)
- **Barriers in neurophysiology.** A scoping review on open neurophysiology data says datasets can be filtered "by task, population, or acquisition conditions" on platforms like OpenNeuro, or traced via data-availability statements. The authors built a searchable app because discovery was hard. — [Open Data In Neurophysiology: Advancements, Solutions & Challenges (PMC11247910)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11247910/) (summarised from search results, not read in full)
- **NIDM-Terms (neuroimaging annotation).** Searching across diverse neuroimaging datasets is described as difficult. It typically requires sufficient annotation of study variables to understand what was collected. — [NIDM-Terms, PMC10392125](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10392125/) (summarised from search results)
- **INCF survey (2022).** INCF ran a "global survey on data sharing barriers in neuroscience" from August 2022. I found no published results. — [INCF blog](https://nteu.incf.org/blog/global-survey-data-sharing-barriers-neuroscience-please-share)

### Inferences
- In the neuroimaging literature, the user need is mostly inferred from reuse patterns (methods development, multi-dataset aggregation) and from how repositories index (modality, task via the Cognitive Atlas, BIDS metadata). It has not been measured by asking searchers directly.
- Task and cognitive construct are the main descriptors OpenNeuro's own authors used to characterise content, yet they had to mine them from titles and READMEs. This suggests a search engine must extract these from free text instead of relying on structured fields.
- Multi-dataset reuse is common: 31 of 165 papers used two or more datasets, and one used 40. Some searches are therefore "find all datasets with property X" (recall-oriented aggregation), not "find the one best dataset."

### Gaps
- I found no survey or interview study of neuroimaging researchers about how they discover datasets (queries used, filters wanted, blocking factors). The same holds for OHBM, ReproNim, CONP, EBRAINS, DANDI and the NIMH Data Archive: I found nothing published on discovery behaviour or query logs. OpenNeuro, NDA and DANDI query logs do not appear to be published.
- INCF 2022 survey results were not found.
- Exact percentages from Borghi & Van Gulick (for example, which repositories respondents used) were not extracted. The paper does not address dataset selection.

---

## 2. Query-log analyses of dataset search engines: what fraction of queries mention geography, time, organism, measurement type?

### Takeaway
On open-government data portals, dataset queries are very short. Most are one word, with a mean of about 1.5 words on the European Data Portal. Time and place dominate data requests: temporal expressions are about 5x more frequent than in web search, and crowdsourced dataset queries had about 7x more geospatial and temporal mentions than portal queries. **No published query-log analysis exists for a biomedical or neuroimaging dataset search engine.** Google's 2020 paper analyses the datasets that appear in results, not query text.

### Cited Findings
- **Kacprzak, Koesten, Ibáñez, Blount, Tennison & Simperl 2019, "Characterising dataset search — An analysis of search logs and data requests" (Journal of Web Semantics 55:37–55).**
  - Logs from four national open data portals, plus a qualitative analysis of free-text data requests sent to one of them.
  - Portal search "appears to be used more for exploration than for finding a specific resource."
  - "geospatial and temporal attributes, as well as information on the required granularity of the data are the most common features in data requests."
  - Sources: [Southampton eprints 426734](https://eprints.soton.ac.uk/426734/); [univie eprints 7656](https://eprints.cs.univie.ac.at/7656) (abstract only; full text not reachable)
- **The same group's summary of that work (EDP Analytical Report 18, 2020):**
  - Most internal portal queries were in business and economy. External queries (from web search engines) were more diverse and "a larger percentage of external queries included question queries, which might be due to the increasing ability of large search engines to support natural language type queries."
  - "Queries including some indication of the time were almost five times more frequent than in web search (Nunes et al., 2008)."
  - Data requests "describe the data by using boundaries and restrictions about location, temporal information, specific data type and/or specific granularity (e.g., year/month/day)."
  - Source: [Kacprzak, Simperl et al., EDP Analytical Report 18 (2020)](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)
- **Kacprzak et al. 2018, "Characterising Dataset Search Queries" (WWW '18 Companion).**
  - Queries written by crowd workers describing a dataset need "were much longer than search queries for datasets on data portals."
  - They "contained seven times more mentions of geospatial and of temporal information."
  - Source: [univie eprints 7688](https://eprints.cs.univie.ac.at/7688) (abstract via search results)
- **Kacprzak et al. 2017, "A Query Log Analysis of Dataset Search" (ICWE 2017).** Data-portal queries differ from web search queries "in their length and structure." — [univie eprints 7700](https://eprints-dev5.cs.univie.ac.at/7700) (abstract via search results)
- **European Data Portal log analysis, April 2018 – June 2020 (EDP Analytical Report 18, 2020).**
  - For all three portal versions, query length had mode 1 and median 1. The mean was "close to 1.5 for all queries and close to 2 for unique queries."
  - The authors' reading: "users may be using the search box in a similar way to a facet ... rather than writing longer, more complex queries or ask questions."
  - Single-word queries contained less than 3% temporal, format or data-type terms, but more than 10% each of country names and other locations. Multi-word queries had somewhat more temporal and data-related keywords.
  - Facet-only searching was "more common than the two other strategies combined." Country and category were the most-used facets. "Format and licence are the least popular."
  - Most-used facet pairs, by portal version:
    - EDPv1, facet-only sessions: (Country, Category) 5.4%.
    - EDPv2, sessions using both keywords and facets: (Country, Category) 23.1%.
    - EDPv3: (Country, Category) about 17–19%.
  - 80% of a sample of 386 multi-word queries were in English.
  - Keyword-type coding scheme, following Kacprzak 2019:
    - Temporal: years 1000–2017, month names, week(ly)/month(ly)/year(ly)/day(ly).
    - Format: csv, pdf, xls, json, wfs, zip, html, api.
    - Data-related: data, dataset, average, index, graph, table, database, indice, rate, stat, map.
    - Countries.
    - Other locations.
  - Source: [EDP Analytical Report 18](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)
- **Carevic et al. 2020 (GESIS social-science portal, dataset vs publication search), as summarised in the EDP report.** Dataset queries are shorter than publication queries and contain "significantly more numerical digits." — [EDP Analytical Report 18](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)
- **Koesten 2017 interview study, as summarised in the EDP report.** "many users do not expect that the search functionality will be able to provide relevant data for longer and more specific queries and therefore issue short queries." — [EDP Analytical Report 18](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)
- **Benjelloun, Chen & Noy 2020, "Google Dataset Search by the Numbers" (ISWC 2020, LNCS pp. 667–682, doi:10.1007/978-3-030-62466-8_41).**
  - Corpus snapshot of 26 March 2020: 28M datasets from about 3,700 domains. Geosciences and social sciences make up 45.2% of datasets.
  - The paper analyses which datasets appeared in results, not query strings: "2.1M unique datasets from 2.6K domains appeared in the top 100 search results" over 14 days in May 2020. Biology and medicine had a larger share of results than of the corpus, attributed to COVID-19. Exact percentages appear only in a figure.
  - Metadata completeness (share of datasets with each field):
    - description 100%, title 100%
    - provider 84.59%
    - keywords 80.08%
    - URL 68.30%
    - temporal coverage 45.41%
    - data download 44.34%
    - spatial coverage 38.69%
    - date modified 37.46%
    - license 34.80%
    - date published 30.83%
    - isAccessibleForFree 3.04%
  - Sources: [arXiv 2006.06894 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.06894); [Google Research page](https://research.google/pubs/google-dataset-search-by-the-numbers/)

### Inferences
- The "time + place" dominance found in government-portal logs is unlikely to carry over directly to neuroimaging. In OpenNeuro the analogues are probably population or cohort, modality, task, and acquisition site or institution. Numeric constraints (age ranges, sample size, field strength) may play the role that years play in government data, consistent with Carevic's finding of more numeric digits in dataset queries.
- Very short queries reflect low user expectations as much as need (Koesten 2017). When people describe needs freely (crowdsourced, data requests, bioCADDIE competency questions), queries are much longer and carry several constraints. A natural-language interface can draw out this richer form.
- Facet use was dominated by two "where/what" facets (country, category), and licence and format facets were rarely used. Licensing may matter for final selection (see section 5) while rarely being used as a search filter.

### Gaps
- I found no published query-log analysis for DataMed, OpenNeuro, NeuroVault, DANDI, NDA, ICPSR or DataONE that gives the share of queries mentioning organism, disease, measurement type or modality.
- Google has not published query-level statistics (fractions mentioning geography or time) for Dataset Search in the paper reviewed.
- Exact Kacprzak 2019 percentages for each portal (for example, the share of queries with temporal terms) could not be verified because the full text was not reachable.

---

## 3. bioCADDIE / DataMed: the DATS model and how the 2016 challenge queries were structured

### Takeaway
bioCADDIE derived its queries from stakeholder use cases turned into "competency questions." Across the 30 retained questions, the four most frequent constraint types were **data type, disease, biological process and organism**, followed by permissions, data format and quality proxies. The challenge queries were templated natural-language sentences combining two or three of these constraints. Systems had to parse natural language, which the organisers called an "unexpected" challenge.

### Cited Findings
- **Query construction (Cohen, Roberts, Gururaj, Chen, Pournejati, Alter, Hersh, Demner-Fushman, Ohno-Machado & Xu 2017, "A publicly available benchmark for biomedical dataset retrieval: the reference standard for the 2016 bioCADDIE dataset retrieval challenge", Database bax061).**
  - Use cases came from user submissions and a bioCADDIE use-case workshop held in San Diego in March 2015. Working group members turned them into competency questions categorised by metadata element.
  - Questions involving PHI, or returning non-datasets (software, drug–drug interactions), were excluded.
  - "The most commonly occurring entity types in the remaining competency questions (n = 30) were data type, disease type, biological process and organism. In addition, a number of competency questions concerned characteristics of the data sets relating to permissions and data format. A smaller number of the questions concerned surrogate indicators of dataset quality, such as funding source or number of referring publications."
  - These were abstracted into three query templates of constraint sets, for example {'data type = x', 'biological process = y'}. Curators instantiated them into 137 candidate queries, of which 15 were chosen for testing plus 6 for training and guidelines.
  - Source: [Europe PMC PMC5737202](https://europepmc.org/article/PMC/PMC5737202); [Database bax061](https://academic.oup.com/database/article/4085942)
- **Corpus and judgments (Cohen et al. 2017).**
  - 794,992 dataset records from 20 repositories, harvested 24 March 2016. Largest sources: ClinicalTrials 192,500; BioProject 155,850; PDB 113,493; GEO 105,033; Dryad 67,455; ArrayExpress 60,881; Dataverse 60,303.
  - 20,184 judgments for 15 queries: about 4% definitely relevant, 15% partially relevant, 81% not relevant.
  - Inter-judge kappa was 0.67. "Judges often looked beyond metadata to linked PubMed articles" to decide relevance.
  - Source: [Database bax061](https://academic.oup.com/database/article/4085942)
- **Verbatim bioCADDIE example queries, with constraint types where given:**
  - "Find data on the NF-KB signaling pathway in MG (Myasthenia gravis) patients" [biological process + disease]. Keyword form: "NF KB signaling pathway Myasthenia gravis MG". — [Cohen et al. 2017, Table 3](https://europepmc.org/article/PMC/PMC5737202)
  - "Search for all data types related to gene TP53INP1 in relation to p53 activation across all databases" [gene + biological process]. Keyword form: "TP53INP1 p53 activation". — [Cohen et al. 2017, Table 3](https://europepmc.org/article/PMC/PMC5737202)
  - "Search for data on neural brain tissue in transgenic mice related to Huntington's disease" — [Teodoro et al. 2017, Database bax083, Table 3 (PMC5714153)](https://europepmc.org/article/PMC/PMC5714153)
  - "Search for gene expression datasets on photo transduction and regulation of calcium in blind D. melanogaster" — [Teodoro et al. 2017](https://europepmc.org/article/PMC/PMC5714153)
  - "Find data of all types on the regulation of DNA repair related to the estrogen signaling pathway in breast cancer patients across all databases" — [Teodoro et al. 2017](https://europepmc.org/article/PMC/PMC5714153)
  - "Search for protein aggregation and gene expression data regarding aging across all databases" — [Teodoro et al. 2017](https://europepmc.org/article/PMC/PMC5714153)
  - "Find data on T-cell homeostasis related to multiple sclerosis across all databases"; "Find protein sequencing data related to bacterial chemotaxis across all databases"; "Search for gene expression and genetic deletion data that mention CD69 in memory augmentation studies across all databases"; "Search for data of all types related to the ob gene in obese M[ice]" (truncated in source). — [Teodoro et al. 2017](https://europepmc.org/article/PMC/PMC5714153)
  - Full query list: Supplementary Table S1 of Teodoro et al. and [biocaddie.org benchmark data](https://biocaddie.org/benchmark-data).
- **Challenge setup and outcome (Roberts et al. 2017, "Information retrieval for biomedical datasets: the 2016 bioCADDIE dataset retrieval challenge", Database bax068, doi:10.1093/database/bax068).**
  - About 795k datasets and 15 test queries. 10 teams submitted 45 runs. Metrics were infNDCG, infAP, P@10 and others.
  - "The Challenge presented two unexpected subsequent challenges, namely, the need to parse natural language queries, and judgment sets that were sparsely populated."
  - Teams also received 30 additional unjudged training queries.
  - Sources: [bax068 DOI](https://doi.org/10.1093/database/bax068) (abstract text via search results); [Cohen et al. 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5737202/)
  - Note: the author order for bax068 comes from memory (Roberts K first; Hersh W and Xu H among co-authors). Verify before citing.
- **Relevance definition used by annotators (Teodoro et al. summary).**
  - Not relevant: the record has fewer than 50% of the key query concepts.
  - Partially relevant: it has all concepts but does not answer the question, or most but not all concepts.
  - Relevant: it has all key concepts and answers the question.
  - Source: [Teodoro et al. 2017](https://europepmc.org/article/PMC/PMC5714153)
- **What worked.** Approaches included MeSH-based query expansion (OHSU) and word-embedding expansion (SIB). The organisers expanded keywords through UMLS via DataMed's terminology server, because metadata is "terse" and uses varied terminology. — [OHSU, Database bax065 (PMC5737054)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5737054/); [Cohen et al. 2017](https://academic.oup.com/database/article/4085942)
- **DATS model (Sansone, Gonzalez-Beltran, Rocca-Serra et al. 2017, "DATS, the data tag suite to enable discoverability of datasets", Scientific Data 4:170059, doi:10.1038/sdata.2017.59).**
  - Core elements cover "what the dataset is about (using the Material entity), how it was produced and which information it holds (DataType, Information, Method, Platform, Instrument), where it can be found (DatasetDistribution, Access, Licence), when it was released and by whom was it created and funded (Person, Organization and their roles, Grant)."
  - An extended set covers life, environmental and biomedical domains.
  - It was derived top-down from use cases and competency questions, and bottom-up by mapping existing schemas. It is serialised as JSON schema and schema.org JSON-LD.
  - Source: [Sci Data / PMC5460592](https://www.nature.com/articles/sdata201759)
- **DATS JSON schema files.** The official repository lists, among others:
  - anatomical_part, biological_entity, disease, taxonomic_info, molecular_entity, treatment
  - study_group, dimension, data_acquisition, data_analysis, data_type, instrument
  - consent_info, data_use_condition, license, access, place, date_info
  - Source: [github.com/datatagsuite/schema](https://github.com/datatagsuite/schema)
- **DataMed papers.** Ohno-Machado et al. 2017, "Finding useful data across multiple biomedical data repositories using DataMed" (Nature Genetics 49:816–819, doi:10.1038/ng.3864). Chen et al. 2018, "DataMed – an open source discovery index for finding biomedical datasets" (JAMIA 25(3):300–308, doi:10.1093/jamia/ocx121). — listed in [DATS README](https://github.com/datatagsuite/README)

### Inferences
- The bioCADDIE template ({data type} + {biological process} + {disease} + {organism/gene}) maps closely onto a BIDS search schema:
  - data type → modality / BIDS datatype (anat, func, dwi, eeg, meg, pet)
  - biological process → task / cognitive construct (Cognitive Atlas)
  - disease → participant diagnosis / clinical group
  - organism → species
  - DATS anatomical_part, study_group, dimension, consent_info and data_use_condition correspond to BIDS-adjacent fields: brain region of interest, participants.tsv groups, measured variables, and licence/consent.
- About 81% of pooled candidates were non-relevant, and judges needed linked papers. Precision on multi-constraint queries is hard when metadata is thin. Checking each constraint separately (the "all key concepts" rule) is a reasonable relevance operationalisation for an NL search engine.
- The bioCADDIE queries were written by curators from templates, not mined from logs. They show realistic information needs, but their phrasing ("Search for ... across all databases") is synthetic.

### Gaps
- I did not retrieve the full list of 15 test queries (it is in a supplement) or the best-run scores from the overview paper.
- DATS extended-entity membership was inferred from the schema file list, not from the paper's figure.

---

## 4. How users phrase dataset queries (keyword vs natural language, length, entities)

### Takeaway
In-portal queries are overwhelmingly short keyword or facet-like strings (mode 1 word). When users can express their needs freely, as in data requests, crowdsourced descriptions, survey answers or competency questions, they use longer multi-constraint descriptions built from location/time or, in biomedicine, data type + process + disease + organism. Researchers also say data search is "specific" and parameter-driven, and that it often starts from a paper or a known source (known-item search).

### Cited Findings
- Portal queries:
  - EDP: mode and median 1 word, mean about 1.5 (about 2 for unique queries).
  - The large share of one-word queries "is consistent with results previously reported for UK open government data portals."
  - Source: [EDP Analytical Report 18](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)
- External (web-search-referred) dataset queries contain more question-form queries than internal portal queries. — [EDP Analytical Report 18, summarising Kacprzak et al. 2019](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)
- Crowdsourced dataset descriptions are "much longer" and have 7x more geospatial and temporal mentions than portal queries. — [Kacprzak et al. 2018](https://eprints.cs.univie.ac.at/7688)
- **Gregory, Groth, Scharnhorst & Wyatt 2020, "Lost or Found? Discovering Data Needed for Research" (Harvard Data Science Review 2(2), doi:10.1162/99608f92.e38165eb).** Survey of n=1,677 people from 105 countries, response rate 1.1%. 82% were researchers and 69% worked at universities.
  - Searching is described as specific: "Much of data searching is very specific. Participants rely on particular, known data repositories and sources."
  - Respondent quote: "I tend to search for data by specifying parameters e.g. geographical and date coverage, or by looking for data created by a specific organization. My literature searches are more general and don't have so many search filters applied." (ID 3688)
  - Literature is a gateway to data: "It is common for respondents to first find the literature, and then use the literature as a gateway to locating the data."
  - Quote: "I generally do not search for data blindly but I would normally know that it already exists through some previous interaction (reading scientific publication, personal communication)." (ID 679)
  - 30% reported no difference between finding literature and finding data. 52% said sometimes the same and sometimes different. 18% said always different.
  - Of respondents using general search engines, 38% found their searches successful or very successful, 55% had mixed success, and 7% were rarely or never successful.
  - "only 15% of researchers never make use of personal networks in data discovery."
  - "Roughly two thirds" often or occasionally find data serendipitously.
  - Source: [arXiv 1909.00464](https://arxiv.org/abs/1909.00464); [HDSR DOI](https://doi.org/10.1162/99608f92.e38165eb)
- bioCADDIE challenge queries were full natural-language requests, for example "Search for data on neural brain tissue in transgenic mice related to Huntington's disease." Participants reduced them to keyword forms, for example "TP53INP1 p53 activation." — [Cohen et al. 2017](https://europepmc.org/article/PMC/PMC5737202); [Teodoro et al. 2017](https://europepmc.org/article/PMC/PMC5714153)
- Dataset queries contain more numeric digits than publication queries. — [Carevic et al. 2020 via EDP Report 18](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)

### Inferences
- A natural-language engine for OpenNeuro should handle two populations:
  - Short keyword or known-item queries, such as a dataset name, task name, accession (ds00xxxx), or the author/lab of a paper. Gregory's "literature as gateway" finding suggests many searches start from a paper, so matching on paper titles, DOIs and author names is valuable.
  - Longer multi-constraint descriptions, in the style of bioCADDIE or Kacprzak's crowdsourced queries.
- Institutions and organisations show up as a search parameter ("data created by a specific organization"). This supports institution/lab entity handling.

### Gaps
- I found no evidence on the share of dataset queries that name institutions, species or diseases in any biomedical log.
- I found no study on phrasing behaviour with LLM/chat-based dataset search (2023–2026).

---

## 5. Which criteria matter most for choosing a dataset (general, and for neuroimaging specifically)

### Takeaway
The best quantitative evidence (Gregory et al. 2020) puts **collection conditions and methodology** first: 89% rate it important or extremely important. Next come processing/handling information and topical relevance, then ease of access (73%), source reputation (71%) and creator reputation (62%). For trust, transparency of collection methods and source reputation (81%) dominate. I found no neuroimaging-specific ranking (sample size, demographics, modality, task, quality, licence). Repository design and the bioCADDIE competency questions point to modality/data type, task, population/disease, permissions/licence and format as the key attributes.

### Cited Findings
- **Gregory et al. 2020, criteria for evaluating data for reuse (n=1,677):**
  - "Eighty-nine percent of respondents reported that information about data collection conditions and methodology was important or extremely important in their decisions; information about data processing/handling as well as topical relevance were also ranked highly."
  - "The ease (or difficulty) of accessing data is also very important to 73% of participants."
  - Data source reputation was important to 71%, and creator reputation to 62%.
  - For trust: 81% rated source reputation important versus 61% for creator reputation. "The transparency of data collection methods, followed by the reputation of the source and a minimum of errors are critical in trust development."
  - Source: [arXiv 1909.00464](https://arxiv.org/abs/1909.00464)
- **Gregory et al. 2020, discovery barriers:**
  - Discovering data is "sometimes challenging (73%) or even difficult (19%)."
  - Top obstacles: data not accessible (68% of researchers) and data spread across multiple locations (49%).
  - About one third cited inadequate search tools, lack of search skill, or non-digital data.
  - Source: [arXiv 1909.00464](https://arxiv.org/abs/1909.00464)
- **Gregory et al. 2020, uses and needs:**
  - 71% use data as the basis for a new study, and half use data for teaching.
  - 50% need more than one data type.
  - Medicine respondents needed out-of-discipline data less often (43%) than others.
  - Having "detailed and complete metadata or documentation" was associated with uses such as experimenting with new methods, integration and comparison.
  - Source: [arXiv 1909.00464](https://arxiv.org/abs/1909.00464)
- **bioCADDIE competency questions.** Beyond content constraints, questions also concerned "permissions and data format" and "surrogate indicators of dataset quality, such as funding source or number of referring publications." — [Cohen et al. 2017](https://europepmc.org/article/PMC/PMC5737202)
- **Licensing and access in neuroimaging.** OpenNeuro uses CC0 so that data is "maximally accessible." EU regulatory constraints limit such licences, and Paret et al. found EU researchers report a higher "need for data governance." — [Markiewicz et al. 2021](https://elifesciences.org/articles/71774); [Paret et al. 2022](https://doi.org/10.1016/j.neuroimage.2022.119306)
- **A separate cross-disciplinary survey.** Per a search-result summary, a survey of 3,257 respondents found "proper documentation, openness, and information on the usability of data" important when searching for existing datasets, and researchers in most disciplines struggled to find datasets to reuse. I could not verify the citation details; treat as unconfirmed. — [search summary pointing to PMC7359296 / Sci Data s41597-020-0569-5 (Gregory et al. 2020 dataset descriptor)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7359296/)
- **Koesten et al. 2020 mixed-methods study (as summarised in EDP Report 18).** Its dataset selection criteria (relevance, quality, usability) mirror Gregory's results. — [EDP Analytical Report 18](https://data.europa.eu/sites/default/files/analytical_report_18-characterising_data_search_edp.pdf)

### Inferences
- For OpenNeuro, "collection conditions and methodology" translates to acquisition details: modality, field strength or scanner, sequence, task/paradigm, and the participant population (N, age, diagnosis). These are the fields BIDS standardises in dataset_description.json, participants.tsv and sidecar JSON. They should be surfaced in result cards so users can judge relevance without opening the dataset, consistent with the bioCADDIE judges needing linked papers.
- "Reputation of the source/creator" and "linked publications" suggest showing the associated paper, authors and institution, and reuse or citation counts. Number of referring publications was itself a bioCADDIE quality proxy.
- Access ease (73%) and inaccessibility as the top barrier (68%) argue for showing licence, size, and download or DataLad availability up front. Logs nonetheless show licence is rarely used as a filter.

### Gaps
- I found no study that ranks neuroimaging-specific criteria (sample size, demographics, modality, task, data quality metrics such as MRIQC, licence) by importance to reusers.
- Gregory's figure-level percentages for search sources (literature vs search engines vs domain repositories) and for each evaluation criterion other than those quoted are only in figures and were not extracted.
- The medicine-subset breakdowns in Gregory et al. are in supplementary materials and were not examined.
