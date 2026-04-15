# Duplicate Candidates Report

## Summary

- Input rows:              2808
- Exact duplicates found:  173  (dropped from dedup output)
- Dedup output rows:       2635
- Near-dup candidates:     86  (flagged, not removed)

## Exact Duplicates (dropped)

- row 26 duplicates row 12: 'Are there any datasets that contain EEG data?'
- row 31 duplicates row 17: 'Find datasets funded by CI241/1-1.'
- row 72 duplicates row 17: 'Find datasets funded by CI241/1-1.'
- row 99 duplicates row 69: 'List datasets authored by Kevin King.'
- row 135 duplicates row 48: 'Find datasets funded by ERC-StG-2018-803370.'
- row 310 duplicates row 271: 'List datasets containing both anatomical MRI and diffusion MRI data.'
- row 428 duplicates row 167: 'Datasets that do not have a DOI.'
- row 435 duplicates row 285: 'Show datasets that include both anatomical MRI and field map data.'
- row 452 duplicates row 12: 'Are there any datasets that contain EEG data?'
- row 466 duplicates row 128: 'Are there any datasets that contain PET data?'
- row 475 duplicates row 442: 'Datasets funded by GRANT #2.'
- row 499 duplicates row 396: 'List datasets funded by GRANT #2.'
- row 538 duplicates row 357: 'Identify datasets that contain no functional MRI data.'
- row 559 duplicates row 393: "Search for datasets mentioning 'valence' in their name or description."
- row 567 duplicates row 128: 'Are there any datasets that contain PET data?'
- row 636 duplicates row 194: 'Find datasets funded by GRANT #1.'
- row 637 duplicates row 136: 'Find datasets that do not have a DOI.'
- row 686 duplicates row 402: 'Find datasets containing reactive task fMRI data.'
- row 692 duplicates row 459: 'Show datasets with PD license.'
- row 813 duplicates row 12: 'Are there any datasets that contain EEG data?'
- row 838 duplicates row 128: 'Are there any datasets that contain PET data?'
- row 853 duplicates row 409: 'Top 5 datasets with at least 2 sessions.'
- row 868 duplicates row 219: 'Find datasets containing both anatomical MRI and diffusion MRI.'
- row 882 duplicates row 128: 'Are there any datasets that contain PET data?'
- row 905 duplicates row 451: 'List the top 10 datasets ordered by the number of unique sessions.'
- row 933 duplicates row 681: 'Show datasets authored by Tyler M. Moore.'
- row 936 duplicates row 847: 'Find datasets funded by the Army Research Laboratory (W911NF-10-0-0002).'
- row 942 duplicates row 61: 'Find datasets containing both anatomical and functional MRI data.'
- row 985 duplicates row 73: "Search for datasets mentioning 'task' in the description."
- row 992 duplicates row 61: 'Find datasets containing both anatomical and functional MRI data.'
  … and 143 more

## Near-Duplicate Candidates (different SQL — manual review)

### 1. seq_ratio=0.9901  jaccard=0.6
- Row 812: Find datasets that have no functional MRI datatype.
- Row 863: Find datasets that have no functional MRI datatypes.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 2. seq_ratio=0.9888  jaccard=1.0
- Row 2306: Find the top 5 datasets with the highest number of subjects that have at least 2 sessions.
- Row 2542: Find the top 5 datasets with the highest number of subjects that have at least 3 sessions.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 3. seq_ratio=0.9857  jaccard=1.0
- Row 389: Are there any datasets that have participants with a specific diagnosis?
- Row 2470: Are there any datasets that have participants with specific diagnosis?
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 4. seq_ratio=0.9844  jaccard=1.0
- Row 34: Search for datasets mentioning 'fluid' in their name or description.
- Row 1678: Search for datasets mentioning 'fluid' in the name or description.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 5. seq_ratio=0.9844  jaccard=1.0
- Row 380: Search for datasets mentioning 'brain' in the name or description.
- Row 1124: Search for datasets mentioning 'brain' in their name or description.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 6. seq_ratio=0.9841  jaccard=1.0
- Row 168: Search for datasets mentioning 'participants' in the description.
- Row 2269: Search for datasets mentioning 'participants' in their description.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 7. seq_ratio=0.982  jaccard=0.8571
- Row 1985: Which datasets have at least 2 sessions per subject? List the top 5 by subject count.
- Row 2204: Which datasets have at least 2 sessions per subject? List the top 10 by subject count.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 8. seq_ratio=0.9796  jaccard=0.6
- Row 1638: Find all datasets released under the PDDL license.
- Row 2122: Find all datasets released under the PPDL license.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 9. seq_ratio=0.9787  jaccard=1.0
- Row 2287: Get the top 5 datasets with at least 4 sessions.
- Row 2531: Get the top 5 datasets with at least 2 sessions.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 10. seq_ratio=0.9744  jaccard=1.0
- Row 409: Top 5 datasets with at least 2 sessions.
- Row 889: Top 5 datasets with at least 4 sessions.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 11. seq_ratio=0.9735  jaccard=0.8
- Row 173: Are there any datasets that contain no anatomical MRI data?
- Row 646: Are there any datasets that contain anatomical MRI data?
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 12. seq_ratio=0.9722  jaccard=1.0
- Row 241: Retrieve datasets that include both anatomical MRI and functional MRI data.
- Row 464: Retrieve datasets that include both anatomical and functional MRI data.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 13. seq_ratio=0.9661  jaccard=0.8333
- Row 126: List the top 50 datasets with at least two distinct sessions.
- Row 1739: List the top 50 datasets with at least 3 distinct sessions.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 14. seq_ratio=0.966  jaccard=1.0
- Row 1160: Retrieve datasets that contain both anatomical MRI and intracranial EEG data.
- Row 1428: Retrieve datasets that contain both anatomical MRI and intracranial EEG.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 15. seq_ratio=0.9655  jaccard=0.5714
- Row 600: Find datasets funded by the Healthy Brain Network donor site.
- Row 1123: Find datasets funded by the Healthy Brain Network donors.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 16. seq_ratio=0.9624  jaccard=1.0
- Row 39: Find datasets that feature both anatomical MRI and diffusion MRI.
- Row 2409: Find datasets that feature both anatomical MRI and diffusion MRI data.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 17. seq_ratio=0.9612  jaccard=1.0
- Row 271: List datasets containing both anatomical MRI and diffusion MRI data.
- Row 2202: List datasets containing both anatomical MRI and diffusion MRI.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 18. seq_ratio=0.9612  jaccard=0.4
- Row 2013: Find datasets that do not list a Manufacturer in their objects.
- Row 2658: Find datasets that do not list a manufacturer in their object files.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 19. seq_ratio=0.9593  jaccard=0.625
- Row 8: Find datasets containing both anatomical MRI and field maps.
- Row 434: Find datasets containing both anatomical MRI and field map files.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`

### 20. seq_ratio=0.9587  jaccard=1.0
- Row 144: Identify datasets that include both anatomical MRI and EEG data.
- Row 1493: Identify datasets that include both anatomical MRI and EEG.
- SQL A: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
- SQL B: `SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, d.source_type, d.remote_url, d.validation_status, C`
