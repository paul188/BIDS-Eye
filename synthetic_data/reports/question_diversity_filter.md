# Question Diversity Filter

## Summary

- Input rows: 4398
- Bundle count: 2385
- Kept rows: 2808
- Dropped rows: 1590
- Bundles flagged for review: 567

## Largest Bundles

### 1. bundle_size=49 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [59]
- Drop row ids: [84, 106, 153, 413, 574, 643, 688, 704, 962, 999, 1046, 1088, 1618, 1637, 1791, 2194, 2254, 2335, 2376, 2401, 2488, 2585, 2629, 2652, 2692, 2717, 2740, 2764, 2806, 2851, 2914, 3100, 3227, 3295, 3405, 3475, 3561, 3670, 3693, 3736, 3799, 4032, 4052, 4163, 4189, 4305, 4328, 4394]

- KEEP row 59 (imperative): Show datasets that do not have a DOI assigned. [risky]
- DROP row 84 (imperative): Show datasets that do not have a DOI. [risky]
- DROP row 106 (imperative): Show all datasets that do not have a DOI. [risky / near-dup]
- DROP row 153 (imperative): Find datasets that do not have a DOI. [risky]
- DROP row 413 (imperative): List datasets without a DOI assigned. [risky]
- DROP row 574 (imperative): Find datasets that do not have a DOI. [risky / near-dup]
- DROP row 643 (other): Datasets without DOIs. [risky]
- DROP row 688 (other): Datasets that do not have a DOI. [risky / near-dup]

### 2. bundle_size=48 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [51]
- Drop row ids: [95, 116, 475, 521, 600, 612, 654, 698, 783, 930, 990, 1138, 1210, 1228, 1253, 1275, 1344, 1351, 1472, 1515, 1738, 1779, 1874, 2033, 2119, 2139, 2245, 2305, 2327, 2478, 2573, 2595, 2730, 2794, 2840, 2904, 2954, 3240, 3281, 3393, 3483, 3572, 4084, 4177, 4250, 4316, 4384]

- KEEP row 51 (other): Get the top 10 datasets ordered by the number of participants. [risky]
- DROP row 95 (imperative): List the top 10 datasets ordered by the number of unique subjects. [risky]
- DROP row 116 (imperative): Show the top 10 datasets ordered by the number of unique subjects. [risky / near-dup]
- DROP row 475 (imperative): List the top 10 datasets ordered by subject count. [risky]
- DROP row 521 (imperative): List all datasets ordered by the number of subjects [risky]
- DROP row 600 (other): Top 10 datasets with the most subjects. [risky]
- DROP row 612 (imperative): Show the 10 largest datasets by subject count. [risky]
- DROP row 654 (imperative): Show the 10 datasets with the highest number of subjects. [risky]

### 3. bundle_size=45 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [9]
- Drop row ids: [186, 227, 250, 337, 431, 633, 717, 884, 971, 1036, 1100, 1164, 1186, 1319, 1537, 1626, 1717, 1760, 2076, 2162, 2184, 2236, 2414, 2661, 2683, 2704, 2719, 2769, 2862, 2894, 2974, 3001, 3132, 3196, 3217, 3464, 3508, 3536, 3609, 3637, 3726, 3921, 3961, 4201]

- KEEP row 9 (imperative): Show the top 5 datasets with the most subjects. [risky]
- DROP row 186 (imperative): Show the top 5 datasets by participant count. [risky]
- DROP row 227 (imperative): List the top 5 largest datasets by subject count. [risky]
- DROP row 250 (imperative): List the 5 datasets with the highest number of subjects. [risky]
- DROP row 337 (other): Get the 5 datasets with the highest number of subjects. [risky / near-dup]
- DROP row 431 (other): Retrieve the top 5 datasets with the highest number of subjects. [risky]
- DROP row 633 (other): Get the top 5 datasets by subject count. [risky]
- DROP row 717 (other): Top 5 datasets ordered by the number of unique subjects. [risky]

### 4. bundle_size=35 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [74]
- Drop row ids: [590, 591, 592, 952, 1364, 1610, 1611, 1804, 1805, 1900, 2057, 2365, 2366, 2391, 2392, 2457, 2501, 2502, 2614, 3041, 3042, 3043, 3306, 3326, 3327, 3353, 3986, 4113, 4131, 4339, 4340, 4365, 4366, 4367]

- KEEP row 74 (imperative): Show the 5 largest datasets by participant count. [risky]
- DROP row 590 (imperative): List 5 datasets sorted by subject count. [risky]
- DROP row 591 (imperative): Find the top 5 largest datasets by participation. [risky]
- DROP row 592 (imperative): Show me 5 datasets ordered by the number of subjects enrolled. [risky]
- DROP row 952 (other): Top 5 datasets by participant count. [risky]
- DROP row 1364 (imperative): List the top 5 datasets by subject count. [risky]
- DROP row 1610 (imperative): List the top 5 datasets by participant count. [risky]
- DROP row 1611 (imperative): Show the 5 datasets with the highest number of participants. [risky]

### 5. bundle_size=28 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [41]
- Drop row ids: [42, 235, 236, 324, 326, 439, 440, 1024, 1025, 1148, 1438, 1439, 2668, 2669, 2870, 2871, 3028, 3029, 3122, 3123, 3160, 3161, 3206, 3249, 3250, 4257, 4258]

- KEEP row 41 (imperative): Show datasets that do not have a DOI. [risky]
- DROP row 42 (imperative): Find datasets missing DOI information. [risky]
- DROP row 235 (imperative): Find all datasets that do not have a DOI. [risky]
- DROP row 236 (imperative): List datasets without assigned DOIs. [risky]
- DROP row 324 (imperative): List datasets that do not have a DOI. [risky / near-dup]
- DROP row 326 (imperative): Show datasets without DOI registration. [risky]
- DROP row 439 (other): Datasets that do not have a registered DOI. [risky]
- DROP row 440 (formal): Identify datasets missing a DOI. [risky]

### 6. bundle_size=27 decision=keep_diverse_subset

- Styles: formal, imperative, other, question
- Keep row ids: [52, 144, 3509]
- Drop row ids: [316, 338, 386, 476, 613, 784, 1254, 1455, 1473, 1780, 2246, 2757, 2884, 2930, 3376, 3394, 3423, 3598, 3682, 3727, 3898, 4154, 4202, 4295]

- KEEP row 52 (imperative): Find datasets containing PET data.
- KEEP row 144 (formal): Identify datasets that contain PET modality data.
- DROP row 316 (question): Are there any datasets containing PET data?
- DROP row 338 (question): Are there any datasets containing PET data? [near-dup]
- DROP row 386 (question): Are there any datasets that contain PET data?
- DROP row 476 (question): Are there any datasets that contain PET data? [near-dup]
- DROP row 613 (question): Are there any datasets that contain PET data? [near-dup]
- DROP row 784 (other): Do any datasets exist that contain PET data?

### 7. bundle_size=20 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [215]
- Drop row ids: [216, 307, 597, 1285, 1286, 1682, 1683, 1817, 2069, 3053, 3268, 3317, 3338, 3339, 3717, 4119, 4143, 4282, 4352]

- KEEP row 215 (other): Datasets that do not have a DOI. [risky]
- DROP row 216 (imperative): Find all datasets lacking a DOI. [risky]
- DROP row 307 (other): Datasets that do not have any DOI specified. [risky]
- DROP row 597 (imperative): Find datasets that do not have a DOI. [risky / near-dup]
- DROP row 1285 (imperative): Find datasets lacking a DOI. [risky / near-dup]
- DROP row 1286 (question): Which datasets do not have an associated DOI? [risky]
- DROP row 1682 (imperative): List datasets without a DOI. [risky]
- DROP row 1683 (imperative): Show datasets that lack DOI references. [risky]

### 8. bundle_size=19 decision=keep_diverse_subset

- Styles: formal, imperative, other, question
- Keep row ids: [246, 588, 737]
- Drop row ids: [312, 988, 1137, 1534, 1672, 1777, 1870, 2241, 2569, 2880, 2902, 2972, 3039, 3594, 4175, 4313]

- KEEP row 246 (formal): Identify datasets that include both anatomical MRI and functional MRI modalities.
- DROP row 312 (other): Retrieve datasets featuring both anatomical MRI and functional MRI modalities.
- KEEP row 588 (imperative): Find multimodal datasets containing both anatomical and functional MRI.
- KEEP row 737 (imperative): Find datasets containing both anatomical MRI and functional MRI data.
- DROP row 988 (imperative): Show datasets that include both anatomical and functional MRI data.
- DROP row 1137 (imperative): Find datasets containing both anatomical and functional MRI. [near-dup]
- DROP row 1534 (imperative): Find datasets that contain both anatomical MRI and functional MRI datatypes.
- DROP row 1672 (other): Return datasets that contain both anatomical MRI and functional MRI data.

### 9. bundle_size=19 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [28]
- Drop row ids: [115, 185, 226, 840, 860, 1185, 1274, 1826, 2161, 2266, 2326, 3148, 3280, 3725, 3962, 4043, 4222, 4383]

- KEEP row 28 (imperative): List the top 50 datasets that have longitudinal data consisting of at least 3 sessions. [risky]
- DROP row 115 (imperative): Find datasets with at least 3 distinct sessions, sorted by subject count descending. [risky]
- DROP row 185 (question): Which datasets have longitudinal data spanning at least 3 sessions? [risky]
- DROP row 226 (imperative): Find the top 50 datasets that have at least 3 distinct sessions. [risky]
- DROP row 840 (imperative): Find datasets with at least 3 longitudinal sessions. [risky]
- DROP row 860 (question): Which datasets have at least 3 distinct sessions? [risky]
- DROP row 1185 (other): Retrieve datasets with at least 3 distinct sessions. [risky]
- DROP row 1274 (question): Which datasets possess at least 3 distinct scanning sessions? [risky]

### 10. bundle_size=19 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [50]
- Drop row ids: [313, 451, 474, 1625, 1693, 2010, 2096, 2225, 2304, 2703, 3040, 3068, 3131, 3239, 3258, 3325, 3392, 4364]

- KEEP row 50 (imperative): List the top 50 datasets that contain longitudinal data with at least 4 sessions. [risky]
- DROP row 313 (question): Which datasets have longitudinal data spanning 4 or more sessions? [risky]
- DROP row 451 (question): Which datasets feature longitudinal data with at least 4 distinct sessions? [risky]
- DROP row 474 (other): Retrieve the top 50 datasets that have at least 4 distinct sessions. [risky]
- DROP row 1625 (imperative): Find datasets with at least 4 imaging sessions. [risky]
- DROP row 1693 (imperative): Find datasets with at least 4 sessions. [risky]
- DROP row 2010 (other): Get datasets with at least 4 distinct sessions, sorted by subject count. [risky]
- DROP row 2096 (imperative): Find datasets with at least 4 sessions, ordered by subject count. [risky]

### 11. bundle_size=18 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [141]
- Drop row ids: [206, 402, 496, 544, 632, 671, 989, 1716, 1737, 1778, 1848, 1919, 2183, 2639, 3195, 3482, 3852]

- KEEP row 141 (imperative): Show the top 50 datasets with at least 2 sessions per subject. [risky]
- DROP row 206 (other): Top 50 datasets with at least 2 sessions. [risky]
- DROP row 402 (other): Get the top 50 datasets that have at least 2 sessions. [risky]
- DROP row 496 (imperative): Find datasets that include at least 2 sessions per subject. [risky]
- DROP row 544 (imperative): Show the top 50 datasets with at least 2 sessions. [risky / near-dup]
- DROP row 632 (question): Which datasets have at least 2 sessions? Show the top 50. [risky]
- DROP row 671 (imperative): List datasets featuring longitudinal data with at least two sessions. [risky]
- DROP row 989 (question): Which datasets contain longitudinal data with at least 2 sessions? [risky]

### 12. bundle_size=18 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [294]
- Drop row ids: [295, 296, 763, 1387, 1675, 1676, 2821, 3259, 3260, 3813, 3835, 3836, 4004, 4005, 4069, 4070, 4273]

- KEEP row 294 (imperative): Show the 10 largest datasets by subject count. [risky]
- DROP row 295 (other): Get information for the 10 datasets with the most participants. [risky]
- DROP row 296 (imperative): List the 10 datasets with the highest number of subjects. [risky]
- DROP row 763 (imperative): Show the 10 datasets with the highest number of subjects. [risky / near-dup]
- DROP row 1387 (imperative): List the 10 datasets with the largest number of unique subjects. [risky]
- DROP row 1675 (imperative): Show the 10 datasets with the highest number of subjects. [risky / near-dup]
- DROP row 1676 (imperative): Find the top 10 largest datasets by participant count. [risky]
- DROP row 2821 (question): Which 10 datasets have the largest number of subjects? [risky]

### 13. bundle_size=16 decision=keep_subset_flagged_review

- Styles: formal, imperative, question
- Keep row ids: [12]
- Drop row ids: [208, 228, 699, 718, 1718, 1761, 1969, 2185, 2705, 2976, 3093, 3112, 3197, 3241, 4251]

- KEEP row 12 (question): Are there any datasets that contain EEG data? [risky]
- DROP row 208 (question): Are there any datasets that contain EEG data? [risky / near-dup]
- DROP row 228 (question): Are there any datasets that contain EEG data? [risky / near-dup]
- DROP row 699 (imperative): Find datasets that contain at least one EEG scan. [risky]
- DROP row 718 (question): Are there any datasets that contain EEG data? [risky / near-dup]
- DROP row 1718 (question): Are there any datasets that contain EEG data? [risky / near-dup]
- DROP row 1761 (imperative): List all datasets that contain EEG data. [risky]
- DROP row 1969 (question): Are there any datasets that contain EEG data? [risky / near-dup]

### 14. bundle_size=16 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [299, 953, 4132]
- Drop row ids: [297, 298, 764, 1806, 1807, 1808, 2367, 2955, 3044, 3045, 3328, 3329, 3354]

- DROP row 297 (question): Are there any datasets that contain PET data?
- DROP row 298 (other): Do any studies have PET imaging? [risky]
- KEEP row 299 (other): Search for datasets with PET modality.
- DROP row 764 (question): Are there any datasets that contain PET modality? [risky]
- KEEP row 953 (other): Datasets that have any PET data.
- DROP row 1806 (question): Are there any datasets that contain PET data? [near-dup]
- DROP row 1807 (other): Do any datasets exist that include PET imaging?
- DROP row 1808 (other): Search for datasets containing PET modality files. [risky]

### 15. bundle_size=15 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [910]
- Drop row ids: [1062, 1063, 3091, 3092, 3441, 3442, 3746, 3747, 3853, 3854, 3896, 3897, 4044, 4045]

- KEEP row 910 (imperative): List the top 10 datasets ordered by number of subjects. [risky]
- DROP row 1062 (imperative): List the 10 most populous datasets. [risky]
- DROP row 1063 (other): Return the names of the 10 datasets with the highest participant count. [risky]
- DROP row 3091 (other): Top 10 datasets ordered by participant count. [risky]
- DROP row 3092 (imperative): List 10 largest datasets by number of unique subjects. [risky]
- DROP row 3441 (imperative): List 10 datasets ordered by participant count. [risky / near-dup]
- DROP row 3442 (question): What are the 10 largest datasets by number of participants? [risky]
- DROP row 3746 (imperative): List the 10 datasets with the highest number of subjects. [risky]

### 16. bundle_size=12 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [374]
- Drop row ids: [552, 810, 874, 938, 1528, 1572, 1954, 1996, 2169, 3183, 3621]

- KEEP row 374 (other): Retrieve datasets without a DOI assigned. [risky]
- DROP row 552 (imperative): Find datasets that have no associated DOI. [risky]
- DROP row 810 (imperative): Find datasets with no DOI specified. [risky]
- DROP row 874 (imperative): List datasets that do not have a DOI. [risky]
- DROP row 938 (imperative): Find datasets that do not have a DOI. [risky / near-dup]
- DROP row 1528 (imperative): Find datasets that do not have a registered DOI. [risky]
- DROP row 1572 (other): Filter datasets that do not have a DOI [risky]
- DROP row 1954 (formal): Identify datasets that do not have a DOI. [risky]

### 17. bundle_size=12 decision=keep_subset_flagged_review

- Styles: imperative, other
- Keep row ids: [1454]
- Drop row ids: [1967, 1968, 2209, 2210, 2211, 2754, 2755, 2756, 3373, 3374, 3375]

- KEEP row 1454 (imperative): Find the 10 datasets with the highest number of subjects. [risky]
- DROP row 1967 (other): Retrieve the 10 datasets with the highest number of subjects. [risky]
- DROP row 1968 (imperative): List the top 10 most populated datasets based on participant count. [risky]
- DROP row 2209 (imperative): List 10 datasets with the most subjects. [risky]
- DROP row 2210 (imperative): Show top 10 datasets ranked by subject count. [risky]
- DROP row 2211 (other): Get the 10 largest datasets by participant count. [risky]
- DROP row 2754 (imperative): Show the 10 largest datasets by subject count. [risky]
- DROP row 2755 (imperative): List the ten datasets with the most participants. [risky]

### 18. bundle_size=11 decision=keep_subset_flagged_review

- Styles: imperative, other, question
- Keep row ids: [163]
- Drop row ids: [164, 1297, 1298, 1920, 1921, 1922, 1985, 2267, 2268, 2269]

- KEEP row 163 (imperative): Show the first 10 datasets ordered by participant count. [risky]
- DROP row 164 (other): Rank datasets by the number of subjects and return the top 10. [risky]
- DROP row 1297 (imperative): Show top 10 datasets by participant count. [risky]
- DROP row 1298 (imperative): List the first 10 datasets sorted by the number of unique subjects enrolled. [risky]
- DROP row 1920 (imperative): List the top 10 datasets by participant count. [risky / near-dup]
- DROP row 1921 (imperative): Show me the 10 datasets with the most participants. [risky]
- DROP row 1922 (other): Sort datasets by the number of unique participants and show the first 10. [risky]
- DROP row 1985 (imperative): List the top 10 datasets based on the number of subjects. [risky]

### 19. bundle_size=11 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [229]
- Drop row ids: [300, 339, 1038, 1678, 2643, 2907, 3046, 3330, 3395, 3466]

- KEEP row 229 (imperative): Find datasets that do not contain information about the 'dieting_status' of participants. [risky]
- DROP row 300 (question): Which datasets lack med.bolus information in their participant metadata? [risky]
- DROP row 339 (imperative): Find datasets that do not have a restraint_scale recorded in their participant metadata. [risky]
- DROP row 1038 (imperative): Find datasets not containing dieting_status in participant records. [risky]
- DROP row 1678 (imperative): List datasets that do not have any dieting_status info for participants. [risky]
- DROP row 2643 (imperative): Find datasets that do not list any 'dataset' in their extra participant JSON fields. [risky]
- DROP row 2907 (imperative): Find datasets that do not have any bodyfat participant data. [risky]
- DROP row 3046 (other): Datasets that do not have 'concern_dieting' extra participant info. [risky]

### 20. bundle_size=11 decision=keep_subset_flagged_review

- Styles: formal, imperative, other, question
- Keep row ids: [341]
- Drop row ids: [404, 1475, 1721, 1829, 2014, 2760, 2909, 3794, 4229, 4322]

- KEEP row 341 (imperative): Find datasets that have CoilString information. [risky]
- DROP row 404 (other): Check for existence of datasets. [risky]
- DROP row 1475 (imperative): Show metadata information for datasets containing 'CoilString' key. [risky]
- DROP row 1721 (other): Search for datasets containing EchoTime in their object metadata. [risky]
- DROP row 1829 (question): Which datasets contain any objects? [risky]
- DROP row 2014 (imperative): Find datasets containing any data. [risky]
- DROP row 2760 (imperative): Find datasets containing image type and session info. [risky]
- DROP row 2909 (other): Search for datasets with patient positions defined in the metadata. [risky]
