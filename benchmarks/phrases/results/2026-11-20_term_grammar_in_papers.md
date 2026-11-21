# How neuroimaging text writes its terms

2026-11-20. Question: when a text means a multi-word term ("temporal cortex",
"working memory", "motion correction"), are the words next to each other? If
not, where are they, and which other ways of writing the same thing exist?

## Material

- **Papers:** 1,727 full-text neuroimaging papers (benchmarks/demand/work/epmc_xml,
  the papers that mention OpenNeuro datasets). Abstract, body and captions,
  no reference lists: 13.9 million words, 611,954 sentences.
- **Datasets:** the catalogue snapshot of 2026-11-13 (benchmarks/recall/work/snapshot.jsonl),
  name, description, README and study fields of 1,943 datasets: 30,546 sentences.
- **Parser:** spaCy 3.8 with en_core_web_sm (part of speech and dependency
  parse). It is a general English model, so its links are a measurement with
  error. Every class below was checked by reading 26 to 34 sampled sentences.
- **Terms:** 164 terms picked by hand in seven kinds (anatomy 42, method 40,
  construct 32, disorder 18, task 15, network 9, stimulation 8); 138 have two
  words. List: analysis/grammar/termlist.py.

## 1. Where the two words of a term are

Each sentence that contains both words of a two-word term is one instance.

| arrangement | example | papers | datasets | really the term (read sample) |
|---|---|---|---|---|
| adjacent | temporal cortex | 74.2% | 81.1% | all |
| list sharing the noun | temporal **and prefrontal** cortices | 1.9% | 1.3% | 25 of 26 |
| word(s) inserted, linked by the parser | functional **brain** networks; Alzheimer**'s** disease; salience **(SAL)** network | 2.7% | 2.5% | 26 of 26 |
| near (≤3 words), not linked by the parser | visual **food** stimuli; gray**/white** matter | 3.5% | 3.7% | 18 of 34 |
| noun first | size of the voxels; watching the movie | 2.3% | 2.1% | 8 of 30 |
| modifier belongs to a neighbouring noun | temporal lobe, … and prefrontal **cortex** | 1.7% | 1.0% | 3 of 26 |
| far apart in the sentence | | 13.8% | 8.3% | 0 or 1 of 30 |

Papers: 124,313 instances. Datasets: 4,752 instances.

Weighting each class by the share that is really the term: about 82% of the
sentence instances in papers mean the term, and **91% of those are adjacent**.
The rest is, in this order: inserted words, lists, noun-first rewordings.

Per dataset instead of per sentence (does the dataset have the term at all?):

| | term–dataset pairs |
|---|---|
| adjacent somewhere | 2,622 |
| only as list or with inserted word | 105 |
| only near or noun-first (about half real) | 189 |
| only far apart (almost never real) | 336 |

Strict adjacency would miss about 200 of 2,820 real pairs (7%). Matching the
words anywhere in one sentence adds about 430 false pairs (13%), and anywhere
in the whole text far more (the earlier bag count: 65% extra).

### Distance and order

Share of instances the parser links, by position of the noun relative to the
modifier (papers):

| noun is … | linked |
|---|---|
| 1 after | 100% |
| 2 after | 66% |
| 3 after | 59% |
| 4 after | 40% |
| 5 after | 22% |
| 1 before | 42% |
| 2 before | 73% |
| 3 before | 59% |
| 4 before | 49% |

Half of the unlinked near cases are real (table above), so a gap of 2 in
order is right about 4 times in 5. Beyond 3 words the pair is more often
unrelated than related.

### Differences between kinds of term

| kind | adjacent | list | inserted | far |
|---|---|---|---|---|
| method (flip angle, time series) | 86.0% | 0.8% | 0.6% | 8.2% |
| stimulation (heart rate) | 91.3% | 0.5% | 1.1% | 4.4% |
| anatomy | 78.1% | 3.5% | 0.4% | 9.9% |
| task | 79.0% | 0.2% | 0.6% | 8.3% |
| construct | 66.7% | 0.8% | 0.5% | 23.3% |
| disorder | 65.7% | 3.0% | 18.3% | 5.3% |
| network | 42.8% | 2.3% | 4.3% | 37.6% |

- **Fixed terms** are adjacent 98–100% of the time: basal ganglia, corpus
  callosum, flip angle, spinal cord, reaction time, working memory, resting
  state, white matter, time series, eye tracking.
- **Lobes and cortices** are where lists happen: frontal lobe 17%, parietal
  lobe 11%, occipital lobe 9%, parietal cortex 8%, frontal cortex 8%,
  temporal cortex 5.5%, temporal lobe 4.8%. Also young adults 13%, structural
  connectivity 11% ("structural and functional connectivity").
- **Terms made of two general words** are mostly not the term when apart:
  visual attention (15% adjacent), response time (30%), visual network,
  functional network, brain network, visual stimulus, motor learning.
- **Eponyms** are almost never adjacent: "Alzheimer's disease" and
  "Parkinson's disease" have the possessive between (2–4% adjacent).

## 2. Typical grammar, independent of the term list

All 2.9 million modifier–noun links in the papers:

| | adjective → noun | noun → noun |
|---|---|---|
| adjacent | 67.4% | 82.9% |
| another modifier between | 29.1% | 14.7% |
| list before the noun | 3.6% | 2.4% |

42% of nouns carry at least one modifier, 11% two or more. Dataset texts are
the same within two points (adjective 65.7 / 30.8 / 3.5).

The field stacks modifiers: "left superior temporal gyrus", "resting-state
functional connectivity", "healthy young adults". An adjective is separated
from its noun by a more specific modifier in 3 of 10 links. That is the main
reason a term's words are near but not adjacent.

Patterns seen when reading:

- **List with shared noun:** "the temporal and frontal lobes", "frontal,
  parietal, and temporal cortices", "gray and white matter", "structural or
  functional connectivity", "eyes open or closed".
- **Inserted modifier:** "functional brain networks", "visual food stimuli",
  "healthy older adults", "electrical forepaw stimulation".
- **Inserted abbreviation or bracket:** "the ventral temporal (VT) cortex",
  "Salience (SAL) network", "fear (Pavlovian) conditioning".
- **Slash:** "gray/white matter", "word/nonword reading",
  "salience/ventral attention network".
- **Long-range ellipsis:** "in the anterior [t(28) = −0.85, P = 0.40, …] or
  the posterior hippocampus" (15 tokens between).
- **Latin order:** "Gyrus Lingualis" for lingual gyrus (atlas labels).
- **Legends:** "ACC, anterior cingulate cortex; TP, temporal pole; …". These
  put unrelated region words within a few positions of each other and are
  the main source of false near matches in anatomy.

## 3. Reversed order: same meaning for actions, different meaning for things

For 62 stem pairs (process.py), every co-occurrence within 5 words:

| | papers | datasets |
|---|---|---|
| noun compound in order (motion correction) | 59.6% | 47.4% |
| adjacent, verb form (motion-corrected, skull-stripped) | 7.1% | 11.8% |
| in order with words between | 17.4% | 23.8% |
| reversed (corrected for motion, watched a movie) | 16.0% | 17.0% |

Action terms are often written as a clause:

| term | not the noun compound | typical wording |
|---|---|---|
| movie watching | 43% (papers), 32% (datasets) | participants watched four short movies |
| music listening | 32%, 68% | while they listened to music |
| image viewing | 92%, 80% | participants viewed images |
| skull stripping | 60%, 82% | skull-stripped |
| spatial smoothing | 29% | spatially smoothed |
| sleep deprivation | 22% | sleep-deprived |
| response inhibition | 42% | inhibit a prepotent response |
| decision making | 26% | make decisions |
| stimulus presentation | 72%, 80% | stimuli were presented |

Reading the reversed samples: for action terms the reversed wording means the
same thing. For terms made of two nouns it usually means something else:

- "phase encoding" (acquisition) vs "the encoding phase" (memory task)
- "voxel size" vs "cluster size = 88 voxels"
- "response time" vs "over time in response to"
- "repetition time" vs "fewer repetitions, reducing participant time"
- "attention network" vs "self-attention … neural network" (deep learning)

So word order may be ignored only when the head of the term is an action
(watching, listening, viewing, naming, reading, stripping, smoothing,
correction, deprivation, inhibition).

## 4. Abbreviations

9,808 different abbreviation definitions in the papers, 28,779 counting each
paper (Schwartz–Hearst matching).

- After the definition the short form is used **80%** of the time (274,091
  short uses, 70,105 long uses).
- In 58% of the definitions the long form occurs only once in the paper, at
  the definition.
- The same abbreviation has several meanings: WM (white matter 197 papers,
  working memory 29), FA (fractional anisotropy 79, flip angle 48), AD
  (Alzheimer's disease 144, axial diffusivity 17), PD (Parkinson's 90, proton
  density 5, panic disorder 1), SD (standard deviation 70, sleep deprivation
  7), AI (artificial intelligence 79, anterior insula 10), FD, MD, FC.
- The long form itself varies: "blood oxygen level dependent" /
  "blood oxygenation level dependent"; "framewise" / "frame wise" /
  "frame displacement"; "gray" / "grey matter"; "hemodynamic" / "haemodynamic".

Dataset texts differ. Concept abbreviations rarely stand alone there (WM 2
datasets, FC 1, DMN 0, MDD 3, TLE 1). Modality and acquisition abbreviations
mostly stand alone: EEG 259 datasets without "electroencephalography", TR 178
without "repetition time", BOLD 114, EPI 87, MPRAGE 58, MEG 30, TMS 13, ECoG 9.

## 5. Fused and hyphenated compounds

Share of a region word's uses that are inside a compound (papers):

| word | alone | fused | hyphenated | most common |
|---|---|---|---|---|
| frontal | 5,010 | 3,477 | 136 | prefrontal 2,925, orbitofrontal 528 |
| motor | 6,007 | 3,065 | 487 | sensorimotor 1,658, somatomotor 527, premotor 365 |
| cortical | 10,838 | 3,553 | 357 | subcortical 2,661, thalamocortical 409 |
| parietal | 3,239 | 1,185 | 530 | frontoparietal 669, fronto-parietal 216, temporoparietal 215 |
| temporal | 13,117 | 1,575 | 561 | spatiotemporal 692, frontotemporal 356, intertemporal 298 |
| lateral | 2,388 | 944 | 55 | dorsolateral 639, ventrolateral 165 |
| sensory | 3,817 | 1,444 | 72 | somatosensory 876, multisensory 383 |

- A fused compound is its own word in the index: "frontotemporal" does not
  match "temporal". Mostly that is correct ("spatiotemporal" and
  "intertemporal" are the time sense, "prefrontal" is a different region).
- The same compound is written fused and hyphenated: frontoparietal 669 /
  fronto-parietal 216, spatiotemporal 692 / spatio-temporal 170,
  temporoparietal 215 / temporo-parietal 87, audiovisual 388 / audio-visual
  101, visuospatial 231 / visuo-spatial 94. The index gives the hyphenated
  form extra lexemes for its parts; the fused form has none.

## 6. What the index does with these (checked on the local copy)

| text | index | effect |
|---|---|---|
| Alzheimer's disease | 'alzheim':1 'diseas':3 | `alzheimer <-> disease` does **not** match; a gap of 2 does |
| temporal and occipital lobes | 'tempor':1 'occipit':3 'lobe':4 | gap 3: outside today's `near` (gap ≤ 2) |
| the salience (SAL) network | 'salienc':2 'sal':3 'network':4 | gap 2: inside `near` |
| gray/white matter | slash already replaced by a space (migrate_add_search_tsv.sql) | fine |
| resting‐state (U+2010 hyphen) | 'rest':1 'state':2 | fine |
| gyri / gyrus, cortices / cortex, sulci / sulcus, nuclei / nucleus, stimuli / stimulus | different lexemes | singular does not match plural |

Latin plurals in the dataset texts: 159 datasets have "stimuli" and not
"stimulus", 112 the reverse; "cortices" only 7; "nuclei" only 4. None of
these pairs is in RAG/word_forms.yaml.

## 7. What follows for the search

1. **Keep phrases, but as "in order, gap up to 2", not strict adjacency.**
   Adjacent covers 91% of real mentions; gap 2 in order adds inserted
   modifiers, possessives and bracketed abbreviations and is right about 4
   times in 5. This is what `near` already does for two words.
2. **Do not accept reversed order for noun–noun terms.** `near` accepts both
   orders for two-word terms today. For terms like phase encoding, voxel size
   and response time the reversed order is a different thing.
3. **Accept either order and a wider window (about 4 words) for action
   terms.** The head is an action noun (‑ing, ‑ion, ‑al of a verb). This
   needs a per-term flag; the phrase review could carry it.
4. **Expand lists with a shared noun at index time, for anatomy and
   networks.** 9–17% of lobe mentions and 5–8% of cortex mentions are lists;
   for method terms it is under 1% and not worth it.
5. **Add Latin plurals to word_forms.yaml** (stimulus/stimuli first: 159
   datasets).
6. **Treat fused and hyphenated compounds as one spelling** (frontoparietal =
   fronto-parietal) when the first part ends in "o".
7. **Abbreviations:** for dataset texts a short curated list of modality and
   acquisition abbreviations covers most cases. If paper text is searched,
   abbreviations matter much more, and the ambiguous ones (WM, FA, AD, PD,
   SD) must not be expanded without context.
8. **Terms of two general words** (visual attention, response time, brain
   network, functional network) should always require the phrase: apart they
   are unrelated in 50–85% of the sentences.

## Limits

- The term list is hand-picked; the shares per kind depend on it.
- "Really the term" comes from my reading of about 180 sampled sentences,
  not from a second annotator.
- The parser is not trained on scientific text. Its list and legend handling
  is weak; that is why the "modifier belongs to a neighbouring noun" class is
  mostly wrong.
- The paper corpus is papers that mention OpenNeuro, heavy on data
  descriptors (Scientific Data 170 of 1,727).
- Everything is counted within one sentence. Mentions split across sentences
  are not counted.

## Files

- analysis/grammar/: extract.py (JATS → paragraphs), parse.py (spaCy),
  termlist.py, terms.py (section 1), np.py (section 2), process.py
  (section 3), acronyms.py (section 4), combining.py (section 5),
  temporal_gap.py (the earlier "temporal" count).
- results/: papers_terms.json, ds_terms.json (counts, gaps and up to 12
  sample sentences per term and class), papers_process.json,
  ds_process.json, acronyms.json.
- Not in the repo: the parsed corpus (104 MB) and spaCy, both in the session
  scratch folder. To rebuild: `pip install --target <dir> spacy` plus the
  en_core_web_sm 3.8.0 wheel, then extract.py, parse.py (about 12 minutes on
  7 processes) and the analysis scripts.
