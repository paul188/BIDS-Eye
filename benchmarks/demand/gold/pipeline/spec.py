"""Gold criteria for the 70 real requests (benchmarks/demand/requests.yaml).

Per question:
  relevant_if / partially_if   the written criterion a judge applies
  method       exact   = computed from recorded facts only
               judged  = every pooled candidate is read and labelled
               mixed   = a hard, recorded precondition plus judging
  must(d)      precondition on our database facts; candidates failing it are
               still pooled when their TEXT names the modality (catalogue gaps
               such as EEG stored outside the BIDS eeg folder)
  exact(d)     'relevant' | 'partial' | None from facts (exact / mixed)
  text         regexes over name, description, uploader fields, README and the
               primary paper (ours) and over OpenNeuro's current text
  modality_text  regex that lets a dataset without the recorded modality in
  on           OpenNeuro site searches (advancedSearch) whose hits join the pool
  scored       False = kept for the record but not scored (criterion cannot be
               checked from any data we or OpenNeuro hold)
"""
import re

L = lambda d, k: d.get(k) or []
P = lambda d: d.get("participants") or {}
DT = lambda d: set(L(d, "all_datatypes"))
EEG = lambda d: "electroencephalography" in DT(d)
MEG = lambda d: "magnetoencephalography" in DT(d)
IEEG = lambda d: "intracranial_eeg" in DT(d)
FNIRS = lambda d: "fnirs" in DT(d)
BOLD = lambda d: bool(d.get("n_bold_files")) or "functional_mri" in DT(d)
DWI = lambda d: "diffusion_mri" in DT(d)
MRI = lambda d: bool(DT(d) & {"anatomical_mri", "functional_mri", "diffusion_mri", "field_maps", "perfusion_asl"})
ANAT = lambda d: "anatomical_mri" in DT(d)
ASL = lambda d: "perfusion_asl" in DT(d) or "arterial_spin_labelling" in (d.get("sub_per_suffix") or {})
REST_NAME = re.compile(r"rest|^rs$|^rsfmri|eyes?(open|closed)", re.I)
REST_BOLD = lambda d: bool(d.get("n_rest_files")) or any(REST_NAME.search(t or "") for t in L(d, "bold_raw_tasks")) or "resting_state" in L(d, "bold_tasks")
TASK_BOLD = lambda d: any(not REST_NAME.search(t or "") for t in L(d, "bold_raw_tasks"))
NSUB = lambda d: d.get("n_sub") or 0
SPS = lambda d, s: (d.get("sub_per_suffix") or {}).get(s, 0)
DERIV = lambda d: {str(v.get("pipeline") or "").lower() for v in d.get("derivatives") or []}
HAS_PRIMARY = lambda d: any(p.get("role") == "primary" for p in d.get("papers") or [])
OPEN_LICENCE = re.compile(r"^\s*(cc0|cco|pddl|ppdl|pd\b|public domain)|public domain dedication", re.I)
NONHUMAN = re.compile(r"mouse|mice|rat\b|rats\b|macaque|monkey|primate|sheep|pigeon|bat\b|canine|dog|pig|sus scrofa|phantom|synthetic|marmoset|ferret|zebra|baboon|chimpanzee|rhesus|macaca", re.I)
HUMAN = lambda d: not NONHUMAN.search(d.get("species") or "")
NHP = re.compile(r"macaque|monkey|non-?human primate|marmoset|baboon|rhesus|macaca|chimpanzee", re.I)

EEG_T = r"\bE?EEG\b|electroencephal"
FMRI_T = r"\bf-?MRI\b|\bBOLD\b|functional (MRI|magnetic)"
REST_T = r"resting[- ]?state|\brs-?f?MRI\b|\brest(ing)? (scan|run|block|session|fmri|bold|eeg|condition|period|recording)|eyes[- ](open|closed)"
AD_T = r"alzheimer|\bAD\b|dementia|\bMCI\b|mild cognitive impairment"
AUT_T = r"autis|\bASD\b|asperger"
TUMOUR_T = r"tumou?r|glioma|glioblastoma|meningioma|adenoma|metasta|neoplasm|neuro-?oncolog|astrocytoma|oligodendro|\bGBM\b|brain cancer"
PREPROC_T = r"pre-?processed|fmriprep|derivatives?[ /]|denois|minimally processed|preproc\b"

Q = []
def add(**kw):
    kw.setdefault("scored", True); kw.setdefault("text", []); kw.setdefault("on", [])
    Q.append(kw)

# ── Forum requests ───────────────────────────────────────────────────────────
add(id="qf01_bipolar_schizophrenia", method="judged",
    relevant_if="The participants include people diagnosed with bipolar disorder or with schizophrenia (schizoaffective disorder and first-episode schizophrenia-spectrum psychosis count). Any data type; the asker accepts datasets without imaging.",
    partially_if="People at clinical high risk for psychosis, with schizotypy, or with an unspecified psychotic or mood disorder, but no diagnosed bipolar or schizophrenia group.",
    text=[r"schizo", r"bipolar", r"psychosis|psychotic", r"\bmania\b|manic"],
    dx=r"schizo|bipolar|psycho",
    on=[dict(keywords=["schizophrenia"]), dict(keywords=["bipolar"]), dict(keywords=["psychosis"]), dict(diagnosis="Schizophrenia"), dict(diagnosis="Bipolar Disorder")])

for qid in ("qf02_eeg_autism_controls_like_abide", "qf04_eeg_autism", "qf35_eeg_autism_patients"):
    controls = qid.startswith("qf02")
    add(id=qid, method="mixed", must=EEG, modality_text=EEG_T,
        relevant_if="EEG recordings are shared and the participants include people with an autism spectrum diagnosis"
                    + (", together with a non-autistic comparison group in the same dataset." if controls else "."),
        partially_if=("EEG from autistic participants without a comparison group; or " if controls else "")
                     + "EEG from people selected for autistic traits or family risk (infant siblings) without a diagnosed group.",
        text=[AUT_T], dx=r"autis|asd",
        on=[dict(keywords=["autism"], modality="eeg"), dict(keywords=["autism"]), dict(keywords=["ASD"], modality="eeg"), dict(diagnosis="Autism")])

add(id="qf03_rest_fmri_alcohol_questionnaires", method="mixed", must=REST_BOLD, modality_text=REST_T,
    relevant_if="Resting-state fMRI of healthy participants (a healthy sample or healthy controls), and a measure of their alcohol use (AUDIT, ADS, drinks per week, or similar) is shared per participant or stated to be collected and available.",
    partially_if="Resting-state fMRI with an alcohol-use measure, but only for patients or dependent drinkers; or healthy resting-state fMRI where alcohol use was only an exclusion or screening criterion.",
    text=[r"alcohol", r"\bAUDIT\b", r"drinking|drinks? per|drinker", r"\bADS\b", r"substance use"],
    extra=r"audit|alcohol|drink|\bads\b",
    on=[dict(keywords=["alcohol"]), dict(keywords=["AUDIT"])])

add(id="qf05_task_fmri_fast_tr", method="exact",
    relevant_if="BOLD runs of a task other than rest with a repetition time below 1.0 s (human or animal).",
    exact=lambda d: "relevant" if any(0.1 <= t < 1.0 for t in L(d, "bold_tr_task")) else None,
    review=lambda d: BOLD(d) and (not L(d, "bold_tr") or any(t < 0.1 or t > 20 for t in L(d, "bold_tr"))),
    review_text=[r"TR\s*(=|of|:)?\s*0?\.\d+\s*s|TR\s*(=|of|:)?\s*[1-9]\d\d\s*ms|sub-?second|multiband|multi-band|fast TR"])

add(id="qf06_rest_fmri_healthy_patients", method="population", must=REST_BOLD, population="healthy_present",
    relevant_if="Resting-state fMRI of human participants that include healthy people (a healthy sample, or healthy controls next to patients).",
    partially_if=None)

add(id="qf07_decoding_block_design_sessions", method="mixed",
    must=lambda d: BOLD(d) and TASK_BOLD(d) and (d.get("n_sub_ge2ses") or 0) >= 4,
    relevant_if="Task fMRI with at least 4 subjects who each have 2 or more sessions, and a block-design task with several stimulus classes or conditions that can be decoded (not only resting state).",
    partially_if="Task fMRI with at least 4 multi-session subjects and decodable conditions, but event-related or the design is not stated.",
    pool_all_must=True)

add(id="qf08_tof_swi", method="mixed",
    exact=lambda d: "relevant" if (SPS(d, "mr_angiography") or SPS(d, "swi")) else None,
    relevant_if="The dataset contains time-of-flight angiography (TOF/MRA) or susceptibility-weighted (SWI) images.",
    partially_if="Multi-echo GRE or T2*-weighted data from which SWI can be computed, described as such, but no SWI or TOF images.",
    text=[r"\bTOF\b|time[- ]of[- ]flight", r"angiogra|\bMRA\b|venogra", r"\bSWI\b|susceptibility[- ]weighted"],
    on=[dict(keywords=["TOF"]), dict(keywords=["SWI"]), dict(keywords=["angiography"])])

add(id="qf09_pcasl_physio", method="mixed", must=ASL, modality_text=r"\bp?c?ASL\b|arterial spin",
    relevant_if="Pseudo-continuous ASL (pCASL) data together with physiological recordings (respiration and/or cardiac) of the same scans.",
    partially_if="pCASL data without shared physiological recordings.",
    text=[r"pcasl|pseudo-?continuous", r"arterial spin|\bASL\b"], pool_all_must=True,
    on=[dict(keywords=["pcasl"]), dict(keywords=["ASL"])])

add(id="qf10_epilepsy_rest_fmri", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="Resting-state fMRI of people with epilepsy.",
    partially_if="fMRI of people with epilepsy without a resting-state run (task or EEG-fMRI of spikes only).",
    text=[r"epilep", r"seizure"], dx=r"epilep|seizure",
    on=[dict(keywords=["epilepsy"]), dict(diagnosis="Epilepsy"), dict(studyDomains=["Epilepsy"])])

add(id="qf11_preprocessed_fmriprep", method="mixed",
    exact=lambda d: "relevant" if "fmriprep" in DERIV(d) else None,
    relevant_if="fMRIPrep outputs are available for the dataset: inside it (derivatives folder), as its OpenNeuro Derivatives companion, or the dataset itself is an fMRIPrep output.",
    partially_if="The README says the data were processed with fMRIPrep but the outputs are not shared; or other preprocessed fMRI is shared instead.",
    text=[r"fmriprep"], on=[dict(keywords=["fmriprep"])])

add(id="qf12_rest_task_rest_one_day", method="mixed", must=BOLD,
    relevant_if="One scanning session contains resting-state fMRI both before and after a task (rest - task - rest).",
    partially_if="Resting-state runs before and after an intervention that is not a task in the scanner (drug, stimulation, sleep, training outside).",
    text=[r"rest[- ]task[- ]rest", r"(pre|post)-?(task|learning|encoding|training|stress|stimulus|movie|scan)[- ]rest", r"rest(ing)?([- ]state)? (scans?|runs?|fmri|periods?|blocks?) (before|prior|preceding|after|following|pre|post)",
          r"(before|after|pre|post|preceding|following) (and|&) (before|after|pre|post|following)", r"pre-? ?(and|&|/) ?post", r"post-?(task|learning|encoding) rest", r"baseline rest"],
    names=lambda d: len({t for t in L(d, "bold_raw_tasks") if REST_NAME.search(t or "")}) >= 2 and TASK_BOLD(d))

add(id="qf13_natural_images_visual_cortex", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="fMRI recorded while participants view still natural images (photographs of objects or scenes).",
    partially_if="fMRI with natural video, or with still images that are not natural photographs (line drawings, artificial shapes, faces only, words).",
    text=[r"natural(istic)? (image|scene|photo|picture|object|stimul)", r"photograph", r"imagenet|\bCOCO\b|BOLD5000|\bNSD\b|natural scenes dataset|\bTHINGS\b",
          r"object (image|categor|recogn|percept|represent)", r"scene (image|percept|categor|recogn|represent)", r"real-world (object|scene|image)", r"images? of (objects|scenes|animals|natural)"],
    on=[dict(keywords=["natural", "images"]), dict(keywords=["natural", "scenes"]), dict(keywords=["object", "recognition"])])

for qid in ("qf14_eeg_alzheimer_and_normal", "qf33_eeg_alzheimer"):
    controls = qid.startswith("qf14")
    add(id=qid, method="mixed", must=EEG, modality_text=EEG_T,
        relevant_if="EEG recordings of people with Alzheimer's disease" + (" and of cognitively healthy people in the same dataset." if controls else "."),
        partially_if=("EEG of Alzheimer's patients without a healthy group; or " if controls else "") + "EEG of people with mild cognitive impairment or another dementia, without an Alzheimer's group.",
        text=[AD_T], dx=r"alzheimer|dementia|mild_cognitive|mci",
        on=[dict(keywords=["alzheimer"], modality="eeg"), dict(keywords=["dementia"], modality="eeg"), dict(diagnosis="Alzheimers")])

add(id="qf15_rest_fmri", method="mixed", exact=lambda d: "relevant" if REST_BOLD(d) else None,
    relevant_if="The dataset contains resting-state fMRI runs (human or animal).",
    partially_if=None, must=BOLD, text=[REST_T], review_only_unmatched=True)

add(id="qf16_covid_brain_signals", method="judged",
    relevant_if="Functional brain recordings (fMRI, EEG, MEG or fNIRS) of people who had Covid-19.",
    partially_if="Only structural or diffusion MRI of people who had Covid-19; or functional data about the pandemic (lockdown, vaccination) without infected participants.",
    text=[r"covid|sars-cov|coronavirus|long[- ]covid|post-covid|pandemic"], dx=r"covid",
    on=[dict(keywords=["covid"]), dict(keywords=["COVID-19"]), dict(keywords=["SARS-CoV-2"])])

add(id="qf17_language_fmri_and_dti", method="mixed", must=lambda d: BOLD(d) and DWI(d),
    relevant_if="The same dataset contains diffusion MRI and task fMRI with a language task (speech, reading, sentence or word processing, naming, story listening).",
    partially_if=None, pool_all_must=True)

PREPROC_PIPES = {"fmriprep", "xcpd", "c-pac", "cpac", "halfpipe", "fmripost", "nilearn", "ciftify", "hcp", "aroma", "tedana"}
for qid, extra_must, crit in (
        ("qf18_preprocessed_fmri_ml", BOLD, "Preprocessed fMRI data are available for the dataset: fMRIPrep, XCP-D or another preprocessing output inside the dataset or as its OpenNeuro Derivatives companion."),
        ("qf25_preprocessed_rs_fmri", REST_BOLD, "Resting-state fMRI for which preprocessed data are available (inside the dataset or as OpenNeuro Derivatives).")):
    add(id=qid, method="mixed", must=extra_must,
        exact=lambda d, m=extra_must: "relevant" if m(d) and (DERIV(d) & PREPROC_PIPES) else None,
        relevant_if=crit, partially_if="The fMRI was preprocessed according to the README, but only raw data are shared; or only quality-control outputs (MRIQC) or FreeSurfer surfaces exist.",
        text=[PREPROC_T], needs_deriv_review=True)

add(id="qf19_videos_people_talking", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="fMRI recorded while participants watch video in which people talk with their faces visible (films, TV episodes, recorded speeches, interviews, audiovisual stories).",
    partially_if="fMRI with video that has little or no visible talking (silent or animated clips without speech), or with spoken stories without video.",
    text=[r"movie|film\b|films\b|video|audio-?visual|cinema|sitcom|episode|tv (show|series)|television|documentar|lecture|interview|speaker|talk(ing)? (head|face)|conversation|speech"],
    on=[dict(tasks=["movie"]), dict(keywords=["movie"]), dict(keywords=["video"]), dict(keywords=["audiovisual"])])

add(id="qf20_schizophrenia_mri_png", method="mixed", must=MRI,
    relevant_if="(No dataset can match: OpenNeuro holds NIfTI, not PNG or JPG exports.)",
    partially_if="MRI of people with schizophrenia (the format constraint cannot be met; a good answer says so and shows these).",
    text=[r"schizo"], dx=r"schizo", never_relevant=True, on=[dict(keywords=["schizophrenia"]), dict(diagnosis="Schizophrenia")])

add(id="qf21_low_field_paired_3t", method="mixed",
    exact=lambda d: "relevant" if any(0 < f < 1 for f in L(d, "field")) and any(2.8 <= f for f in L(d, "field")) else None,
    relevant_if="MRI below 1 T and MRI at 3 T or above of the same participants in one dataset.",
    partially_if="Low-field MRI (below 1 T) without a paired high-field scan of the same people.",
    text=[r"low[- ]field|ultra-?low", r"0\.064|64 ?mT|hyperfine|portable MRI|point-of-care", r"0\.55 ?T|0\.5 ?T\b|0\.35 ?T|0\.3 ?T\b|0\.2 ?T\b|47 ?mT"],
    on=[dict(keywords=["low-field"]), dict(keywords=["low", "field"]), dict(keywords=["Hyperfine"])])

add(id="qf22_diffusion_tumor", method="mixed", must=DWI, modality_text=r"diffusion|\bDWI\b|\bDTI\b|tractogra",
    relevant_if="Diffusion MRI of patients with a brain tumour.", partially_if=None,
    text=[TUMOUR_T, r"neurosurg|resection"], dx=r"tumo|glio|mening|adenoma",
    on=[dict(keywords=["tumor"]), dict(keywords=["glioma"]), dict(keywords=["tumour"])])

def _appa(d):
    dirs = {x.upper() for x in L(d, "rest_dir")}
    ped = set(L(d, "rest_ped"))
    if {"AP", "PA"} <= dirs or ({"j", "j-"} <= ped):
        return "relevant"
    fd = {x.upper() for x in L(d, "fmap_dir")} | {x.upper() for x in L(d, "bold_dir")}
    if REST_BOLD(d) and {"AP", "PA"} <= fd:
        return "relevant"
    return None
add(id="qf23_fmri_dir_ap_pa_small", method="exact", exact=_appa,
    relevant_if="Resting-state fMRI with both phase-encoding directions present as dir-AP and dir-PA files (BOLD runs or their EPI fieldmaps).")

add(id="qf24_fnirs_alzheimer", method="mixed", must=FNIRS, modality_text=r"fNIRS|near-?infrared",
    relevant_if="fNIRS recordings of people with Alzheimer's disease.",
    partially_if="fNIRS of people with mild cognitive impairment, another dementia, or of older adults studied for cognitive decline.",
    text=[AD_T, r"cognitive (impairment|decline)|older adults|aging|ageing"], pool_all_must=True,
    on=[dict(keywords=["alzheimer"], modality="nirs"), dict(modality="nirs")])

add(id="qf26_fmri_preprocessed_any_design", method="exact", exact=lambda d: "relevant" if BOLD(d) else None,
    relevant_if="Any dataset with fMRI (preprocessing is a preference, not a filter; preprocessed ones should rank first).")

add(id="qf27_7t_rest_fmri_mprage", method="mixed",
    must=lambda d: REST_BOLD(d) and (any(6.5 <= f <= 7.5 for f in L(d, "field_rest") + L(d, "field")) or not L(d, "field")),
    relevant_if="Human resting-state fMRI at 7 T with a T1-weighted anatomical image acquired with MPRAGE (not MP2RAGE).",
    partially_if="Human resting-state fMRI at 7 T whose anatomical image is MP2RAGE or of an unstated sequence.",
    text=[r"\b7 ?T\b|7[- ]?tesla|seven tesla|ultra-?high[- ]field"], need_all_text=True,
    on=[dict(keywords=["7T"], tasks=["rest"]), dict(keywords=["7", "tesla"])])

add(id="qf28_preprocessed_fmri_alzheimer", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="fMRI of people with Alzheimer's disease for which preprocessed data are available (in the dataset or as OpenNeuro Derivatives).",
    partially_if="fMRI of people with Alzheimer's disease or MCI with raw data only; or preprocessed fMRI of MCI or at-risk groups without Alzheimer's patients.",
    text=[AD_T], dx=r"alzheimer|dementia|mild_cognitive|mci", on=[dict(keywords=["alzheimer"]), dict(diagnosis="Alzheimers")])

add(id="qf29_simple_visual_task_20_subjects", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="Task fMRI with a simple low-level visual stimulus (gratings, oriented stripes, checkerboards, flicker) and at least 20 participants. (The 30 trials and randomisation of the request cannot be checked.)",
    partially_if="Such a simple visual task with fewer than 20 participants.",
    text=[r"grating|gabor|stripe", r"checkerboard|checker board|flicker", r"orientation (discrim|decod|select|tun)|oriented", r"simple visual|visual stimulation|contrast (reversing|reversal)", r"retinotop|population receptive|\bpRF\b"],
    on=[dict(keywords=["checkerboard"]), dict(keywords=["gratings"]), dict(keywords=["orientation"])])

add(id="qf30_fmri_emotional_state", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="fMRI in which the participant's own emotional state is induced, regulated or measured (mood or emotion induction, emotional films, music or pictures with ratings, emotion regulation, anxiety or stress induction).",
    partially_if="fMRI about perceiving emotion in others (emotional faces, voices) or about affect-related decisions, without inducing or measuring the participant's own emotional state.",
    text=[r"emotion", r"\bmood\b", r"affective|\baffect\b", r"\bfear|anxi|stress induc|\bsad(ness)?\b|\bhappy|happiness|disgust|\banger\b|\bIAPS\b"],
    on=[dict(keywords=["emotion"]), dict(studyDomains=["Emotion"])])

CONN_T = [r"connectivity matri|connectomes?\b|adjacency|\.graphml|network matri|correlation matri", r"structural connectivity|functional connectivity (matri|network)", r"graph (theor|analy|measure)"]
add(id="qf31_brain_networks_phenotype", method="judged",
    relevant_if="Brain networks are shared as data: structural or functional connectivity matrices (or graphs) per participant.",
    partially_if="Parcellated time series or tractography outputs from which matrices follow directly, but no matrices.",
    text=CONN_T, on=[dict(keywords=["connectome"]), dict(keywords=["connectivity", "matrices"])])
add(id="qf32_connectomes_consciousness", method="judged",
    relevant_if="Functional connectivity networks (matrices or graphs) are shared for a study of states of consciousness: disorders of consciousness, sleep, anaesthesia or sedation, psychedelics, or meditation.",
    partially_if="A study of such a state of consciousness with shared functional recordings (fMRI, EEG) but no connectivity matrices.",
    text=[r"disorders? of consciousness|\bDoC\b|coma\b|vegetative|minimally conscious|unresponsive wake", r"anesthe|anaesthe|sedat|propofol|ketamine|sevoflurane", r"\bsleep", r"meditat|mindful", r"psychedel|psilocybin|\bLSD\b|\bDMT\b|ayahuasca", r"conscious"],
    on=[dict(keywords=["consciousness"]), dict(keywords=["sleep"]), dict(keywords=["meditation"]), dict(keywords=["anesthesia"])])

def _licence(d):
    if not MRI(d): return None
    lic = d.get("license") or ""
    if OPEN_LICENCE.search(lic): return "relevant"
    if re.search(r"CC[- ]?BY|creative commons attribution", lic, re.I): return "partial"
    return "unknown"
add(id="qf34_mri_redistributable", method="exact", exact=_licence,
    relevant_if="MRI datasets under a public-domain dedication (CC0, PDDL), which allows redistribution without restrictions.",
    partially_if="MRI datasets under an attribution licence (CC-BY, CC-BY-SA): redistribution allowed, with conditions.")

add(id="qf36_brain_tumor_classes", method="mixed", must=MRI,
    relevant_if="MRI of brain tumour patients with the tumour type or diagnosis recorded per participant (usable as class labels).",
    partially_if="MRI of brain tumour patients of a single type, or without per-participant tumour type.",
    text=[TUMOUR_T], dx=r"tumo|glio|mening|adenoma", on=[dict(keywords=["tumor"]), dict(keywords=["glioma"]), dict(keywords=["meningioma"]), dict(keywords=["adenoma"])])
add(id="qf37_tumor_segmentation_ground_truth", method="mixed", must=MRI,
    relevant_if="MRI of brain tumour patients with tumour segmentation masks shared (manual or expert-corrected).",
    partially_if="MRI of brain tumour patients without segmentation masks; or lesion masks for another pathology.",
    text=[TUMOUR_T], dx=r"tumo|glio|mening|adenoma", on=[dict(keywords=["tumor", "segmentation"]), dict(keywords=["glioma"]), dict(keywords=["tumor"])])

add(id="qf38_eeg_depression_labeled", method="mixed", must=EEG, modality_text=EEG_T,
    relevant_if="EEG recordings of people with depression and of a comparison group, with the group of each participant given (usable as labels).",
    partially_if="EEG with depression scores (BDI, PHQ) in a non-clinical sample, or EEG of depressed patients without a comparison group.",
    text=[r"depress|\bMDD\b|dysthym|mood disorder|\bBDI\b|\bPHQ"], dx=r"depress|mdd",
    on=[dict(keywords=["depression"], modality="eeg"), dict(keywords=["MDD"], modality="eeg"), dict(diagnosis="Major Depressive Disorder")])

add(id="qf39_eeg_rest_adult_man_healthy", method="mixed", must=EEG, population="healthy_present",
    relevant_if="Resting-state EEG (eyes open or closed) of healthy adults that include men.",
    partially_if=None,
    text=[REST_T], names=lambda d: EEG(d) and any(REST_NAME.search(t or "") for t in L(d, "raw_tasks")))

add(id="qf40_mri_ct_paired", method="mixed",
    exact=lambda d: "relevant" if SPS(d, "ct_scan") and MRI(d) else None,
    relevant_if="CT and MRI images of the same subjects are both shared.",
    partially_if=None, text=[r"\bCT\b|computed tomograph"], must=lambda d: MRI(d) or IEEG(d),
    on=[dict(keywords=["CT"]), dict(keywords=["computed", "tomography"])])

add(id="qf41_eeg_digitization", method="mixed", must=EEG, modality_text=EEG_T,
    relevant_if="EEG with measured (digitised) electrode positions shared per participant.",
    partially_if="EEG with an electrodes file whose positions are a template or of unstated origin.",
    text=[r"digiti[sz]", r"polhemus|captrak|fastrak|zebris|xensor|structure sensor|3d scan|photogramm", r"(individual|measured|subject-specific|recorded) electrode (position|location|coordinate)s?"],
    exact=lambda d: "partial" if d.get("n_eeg_electrodes") else None,
    on=[dict(keywords=["digitized"], modality="eeg"), dict(keywords=["electrode", "positions"], modality="eeg")])

add(id="qf42_eeg_image_pairs_scenes", method="mixed", must=EEG, modality_text=EEG_T,
    relevant_if="EEG recorded while viewing images of natural scenes or objects, with the stimulus images shared (or openly obtainable as a named image set).",
    partially_if="EEG recorded while viewing natural images whose image files are not shared or not identifiable.",
    text=[r"natural(istic)? (image|scene|photo|picture|object)", r"scene", r"photograph", r"imagenet|\bCOCO\b|\bTHINGS\b", r"object (image|categor|recogn)", r"images? of", r"\bRSVP\b|rapid serial visual", r"visual (object|categor)"],
    on=[dict(keywords=["images"], modality="eeg"), dict(keywords=["scenes"], modality="eeg"), dict(keywords=["THINGS"])])

add(id="qf43_eeg_emotion_like_deap", method="mixed", must=EEG, modality_text=EEG_T,
    relevant_if="EEG recorded while emotions are induced in the participant (music, video or film clips, pictures), with emotion labels or ratings: suitable for emotion recognition.",
    partially_if="EEG about perceiving emotional faces or words, or with affect questionnaires only, without emotion induction.",
    text=[r"emotion", r"valence|arousal", r"\bmood\b|affective", r"\bDEAP\b"],
    on=[dict(keywords=["emotion"], modality="eeg"), dict(studyDomains=["Emotion"], modality="eeg")])

add(id="qf44_flair_min_subjects", method="exact", exact=lambda d: "relevant" if SPS(d, "flair_mri") >= 50 else None,
    relevant_if="At least 50 subjects have a FLAIR image. (The threshold of 50 is ours; the request names none.)")

# ── Requests for a search capability ─────────────────────────────────────────
add(id="qx01_fieldmap_effective_echo_spacing", method="exact",
    exact=lambda d: "relevant" if d.get("n_fmap") and (d.get("n_fmap_ees") or d.get("n_bold_ees")) else None,
    relevant_if="The dataset has fieldmaps and EffectiveEchoSpacing is recorded in the sidecars of the fieldmaps or of the BOLD runs they correct.")
add(id="qx02_spinal", method="judged",
    relevant_if="The dataset is about the spinal cord or spine (spinal cord imaging, spinal cord injury, spinal stimulation).",
    partially_if=None, text=[r"spinal|spine\b"], on=[dict(keywords=["spinal"]), dict(keywords=["spinal", "cord"])])
add(id="qx03_male_40yo", method="exact", exact=lambda d: "relevant" if (d.get("n_male40_fmap") or 0) >= 1 else None,
    relevant_if="At least one male participant aged 40 has fieldmap files.")
GE_PH = re.compile(r"^\s*(ge\b|ge[ _]|general electric|philips)", re.I)
add(id="qx04_non_siemens", method="mixed",
    exact=lambda d: "relevant" if any(GE_PH.search(m or "") for m in L(d, "manufacturer_bold")) else None,
    relevant_if="fMRI acquired on a GE or Philips scanner.", partially_if=None,
    review=lambda d: BOLD(d) and not L(d, "manufacturer_bold"),
    review_text=[r"philips|achieva|ingenia|intera\b", r"\bGE\b|general electric|signa\b|discovery mr|mr750"])
add(id="qx05_sessions", method="exact",
    exact=lambda d: ("relevant" if (d.get("ses_min") or 0) >= 3 else "partial" if (d.get("n_sub_ge3ses") or 0) >= 1 else None),
    relevant_if="Every subject has at least 3 sessions.", partially_if="Some, but not all, subjects have at least 3 sessions.")
add(id="qx06_fmriprep_version", method="exact", exact=lambda d: "relevant" if "mriqc" in DERIV(d) else None,
    relevant_if="MRIQC outputs are available (inside the dataset or as its OpenNeuro Derivatives companion).")
add(id="qx07_meeg", method="exact", exact=lambda d: "relevant" if (EEG(d) or MEG(d)) else "partial" if IEEG(d) else None,
    relevant_if="The dataset contains MEG or scalp EEG.", partially_if="Only intracranial EEG.")
add(id="qx08_seeg", method="mixed", must=IEEG, pool_all_must=True,
    exact=lambda d: "relevant" if (d.get("seeg_ch") or 0) > 0 else None,
    relevant_if="Intracranial EEG recorded with depth (stereo-EEG) electrodes.", partially_if=None,
    text=[r"\bs-?EEG\b|stereo-?(tactic )?(EEG|electroenceph)|depth electrode"], on=[dict(keywords=["sEEG"]), dict(keywords=["stereo-EEG"])])
def _female_only(d):
    p = P(d)
    if p.get("n_part") and p.get("n_female") and not p.get("n_male") and p["n_female"] == p["n_part"]: return "relevant"
    return None
add(id="qx09_female_only", method="mixed", exact=_female_only,
    relevant_if="All participants are female (human).", partially_if=None, must=lambda d: HUMAN(d) and not P(d).get("n_male"),
    text=[r"\b(all|only|exclusively)[- ](female|women)|(female|women)[- ]only|all (participants|subjects) (were|are) (female|women)", r"\d+ (healthy )?(young |adult )?(women|females)\b", r"pregnan|postpartum|mothers\b|breast cancer|menstrua|contracepti|menopaus"],
    )
add(id="qx10_nhp", method="mixed", exact=lambda d: "relevant" if NHP.search(d.get("species") or "") else None,
    relevant_if="The data are from non-human primates.", partially_if=None,
    text=[r"macaque|monkey|marmoset|non-?human primate|\bNHP\b|rhesus|baboon|chimpanzee|macaca"], on=[dict(keywords=["macaque"]), dict(keywords=["monkey"]), dict(keywords=["marmoset"]), dict(keywords=["NHP"])])
add(id="qx11_bids_filename", method="exact", exact="files_desc_preproc_bold",
    relevant_if="The dataset contains files whose names end in _desc-preproc_bold.nii.gz.")
add(id="qx12_metadata_sheet", method="exact",
    exact=lambda d: "relevant" if EEG(d) and HUMAN(d) and (d.get("size_bytes") or 10**18) < 10 * 1024**3 and NSUB(d) > 30 and HAS_PRIMARY(d) else None,
    relevant_if="Human EEG dataset smaller than 10 GB with more than 30 subjects and a linked primary paper.")

# ── Selection rules from papers ───────────────────────────────────────────────
add(id="qp01_t1w_and_bold_per_subject", method="exact",
    exact=lambda d: "relevant" if NSUB(d) and d.get("n_sub_t1_bold") == NSUB(d) else None,
    relevant_if="Every subject has at least one T1w image and at least one BOLD run.")
add(id="qp02_t1w_five_adult_hc", method="population", population="healthy_adult_t1w_5",
    relevant_if="At least five healthy adults (18 or older) have a T1w image.")
add(id="qp03_t1w_healthy_children_2_12", method="population", population="healthy_child_t1w_5",
    relevant_if="At least five healthy children aged 2 to 12 have a T1w image.")
MPRAGE_T = re.compile(r"mp-?rage|tfl3d|\bbravo\b|spgr|3d[- ]?tfe|\bt1[- ]?tfe", re.I)
MP2_T = re.compile(r"mp2-?rage", re.I)
VNAV_T = re.compile(r"vnav|\bpromo\b|prospective(ly)? motion|volumetric navigator|\bmoco\b", re.I)
def _qp04(d, text=""):
    if not (SPS(d, "t1_weighted_mri") and any(2.8 <= f <= 3.1 for f in (L(d, "field_t1") or L(d, "field"))) and (P(d).get("n_5_21") or 0) >= 1):
        return None
    seq = " ".join(L(d, "t1_seq"))
    both = seq + " " + text
    if VNAV_T.search(both): return "partial"
    if re.search(r"mp-?rage|tfl3d", MP2_T.sub("", both), re.I): return "relevant"
    if MP2_T.search(both) and not re.search(r"mp-?rage|tfl3d", MP2_T.sub("", both), re.I): return None
    return "unknown"
add(id="qp04_3t_mprage_age_5_21_no_pmc", method="exact", exact=_qp04, exact_text=True,
    must=lambda d: SPS(d, "t1_weighted_mri") and any(2.8 <= f <= 3.1 for f in (L(d, "field_t1") or L(d, "field"))) and (P(d).get("n_5_21") or 0) >= 1,
    relevant_if="3 T T1w images acquired with MPRAGE, at least one participant aged 5 to 21, and no prospective motion correction (vNav, PROMO).",
    partially_if="3 T MPRAGE with a participant aged 5 to 21, acquired with prospective motion correction.")
add(id="qp05_rest_epi_t1w_same_session_tr", method="exact",
    exact=lambda d: "relevant" if REST_BOLD(d) and (d.get("n_sub_t1_bold_same_ses") or 0) > 12 and any(t > 1.5 for t in L(d, "bold_tr_rest")) else None,
    relevant_if="Resting-state fMRI with more than 12 subjects who each have a BOLD run and a T1w image in the same session, and a resting-state TR above 1.5 s. (Voxel size and number of volumes are not recorded and are not checked.)")
add(id="qp06_language_50_healthy_adults_descriptor", method="mixed", must=lambda d: MRI(d) and NSUB(d) > 50,
    relevant_if="MRI dataset whose subject is language (speech, reading, sentence or word processing), with more than 50 healthy adults and a published data descriptor or paper.",
    partially_if="A language MRI dataset with more than 50 participants who are not healthy adults (children, patients), or without a published paper.",
    text=[r"language|linguistic", r"speech|sentence|reading|semantic|syntax|syntactic|lexical|phonolog|narrative|story|stories|\bwords?\b|bilingual"],
    on=[dict(keywords=["language"]), dict(studyDomains=["Language"])])
add(id="qp07_isolated_words", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="fMRI experiments that present isolated single words (visual or spoken), not sentences or stories.",
    partially_if="fMRI with words embedded in sentences, stories, or word pairs or lists only as memory items without a word-processing aim.",
    text=[r"\bwords?\b", r"lexical|pseudo-?word|non-?word|letter string", r"\bnouns?\b|\bverbs?\b", r"visual word|word (reading|recognition|form|naming|generation|repetition)"],
    on=[dict(keywords=["words"]), dict(keywords=["lexical"]), dict(keywords=["reading"])])
add(id="qp08_eating_disorders_mri", method="mixed", must=MRI,
    relevant_if="MRI of people with an eating disorder (anorexia nervosa, bulimia nervosa, binge-eating disorder).",
    partially_if="MRI about disordered eating, dieting or obesity without a diagnosed eating-disorder group.",
    text=[r"eating disorder|anorexi|bulimi|binge[- ]eating", r"obes|dieting|overeating|food addiction"], dx=r"anorex|bulimi|eating|binge",
    on=[dict(keywords=["anorexia"]), dict(keywords=["bulimia"]), dict(keywords=["eating", "disorder"]), dict(keywords=["binge"])])
add(id="qp09_eeg_stress", method="mixed", must=EEG, modality_text=EEG_T,
    relevant_if="EEG recorded under induced stress, or with stress as the studied state (stress induction, stress versus relaxation, stress level labels).",
    partially_if="EEG with a stress questionnaire or cortisol measure only, without stress as a condition.",
    text=[r"stress", r"cortisol|\bTSST\b|trier social|cold pressor|\bMIST\b|mental arithmetic"],
    on=[dict(keywords=["stress"], modality="eeg")])
add(id="qp10_task_fmri_with_paper", method="exact",
    exact=lambda d: "relevant" if BOLD(d) and TASK_BOLD(d) and HAS_PRIMARY(d) else None,
    relevant_if="Task fMRI dataset with a linked primary publication. (Whether the publication reports group-level activation is not checked.)")
add(id="qp11_memory_consolidation", method="mixed", must=BOLD, modality_text=FMRI_T,
    relevant_if="fMRI study of memory consolidation (post-encoding rest or sleep, reactivation or replay, systems consolidation, retrieval after a delay of a day or more), with task events files.",
    partially_if="fMRI of memory encoding and retrieval without a consolidation question, or a consolidation study without events files.",
    text=[r"consolidat", r"\breplay\b|reactivat", r"(over ?night|next[- ]day|24[- ]?h|one week|delayed) (retrieval|recall|test|memory)|remote memor", r"post-?encoding", r"sleep.{0,40}memor|memor.{0,40}sleep"],
    on=[dict(keywords=["consolidation"]), dict(keywords=["memory", "consolidation"])])
add(id="qp12_normative_healthy", method="population", must=MRI, population="healthy_only", scored=False,
    relevant_if="MRI of healthy people only. (Not scored: 'normative database' implies a size and curation the request does not define.)")
add(id="qp13_bci_paradigms", method="mixed", must=EEG, modality_text=EEG_T,
    relevant_if="EEG with a motor imagery task, an SSVEP task, or a P300 / oddball-type ERP paradigm of the kind used for brain-computer interfaces.",
    partially_if="EEG with other event-related designs (cognitive ERP experiments) that are not one of the three BCI paradigms.",
    text=[r"motor imagery|imagined (movement|motor)|\bMI\b", r"SSVEP|steady[- ]state visual", r"P300|P3b?\b|oddball|speller", r"\bBCI\b|brain[- ]computer|brain[- ]machine"],
    on=[dict(keywords=["motor", "imagery"]), dict(keywords=["SSVEP"]), dict(keywords=["P300"]), dict(keywords=["BCI"]), dict(tasks=["oddball"], modality="eeg")])
def _eeg200(d):
    if not EEG(d): return None
    lic = d.get("license") or ""
    ok_lic = OPEN_LICENCE.search(lic) or re.search(r"CC[- ]?BY|creative commons attribution", lic, re.I)
    sf = L(d, "eeg_sfreq")
    if not sf: return "unknown"
    if any(s >= 200 for s in sf) and ok_lic: return "relevant"
    if any(s >= 200 for s in sf): return "unknown"
    return None
add(id="qp14_eeg_200hz_redistributable", method="exact", exact=_eeg200,
    relevant_if="EEG recordings sampled at 200 Hz or more, under a licence that allows redistributing converted copies (CC0, PDDL, CC-BY, CC-BY-SA).")

BY_ID = {q["id"]: q for q in Q}

ON_MOD = {"eeg": r"\beeg\b", "mri": r"mri", "nirs": r"nirs", "ieeg": r"ieeg", "meg": r"\bmeg\b"}
for q in Q:
    m = q.get("must")
    q["on_mod"] = ("eeg" if m is EEG else "nirs" if m is FNIRS else "ieeg" if m is IEEG else
                   "mri" if m in (BOLD, REST_BOLD, DWI, MRI, ASL) or q["id"].startswith(("qf17", "qf27", "qp06", "qf40")) else None)
    if q["id"].startswith(("qf07", "qp04", "qx09")): q["on_skip"] = True     # need participant or session facts OpenNeuro's summary lacks
