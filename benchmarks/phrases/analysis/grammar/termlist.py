"""Terms to study, by kind. A word may list alternates with |."""
TERMS = {
 "anatomy": """temporal lobe; temporal cortex|cortices; frontal lobe; frontal cortex|cortices; prefrontal cortex|cortices; parietal cortex|cortices;
   parietal lobe; occipital cortex|cortices; occipital lobe; visual cortex|cortices; motor cortex|cortices; auditory cortex|cortices;
   somatosensory cortex|cortices; cingulate cortex|cortices; insular cortex|cortices; orbitofrontal cortex|cortices; cerebral cortex|cortices;
   anterior cingulate; posterior cingulate; temporal gyrus|gyri; frontal gyrus|gyri; fusiform gyrus|gyri; angular gyrus|gyri;
   temporal pole; temporal sulcus|sulci; white matter; gray|grey matter; basal ganglia; corpus callosum; spinal cord; brain stem;
   nucleus accumbens; ventral striatum; dorsal striatum; medial temporal lobe; superior temporal gyrus|gyri; inferior frontal gyrus|gyri;
   posterior cingulate cortex; anterior cingulate cortex; dorsolateral prefrontal cortex; supplementary motor area; primary visual cortex""",
 "network": """default mode network; salience network; attention network; frontoparietal network; visual network; motor network; brain network;
   functional network; language network""",
 "method": """functional connectivity; effective connectivity; structural connectivity; resting state; field map; phase encoding; echo time;
   repetition time; flip angle; slice timing; motion correction; spatial smoothing; spatial normalization|normalisation; head motion;
   diffusion tensor; voxel size; field strength; head coil; time series; time course; temporal resolution; spatial resolution;
   eye tracking; source localization|localisation; brain extraction; skull stripping; cortical thickness; cortical surface;
   band pass; high pass; signal change; effect size; multiple comparisons; independent component analysis; general linear model;
   region of interest; signal to noise ratio; hemodynamic|haemodynamic response function; framewise displacement; global signal""",
 "construct": """working memory; episodic memory; long term memory; short term memory; semantic memory; cognitive control; inhibitory control;
   response inhibition; emotion regulation; reward processing; decision making; face perception; face processing; speech perception;
   language comprehension; motor imagery; visual attention; selective attention; spatial attention; reinforcement learning;
   fear conditioning; mental rotation; theory of mind; executive function; social cognition; motor learning; prediction error;
   reaction time; response time; visual stimulus|stimuli; auditory stimulus|stimuli; emotional faces|face""",
 "task": """n back; go no go|nogo; stop signal; eyes closed; eyes open; finger tapping; movie watching; naturalistic viewing;
   oddball task; stroop task; flanker task; resting state fmri; passive viewing; picture naming; word reading""",
 "disorder": """major depressive disorder; bipolar disorder; autism spectrum disorder; parkinson disease; alzheimer disease; multiple sclerosis;
   traumatic brain injury; mild cognitive impairment; healthy controls|control; healthy adults|adult; older adults|adult; young adults|adult;
   temporal lobe epilepsy; chronic pain; sleep deprivation; substance use; anxiety disorder; depressive symptoms|symptom""",
 "stimulation": """transcranial magnetic stimulation; deep brain stimulation; heart rate; skin conductance; blood flow; blood pressure;
   magnetic field; electrical stimulation""",
}
def terms():
    out = []
    for kind, block in TERMS.items():
        for t in block.replace("\n", " ").split(";"):
            words = [w.split("|") for w in t.split()]
            if words: out.append((kind, " ".join(w[0] for w in words), words))
    return out
if __name__ == "__main__":
    ts = terms(); print(len(ts)); 
    from collections import Counter; print(Counter(k for k, _, _ in ts), Counter(len(w) for _, _, w in ts))
