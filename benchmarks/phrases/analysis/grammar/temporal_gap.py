"""For each 'temporal' in the snapshot: how far away is the word that decides its sense?"""
import json, re, sys
from collections import Counter, defaultdict
ANAT = set("lobe lobes cortex cortices cortical gyrus gyri sulcus sulci pole poles region regions area areas "
           "lobectomy epilepsy junction operculum plane lesion lesions electrodes electrode white grey gray".split())
TIME = set("resolution resolutions dynamics snr signal-to-noise sampling interpolation filtering filter derivative "
           "derivatives smoothing autocorrelation precision scale scales correlations evolution".split())
COG = set("expectation discounting difference order attention prediction judgment judgments judgement integration "
          "context sequence duration perception processing".split())
SENSE = {**{w: "anat" for w in ANAT}, **{w: "time" for w in TIME}, **{w: "cog" for w in COG}}
TOK = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[.;:!?()]")
BOUND = set(".;:!?()")
WIN = 8
dist = Counter(); per_ds = defaultdict(set); ex = defaultdict(list); both = 0; total = 0
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except Exception: continue
    text = " ".join(str(d.get(f) or "") for f in ("name", "description", "readme", "dx_status", "study_domain", "study_design")).lower()
    toks = TOK.findall(text)
    for i, t in enumerate(toks):
        if t != "temporal": continue
        total += 1
        found = []
        for step in (1, -1):                          # right first, then left
            for k in range(1, WIN + 1):
                j = i + step * k
                if j < 0 or j >= len(toks) or toks[j] in BOUND: break
                if toks[j] in SENSE: found.append((k, step, SENSE[toks[j]])); break
        if not found:
            dist["none within sentence/8"] += 1; per_ds["none"].add(d.get("accession_id") or id(d)); 
            if len(ex["none"]) < 400: ex["none"].append(" ".join(toks[max(0,i-6):i+7]))
            continue
        k, step, s = min(found)
        if len({f[2] for f in found}) > 1: both += 1
        key = f"{'right' if step>0 else 'left'} gap {k if k<5 else '5+'}"
        dist[key] += 1
        if k > 1 and len(ex[key]) < 400: ex[key].append(f"[{s}] " + " ".join(toks[max(0,i-6):i+8]))
print("occurrences", total, "| two senses' words nearby:", both)
for k, v in sorted(dist.items()): print(f"{k:28} {v}")
import random; random.seed(1)
for k in sorted(ex):
    print("\n==", k)
    for s in random.sample(ex[k], min(8, len(ex[k]))): print("  ", s)
