"""Multi-word terms: datasets with all words anywhere vs. within 2 positions vs. adjacent.
Also mine bigrams that behave like fixed terms (high adjacency share)."""
import json, sys, math
from collections import Counter, defaultdict

S = sys.argv[1]
docs = json.load(open(f"{S}/docs.json"))
STOP = set("a an and as at by for from in of on or the to with is are was were be been this that these those it its we our not no all each per can will please if into than also using used use".split())

TERMS = """resting state|working memory|default mode network|white matter|gray matter|grey matter|event related|block design|
diffusion weighted|functional connectivity|motor imagery|visual cortex|theory of mind|reward processing|face processing|
mismatch negativity|stop signal|go nogo|n back|phase encoding|field map|magnetic resonance|slice timing|eye tracking|
eye movements|sleep deprivation|deep brain stimulation|transcranial magnetic stimulation|cerebral blood flow|arterial spin labeling|
spinal cord|auditory cortex|semantic memory|episodic memory|fear conditioning|emotion regulation|social cognition|cognitive control|
decision making|multiple sclerosis|healthy controls|major depressive disorder|mild cognitive impairment|traumatic brain injury|
inter trial interval|skull stripping|head motion|motion correction|signal to noise|mental arithmetic|language comprehension|
naturalistic viewing|movie watching|pattern separation|prediction error|selective attention|visual attention|spatial attention|
mental imagery|visual search|motor learning|statistical learning|reinforcement learning|associative learning|free recall|
attention deficit|autism spectrum|eating disorder|bipolar disorder|substance use|alcohol use|cognitive aging|healthy aging|
temporal lobe|frontal lobe|prefrontal cortex|basal ganglia|medial temporal lobe|visual word form|fusiform face area|
evoked potentials|event related potentials|steady state|resting state fmri|pain intensity|heart rate|skin conductance|
time series|sampling rate|repetition time|echo time|flip angle|multiband acceleration|independent component|surface based|
single trial|high density|low frequency|mean age|age range|face recognition|object recognition|word recognition|
emotion recognition|speech perception|music listening|language network|hand movements|finger tapping|motor cortex""".replace("\n", "")
TERMS = [t.split() for t in TERMS.split("|")]

def positions(toks):
    pos = defaultdict(list)
    for i, t in enumerate(toks): pos[t].append(i)
    return pos

POS = {ds: positions(t) for ds, t in docs.items()}

def match(pos, words, maxgap):
    """words in order, each next one within maxgap positions (1 = adjacent)."""
    if any(w not in pos for w in words): return False
    starts = pos[words[0]]
    for s in starts:
        cur = {s}
        for w in words[1:]:
            nxt = {p for p in pos[w] for c in cur if 0 < p - c <= maxgap}
            if not nxt: break
            cur = nxt
        else:
            return True
    return False

rows = []
for words in TERMS:
    bag = sum(1 for p in POS.values() if all(w in p for w in words))
    if not bag: rows.append((" ".join(words), 0, 0, 0)); continue
    near = sum(1 for p in POS.values() if match(p, words, 3))
    adj = sum(1 for p in POS.values() if match(p, words, 1))
    rows.append((" ".join(words), bag, near, adj))
print(f"{'term':32} {'bag':>5} {'near3':>5} {'adj':>5}  extra-hits-of-bag")
for t, b, n, a in sorted(rows, key=lambda r: -(r[1] - r[3])):
    share = f"{100*(b-a)/b:4.0f}%" if b else "  -"
    print(f"{t:32} {b:5} {n:5} {a:5}  {share}")
json.dump(rows, open(f"{S}/phrases.json", "w"))

# bigram mining
uni = Counter(); bi = Counter()
for toks in docs.values():
    uni.update(set(toks))
    bi.update({(a, b) for a, b in zip(toks, toks[1:]) if a not in STOP and b not in STOP
               and not a.isdigit() and not b.isdigit() and len(a) > 1 and len(b) > 1})
N = len(docs)
cand = []
for (a, b), c in bi.items():
    if c < 15: continue
    co = sum(1 for p in POS.values() if a in p and b in p)
    pmi = math.log(c * N / (uni[a] * uni[b]))
    cand.append((a + " " + b, c, co, uni[a], uni[b], pmi))
json.dump(cand, open(f"{S}/bigrams.json", "w"))
print("\nbigrams df>=15:", len(cand))
