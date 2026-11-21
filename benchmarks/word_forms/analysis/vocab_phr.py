import json, re, sys, yaml
S=sys.argv[1]
docs=json.load(open(f"{S}/docs.json"))
STOP=set("a an and as at by for from in of on or the to with".split())
terms=set()
def walk(x):
    if isinstance(x,str):
        w=[t for t in re.findall(r"[a-z0-9]+",x.lower()) if t not in STOP]
        if 2<=len(w)<=4: terms.add(tuple(w))
    elif isinstance(x,dict):
        for k,v in x.items(): walk(k); walk(v)
    elif isinstance(x,list):
        for v in x: walk(v)
for f in ["reference_vocabulary.yaml","cognitive_atlas_constructs.yaml"]:
    walk(yaml.safe_load(open(f)))
print("multi-word vocabulary entries:",len(terms))
idx={}
for ds,toks in docs.items():
    toks=[t for t in toks if t not in STOP]
    pos={}
    for i,t in enumerate(toks): pos.setdefault(t,[]).append(i)
    idx[ds]=pos
def adj(pos,w):
    return any(all((s+k) in set(pos.get(w[k],())) for k in range(1,len(w))) for s in pos[w[0]])
rows=[]
for w in terms:
    bag=[p for p in idx.values() if all(x in p for x in w)]
    if not bag: continue
    a=sum(1 for p in bag if adj(p,w))
    rows.append((" ".join(w),len(bag),a))
found=[r for r in rows if r[2]>0]
print("entries with at least one dataset as a phrase:",len(found))
B=sum(r[1] for r in found); A=sum(r[2] for r in found)
print(f"over those: bag matches {B}, phrase matches {A}, bag extra {100*(B-A)/B:.0f}%")
print("entries never written as a phrase but all words present somewhere:",sum(1 for r in rows if r[2]==0), "datasets", sum(r[1] for r in rows if r[2]==0))
for r in sorted(found,key=lambda r:-(r[1]-r[2]))[:25]: print(f"  {r[0]:40} bag {r[1]:4} phrase {r[2]:4}")
