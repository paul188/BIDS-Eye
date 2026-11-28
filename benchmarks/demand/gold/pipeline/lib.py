"""Local store for building the gold answers: our database facts + OpenNeuro's own catalogue."""
import json, re, os
G = os.path.dirname(os.path.abspath(__file__))

def _load(name):
    return [json.loads(l) for l in open(f"{G}/{name}") if l.startswith("{")]

def load():
    D = {}
    for name in ("f4.jsonl", "f1.jsonl", "f2a.jsonl", "f2b.jsonl", "f3.jsonl", "f5.jsonl"):
        for r in _load(name):
            if r.get("acc"):
                D.setdefault(r["acc"], {}).update({k: v for k, v in r.items() if v is not None or k not in D.get(r["acc"], {})})
    D = {k: v for k, v in D.items() if "name" in v}          # openneuro rows only (f4)
    ON = {}
    for r in _load("openneuro_all.jsonl"):
        ON[r["id"]] = r
    return D, ON

def on_text(o):
    """All text OpenNeuro holds for a dataset."""
    s = o.get("latestSnapshot") or {}
    desc = s.get("description") or {}
    m = o.get("metadata") or {}
    parts = [desc.get("Name"), m.get("datasetName"), m.get("dxStatus"), m.get("studyDomain"), m.get("studyDesign"),
             " ".join(m.get("tasksCompleted") or []) if isinstance(m.get("tasksCompleted"), list) else m.get("tasksCompleted"),
             s.get("readme")]
    return "\n".join(str(p) for p in parts if p)

def our_text(d, papers=True):
    parts = [d.get("name"), d.get("description"), d.get("dx_status"), d.get("study_domain"), d.get("study_design"), d.get("readme")]
    if papers:
        for p in d.get("papers") or []:
            if p.get("role") == "primary":
                parts += [p.get("title"), p.get("abstract"), " ".join(p.get("mesh") or []), " ".join(p.get("keywords") or [])]
    return "\n".join(str(p) for p in parts if p)

def snippets(text, pattern, radius=140, limit=3):
    out = []
    for m in re.finditer(pattern, text, re.I):
        a, b = max(0, m.start() - radius), min(len(text), m.end() + radius)
        s = re.sub(r"\s+", " ", text[a:b])
        if not any(s[:60] in o for o in out):
            out.append(s)
        if len(out) >= limit:
            break
    return out
