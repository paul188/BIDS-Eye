import json, os
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ("relevant", "partially_relevant", "not_relevant", "unsure")
POPS = ("healthy_only", "patients_and_healthy", "patients_only", "special_non_clinical", "not_human", "unclear")
YNU = ("yes", "no", "unknown")

def load_batch(name):
    b = json.load(open(f"{G}/batches/{name}.json"))
    if "questions" in b:
        need = [(q["query_id"], a) for q in b["questions"] for a in q["candidates"]]
    else:
        need = [("pop", a) for a in sorted(b["cards"])]
    return b, need

def done_pairs(name):
    path = f"{G}/judged/{name}.jsonl"
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            done[(r.get("query", "pop"), r.get("acc"))] = r
    return done
