"""Split the stored SQL of each request into the top-level conditions of its candidate query."""
import json, re, sys
import sqlglot
from sqlglot import exp

RUN = "../results/2026-11-14_demand_first_run_5f91230.json"
TRUTH = "../gold/requests_truth.json"

def load():
    run = {r["id"]: r for r in json.load(open(RUN))["results"] if r["id"][:2] in ("qf", "qx", "qp")}
    truth = json.load(open(TRUTH))["queries"]
    return run, truth

def flatten_and(e):
    e = e.unnest() if isinstance(e, exp.Paren) else e
    if isinstance(e, exp.And):
        return flatten_and(e.left) + flatten_and(e.right)
    return [e]

def parts(sql):
    """(base select, WHERE conjuncts, HAVING conjuncts, outer tier filter text) of a stored query."""
    tree = sqlglot.parse_one(sql, read="postgres")
    base = None
    for sub in tree.find_all(exp.Subquery):
        if sub.alias == "tier_base":
            base = sub.this
            break
    outer = tree.args.get("where").this.sql(dialect="postgres") if base is not None and tree.args.get("where") else ""
    if base is None:
        base = tree
    where = flatten_and(base.args["where"].this) if base.args.get("where") else []
    having = flatten_and(base.args["having"].this) if base.args.get("having") else []
    return base, where, having, outer

def short(e, n=150):
    s = e.sql(dialect="postgres")
    s = re.sub(r"TO_TSQUERY\('english', ('(?:[^']|'')*')\)", r"TSQ(\1)", s)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 3] + "..."

if __name__ == "__main__":
    run, truth = load()
    import collections
    shape = collections.Counter()
    for qid, r in run.items():
        sql = r.get("sql") or ""
        if not sql:
            shape["no sql"] += 1; continue
        try:
            base, where, having, outer = parts(sql)
        except Exception as ex:
            shape["parse error"] += 1; print(qid, "PARSE", str(ex)[:80]); continue
        shape[("tier" if "tier_base" in sql else "plain", "having" if having else "", "fallback" if r.get("fallback") else "")] += 1
        if len(sys.argv) > 1 and qid.startswith(tuple(sys.argv[1].split(","))):
            print("##", qid, "| returned", len(r.get("returned") or []), "| outer:", outer[:120])
            for c in where: print("   W", short(c))
            for c in having: print("   H", short(c))
    print(shape.most_common())
