"""Write blame.sql: per request one read-only query that evaluates every top-level condition for every dataset.

Output rows (psql -At -F '|'):  qid|accession|bits   with one 0/1 per condition, WHERE conditions first, then HAVING.
"""
import json, sys
from sqlglot import exp
import conds

def outer_aliases(e):
    """Table aliases a condition uses outside of any subquery."""
    out = set()
    for col in e.find_all(exp.Column):
        node, inside = col.parent, False
        while node is not None and node is not e.parent:
            if isinstance(node, (exp.Subquery, exp.Select, exp.Exists)):
                inside = True; break
            node = node.parent
        if not inside and col.table:
            out.add(col.table)
    return out

run, truth = conds.load()
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
meta, sql_out = {}, ["\\set ON_ERROR_STOP off", "SET statement_timeout = '240s';"]
for qid, r in run.items():
    if only and not qid.startswith(tuple(only)): continue
    if not r.get("sql"): continue
    base, where, having, outer = conds.parts(r["sql"])
    frm = (base.args.get("from") or base.args.get("from_")).this
    assert frm.alias_or_name == "d", (qid, frm.sql())
    joins = base.args.get("joins") or []
    need_join = bool(having) or any(outer_aliases(c) - {"d"} for c in where)
    cols = []
    for c in where:
        s = c.sql(dialect="postgres")
        cols.append(f"BOOL_OR(COALESCE(({s}), FALSE))" if (need_join and outer_aliases(c) - {"d"}) else f"COALESCE(({s}), FALSE)")
    for c in having:
        cols.append(f"COALESCE(({c.sql(dialect='postgres')}), FALSE)")
    if not cols:
        meta[qid] = {"where": [], "having": [], "outer": outer}; continue
    bits = " || ".join(f"CASE WHEN {c} THEN '1' ELSE '0' END" for c in cols)
    from_sql = "bids_datasets AS d" + ("".join(" " + j.sql(dialect="postgres") for j in joins) if need_join else "")
    group = " GROUP BY d.id, d.accession_id" if need_join else ""
    sql_out.append(f"\\echo ##{qid}")
    sql_out.append(f"SELECT '{qid}', d.accession_id, {bits} FROM {from_sql}{group};")
    meta[qid] = {"where": [conds.short(c, 400) for c in where], "having": [conds.short(c, 400) for c in having], "outer": outer, "joined": need_join}
open("blame.sql", "w").write("\n".join(sql_out) + "\n")
json.dump(meta, open("blame_meta.json", "w"), indent=1)
print(len(meta), "requests;", sum(1 for m in meta.values() if m.get("joined")), "need the objects join;", sum(len(m["where"]) + len(m["having"]) for m in meta.values()), "conditions")
