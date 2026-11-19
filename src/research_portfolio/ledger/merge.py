"""Record merging shared by build_ledger (within a department) and the sync step (against Neo4j)."""
from research_portfolio.common import norm_key

STATUS_RANK = {"published": 5, "funded": 5, "accepted": 4, "currently under review": 3, "submitted": 3,
               "in preparation": 2, "not yet submitted": 1, "not funded": 1}
UNION_LISTS = ("dm_record_ids", "claimed_by", "doi_candidates")

def merge_contributors(base, extra):
    out = [dict(x) for x in base]
    def find(x):
        for y in out:
            if x.get("person_id") and y.get("person_id") == x["person_id"]:
                return y
            if norm_key(x.get("name")) and norm_key(y.get("name")) == norm_key(x.get("name")):
                return y
        return None
    for x in extra:
        hit = find(x)
        if hit is None:
            out.append(dict(x))
            continue
        for k, v in x.items():                         # fill gaps (e.g. person_id, role) on the match
            if v not in (None, "", False) and not hit.get(k):
                hit[k] = v
    return out

def merge_records(a, b):
    """Prefer the more advanced status; fill empty fields; longer abstract wins; union lists/contributors."""
    rank = lambda r: STATUS_RANK.get(norm_key(r.get("status")), 0)
    prim, sec = (b, a) if rank(b) > rank(a) else (a, b)
    out = dict(prim)
    for k, v in sec.items():
        if k in UNION_LISTS or k == "contributors":
            continue
        if out.get(k) in (None, "", []) and v not in (None, "", []):
            out[k] = v
    if len(sec.get("abstract") or "") > len(out.get("abstract") or ""):
        out["abstract"] = sec["abstract"]
    for k in UNION_LISTS:
        vals = sorted(set(a.get(k, [])) | set(b.get(k, [])))
        if vals:
            out[k] = vals
    out["contributors"] = merge_contributors(prim.get("contributors", []), sec.get("contributors", []))
    return out

def identity_keys(x, include_doi=True):
    """Exact identity keys. Undated items get no title key, so they never merge by title alone."""
    kind = x["kind"]
    ks = [f"{kind}:dmrec:{r}" for r in x.get("dm_record_ids", []) if r]
    if include_doi and x.get("doi"):
        ks.append(f"{kind}:doi:{x['doi']}")
    if x.get("year"):
        ks.append(f"{kind}:ty:{norm_key(x['title'])}|{x['year']}")
    return ks
