#!/usr/bin/env python3
"""Enrich publication ledger records from OpenAlex. Never overwrites DM fields; writes openalex_enrichment.json.
Matching: DOI exact first; otherwise title search requiring normalized-title >= 92, year +/-1, and author-family overlap.
Set OPENALEX_MAILTO for polite-pool access."""
import argparse, json, os, time
from datetime import datetime, timezone
from urllib.parse import quote
import requests
from rapidfuzz import fuzz
from research_portfolio.common import load_config, department_dir, read_json, write_atomic, norm_key
BASE="https://api.openalex.org"
MAILTO=os.getenv("OPENALEX_MAILTO")
def reconstruct(inv):
    if not inv: return None
    words=[]
    for word,positions in inv.items():
        for pos in positions: words.append((pos,word))
    return " ".join(w for _,w in sorted(words))
def get(url,params=None):
    params=dict(params or {});
    if MAILTO: params["mailto"]=MAILTO
    r=requests.get(url,params=params,timeout=45); r.raise_for_status(); return r.json()
def families(pub): return {norm_key(c.get("family_name")) for c in pub.get("contributors",[]) if c.get("family_name")}
def candidate(pub):
    doi=pub.get("doi")
    if doi:
        try: return get(f"{BASE}/works/https://doi.org/{quote(doi,safe='')}") , "doi"
        except requests.HTTPError as e:
            if e.response.status_code != 404: raise
    data=get(f"{BASE}/works",{"search":pub["title"],"per-page":10})
    best=None
    for w in data.get("results",[]):
        ts=fuzz.token_sort_ratio(norm_key(pub["title"]),norm_key(w.get("title")))
        year_ok=not pub.get("year") or not w.get("publication_year") or abs(pub["year"]-w["publication_year"])<=1
        wf={norm_key(a.get("author",{}).get("display_name","").split()[-1]) for a in w.get("authorships",[])}
        shared=bool(families(pub)&wf)
        if ts>=92 and year_ok and shared and (best is None or ts>best[0]): best=(ts,w)
    return (best[1],"title_year_author") if best else (None,None)
def compact_work(w,method):
    return {"openalex_id":w.get("id"),"matched_by":method,"title":w.get("title"),"doi":w.get("doi"),
      "year":w.get("publication_year"),"abstract":reconstruct(w.get("abstract_inverted_index")),
      "primary_topic":(w.get("primary_topic") or {}).get("display_name"),
      "topics":[{"id":t.get("id"),"name":t.get("display_name"),"score":t.get("score")} for t in w.get("topics",[])[:10]],
      "source":((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
      "cited_by_count":w.get("cited_by_count"),"authorships":[{"name":a.get("author",{}).get("display_name"),
      "orcid":a.get("author",{}).get("orcid")} for a in w.get("authorships",[])],
      "retrieved_at":datetime.now(timezone.utc).isoformat()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--refresh",action="store_true"); ap.add_argument("--delay",type=float,default=.12); a=ap.parse_args()
    cfg=load_config(a.config); out=department_dir(cfg); ledger=read_json(out/"ledger.json"); path=out/"openalex_enrichment.json"
    saved={} if a.refresh or not path.exists() else read_json(path).get("works",{})
    pubs=ledger["entities"]["publication"]
    for i,p in enumerate(pubs,1):
        lid=p["ledger_id"]
        if lid in saved: continue
        try:
            w,method=candidate(p); saved[lid]=compact_work(w,method) if w else {"not_found":True,"retrieved_at":datetime.now(timezone.utc).isoformat()}
        except Exception as e: saved[lid]={"error":str(e),"retrieved_at":datetime.now(timezone.utc).isoformat()}
        if i%20==0: write_atomic(path,{"schema_version":1,"works":saved})
        print(f"[{i}/{len(pubs)}] {p['title'][:65]} -> {saved[lid].get('openalex_id','not found')}")
        time.sleep(a.delay)
    result={"schema_version":1,"department":cfg["department_slug"],"generated_at":datetime.now(timezone.utc).isoformat(),"works":saved}
    write_atomic(path,result); print(f"Matched {sum(bool(x.get('openalex_id')) for x in saved.values())}/{len(pubs)} -> {path}"); return 0
if __name__=="__main__": raise SystemExit(main())
