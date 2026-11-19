#!/usr/bin/env python3
"""Apply OpenAlex data to ledger as provenance-preserving additions, then re-index normally."""
import argparse
from research_portfolio.common import load_config, department_dir, read_json, write_atomic

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args(); cfg=load_config(a.config); out=department_dir(cfg)
    ledger=read_json(out/"ledger.json"); enrich=read_json(out/"openalex_enrichment.json")["works"]
    changed=0
    for p in ledger["entities"]["publication"]:
        x=enrich.get(p["ledger_id"],{})
        if not x.get("openalex_id"): continue
        p["openalex_id"]=x["openalex_id"]; p["openalex_topics"]=x.get("topics",[]); p["primary_topic"]=x.get("primary_topic")
        p["cited_by_count"]=x.get("cited_by_count"); p["openalex_source"]=x.get("source")
        prov=p.setdefault("provenance",{})
        for field in ("openalex_id","openalex_topics","primary_topic","cited_by_count","openalex_source"): prov[field]="openalex"
        if not p.get("abstract") and x.get("abstract"):
            p["abstract"]=x["abstract"]; prov["abstract"]="openalex"; changed+=1
        else: prov.setdefault("abstract","digital_measures" if p.get("abstract") else None)
    write_atomic(out/"ledger.json",ledger); print(f"Added {changed} missing abstracts; retained provenance -> {out/'ledger.json'}"); return 0
if __name__=="__main__": raise SystemExit(main())
