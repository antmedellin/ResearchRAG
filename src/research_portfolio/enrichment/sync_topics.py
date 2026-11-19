#!/usr/bin/env python3
"""Create controlled OpenAlex Topic nodes and ABOUT links after graph sync."""
import argparse, os
from neo4j import GraphDatabase
from research_portfolio.common import load_config, department_dir, read_json, norm_key

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args(); cfg=load_config(a.config); L=read_json(department_dir(cfg)/"ledger.json")
    rows=[]
    for p in L["entities"]["publication"]:
        for t in p.get("openalex_topics",[]):
            if t.get("id") and t.get("name"): rows.append({"ledger_id":p["ledger_id"],"topic_id":"oa:"+t["id"].rsplit('/',1)[-1],"name":t["name"],"score":t.get("score")})
    uri=os.getenv("NEO4J_URI","bolt://neo4j:7687"); auth=(os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"]); db=os.getenv("NEO4J_DATABASE","neo4j")
    with GraphDatabase.driver(uri,auth=auth) as d:
        d.execute_query("CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",database_=db)
        # Find publication by its ledger key IdentityKey; ledger_id is provenance only, so exact title/year key is normally present.
        # We attach by openalex_id when available, which is persisted on Publication after apply + graph sync.
        pubs={p.get("openalex_id"):p for p in L["entities"]["publication"] if p.get("openalex_id")}
        batch=[{"openalex_id":oid,"topics":[{"topic_id":"oa:"+t["id"].rsplit('/',1)[-1],"name":t["name"],"score":t.get("score")} for t in p.get("openalex_topics",[])]} for oid,p in pubs.items()]
        d.execute_query("""UNWIND $rows AS r MATCH (p:Publication {openalex_id:r.openalex_id})
          UNWIND r.topics AS x MERGE (t:Topic {topic_id:x.topic_id}) SET t.name=x.name,t.source='openalex'
          MERGE (p)-[a:ABOUT]->(t) SET a.score=x.score""",rows=batch,database_=db)
    print(f"Synced topics for {len(batch)} publications"); return 0
if __name__=="__main__": raise SystemExit(main())
