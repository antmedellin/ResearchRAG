#!/usr/bin/env python3
"""Structured Neo4j queries: exhaustive lists/counts/collaborators, not semantic RAG."""
import argparse, os
from neo4j import GraphDatabase
URI=os.getenv("NEO4J_URI","bolt://neo4j:7687"); DB=os.getenv("NEO4J_DATABASE","neo4j")
AUTH=(os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"])
def resolve(s,person):
    rows=list(s.run("""MATCH (p:Person) WHERE p.person_id=$v OR toLower(p.display_name)=toLower($v)
      RETURN p.person_id AS id,p.display_name AS name LIMIT 2""",v=person))
    if len(rows)!=1: raise ValueError(f"Expected one person for {person!r}; found {len(rows)}")
    return dict(rows[0])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--person",required=True); ap.add_argument("--since",type=int); ap.add_argument("--limit",type=int,default=100)
    g=ap.add_mutually_exclusive_group(required=True); g.add_argument("--list-publications",action="store_true"); g.add_argument("--list-grants",action="store_true"); g.add_argument("--collaborators",action="store_true")
    a=ap.parse_args()
    with GraphDatabase.driver(URI,auth=AUTH) as d, d.session(database=DB) as s:
        p=resolve(s,a.person); print(f"{p['name']} ({p['id']})\n")
        if a.list_publications:
            rows=s.run("""MATCH (:Person {person_id:$id})-[:AUTHORED]->(x:Publication)
              WHERE $since IS NULL OR x.year >= $since RETURN x.title AS title,x.year AS year,x.type AS type,x.doi AS doi
              ORDER BY year DESC,toLower(title) LIMIT $limit""",id=p["id"],since=a.since,limit=a.limit)
        elif a.list_grants:
            rows=s.run("""MATCH (:Person {person_id:$id})-[r:INVESTIGATOR_ON]->(x:Grant)
              WHERE $since IS NULL OR x.year >= $since RETURN x.title AS title,x.year AS year,x.status AS status,
              x.sponsor AS sponsor,r.role AS role ORDER BY year DESC,toLower(title) LIMIT $limit""",id=p["id"],since=a.since,limit=a.limit)
        else:
            rows=s.run("""MATCH (:Person {person_id:$id})-[:AUTHORED|INVESTIGATOR_ON|PRESENTED]->(x)
              <-[:AUTHORED|INVESTIGATOR_ON|PRESENTED]-(c:Person) WHERE c.person_id<>$id
              RETURN c.display_name AS collaborator,c.person_id AS person_id,count(DISTINCT x) AS shared
              ORDER BY shared DESC,collaborator LIMIT $limit""",id=p["id"],limit=a.limit)
        for i,r in enumerate(rows,1): print(f"{i:3}. "+" | ".join(f"{k}={v}" for k,v in dict(r).items() if v is not None))
    return 0
if __name__=="__main__": raise SystemExit(main())
