#!/usr/bin/env python3
"""UTRGV Research Portfolio web API v1.4."""
import asyncio,json,os,re,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal
import yaml
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import FileResponse
from neo4j import GraphDatabase
from pydantic import BaseModel,field_validator
from research_portfolio.matching.faculty_match import recommend as recommend_faculty_team
from research_portfolio.common import fetch_json,dm_records,index_values
from research_portfolio.retrieval import search as R
from research_portfolio.answer_guard import guard_query_sync, guard_query_async  # RAG_FIXES_V1

ROOT=Path(os.getenv("RAG_REPO_ROOT","/app")).resolve();CONFIG_DIR=ROOT/"config"/"departments";DATA_DIR=ROOT/"data"/"departments";STATIC=Path(__file__).parent/"static"
DM_URL="https://webapps.utrgv.edu/aa/dm/api/DMUser/GetDMList";PROFILE_URL="https://webapps.utrgv.edu/aa/dm/api/DMUser/GetDMuser?username=";SLUG=re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")
TYPES=("bio","interests","education","experience","award","license","publication","grant","presentation");JOBS={};CACHE={"rows":None};app=FastAPI(title="UTRGV Research Portfolio",version="1.5")

# GRAPH_LR_V1: LightRAG-style knowledge graph (/graph-view, /api/kg/*)
import importlib.util as _kg_ilu
from pathlib import Path as _KgPath
_kg_spec = _kg_ilu.spec_from_file_location(
    "rr_graph_kg", _KgPath(__file__).with_name("graph_kg.py"))
_graph_kg = _kg_ilu.module_from_spec(_kg_spec)
_kg_spec.loader.exec_module(_graph_kg)
app.include_router(_graph_kg.router)

SPECIAL={"Department of Manufacturing and Industrial Engineering":"maie","Department of Informatics and Engineering Systems":"ies","Department of Electrical and Computer Engineering":"ece","Department of Mechanical Engineering":"me","Department of Civil Engineering":"ce","Department of Computer Science":"cs"}

def slugify(n):
    if n in SPECIAL:return SPECIAL[n]
    v=n.lower()
    for p in ("department of ","school of ","college of "):
        if v.startswith(p):v=v[len(p):];break
    return re.sub(r"[^a-z0-9]+","-",v).strip("-")
def graph():return GraphDatabase.driver(os.getenv("NEO4J_URI","bolt://neo4j:7687"),auth=(os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"]))
def configs():
    out={}
    for p in CONFIG_DIR.glob("*.yaml"):
        try:
            c=yaml.safe_load(p.read_text()) or {}
            if c.get("department_name"):out[c["department_name"]]=(p,c)
        except Exception:pass
    return out
def directory(force=False):
    if CACHE["rows"] is not None and not force:return CACHE["rows"]
    _,data=fetch_json(DM_URL);counts={};known=configs()
    for row in dm_records(data):
        for n in index_values(row,"DEPARTMENT"):counts[n]=counts.get(n,0)+1
    out=[]
    for n,cnt in sorted(counts.items()):
        p,c=known.get(n,(None,None));slug=c.get("department_slug") if c else slugify(n);valid=False;vp=DATA_DIR/slug/"validation.json"
        if vp.is_file():
            try:valid=json.loads(vp.read_text()).get("passed") is True
            except Exception:pass
        out.append({"name":n,"slug":slug,"count":cnt,"configured":p is not None,"validated":valid})
    CACHE["rows"]=out;return out
def ensure_config(slug):
    if not SLUG.match(slug):raise HTTPException(400,"Invalid organization")
    row=next((x for x in directory() if x["slug"]==slug),None)
    if not row:raise HTTPException(404,"Organization not found")
    p=CONFIG_DIR/f"{slug}.yaml"
    if not p.exists():
        CONFIG_DIR.mkdir(parents=True,exist_ok=True);c={"institution":"University of Texas Rio Grande Valley","department_name":row["name"],"department_slug":slug,"sources":{"faculty":{"type":"digital_measures","source_name":"UTRGV Digital Measures","api_url":DM_URL,"profile_api_url":PROFILE_URL,"department_name":row["name"],"department_aliases":[]}}};p.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True));CACHE["rows"]=None
    return p
def person(pid):
    with graph() as d:r,_,_=d.execute_query("MATCH (p:Person {person_id:$id}) OPTIONAL MATCH (p)-[:MEMBER_OF]->(x:Department) RETURN p.person_id AS person_id,p.display_name AS display_name,p.rank AS rank,collect(DISTINCT {slug:x.slug,name:x.name}) AS departments",id=pid,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    if not r:raise HTTPException(400,"Select a valid faculty member")
    return dict(r[0])

def source(point,i):
    p=point["payload"]
    return {"citation":i,"point_id":str(point["id"]),"entity_id":p.get("entity_id"),"entity_type":p.get("entity_type"),"chunk_type":p.get("chunk_type"),"title":p.get("title") or p.get("display_name"),"text":p.get("text"),"year":p.get("year"),"status":p.get("status"),"sponsor":p.get("sponsor"),"people":p.get("faculty_names",[]),"person_ids":p.get("person_ids",[]),"departments":p.get("department_slugs",[]),"rrf_score":point.get("score"),"rerank_score":point.get("rerank_score"),"payload":p}
def query_one(q,dept,pid,types,mode,limit,since,grant_status,sponsor,investigator_role,publication_type,refereed,has_abstract):
    person_obj=R.resolve_person(pid) if pid else None;depts=[dept] if dept else None
    if person_obj and mode=="answer" and not types:points=R.balanced_person_evidence(q,person_obj["person_id"],candidates=max(20,limit))
    else:
        flt=R.build_filter(types=types or None,depts=depts,since=since,people=[pid] if pid else None,funded_only=False)
        must=(flt or {"must":[]})["must"]
        if grant_status: must.append({"key":"status","match":{"value":grant_status}})
        if sponsor: must.append({"key":"sponsor","match":{"value":sponsor}})
        if publication_type: must.append({"key":"type","match":{"value":publication_type}})
        if refereed is not None: must.append({"key":"refereed","match":{"value":refereed}})
        flt={"must":must} if must else None
        candidates=R.hybrid(q,k=max(40,limit),prefetch=100,flt=flt)
        if investigator_role:
            candidates=[x for x in candidates if any(c.get("role")==investigator_role for c in x.get("payload",{}).get("contributors",[]))]
        if has_abstract is not None:
            candidates=[x for x in candidates if bool(x.get("payload",{}).get("abstract"))==has_abstract]
        points=R.rerank(q,candidates,top_n=len(candidates) if mode=="faculty" else limit)
    if mode=="faculty":
        allowed=R.department_member_ids(depts);rows=R.rank_faculty(points,top=limit,allowed_person_ids=allowed)
        return {"kind":"faculty","faculty":[{"score":score,"person_id":pid,"name":name,"evidence":[source(x,i+1) for i,x in enumerate(ev)]} for score,pid,name,ev in rows]}
    points=R.collapse_by_entity(points,limit)
    answer=R.ask(q,points,person=person_obj) if mode=="answer" and points else None
    return {"kind":"answer" if answer else "evidence","answer":answer,"sources":[source(x,i+1) for i,x in enumerate(points)]}

class QueryRequest(BaseModel):
    mode:Literal["answer","faculty","evidence","publications","grants","collaborators"]
    query:str|None=None;department:str|None=None;person_ids:list[str]=[];types:list[str]=[];since:int|None=None;limit:int=50
    grant_status:str|None=None;sponsor:str|None=None;investigator_role:str|None=None
    publication_type:str|None=None;refereed:bool|None=None;has_abstract:bool|None=None
    @field_validator("types")
    @classmethod
    def vt(cls,v):
        if set(v)-set(TYPES):raise ValueError("Invalid evidence type")
        return list(dict.fromkeys(v))
class JobRequest(BaseModel):department:str;action:Literal["discover","fetch","build","dry_run","apply","validate","auto_ingest"];confirmation:str|None=None

def jobcmd(r):
    cfg=str(ensure_config(r.department).relative_to(ROOT));py=[sys.executable]
    m={"discover":["src/research_portfolio/ledger/discover_faculty.py","--config",cfg],"fetch":["src/research_portfolio/ledger/fetch_profiles.py","--config",cfg],"build":["src/research_portfolio/ledger/build_ledger.py","--config",cfg],"dry_run":["src/research_portfolio/run_department.py","--config",cfg,"--skip-discover","--skip-fetch"],"validate":["-m","research_portfolio.workflow.validate_department","--config",cfg],"apply":["-m","research_portfolio.workflow.safe_run_department","--config",cfg,"--skip-discover","--skip-fetch","--apply"],"auto_ingest":["-m","research_portfolio.workflow.safe_run_department","--config",cfg,"--apply"]}
    return py+m[r.action]
async def execute(j,c):
    x=JOBS[j];x["status"]="running";p=await asyncio.create_subprocess_exec(*c,cwd=ROOT,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT);x["pid"]=p.pid
    while True:
        l=await p.stdout.readline()
        if not l:break
        x["log"]=(x["log"]+l.decode(errors="replace"))[-100000:]
    rc=await p.wait();x["status"]="succeeded" if rc==0 else "failed";x["return_code"]=rc;CACHE["rows"]=None

@app.get("/")
def home():return FileResponse(STATIC/"index.html")
@app.get("/api/options")
def options():return {"evidence_types":TYPES}
@app.get("/api/departments")
def departments(refresh:bool=False):return directory(refresh)
@app.get("/api/people")
def people(q:str=Query(min_length=2),department:str|None=None):
    with graph() as d:r,_,_=d.execute_query("MATCH (p:Person) WHERE p.in_roster=true AND toLower(p.display_name) CONTAINS toLower($q) OPTIONAL MATCH (p)-[:MEMBER_OF]->(x:Department) WITH p,collect(DISTINCT {slug:x.slug,name:x.name}) AS ds WHERE $dep IS NULL OR any(v IN ds WHERE v.slug=$dep) RETURN p.person_id AS person_id,p.display_name AS display_name,p.rank AS rank,ds AS departments ORDER BY p.display_name LIMIT 20",q=q,dep=department,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return [dict(x) for x in r]
@app.get("/api/years")
def years(mode:str,department:str|None=None,person_ids:list[str]=Query(default=[])):
    label="Grant" if mode=="grants" else "Publication" if mode=="publications" else None
    with graph() as d:
        if label:
            rel="INVESTIGATOR_ON" if label=="Grant" else "AUTHORED"
            q=f"MATCH (p:Person)-[:{rel}]->(x:{label}) WHERE (size($ids)=0 OR p.person_id IN $ids) AND ($dept IS NULL OR $dept IN coalesce(x.department_slugs,[])) AND x.year IS NOT NULL RETURN DISTINCT x.year AS year ORDER BY year DESC"
        else:q="MATCH (p:Person)-[:AUTHORED|INVESTIGATOR_ON|PRESENTED]->(x) WHERE (size($ids)=0 OR p.person_id IN $ids) AND ($dept IS NULL OR $dept IN coalesce(x.department_slugs,[])) AND x.year IS NOT NULL RETURN DISTINCT x.year AS year ORDER BY year DESC"
        r,_,_=d.execute_query(q,ids=person_ids,dept=department,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return {"years":[x["year"] for x in r if isinstance(x["year"],int)]}
@app.get("/api/filter-options")
def filter_options(department:str|None=None,person_ids:list[str]=Query(default=[])):
    """Dynamic advanced filter values from the current Neo4j corpus."""
    with graph() as d:
        params={"dept":department,"ids":person_ids}
        grants,_,_=d.execute_query("""
          MATCH (p:Person)-[r:INVESTIGATOR_ON]->(g:Grant)
          WHERE (size($ids)=0 OR p.person_id IN $ids)
            AND ($dept IS NULL OR $dept IN coalesce(g.department_slugs,[]))
          RETURN collect(DISTINCT g.status) AS statuses,
                 collect(DISTINCT g.sponsor) AS sponsors,
                 collect(DISTINCT r.role) AS roles
        """,**params,database_=os.getenv("NEO4J_DATABASE","neo4j"))
        pubs,_,_=d.execute_query("""
          MATCH (p:Person)-[:AUTHORED]->(x:Publication)
          WHERE (size($ids)=0 OR p.person_id IN $ids)
            AND ($dept IS NULL OR $dept IN coalesce(x.department_slugs,[]))
          RETURN collect(DISTINCT x.type) AS types
        """,**params,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    g=dict(grants[0]) if grants else {}; q=dict(pubs[0]) if pubs else {}
    clean=lambda xs:sorted({str(x).strip() for x in (xs or []) if x not in (None,"")},key=str.casefold)
    return {"grant_statuses":clean(g.get("statuses")),"sponsors":clean(g.get("sponsors")),"investigator_roles":clean(g.get("roles")),"publication_types":clean(q.get("types"))}

# structured_records_v13
def structured_records(mode,people,since,limit,grant_status=None,sponsor=None,investigator_role=None,publication_type=None,refereed=None,has_abstract=None):
    rows=[]
    with graph() as d:
        for p in people:
            if mode=="grants":
                recs,_,_=d.execute_query("""
                  MATCH (p:Person {person_id:$pid})-[rel:INVESTIGATOR_ON]->(g:Grant)
                  WHERE ($since IS NULL OR g.year >= $since)
                    AND ($status IS NULL OR g.status=$status)
                    AND ($sponsor IS NULL OR g.sponsor=$sponsor)
                    AND ($role IS NULL OR rel.role=$role)
                  RETURN p.display_name AS person,p.person_id AS person_id,g.title AS title,g.year AS year,
                         g.status AS status,g.sponsor AS sponsor,rel.role AS role,g.amount AS amount,
                         g.abstract AS abstract,g.id AS entity_id
                  ORDER BY g.year DESC,toLower(g.title) LIMIT $limit
                """,pid=p["person_id"],since=since,status=grant_status,sponsor=sponsor,role=investigator_role,limit=min(limit,500),database_=os.getenv("NEO4J_DATABASE","neo4j"))
            elif mode=="publications":
                recs,_,_=d.execute_query("""
                  MATCH (p:Person {person_id:$pid})-[:AUTHORED]->(x:Publication)
                  WHERE ($since IS NULL OR x.year >= $since)
                    AND ($ptype IS NULL OR x.type=$ptype)
                    AND ($refereed IS NULL OR coalesce(x.refereed,false)=$refereed)
                    AND ($has_abs IS NULL OR (x.abstract IS NOT NULL AND trim(x.abstract)<>"")=$has_abs)
                  RETURN p.display_name AS person,p.person_id AS person_id,x.title AS title,x.year AS year,
                         x.type AS type,x.refereed AS refereed,x.publisher AS publisher,x.doi AS doi,
                         x.abstract AS abstract,x.id AS entity_id
                  ORDER BY x.year DESC,toLower(x.title) LIMIT $limit
                """,pid=p["person_id"],since=since,ptype=publication_type,refereed=refereed,has_abs=has_abstract,limit=min(limit,500),database_=os.getenv("NEO4J_DATABASE","neo4j"))
            else:
                recs,_,_=d.execute_query("""
                  MATCH (p:Person {person_id:$pid})-[:AUTHORED|INVESTIGATOR_ON|PRESENTED]->(x)<-[:AUTHORED|INVESTIGATOR_ON|PRESENTED]-(c:Person)
                  WHERE c.person_id<>p.person_id
                  RETURN p.display_name AS person,p.person_id AS person_id,c.display_name AS collaborator,
                         c.person_id AS collaborator_id,count(DISTINCT x) AS shared
                  ORDER BY shared DESC,collaborator LIMIT $limit
                """,pid=p["person_id"],limit=min(limit,500),database_=os.getenv("NEO4J_DATABASE","neo4j"))
            rows.extend(dict(x) for x in recs)
    return {"kind":"records","record_type":mode[:-1] if mode.endswith("s") else mode,"records":rows}

async def _query__impl(r:QueryRequest):
    people=[person(x) for x in r.person_ids]
    if r.mode in ("publications","grants","collaborators"):
        if not people: raise HTTPException(400,"Select at least one person")
        return structured_records(r.mode,people,r.since,r.limit,r.grant_status,r.sponsor,r.investigator_role,r.publication_type,r.refereed,r.has_abstract)
    if not r.query:raise HTTPException(400,"Enter a question or topic")
    if len(people)>1:return {"kind":"comparison","results":[{"person":p,"result":await asyncio.to_thread(query_one,r.query,r.department,p["person_id"],r.types,r.mode,r.limit,r.since,r.grant_status,r.sponsor,r.investigator_role,r.publication_type,r.refereed,r.has_abstract)} for p in people]}
    return await asyncio.to_thread(query_one,r.query,r.department,people[0]["person_id"] if people else None,r.types,r.mode,r.limit,r.since,r.grant_status,r.sponsor,r.investigator_role,r.publication_type,r.refereed,r.has_abstract)


# RAG_FIXES_V1: fuzzy names + cited-answer guard
@app.post("/api/query")
async def query(r: QueryRequest):
    return await guard_query_async(_query__impl, r=r)


@app.get("/api/people/suggest")
def people_suggest(q: str):
    from research_portfolio.person_match import resolve_name
    return resolve_name(q, limit=8)["candidates"]
@app.get("/api/chunks/{point_id}")
def chunk(point_id:str):
    import requests
    res=requests.post(f"{R.QDRANT}/collections/{R.COLL}/points",json={"ids":[point_id],"with_payload":True,"with_vector":False},timeout=30);res.raise_for_status();pts=res.json().get("result",[])
    if not pts:raise HTTPException(404,"Chunk not found")
    return {"point_id":str(pts[0]["id"]),"payload":pts[0].get("payload",{})}
@app.get("/api/graph/options")
def graph_options():
    with graph() as d:
        labels,_,_=d.execute_query("CALL db.labels() YIELD label RETURN label ORDER BY label",database_=os.getenv("NEO4J_DATABASE","neo4j"))
        rels,_,_=d.execute_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType",database_=os.getenv("NEO4J_DATABASE","neo4j"))
        years,_,_=d.execute_query("MATCH (n) WHERE n.year IS NOT NULL RETURN min(n.year) AS min_year,max(n.year) AS max_year",database_=os.getenv("NEO4J_DATABASE","neo4j"))
    y=dict(years[0]) if years else {}
    return {"node_types":[x["label"] for x in labels if x["label"]!="IdentityKey"],"relationship_types":[x["relationshipType"] for x in rels],"min_year":y.get("min_year"),"max_year":y.get("max_year")}

@app.get("/api/graph/overview")
def graph_overview(department:str|None=None,node_types:list[str]=Query(default=[]),relationship_types:list[str]=Query(default=[]),year_from:int|None=None,year_to:int|None=None,limit:int=Query(80,ge=10,le=250)):
    with graph() as d:
        rows,_,_=d.execute_query("""
          MATCH (n)
          WHERE NOT n:IdentityKey
            AND (size($types)=0 OR any(x IN labels(n) WHERE x IN $types))
            AND ($dept IS NULL OR n:Department OR n:Person OR $dept IN coalesce(n.department_slugs,[]))
            AND ($yf IS NULL OR n.year IS NULL OR n.year >= $yf)
            AND ($yt IS NULL OR n.year IS NULL OR n.year <= $yt)
          OPTIONAL MATCH (n:Person)-[:MEMBER_OF]->(dep:Department)
          WITH n,collect(DISTINCT dep.slug) AS pdeps
          WHERE $dept IS NULL OR NOT n:Person OR $dept IN pdeps
          WITH n ORDER BY CASE WHEN n:Person THEN 0 WHEN n:Department THEN 1 ELSE 2 END,coalesce(n.year,0) DESC LIMIT $limit
          WITH collect(n) AS nodes
          UNWIND nodes AS a OPTIONAL MATCH (a)-[r]-(b) WHERE b IN nodes AND (size($rels)=0 OR type(r) IN $rels)
          RETURN nodes,collect(DISTINCT {source:coalesce(startNode(r).person_id,startNode(r).id,startNode(r).topic_id),target:coalesce(endNode(r).person_id,endNode(r).id,endNode(r).topic_id),label:type(r)}) AS edges
        """,dept=department,types=node_types,rels=relationship_types,yf=year_from,yt=year_to,limit=limit,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return _shape_graph(rows[0]["nodes"] if rows else [],rows[0]["edges"] if rows else [])

def _shape_graph(nodes,edges):
    out=[]
    for n in nodes:
        p=dict(n); nid=p.get("person_id") or p.get("id") or p.get("topic_id")
        if not nid: continue
        out.append({"id":nid,"label":p.get("display_name") or p.get("title") or p.get("name") or nid,"type":list(n.labels)[0] if n.labels else "Node","year":p.get("year")})
    valid={x["id"] for x in out}; seen=set(); clean=[]
    for i,e in enumerate(edges):
        if not e.get("source") or not e.get("target") or e["source"] not in valid or e["target"] not in valid: continue
        k=(e["source"],e["target"],e.get("label"))
        if k in seen: continue
        seen.add(k);clean.append({"id":f"e{i}",**e})
    return {"nodes":out,"edges":clean}

@app.get("/api/graph/expand")
def graph_expand(node_id:str,node_types:list[str]=Query(default=[]),relationship_types:list[str]=Query(default=[]),year_from:int|None=None,year_to:int|None=None,offset:int=Query(0,ge=0),limit:int=Query(100,ge=10,le=250)):
    with graph() as d:
        count_rows,_,_=d.execute_query("""MATCH (c) WHERE c.id=$id OR c.person_id=$id OR c.topic_id=$id MATCH (c)-[r]-(n) WHERE NOT n:IdentityKey AND (size($types)=0 OR any(x IN labels(n) WHERE x IN $types)) AND (size($rels)=0 OR type(r) IN $rels) AND ($yf IS NULL OR n.year IS NULL OR n.year >= $yf) AND ($yt IS NULL OR n.year IS NULL OR n.year <= $yt) RETURN count(DISTINCT n) AS total""",id=node_id,types=node_types,rels=relationship_types,yf=year_from,yt=year_to,database_=os.getenv("NEO4J_DATABASE","neo4j"))
        rows,_,_=d.execute_query("""MATCH (c) WHERE c.id=$id OR c.person_id=$id OR c.topic_id=$id MATCH (c)-[r]-(n) WHERE NOT n:IdentityKey AND (size($types)=0 OR any(x IN labels(n) WHERE x IN $types)) AND (size($rels)=0 OR type(r) IN $rels) AND ($yf IS NULL OR n.year IS NULL OR n.year >= $yf) AND ($yt IS NULL OR n.year IS NULL OR n.year <= $yt) WITH c,n,r ORDER BY coalesce(n.year,0) DESC SKIP $offset LIMIT $limit RETURN collect(DISTINCT n) AS nodes,collect(DISTINCT {source:coalesce(startNode(r).person_id,startNode(r).id,startNode(r).topic_id),target:coalesce(endNode(r).person_id,endNode(r).id,endNode(r).topic_id),label:type(r)}) AS edges""",id=node_id,types=node_types,rels=relationship_types,yf=year_from,yt=year_to,offset=offset,limit=limit,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    center_rows,_,_=d.execute_query(
        "MATCH (c) WHERE c.id=$id OR c.person_id=$id OR c.topic_id=$id RETURN c",
        id=node_id,database_=os.getenv("NEO4J_DATABASE","neo4j")
    )
    center=[center_rows[0]["c"]] if center_rows else []
    neighbors=rows[0]["nodes"] if rows else []
    shaped=_shape_graph(center+neighbors,rows[0]["edges"] if rows else [])
    total=count_rows[0]["total"] if count_rows else 0
    shaped.update(total=total,offset=offset,returned=len(shaped["nodes"]),has_more=offset+len(shaped["nodes"])<total)
    return shaped

@app.get("/api/graph/scoped-options")
def graph_scoped_options(department:str|None=None,year_from:int|None=None,year_to:int|None=None):
    """Return only node/relationship types and years available in the active scope."""
    with graph() as d:
        rows,_,_=d.execute_query("""
          MATCH (n)
          WHERE NOT n:IdentityKey
            AND ($dept IS NULL OR n:Department OR n:Person OR $dept IN coalesce(n.department_slugs,[]))
            AND ($yf IS NULL OR n.year IS NULL OR n.year >= $yf)
            AND ($yt IS NULL OR n.year IS NULL OR n.year <= $yt)
          OPTIONAL MATCH (n:Person)-[:MEMBER_OF]->(dep:Department)
          WITH n,collect(DISTINCT dep.slug) AS pdeps
          WHERE $dept IS NULL OR NOT n:Person OR $dept IN pdeps
          OPTIONAL MATCH (n)-[r]-(m)
          WHERE NOT m:IdentityKey
          RETURN collect(DISTINCT labels(n)) AS label_sets,
                 collect(DISTINCT type(r)) AS rels,
                 collect(DISTINCT n.year) AS years
        """,dept=department,yf=year_from,yt=year_to,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    row=dict(rows[0]) if rows else {}
    labels=sorted({label for labels in row.get("label_sets",[]) for label in labels if label!="IdentityKey"})
    rels=sorted({x for x in row.get("rels",[]) if x})
    years=sorted({x for x in row.get("years",[]) if isinstance(x,int)},reverse=True)
    return {"node_types":labels,"relationship_types":rels,"years":years}

@app.get("/api/graph/focus")
def graph_focus(node_id:str,node_types:list[str]=Query(default=[]),relationship_types:list[str]=Query(default=[]),year_from:int|None=None,year_to:int|None=None,limit:int=Query(200,ge=10,le=500)):
    """Return only the selected node and its filtered connected neighborhood."""
    with graph() as d:
        rows,_,_=d.execute_query("""
          MATCH (c) WHERE c.id=$id OR c.person_id=$id OR c.topic_id=$id
          OPTIONAL MATCH (c)-[r]-(n)
          WHERE NOT n:IdentityKey
            AND (size($types)=0 OR any(x IN labels(n) WHERE x IN $types))
            AND (size($rels)=0 OR type(r) IN $rels)
            AND ($yf IS NULL OR n.year IS NULL OR n.year >= $yf)
            AND ($yt IS NULL OR n.year IS NULL OR n.year <= $yt)
          WITH c,n,r ORDER BY coalesce(n.year,0) DESC LIMIT $limit
          RETURN c AS center,collect(DISTINCT n) AS neighbors,
                 collect(DISTINCT {source:coalesce(startNode(r).person_id,startNode(r).id,startNode(r).topic_id),
                                   target:coalesce(endNode(r).person_id,endNode(r).id,endNode(r).topic_id),
                                   label:type(r)}) AS edges
        """,id=node_id,types=node_types,rels=relationship_types,yf=year_from,yt=year_to,limit=limit,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    if not rows: raise HTTPException(404,"Graph node not found")
    row=rows[0]
    nodes=[row["center"]]+[x for x in row["neighbors"] if x is not None]
    return _shape_graph(nodes,row["edges"])

@app.get("/api/graph/search")
def graph_search(q:str=Query(min_length=2,max_length=100),department:str|None=None,limit:int=Query(20,ge=1,le=50)):
    with graph() as d:
        rows,_,_=d.execute_query("""
          MATCH (n)
          WHERE (n:Person OR n:Publication OR n:Grant OR n:Presentation)
            AND toLower(coalesce(n.display_name,n.title,'')) CONTAINS toLower($q)
            AND ($dept IS NULL OR n:Person OR $dept IN coalesce(n.department_slugs,[]))
          OPTIONAL MATCH (n:Person)-[:MEMBER_OF]->(dep:Department)
          WITH n,collect(DISTINCT dep.slug) AS person_depts
          WHERE $dept IS NULL OR NOT n:Person OR $dept IN person_depts
          RETURN coalesce(n.person_id,n.id) AS id,coalesce(n.display_name,n.title) AS label,
                 labels(n)[0] AS type,coalesce(n.department_slugs,person_depts) AS departments
          ORDER BY label LIMIT $limit
        """,q=q,dept=department,limit=limit,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return [dict(x) for x in rows]

@app.get("/api/graph/subgraph")
def graph_subgraph(node_id:str,limit:int=Query(80,ge=10,le=200)):
    with graph() as d:
        rows,_,_=d.execute_query("""
          MATCH (c) WHERE c.id=$id OR c.person_id=$id OR c.topic_id=$id
          OPTIONAL MATCH (c)-[r]-(n)
          WITH c,r,n LIMIT $limit
          RETURN coalesce(c.person_id,c.id,c.topic_id) AS center_id,
                 coalesce(c.display_name,c.title,c.name) AS center_label,labels(c)[0] AS center_type,
                 collect({id:coalesce(n.person_id,n.id,n.topic_id),label:coalesce(n.display_name,n.title,n.name),
                          type:head(labels(n)),rel:type(r),outgoing:startNode(r)=c}) AS neighbors
        """,id=node_id,limit=limit,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    if not rows: raise HTTPException(404,"Graph node not found")
    x=dict(rows[0]);nodes={x["center_id"]:{"id":x["center_id"],"label":x["center_label"],"type":x["center_type"]}};edges=[]
    for i,n in enumerate(x["neighbors"]):
        if not n.get("id"): continue
        nodes[n["id"]]={"id":n["id"],"label":n.get("label") or n["id"],"type":n.get("type") or "Node"}
        src=x["center_id"] if n.get("outgoing") else n["id"]
        dst=n["id"] if n.get("outgoing") else x["center_id"]
        edges.append({"id":f"e{i}","source":src,"target":dst,"label":n.get("rel")})
    return {"center":x["center_id"],"nodes":list(nodes.values()),"edges":edges}

@app.get("/api/graph/node/{node_id}")
def graph_node(node_id:str):
    with graph() as d:
        rows,_,_=d.execute_query("""
          MATCH (n) WHERE n.id=$id OR n.person_id=$id OR n.topic_id=$id
          OPTIONAL MATCH (n)-[r]-(m)
          RETURN labels(n) AS labels,properties(n) AS properties,
                 collect({relationship:type(r),other_id:coalesce(m.person_id,m.id,m.topic_id),
                          other:coalesce(m.display_name,m.title,m.name)}) AS relationships
        """,id=node_id,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    if not rows: raise HTTPException(404,"Graph node not found")
    return dict(rows[0])

@app.get("/api/faculty-match")
def faculty_match(query:str,team_size:str="auto",top_people:int=15,top_teams:int=6):
    if team_size!="auto" and (not team_size.isdigit() or not 2<=int(team_size)<=6):raise HTTPException(400,"team_size must be auto or 2-6")
    return recommend_faculty_team(query,None,team_size,min(max(top_people,8),20),min(max(top_teams,2),6))

@app.post("/api/jobs")
async def jobs(r:JobRequest):
    if r.action in ("apply","auto_ingest") and r.confirmation!=f"APPLY {r.department}":raise HTTPException(400,f"Type APPLY {r.department}")
    c=jobcmd(r);j=uuid.uuid4().hex[:12];JOBS[j]={"status":"queued","log":"","department":r.department};asyncio.create_task(execute(j,c));return {"job_id":j,"department":r.department}
@app.get("/api/jobs/{j}")
def getjob(j:str):return JOBS.get(j) or (_ for _ in ()).throw(HTTPException(404,"Unknown job"))

@app.post("/api/jobs/{j}/cancel")
def cancel(j:str):
    x=JOBS.get(j)
    if not x or x.get("status")!="running":raise HTTPException(409,"Job is not running")
    try:os.kill(x["pid"],15)
    except ProcessLookupError:pass
    x["status"]="cancelled";return {"ok":True}
