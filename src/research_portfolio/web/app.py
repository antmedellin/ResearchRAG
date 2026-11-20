#!/usr/bin/env python3
"""UTRGV Research Portfolio web API v1.7."""
import time,asyncio,json,os,re,sys,uuid,sqlite3
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal
import yaml
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import FileResponse
from neo4j import GraphDatabase
from pydantic import BaseModel,field_validator, Field
from research_portfolio.matching.faculty_match import recommend as recommend_faculty_team
from research_portfolio.common import fetch_json,dm_records,index_values
from research_portfolio.retrieval import search as R
from research_portfolio.answer_guard import guard_query_sync, guard_query_async  # RAG_FIXES_V1

ROOT=Path(os.getenv("RAG_REPO_ROOT","/app")).resolve();CONFIG_DIR=ROOT/"config"/"departments";DATA_DIR=ROOT/"data"/"departments";STATIC=Path(__file__).parent/"static"
DM_URL="https://webapps.utrgv.edu/aa/dm/api/DMUser/GetDMList";PROFILE_URL="https://webapps.utrgv.edu/aa/dm/api/DMUser/GetDMuser?username=";SLUG=re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")
TYPES=("bio","interests","education","experience","award","license","publication","grant","presentation");JOBS={};CACHE={"rows":None};app=FastAPI(title="UTRGV Research Portfolio",version="1.7")

# SAVED_RESEARCH_V1: shared research prompt and answer library
SAVED_RESEARCH_DB = ROOT / "data" / "saved_research.db"


def init_saved_research_db():
    """Create the shared saved-research database if it does not exist."""
    SAVED_RESEARCH_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(SAVED_RESEARCH_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

# GRAPH_LR_V1: LightRAG-style knowledge graph (/graph-view, /api/kg/*)
import importlib.util as _kg_ilu
from pathlib import Path as _KgPath
_kg_spec = _kg_ilu.spec_from_file_location(
    "rr_graph_kg", _KgPath(__file__).with_name("graph_kg.py"))
_graph_kg = _kg_ilu.module_from_spec(_kg_spec)
_kg_spec.loader.exec_module(_graph_kg)
app.include_router(_graph_kg.router)
# NO_STORE_V1: never cache API/HTML responses (new departments appear without hard refresh)
@app.middleware("http")
async def _no_store(request,call_next):
    resp=await call_next(request)
    pth=request.url.path
    if pth=="/" or pth.startswith("/api/") or pth=="/graph-view":
        resp.headers["Cache-Control"]="no-store, max-age=0"
    return resp

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

# DEPT_SCOPE_V3: strict (multi-)department scoping helpers
PERSON_CHUNKS={"bio","interests","education","experience","award","license"}
def norm_depts(departments=None,department=None):
    out=[]
    for x in list(departments or [])+([department] if department else []):
        if x and x not in out:out.append(x)
    return out
def dept_member_ids(depts):
    """Validated roster members of the selected department(s) (exact MEMBER_OF match)."""
    if isinstance(depts,str):depts=[depts]
    if not depts:return None
    with graph() as d:
        r,_,_=d.execute_query("MATCH (p:Person)-[:MEMBER_OF]->(x:Department) WHERE x.slug IN $s RETURN collect(DISTINCT p.person_id) AS ids",s=list(depts),database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return set(r[0]["ids"] or []) if r else set()
def dept_member_names(ids):
    if not ids:return []
    with graph() as d:
        r,_,_=d.execute_query("MATCH (p:Person) WHERE p.person_id IN $ids RETURN p.display_name AS n ORDER BY toLower(n)",ids=list(ids),database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return [x["n"] for x in r if x["n"]]
def scope_points(points,allowed,since=None):
    """Drop points that do not involve a selected-department member, and points older than `since`."""
    out=[]
    for x in points:
        p=x.get("payload",{}) or {}
        if allowed is not None and not (set(p.get("person_ids") or [])&allowed):continue
        y=p.get("year")
        if since and isinstance(y,int) and y<since:continue
        out.append(x)
    return out
def source(point,i,allowed=None):
    p=point["payload"];names=p.get("faculty_names",[]) or [];ids=p.get("person_ids",[]) or []
    outside=0
    if allowed is not None and len(names)==len(ids):
        keep=[(n,k) for n,k in zip(names,ids) if k in allowed];outside=len(names)-len(keep)
        names=[n for n,_ in keep];ids=[k for _,k in keep]
    elif allowed is not None:
        ids=[k for k in ids if k in allowed]
    return {"citation":i,"point_id":str(point["id"]),"entity_id":p.get("entity_id"),"entity_type":p.get("entity_type"),"chunk_type":p.get("chunk_type"),"title":p.get("title") or p.get("display_name"),"text":p.get("text"),"year":p.get("year"),"status":p.get("status"),"sponsor":p.get("sponsor"),"amount":p.get("amount"),"people":names,"person_ids":ids,"outside_department_people":outside,"departments":p.get("department_slugs",[]),"rrf_score":point.get("score"),"rerank_score":point.get("rerank_score"),"payload":p}
def query_one(q,depts,pid,types,mode,limit,since,grant_status,sponsor,investigator_role,publication_type,refereed,has_abstract):
    depts=list(depts or []) or None
    person_obj=R.resolve_person(pid) if pid else None
    allowed=dept_member_ids(depts) if depts else None
    if depts and not allowed:
        return {"kind":"evidence","answer":None,"sources":[],"scope_note":f"No validated roster members found for: {', '.join(depts)}."}
    if person_obj and mode=="answer" and not types:
        points=R.balanced_person_evidence(q,person_obj["person_id"],candidates=max(20,limit))
        points=scope_points(points,None,since)
    else:
        flt=R.build_filter(types=types or None,depts=depts,since=since,people=[pid] if pid else None,funded_only=False)
        must=list((flt or {"must":[]}).get("must",[]))
        # strict scope: every chunk must involve at least one member of the selected department(s)
        if allowed and not pid: must.append({"key":"person_ids","match":{"any":sorted(allowed)}})
        if grant_status: must.append({"key":"status","match":{"value":grant_status}})
        if sponsor: must.append({"key":"sponsor","match":{"value":sponsor}})
        if publication_type: must.append({"key":"type","match":{"value":publication_type}})
        if refereed is not None: must.append({"key":"refereed","match":{"value":refereed}})
        flt={"must":must} if must else None
        candidates=R.hybrid(q,k=max(40,limit),prefetch=100,flt=flt)
        candidates=scope_points(candidates,allowed,since)
        if investigator_role:
            candidates=[x for x in candidates if any(c.get("role")==investigator_role and (allowed is None or c.get("person_id") in allowed or not c.get("person_id")) for c in x.get("payload",{}).get("contributors",[]))]
        if has_abstract is not None:
            candidates=[x for x in candidates if bool(x.get("payload",{}).get("abstract"))==has_abstract]
        points=R.rerank(q,candidates,top_n=len(candidates) if mode=="faculty" else limit) if candidates else []
    if mode=="faculty":
        allowed_rank=allowed if allowed is not None else R.department_member_ids(depts)
        rows=R.rank_faculty(points,top=limit,allowed_person_ids=allowed_rank)
        return {"kind":"faculty","faculty":[{"score":score,"person_id":fid,"name":name,"evidence":[source(x,i+1,allowed) for i,x in enumerate(ev)]} for score,fid,name,ev in rows]}
    points=R.collapse_by_entity(points,limit)
    ask_q=q
    if allowed and not person_obj:
        names=dept_member_names(allowed)
        ask_q=(f"{q}\n\n[Scope: department(s) {', '.join(depts)}. Only describe or name these faculty: "
               f"{'; '.join(names)}. Other people in the sources are outside the selected department(s); "
               f"mention them only as co-authors/co-investigators if necessary.]")
    answer=R.ask(ask_q,points,person=person_obj) if mode=="answer" and points else None
    return {"kind":"answer" if answer else "evidence","answer":answer,"sources":[source(x,i+1,allowed) for i,x in enumerate(points)]}

class QueryRequest(BaseModel):
    mode:Literal["answer","faculty","evidence","publications","grants","collaborators","funding"]
    query:str|None=None;department:str|None=None;departments:list[str]=[];person_ids:list[str]=[];types:list[str]=[];since:int|None=None;limit:int=50
    grant_status:str|None=None;sponsor:str|None=None;investigator_role:str|None=None
    publication_type:str|None=None;refereed:bool|None=None;has_abstract:bool|None=None
    @field_validator("types")
    @classmethod
    def vt(cls,v):
        if set(v)-set(TYPES):raise ValueError("Invalid evidence type")
        return list(dict.fromkeys(v))

class SavedResearchRequest(BaseModel):
    prompt: str
    answer: str
    sources: list[dict] = Field(default_factory=list)

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
def people(q:str=Query(min_length=2),department:list[str]=Query(default=[])):
    with graph() as d:r,_,_=d.execute_query("MATCH (p:Person) WHERE p.in_roster=true AND toLower(p.display_name) CONTAINS toLower($q) OPTIONAL MATCH (p)-[:MEMBER_OF]->(x:Department) WITH p,collect(DISTINCT {slug:x.slug,name:x.name}) AS ds WHERE size($dep)=0 OR any(v IN ds WHERE v.slug IN $dep) RETURN p.person_id AS person_id,p.display_name AS display_name,p.rank AS rank,ds AS departments ORDER BY p.display_name LIMIT 20",q=q,dep=department,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return [dict(x) for x in r]
@app.get("/api/years")
def years(mode:str,department:list[str]=Query(default=[]),person_ids:list[str]=Query(default=[])):
    label="Grant" if mode in ("grants","funding") else "Publication" if mode=="publications" else None
    with graph() as d:
        if label:
            rel="INVESTIGATOR_ON" if label=="Grant" else "AUTHORED"
            q=f"MATCH (p:Person)-[:{rel}]->(x:{label}) WHERE (size($ids)=0 OR p.person_id IN $ids) AND (size($dept)=0 OR any(s IN coalesce(x.department_slugs,[]) WHERE s IN $dept)) AND x.year IS NOT NULL RETURN DISTINCT x.year AS year ORDER BY year DESC"
        else:q="MATCH (p:Person)-[:AUTHORED|INVESTIGATOR_ON|PRESENTED]->(x) WHERE (size($ids)=0 OR p.person_id IN $ids) AND (size($dept)=0 OR any(s IN coalesce(x.department_slugs,[]) WHERE s IN $dept)) AND x.year IS NOT NULL RETURN DISTINCT x.year AS year ORDER BY year DESC"
        r,_,_=d.execute_query(q,ids=person_ids,dept=department,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return {"years":[x["year"] for x in r if isinstance(x["year"],int)]}
@app.get("/api/filter-options")
def filter_options(department:list[str]=Query(default=[]),person_ids:list[str]=Query(default=[])):
    """Dynamic advanced filter values from the current Neo4j corpus."""
    with graph() as d:
        params={"dept":department,"ids":person_ids}
        grants,_,_=d.execute_query("""
          MATCH (p:Person)-[r:INVESTIGATOR_ON]->(g:Grant)
          WHERE (size($ids)=0 OR p.person_id IN $ids)
            AND (size($dept)=0 OR any(s IN coalesce(g.department_slugs,[]) WHERE s IN $dept))
          RETURN collect(DISTINCT g.status) AS statuses,
                 collect(DISTINCT g.sponsor) AS sponsors,
                 collect(DISTINCT r.role) AS roles
        """,**params,database_=os.getenv("NEO4J_DATABASE","neo4j"))
        pubs,_,_=d.execute_query("""
          MATCH (p:Person)-[:AUTHORED]->(x:Publication)
          WHERE (size($ids)=0 OR p.person_id IN $ids)
            AND (size($dept)=0 OR any(s IN coalesce(x.department_slugs,[]) WHERE s IN $dept))
          RETURN collect(DISTINCT x.type) AS types
        """,**params,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    g=dict(grants[0]) if grants else {}; q=dict(pubs[0]) if pubs else {}
    clean=lambda xs:sorted({str(x).strip() for x in (xs or []) if x not in (None,"")},key=str.casefold)
    return {"grant_statuses":clean(g.get("statuses")),"sponsors":clean(g.get("sponsors")),"investigator_roles":clean(g.get("roles")),"publication_types":clean(q.get("types"))}

# structured_records_v15: multi-department scoping, since-year, grouped by person (A-Z), collaborator shared records
DEPT_OK="(size($depts)=0 OR any(s IN coalesce({v}.department_slugs,[]) WHERE s IN $depts))"
def structured_records(mode,people,since,limit,grant_status=None,sponsor=None,investigator_role=None,publication_type=None,refereed=None,has_abstract=None,depts=None):
    depts=list(depts or [])
    people=sorted(people,key=lambda p:(p.get("display_name") or "").casefold())
    groups=[];db=os.getenv("NEO4J_DATABASE","neo4j")
    with graph() as d:
        for p in people:
            if mode=="grants":
                recs,_,_=d.execute_query(f"""
                  MATCH (p:Person {{person_id:$pid}})-[rel:INVESTIGATOR_ON]->(g:Grant)
                  WHERE ($since IS NULL OR g.year >= $since)
                    AND {DEPT_OK.format(v='g')}
                    AND ($status IS NULL OR g.status=$status)
                    AND ($sponsor IS NULL OR g.sponsor=$sponsor)
                    AND ($role IS NULL OR rel.role=$role)
                  RETURN p.display_name AS person,p.person_id AS person_id,g.title AS title,g.year AS year,
                         g.status AS status,g.sponsor AS sponsor,rel.role AS role,g.amount AS amount,
                         g.abstract AS abstract,g.id AS entity_id
                  ORDER BY g.year DESC,toLower(g.title) LIMIT $limit
                """,pid=p["person_id"],since=since,depts=depts,status=grant_status,sponsor=sponsor,role=investigator_role,limit=min(limit,500),database_=db)
            elif mode=="publications":
                recs,_,_=d.execute_query(f"""
                  MATCH (p:Person {{person_id:$pid}})-[:AUTHORED]->(x:Publication)
                  WHERE ($since IS NULL OR x.year >= $since)
                    AND {DEPT_OK.format(v='x')}
                    AND ($ptype IS NULL OR x.type=$ptype)
                    AND ($refereed IS NULL OR coalesce(x.refereed,false)=$refereed)
                    AND ($has_abs IS NULL OR (x.abstract IS NOT NULL AND trim(x.abstract)<>"")=$has_abs)
                  RETURN p.display_name AS person,p.person_id AS person_id,x.title AS title,x.year AS year,
                         x.type AS type,x.refereed AS refereed,x.publisher AS publisher,x.doi AS doi,
                         x.abstract AS abstract,x.id AS entity_id
                  ORDER BY x.year DESC,toLower(x.title) LIMIT $limit
                """,pid=p["person_id"],since=since,depts=depts,ptype=publication_type,refereed=refereed,has_abs=has_abstract,limit=min(limit,500),database_=db)
            else:
                recs,_,_=d.execute_query("""
                  MATCH (p:Person {person_id:$pid})-[r1:AUTHORED|INVESTIGATOR_ON|PRESENTED]->(x)<-[r2:AUTHORED|INVESTIGATOR_ON|PRESENTED]-(c:Person)
                  WHERE c.person_id<>p.person_id
                    AND ($since IS NULL OR x.year IS NULL OR x.year >= $since)
                    AND (size($depts)=0 OR size([(c)-[:MEMBER_OF]->(md:Department) WHERE md.slug IN $depts | 1])>0)
                  WITH p,c,x,collect(DISTINCT r1.role)[0] AS my_role,collect(DISTINCT r2.role)[0] AS their_role
                  ORDER BY coalesce(x.year,0) DESC,toLower(coalesce(x.title,''))
                  WITH p,c,collect(DISTINCT {entity_id:x.id,title:x.title,year:x.year,type:head(labels(x)),
                                             status:x.status,sponsor:x.sponsor,amount:x.amount,
                                             my_role:my_role,their_role:their_role}) AS shared_records
                  OPTIONAL MATCH (c)-[:MEMBER_OF]->(cd:Department)
                  WITH p,c,shared_records,collect(DISTINCT cd.slug) AS cdepts
                  RETURN p.display_name AS person,p.person_id AS person_id,c.display_name AS collaborator,
                         c.person_id AS collaborator_id,size(shared_records) AS shared,
                         [s IN shared_records WHERE s.type='Grant' | 1] AS _g,
                         [s IN shared_records WHERE s.type='Publication' | 1] AS _p,
                         [s IN shared_records WHERE s.type='Presentation' | 1] AS _r,
                         shared_records,cdepts AS collaborator_departments
                  ORDER BY shared DESC,toLower(collaborator) LIMIT $limit
                """,pid=p["person_id"],since=since,depts=depts,limit=min(limit,500),database_=db)
            rows=[]
            for x in recs:
                x=dict(x)
                if mode=="collaborators":
                    x["shared_grants"]=len(x.pop("_g") or []);x["shared_publications"]=len(x.pop("_p") or []);x["shared_presentations"]=len(x.pop("_r") or [])
                rows.append(x)
            groups.append({"person":p.get("display_name"),"person_id":p["person_id"],"count":len(rows),"records":rows})
    flat=[r for g in groups for r in g["records"]]
    return {"kind":"records","record_type":mode[:-1] if mode.endswith("s") else mode,"groups":groups,"records":flat,"since":since,"departments":depts}

# FUNDING_SUMMARY_V2: award totals from Qdrant grant payloads (Neo4j fallback for missing amounts)
FUNDING_ORDER=["Awarded","Under review","Denied / not funded","Withdrawn / other"]
def funding_category(status):
    s=(status or "").strip().lower()
    if not s:return "Withdrawn / other"
    if any(k in s for k in ("not funded","unfunded","declin","denied","reject","not award","unsuccessful")):return "Denied / not funded"
    if any(k in s for k in ("funded","award","active","complete","closed")):return "Awarded"
    if any(k in s for k in ("review","pending","submitted","under consideration","in progress")):return "Under review"
    return "Withdrawn / other"
def parse_amount(v):
    if v is None or v=="" or isinstance(v,bool):return None
    if isinstance(v,(int,float)):return float(v)
    try:return float(re.sub(r"[^0-9.\-]","",str(v)) or "x")
    except ValueError:return None
def qdrant_grant_points(person_ids,since):
    import requests
    must=[{"key":"person_ids","match":{"any":list(person_ids)}}]
    if since:must.append({"key":"year","range":{"gte":since}})
    flt={"must":must,"should":[{"key":"chunk_type","match":{"any":["grant","Grant"]}},{"key":"entity_type","match":{"any":["grant","Grant"]}}]}
    out=[];offset=None
    for _ in range(200):
        body={"filter":flt,"limit":256,"with_payload":True,"with_vector":False}
        if offset is not None:body["offset"]=offset
        res=requests.post(f"{R.QDRANT}/collections/{R.COLL}/points/scroll",json=body,timeout=60);res.raise_for_status()
        r=res.json().get("result",{});out.extend(r.get("points",[]));offset=r.get("next_page_offset")
        if offset is None:break
    return out
def funding_summary(people,since,depts=None,sponsor=None,investigator_role=None,grant_status=None):
    depts=list(depts or [])
    people=sorted(people,key=lambda p:(p.get("display_name") or "").casefold())
    ids={p["person_id"] for p in people};names={p["person_id"]:p.get("display_name") for p in people}
    grants={}
    for pt in qdrant_grant_points(ids,since):
        p=pt.get("payload",{}) or {};eid=p.get("entity_id") or p.get("grant_id") or str(pt.get("id"))
        g=grants.setdefault(eid,{"entity_id":eid,"point_id":str(pt.get("id")),"title":p.get("title"),"year":p.get("year"),"status":p.get("status"),"sponsor":p.get("sponsor"),"amount":parse_amount(p.get("amount")),"person_ids":set(),"roles":{},"departments":set(),"amount_source":"qdrant" if p.get("amount") not in (None,"") else None})
        g["person_ids"].update(x for x in (p.get("person_ids") or []) if x in ids)
        g["departments"].update(p.get("department_slugs") or [])
        if g["amount"] is None and p.get("amount") not in (None,""):g["amount"]=parse_amount(p.get("amount"));g["amount_source"]="qdrant"
        for k in ("title","year","status","sponsor"):
            if not g.get(k) and p.get(k):g[k]=p.get(k)
        if p.get("person_id") in ids and p.get("role"):g["roles"][p["person_id"]]=p["role"]
        for c in p.get("contributors") or []:
            if isinstance(c,dict) and c.get("person_id") in ids and c.get("role"):g["roles"][c["person_id"]]=c["role"]
    if grants:
        with graph() as d:
            rows,_,_=d.execute_query("""
              MATCH (p:Person)-[r:INVESTIGATOR_ON]->(g:Grant) WHERE g.id IN $eids AND p.person_id IN $ids
              RETURN g.id AS eid,g.amount AS amount,g.status AS status,g.sponsor AS sponsor,g.year AS year,p.person_id AS pid,r.role AS role
            """,eids=list(grants),ids=list(ids),database_=os.getenv("NEO4J_DATABASE","neo4j"))
        for x in rows:
            g=grants.get(x["eid"])
            if not g:continue
            if g["amount"] is None and parse_amount(x["amount"]) is not None:g["amount"]=parse_amount(x["amount"]);g["amount_source"]="neo4j"
            for k in ("status","sponsor","year"):
                if not g.get(k) and x[k] is not None:g[k]=x[k]
            g["person_ids"].add(x["pid"])
            if x["role"] and x["pid"] not in g["roles"]:g["roles"][x["pid"]]=x["role"]
    rows=[]
    for g in grants.values():
        if since and isinstance(g["year"],int) and g["year"]<since:continue
        if depts and g["departments"] and not (g["departments"]&set(depts)):continue
        if sponsor and g["sponsor"]!=sponsor:continue
        if grant_status and g["status"]!=grant_status:continue
        if investigator_role and not any(v==investigator_role for v in g["roles"].values()):continue
        g["category"]=funding_category(g["status"]);rows.append(g)
    def tally(items):
        t={c:{"category":c,"count":0,"amount":0.0,"missing_amount":0} for c in FUNDING_ORDER}
        for g in items:
            b=t[g["category"]];b["count"]+=1
            if g["amount"] is None:b["missing_amount"]+=1
            else:b["amount"]+=g["amount"]
        return [t[c] for c in FUNDING_ORDER]
    def shape(g,pid=None):
        return {"entity_id":g["entity_id"],"point_id":g["point_id"],"title":g["title"],"year":g["year"],"status":g["status"],"category":g["category"],"sponsor":g["sponsor"],"amount":g["amount"],"amount_source":g["amount_source"],"role":g["roles"].get(pid) if pid else None,"roles":{names.get(k,k):v for k,v in g["roles"].items()},"people":sorted((names.get(x) or x for x in g["person_ids"]),key=str.casefold)}
    sortkey=lambda g:(FUNDING_ORDER.index(g["category"]),-(g["year"] or 0),(g["title"] or "").casefold())
    per=[]
    for p in people:
        mine=sorted((g for g in rows if p["person_id"] in g["person_ids"]),key=sortkey)
        per.append({"person":p.get("display_name"),"person_id":p["person_id"],"totals":tally(mine),"grants":[shape(g,p["person_id"]) for g in mine]})
    allg=sorted(rows,key=sortkey)
    return {"kind":"funding","since":since,"departments":depts,"order":FUNDING_ORDER,
            "group":{"people":[p.get("display_name") for p in people],"totals":tally(allg),"grants":[shape(g) for g in allg]},
            "people":per,
            "note":"Amounts are the full recorded grant amount, not an individual's share. Group totals count each grant once even when several selected people are on it."}

# RECORD_DETAIL_V1: full details for any Grant/Publication/Presentation (clickable rows/cards)
@app.get("/api/record/{entity_id:path}")
def record_detail(entity_id:str):
    with graph() as d:
        rows,_,_=d.execute_query("""
          MATCH (x) WHERE x.id=$id AND (x:Grant OR x:Publication OR x:Presentation)
          OPTIONAL MATCH (p:Person)-[r:AUTHORED|INVESTIGATOR_ON|PRESENTED]->(x)
          OPTIONAL MATCH (p)-[:MEMBER_OF]->(dep:Department)
          WITH x,p,r,collect(DISTINCT dep.slug) AS pdeps
          RETURN labels(x) AS labels,properties(x) AS props,
                 collect(DISTINCT CASE WHEN p IS NULL THEN NULL ELSE {person_id:p.person_id,name:p.display_name,
                         role:r.role,relationship:type(r),departments:pdeps,in_roster:p.in_roster} END) AS people
        """,id=entity_id,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    if not rows:raise HTTPException(404,"Record not found")
    x=dict(rows[0]);props={k:v for k,v in (x["props"] or {}).items() if not isinstance(v,(bytes,bytearray))}
    for k,v in list(props.items()):
        if not isinstance(v,(str,int,float,bool,list,type(None))):props[k]=str(v)
    people=sorted([p for p in x["people"] if p],key=lambda p:(p.get("name") or "").casefold())
    return {"entity_id":entity_id,"type":(x["labels"] or ["Record"])[0],"properties":props,"people":people}

async def _query__impl(r:QueryRequest):
    depts=norm_depts(r.departments,r.department)
    people=[person(x) for x in r.person_ids]
    if depts and people:
        bad=[p["display_name"] for p in people if not any(d.get("slug") in depts for d in p.get("departments",[]))]
        if bad:raise HTTPException(400,f"Not in the selected department(s): {', '.join(bad)}")
    if r.mode=="funding":
        if not people and depts:
            people=[person(x) for x in sorted(dept_member_ids(depts) or [])]
        if not people:raise HTTPException(400,"Select at least one person or a department")
        return await asyncio.to_thread(funding_summary,people,r.since,depts,r.sponsor,r.investigator_role,r.grant_status)
    if r.mode in ("publications","grants","collaborators"):
        if not people: raise HTTPException(400,"Select at least one person")
        return structured_records(r.mode,people,r.since,r.limit,r.grant_status,r.sponsor,r.investigator_role,r.publication_type,r.refereed,r.has_abstract,depts)
    if not r.query:raise HTTPException(400,"Enter a question or topic")
    args=(r.types,r.mode,r.limit,r.since,r.grant_status,r.sponsor,r.investigator_role,r.publication_type,r.refereed,r.has_abstract)
    if len(people)>1:
        people=sorted(people,key=lambda p:(p.get("display_name") or "").casefold())
        return {"kind":"comparison","results":[{"person":p,"result":await asyncio.to_thread(query_one,r.query,depts,p["person_id"],*args)} for p in people]}
    return await asyncio.to_thread(query_one,r.query,depts,people[0]["person_id"] if people else None,*args)


# RAG_FIXES_V1: fuzzy names + cited-answer guard
@app.post("/api/query")
async def query(r: QueryRequest):
    return await guard_query_async(_query__impl, r=r)


@app.get("/api/people/suggest")
def people_suggest(q: str):
    from research_portfolio.person_match import resolve_name
    return resolve_name(q, limit=8)["candidates"]
@app.post("/api/saved-research")
def save_research(r: SavedResearchRequest):
    prompt = r.prompt.strip()
    answer = r.answer.strip()

    if not prompt:
        raise HTTPException(400, "Research prompt cannot be empty")
    if not answer:
        raise HTTPException(400, "Research answer cannot be empty")

    init_saved_research_db()

    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(SAVED_RESEARCH_DB) as conn:
        cursor = conn.execute(
            """
            INSERT INTO saved_research (prompt, answer, sources, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (prompt, answer, json.dumps(r.sources), created_at)
        )
        saved_id = cursor.lastrowid

    return {
        "id": saved_id,
        "prompt": prompt,
        "answer": answer,
        "sources": r.sources,
        "created_at": created_at
    }
@app.get("/api/saved-research")
def list_saved_research():
    init_saved_research_db()

    with sqlite3.connect(SAVED_RESEARCH_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, prompt, answer, sources, created_at
            FROM saved_research
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()

    items = []
    for row in rows:
        items.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "answer": row["answer"],
            "sources": json.loads(row["sources"]),
            "created_at": row["created_at"]
        })

    return {"items": items}
@app.delete("/api/saved-research/{saved_id}")
def delete_saved_research(saved_id: int):
    init_saved_research_db()

    with sqlite3.connect(SAVED_RESEARCH_DB) as conn:
        cursor = conn.execute(
            "DELETE FROM saved_research WHERE id = ?",
            (saved_id,)
        )

        if cursor.rowcount == 0:
            raise HTTPException(404, "Saved research entry not found")

    return {"deleted": saved_id}
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
    if _batch_active():raise HTTPException(409,"An all-departments batch ingest is running; cancel it or wait")  # BATCH_INGEST_V1
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


# ============================================================================
# BATCH_INGEST_V1: sequential auto-ingest of all departments
# ============================================================================
BATCH={"current":None,"task":None}
BATCH_STATE=DATA_DIR/"_batch_ingest.json"
BATCH_PHRASE="APPLY ALL"
class BatchRequest(BaseModel):
    scope:Literal["pending","failed","all","selected"]="pending"
    departments:list[str]=[]          # used when scope == "selected"
    exclude:list[str]=[]
    min_members:int=3
    order:Literal["size_asc","size_desc","name"]="size_asc"
    timeout_minutes:int=180
    stop_after_failures:int=0         # 0 = never stop early
    confirmation:str|None=None
def _now():return datetime.now(timezone.utc).isoformat(timespec="seconds")
def _batch_public(b):
    if not b:return None
    return {k:v for k,v in b.items() if k not in ("task",)}
def _batch_save():
    b=BATCH["current"]
    if not b:return
    try:
        DATA_DIR.mkdir(parents=True,exist_ok=True)
        tmp=BATCH_STATE.with_suffix(".tmp");tmp.write_text(json.dumps(_batch_public(b),indent=1,default=str));tmp.replace(BATCH_STATE)
    except Exception as e:print("batch state save failed:",e,file=sys.stderr)
def _batch_load():
    try:return json.loads(BATCH_STATE.read_text())
    except Exception:return None
def _batch_active():
    b=BATCH["current"];return bool(b and b.get("status") in ("queued","running","cancelling"))
def _dept_validated(slug):
    vp=DATA_DIR/slug/"validation.json"
    try:return json.loads(vp.read_text()).get("passed") is True
    except Exception:return False
# restore last batch on startup; a batch that was running when the server stopped is "interrupted"
_prev=_batch_load()
if _prev:
    if _prev.get("status") in ("queued","running","cancelling"):
        _prev["status"]="interrupted";_prev["finished"]=_prev.get("finished") or _now()
        for _it in _prev.get("items",[]):
            if _it.get("status") in ("queued","running"):_it["status"]="interrupted"
    BATCH["current"]=_prev
def _batch_plan(scope,min_members,order,departments=(),exclude=()):
    rows=directory(True);prev=_batch_load() or {};last={i["slug"]:i.get("status") for i in prev.get("items",[])}
    sel=set(departments or []);exc=set(exclude or [])
    out=[]
    for r in rows:
        slug=r["slug"];valid=_dept_validated(slug);reason=None
        if not SLUG.match(slug or ""):reason="invalid slug"
        elif slug in exc:reason="excluded"
        elif r["count"]<min_members:reason=f"fewer than {min_members} faculty"
        elif scope=="pending" and valid:reason="already validated"
        elif scope=="failed" and last.get(slug) not in ("failed","unvalidated","timeout","interrupted"):reason="did not fail in last batch"
        elif scope=="selected" and slug not in sel:reason="not selected"
        out.append({"slug":slug,"name":r["name"],"count":r["count"],"configured":r.get("configured"),"validated":valid,
                    "last_status":last.get(slug),"include":reason is None,"reason":reason})
    key={"size_asc":lambda x:(x["count"],x["name"].casefold()),"size_desc":lambda x:(-x["count"],x["name"].casefold()),"name":lambda x:x["name"].casefold()}[order]
    return sorted(out,key=key)
async def _batch_run(b):
    fails=0
    try:
        b["status"]="running";_batch_save()
        for it in b["items"]:
            if b.get("cancel"):
                it["status"]="skipped";continue
            slug=it["slug"];it["status"]="running";it["started"]=_now();b["active_slug"]=slug;b["job_id"]=None;_batch_save()
            t0=time.monotonic()
            try:
                c=jobcmd(JobRequest(department=slug,action="auto_ingest",confirmation=f"APPLY {slug}"))
            except HTTPException as e:
                it.update(status="failed",error=str(e.detail),finished=_now(),seconds=0);fails+=1;_batch_save();continue
            j=uuid.uuid4().hex[:12];JOBS[j]={"status":"queued","log":"","department":slug,"batch_id":b["id"]}
            it["job_id"]=j;b["job_id"]=j;_batch_save()
            try:
                await asyncio.wait_for(execute(j,c),timeout=max(1,b["timeout_minutes"])*60)
            except asyncio.TimeoutError:
                pid=JOBS[j].get("pid")
                if pid:
                    try:os.kill(pid,15)
                    except ProcessLookupError:pass
                JOBS[j]["status"]="timeout";JOBS[j]["log"]+=f"\n[batch] timed out after {b['timeout_minutes']} min\n"
            except asyncio.CancelledError:
                raise
            except Exception as e:
                JOBS[j]["status"]="failed";JOBS[j]["log"]+=f"\n[batch] runner error: {e}\n"
            st=JOBS[j].get("status");valid=_dept_validated(slug)
            if b.get("cancel"):it["status"]="cancelled"
            elif st=="succeeded" and valid:it["status"]="succeeded"
            elif st=="succeeded":it["status"]="unvalidated"
            elif st=="timeout":it["status"]="timeout"
            else:it["status"]="failed"
            it.update(return_code=JOBS[j].get("return_code"),finished=_now(),seconds=round(time.monotonic()-t0),validated=valid,log_tail=(JOBS[j].get("log") or "")[-6000:])
            CACHE["rows"]=None
            if it["status"] in ("failed","unvalidated","timeout"):
                fails+=1
                if b["stop_after_failures"] and fails>=b["stop_after_failures"]:
                    b["cancel"]=True;b["stop_reason"]=f"stopped after {fails} failure(s)"
            _batch_save()
        b["status"]="cancelled" if b.get("cancel") else "completed"
    except asyncio.CancelledError:
        b["status"]="cancelled"
    except Exception as e:
        b["status"]="failed";b["error"]=str(e)
    finally:
        b["active_slug"]=None;b["job_id"]=None;b["finished"]=_now();CACHE["rows"]=None;_batch_save()
@app.get("/api/batch-ingest/plan")
def batch_plan(scope:Literal["pending","failed","all","selected"]="pending",min_members:int=Query(3,ge=0,le=500),
               order:Literal["size_asc","size_desc","name"]="size_asc",departments:list[str]=Query(default=[]),exclude:list[str]=Query(default=[])):
    rows=_batch_plan(scope,min_members,order,departments,exclude)
    return {"departments":rows,"included":sum(r["include"] for r in rows),"total":len(rows),"phrase":BATCH_PHRASE,"active":_batch_active()}
@app.post("/api/batch-ingest")
async def batch_start(r:BatchRequest):
    if r.confirmation!=BATCH_PHRASE:raise HTTPException(400,f"Type {BATCH_PHRASE}")
    if _batch_active():raise HTTPException(409,"A batch ingest is already running")
    if any(x.get("status") in ("queued","running") for x in JOBS.values()):raise HTTPException(409,"A department job is running; wait for it to finish")
    plan=[x for x in _batch_plan(r.scope,r.min_members,r.order,r.departments,r.exclude) if x["include"]]
    if r.scope=="selected" and r.departments:
        pos={s:i for i,s in enumerate(r.departments)};plan.sort(key=lambda x:pos.get(x["slug"],1e9))
    if not plan:raise HTTPException(400,"No departments match this plan")
    b={"id":uuid.uuid4().hex[:12],"status":"queued","created":_now(),"finished":None,"scope":r.scope,"min_members":r.min_members,
       "order":r.order,"timeout_minutes":max(5,min(r.timeout_minutes,24*60)),"stop_after_failures":max(0,r.stop_after_failures),
       "cancel":False,"stop_reason":None,"active_slug":None,"job_id":None,
       "items":[{"slug":x["slug"],"name":x["name"],"count":x["count"],"status":"queued"} for x in plan]}
    BATCH["current"]=b;_batch_save();BATCH["task"]=asyncio.create_task(_batch_run(b))
    return _batch_public(b)
@app.get("/api/batch-ingest/current")
def batch_current():
    b=BATCH["current"]
    if not b:return {"batch":None}
    items=b.get("items",[]);done=[i for i in items if i.get("status") not in ("queued","running")]
    secs=[i.get("seconds") for i in items if i.get("seconds")]
    remaining=sum(1 for i in items if i.get("status") in ("queued","running"))
    summary={"total":len(items),"done":len(done),"remaining":remaining,
             **{k:sum(1 for i in items if i.get("status")==k) for k in ("succeeded","failed","unvalidated","timeout","cancelled","skipped","interrupted")},
             "eta_seconds":round(sum(secs)/len(secs)*remaining) if secs and _batch_active() else None}
    log=None
    if b.get("job_id") and b["job_id"] in JOBS:log=(JOBS[b["job_id"]].get("log") or "")[-8000:]
    return {"batch":_batch_public(b),"summary":summary,"active":_batch_active(),"log":log}
@app.post("/api/batch-ingest/cancel")
def batch_cancel():
    b=BATCH["current"]
    if not _batch_active():raise HTTPException(409,"No batch is running")
    b["cancel"]=True;b["status"]="cancelling";b["stop_reason"]="cancelled by user"
    j=JOBS.get(b.get("job_id") or "")
    if j and j.get("pid") and j.get("status")=="running":
        try:os.kill(j["pid"],15)
        except ProcessLookupError:pass
    _batch_save();return {"ok":True}
