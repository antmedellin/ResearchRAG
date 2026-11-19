"""Auditable multi-signal faculty and team matching for Research Portfolio."""
from __future__ import annotations
import itertools, math, os, re
from collections import defaultdict
from datetime import datetime
from neo4j import GraphDatabase
from research_portfolio.retrieval import search as R

CURRENT_YEAR=datetime.now().year
INDIVIDUAL_WEIGHTS={"technical":.45,"grant_relevance":.25,"recency":.15,"evidence_strength":.15}
TEAM_WEIGHTS={"coverage":.45,"member_quality":.25,"collaboration":.15,"complementarity":.15}
STATUS_WEIGHT={"funded":1.0,"currently under review":.65,"not funded":.35}
STOP=set("a an and are as at be by for from grant in into is of on or that the this to with research project program development applications using use new advanced related through".split())
PHRASES=("additive manufacturing","advanced manufacturing","materials characterization","nondestructive evaluation","machine learning","artificial intelligence","computer vision","supply chain","queueing theory","border security","workforce development","engineering education","renewable energy","energy systems","robotics and automation","process safety","digital manufacturing","cybersecurity","data analytics","healthcare","simulation modeling")

def driver():
    return GraphDatabase.driver(os.getenv("NEO4J_URI","bolt://neo4j:7687"),auth=(os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"]))
def clamp(x): return max(0.0,min(1.0,float(x)))
def pscore(p): return float(p.get("rerank_score") if p.get("rerank_score") is not None else p.get("score",0))
def normalize_rerank(x):
    # Cross-encoder scores vary sharply by query. Log transform preserves ordering without treating 0.05 as zero.
    x=max(0.0,float(x)); return clamp(math.log1p(20*x)/math.log(21))
def entity_key(p): return p.get("payload",{}).get("entity_id") or str(p.get("id"))
def serial(p):
    q=p.get("payload",{})
    return {"point_id":str(p.get("id")),"entity_id":q.get("entity_id"),"chunk_type":q.get("chunk_type"),"title":q.get("title") or q.get("display_name"),"year":q.get("year"),"status":q.get("status"),"sponsor":q.get("sponsor"),"text":q.get("text"),"score":pscore(p),"departments":q.get("department_slugs",[])}

def extract_areas(text,max_areas=8):
    t=re.sub(r"[^a-zA-Z0-9+/# -]+"," ",text.lower()); areas=[]
    for phrase in PHRASES:
        if phrase in t: areas.append(phrase.title())
    tokens=[x for x in re.findall(r"[a-zA-Z][a-zA-Z0-9+/#-]+",t) if len(x)>3 and x not in STOP]
    counts=defaultdict(int)
    for n in (3,2):
        for i in range(len(tokens)-n+1):
            phrase=" ".join(tokens[i:i+n]); counts[phrase]+=1
    for phrase,_ in sorted(counts.items(),key=lambda x:(-x[1],-len(x[0]))):
        title=phrase.title()
        if all(phrase not in a.lower() and a.lower() not in phrase for a in areas): areas.append(title)
        if len(areas)>=max_areas: break
    if not areas:
        areas=[x.title() for x in tokens[:max_areas]]
    return areas[:max_areas]

def member_ids(department=None):
    with driver() as d:
        rows,_,_=d.execute_query("""
          MATCH (p:Person)-[:MEMBER_OF]->(dep:Department)
          WITH p,collect(DISTINCT dep.slug) AS departments
          RETURN p.person_id AS id,p.display_name AS name,departments
        """,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return {r["id"]:{"name":r["name"],"departments":r["departments"]} for r in rows}

def relevant_points(query,department=None,candidates=120):
    flt=R.build_filter(types=["bio","interests","publication","grant","presentation"])
    pts=R.hybrid(query,k=candidates,prefetch=max(100,candidates),flt=flt)
    return R.rerank(query,pts,top_n=len(pts))

def collaboration_matrix(ids):
    if not ids:return {}
    with driver() as d:
        rows,_,_=d.execute_query("""
          MATCH (a:Person)-[ra:AUTHORED|INVESTIGATOR_ON|PRESENTED]->(x)<-[rb:AUTHORED|INVESTIGATOR_ON|PRESENTED]-(b:Person)
          WHERE a.person_id IN $ids AND b.person_id IN $ids AND a.person_id < b.person_id
          RETURN a.person_id AS a,b.person_id AS b,
                 count(DISTINCT CASE WHEN x:Grant AND toLower(coalesce(x.status,''))='funded' THEN x END) AS funded_grants,
                 count(DISTINCT CASE WHEN x:Grant THEN x END) AS grants,
                 count(DISTINCT CASE WHEN x:Publication THEN x END) AS publications,
                 count(DISTINCT CASE WHEN x:Presentation THEN x END) AS presentations
        """,ids=ids,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    out={}
    for r in rows:
        value=0
        if r["funded_grants"]: value=max(value,1.0)
        elif r["grants"]: value=max(value,.85)
        if r["publications"]: value=max(value,.65)
        if r["presentations"]: value=max(value,.30)
        out[(r["a"],r["b"]) ]={"score":value,**dict(r)}
    return out

def score_faculty(query,department=None,areas=None,top_n=20):
    areas=areas or extract_areas(query)
    members=member_ids(); pts=relevant_points(query); by=defaultdict(dict)
    for p in pts:
        for pid in p.get("payload",{}).get("person_ids",[]):
            if pid in members:
                k=entity_key(p)
                if k not in by[pid] or pscore(p)>pscore(by[pid][k]):by[pid][k]=p
        if p.get("payload",{}).get("entity_type")=="person":
            pid=p["payload"].get("person_id")
            if pid in members:by[pid][entity_key(p)]=p
    result=[]
    for pid,entities in by.items():
        ev=sorted(entities.values(),key=pscore,reverse=True); research=[p for p in ev if p["payload"].get("chunk_type") in ("publication","grant","presentation")]; profile=[p for p in ev if p["payload"].get("chunk_type") in ("bio","interests")]; grants=[p for p in ev if p["payload"].get("chunk_type")=="grant"]
        tech=clamp(.8*(sum(normalize_rerank(pscore(x)) for x in research[:4])/max(1,min(4,len(research))))+.2*(max([normalize_rerank(pscore(x)) for x in profile] or [0])))
        grant_scores=[]
        for g in grants[:5]: grant_scores.append(normalize_rerank(pscore(g))*STATUS_WEIGHT.get(str(g["payload"].get("status","")).lower(),.5))
        grant=clamp((max(grant_scores) if grant_scores else 0)+.08*max(0,len(grant_scores)-1))
        relevant_years=[x["payload"].get("year") for x in ev[:8] if isinstance(x["payload"].get("year"),int)]
        recency=sum(1.0 if CURRENT_YEAR-y<=2 else .85 if CURRENT_YEAR-y<=5 else .65 if CURRENT_YEAR-y<=8 else .4 if CURRENT_YEAR-y<=12 else .2 for y in relevant_years)/max(1,len(relevant_years))
        diversity=len({x["payload"].get("chunk_type") for x in ev[:8]}); evidence=clamp(.15*min(len(ev),5)+.12*min(diversity,3)+.10*sum(bool(x["payload"].get("abstract")) for x in ev[:5]))
        signals={"technical":tech,"grant_relevance":grant,"recency":recency,"evidence_strength":evidence}
        total=sum(signals[k]*w for k,w in INDIVIDUAL_WEIGHTS.items())
        area_scores={}
        text=" ".join((x["payload"].get("text") or "").lower() for x in ev[:8])
        for area in areas:
            toks=[x for x in area.lower().split() if x not in STOP]; area_scores[area]=clamp(sum(1 for x in toks if x in text)/max(1,len(toks))*.65+tech*.35)
        result.append({"person_id":pid,"name":members[pid]["name"],"departments":members[pid]["departments"],"score":round(total*100,1),"signals":{k:round(v*100,1) for k,v in signals.items()},"areas":{k:round(v*100,1) for k,v in area_scores.items()},"evidence":[serial(x) for x in ev[:8]]})
    return sorted(result,key=lambda x:x["score"],reverse=True)[:top_n]

def team_score(team,areas,collab):
    cover={a:max(m["areas"].get(a,0) for m in team)/100 for a in areas}; coverage=sum(cover.values())/max(1,len(areas)); scores=sorted([m["score"]/100 for m in team],reverse=True); quality=.5*scores[0]+.3*(scores[1] if len(scores)>1 else scores[0])+.2*(sum(scores[2:])/len(scores[2:]) if len(scores)>2 else scores[-1]); pairs=list(itertools.combinations(sorted(m["person_id"] for m in team),2)); pairvals=[collab.get(p,{"score":0})["score"] for p in pairs]; connected=sum(v>0 for v in pairvals); collaboration=clamp((sum(pairvals)/max(1,len(pairvals)))*.65+(connected/max(1,len(team)-1))*.35); winners=[max(team,key=lambda m:m["areas"].get(a,0))["person_id"] for a in areas]; complementarity=clamp(len(set(winners))/max(1,min(len(team),len(areas)))); signals={"coverage":coverage,"member_quality":quality,"collaboration":collaboration,"complementarity":complementarity}; total=sum(signals[k]*w for k,w in TEAM_WEIGHTS.items()); return round(total*100,1),{k:round(v*100,1) for k,v in signals.items()},cover

def recommend(query,department,team_size="auto",top_people=12,top_teams=3):
    areas=extract_areas(query); faculty=score_faculty(query,department,areas,top_people); collab=collaboration_matrix([x["person_id"] for x in faculty]); sizes=range(2,min(6,len(faculty))+1) if team_size=="auto" else [int(team_size)]; teams=[]
    for size in sizes:
        for combo in itertools.combinations(faculty,size):
            score,signals,coverage=team_score(combo,areas,collab); min_cover=sum(v>=.55 for v in coverage.values())/max(1,len(areas)); adjusted=score-(size-2)*1.5 if team_size=="auto" else score
            teams.append({"score":round(score,1),"selection_score":round(adjusted,1),"signals":signals,"coverage":{k:round(v*100,1) for k,v in coverage.items()},"members":[{"person_id":x["person_id"],"name":x["name"],"score":x["score"]} for x in combo],"size":size,"coverage_threshold":round(min_cover*100,1)})
    teams.sort(key=lambda x:(x["selection_score"],x["coverage_threshold"]),reverse=True)
    chosen=[];seen=set()
    for t in teams:
        ids=tuple(x["person_id"] for x in t["members"])
        if ids not in seen:chosen.append(t);seen.add(ids)
        if len(chosen)>=top_teams:break
    def top_unique(key,n=2):
        ranked=sorted(teams,key=key,reverse=True); out=[]; seen=set()
        for t in ranked:
            ids=tuple(sorted(x["person_id"] for x in t["members"]))
            if ids in seen: continue
            seen.add(ids); out.append(t)
            if len(out)>=n: break
        return out
    categories={
      "overall":top_unique(lambda t:(t["selection_score"],t["signals"]["coverage"])),
      "collaboration":top_unique(lambda t:(t["signals"]["collaboration"],t["selection_score"])),
      "complementarity":top_unique(lambda t:(t["signals"]["complementarity"],t["signals"]["coverage"],t["selection_score"])),
    }
    edges=[]
    for (a,b),v in collab.items():
        if v["score"]:edges.append({"source":a,"target":b,**v})
    return {"query":query,"scope":"UTRGV","areas":areas,"weights":{"individual":INDIVIDUAL_WEIGHTS,"team":TEAM_WEIGHTS},"faculty":faculty,"teams":chosen,"team_categories":categories,"collaboration_edges":edges}
