"""Auditable multi-signal faculty and team matching for Research Portfolio."""
from __future__ import annotations
import itertools, json, math, os, re
from urllib import request
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

def extract_areas_legacy(text,max_areas=8):
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

def _find_areas_json(content):
    """Return the first JSON object containing an "areas" list."""
    decoder = json.JSONDecoder()
    index = content.find("{")
    while index >= 0:
        try:
            obj, _ = decoder.raw_decode(content[index:])
            if isinstance(obj, dict) and isinstance(obj.get("areas"), list):
                return obj
        except ValueError:
            pass
        index = content.find("{", index + 1)
    raise ValueError("no complete JSON object with areas found")


def extract_areas(text, max_areas=6):
    """Extract weighted research areas with the chat model via llama-swap."""
    max_areas = max(1, min(6, int(max_areas)))
    url = os.getenv("CHAT_URL", "http://llama-swap:8080/v1/chat/completions")
    model = os.getenv("CHAT_MODEL", "chat-vllm")

    prompt = (
        f"Extract the minimum set of distinct, non-overlapping research "
        f"capabilities required by the request. Return 1 to {max_areas} areas; "
        f"prefer 2-4 for a short request. Do not return a broad area that "
        f"duplicates a more specific one. Return one JSON object only, no prose: "
        '{"areas":[{"name":"display name",'
        '"search_query":"faculty search query with synonyms","weight":0.0}]}. '
        "Weights must be nonnegative and reflect importance. "
        "Request: " + text
    )

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system",
             "content": "Extract research capabilities. Return one complete JSON object only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 2000,
    }).encode("utf-8")

    try:
        req = request.Request(url, data=body,
                              headers={"Content-Type": "application/json"},
                              method="POST")
        with request.urlopen(req, timeout=float(os.getenv("CHAT_TIMEOUT", "120"))) as response:
            data = json.loads(response.read().decode("utf-8"))

        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning") or "").strip()

        try:
            parsed = _find_areas_json(content)
        except ValueError:
            parsed = _find_areas_json(reasoning)

        areas = []
        seen = set()
        for item in parsed["areas"][:max_areas]:
            name = str(item.get("name", "")).strip()
            query = str(item.get("search_query", "")).strip()
            if not name or not query or name.casefold() in seen:
                continue
            seen.add(name.casefold())
            try:
                weight = max(0.0, float(item.get("weight", 0.0)))
            except (TypeError, ValueError):
                weight = 0.0
            areas.append({"name": name, "search_query": query, "weight": weight})

        if not areas:
            raise ValueError("no valid research areas")

        total = sum(a["weight"] for a in areas)
        if total <= 0:
            for a in areas:
                a["weight"] = 1.0
            total = float(len(areas))
        for a in areas:
            a["weight"] = round(a["weight"] / total, 6)
        return areas

    except Exception as exc:
        print("Area extraction fallback:", type(exc).__name__, exc, flush=True)
        fallback = extract_areas_legacy(text, max_areas)
        weight = round(1.0 / max(1, len(fallback)), 6)
        return [{"name": a, "search_query": a, "weight": weight} for a in fallback]

def member_ids(department=None):
    with driver() as d:
        rows,_,_=d.execute_query("""
          MATCH (p:Person)-[:MEMBER_OF]->(dep:Department)
          WITH p,collect(DISTINCT dep.slug) AS departments
          RETURN p.person_id AS id,p.display_name AS name,departments
        """,database_=os.getenv("NEO4J_DATABASE","neo4j"))
    return {r["id"]:{"name":r["name"],"departments":r["departments"]} for r in rows}

def relevant_points(query,department=None,candidates=None):
    candidates=int(candidates or os.getenv("RERANK_CANDIDATES","20"))
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

def score_faculty(
    query,
    department=None,
    areas=None,
    top_n=20,
    points=None,
    allowed_person_ids=None,
    ):

    members = member_ids()

    if allowed_person_ids is not None:
        allowed = set(allowed_person_ids)
        members = {
            pid: info
            for pid, info in members.items()
            if pid in allowed
        }

    pts = points if points is not None else relevant_points(query)
    by = defaultdict(dict)

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
       
        signals = {
            "technical": tech,
            "grant_relevance": grant,
            "recency": recency,
            "evidence_strength": evidence,
        }

        total = clamp(
            sum(
                signals[name] * weight
                for name, weight in INDIVIDUAL_WEIGHTS.items()
            )
        )

        result.append({
            "person_id": pid,
            "name": members[pid]["name"],
            "departments": members[pid]["departments"],
            "score": round(total, 4),
            "signals": {
                name: round(value, 4)
                for name, value in signals.items()
            },
            "evidence": [
                serial(point)
                for point in ev[:int(os.getenv("RERANK_CANDIDATES", "20"))]
            ],
        })

    return sorted(result,key=lambda x:x["score"],reverse=True)[:top_n]

def _member_quality(team):
    """Calculate team quality from the unified 0-1 faculty scores."""
    scores = sorted(
        (member["score"] for member in team),
        reverse=True,
    )

    if len(scores) == 1:
        return scores[0]

    if len(scores) == 2:
        return 0.60 * scores[0] + 0.40 * scores[1]

    remaining_mean = sum(scores[2:]) / len(scores[2:])

    return (
        0.50 * scores[0]
        + 0.30 * scores[1]
        + 0.20 * remaining_mean
    )


def _complementarity(team, areas):
    """
    Calculate concentration-based complementarity.

    Returns 0 when one member wins every area and approaches 1 when
    weighted area leadership is evenly distributed across the team.
    """
    if len(team) <= 1 or not areas:
        return 0.0

    won_weights = defaultdict(float)

    for area in areas:
        area_name = area["name"]

        winner = max(
            team,
            key=lambda member: member.get(
                "area_scores", {}
            ).get(area_name, 0.0),
        )

        won_weights[winner["person_id"]] += area["weight"]

    concentration = sum(
        weight * weight
        for weight in won_weights.values()
    )

    maximum_diversity = 1.0 - (1.0 / len(team))

    if maximum_diversity <= 0:
        return 0.0

    return clamp(
        (1.0 - concentration) / maximum_diversity
    )


def team_score(team, areas, collab):
    """Calculate the unified 0-1 score for one faculty team."""
    coverage = {}

    for area in areas:
        area_name = area["name"]

        coverage[area_name] = max(
            member.get("area_scores", {}).get(area_name, 0.0)
            for member in team
        )

    coverage_score = sum(
        area["weight"] * coverage[area["name"]]
        for area in areas
    )

    quality = _member_quality(team)

    pairs = list(itertools.combinations(team, 2))
    pair_scores = []
    connected_pairs = 0

    for left, right in pairs:
        key = tuple(sorted((
            left["person_id"],
            right["person_id"],
        )))

        pair_score = collab.get(
            key,
            {"score": 0.0},
        )["score"]

        pair_scores.append(pair_score)

        if pair_score > 0:
            connected_pairs += 1

    if not pairs:
        collaboration = 0.0
    else:
        collaboration = clamp(
            0.65 * (
                sum(pair_scores) / len(pair_scores)
            )
            + 0.35 * (
                connected_pairs / len(pairs)
            )
        )

    complementarity = _complementarity(team, areas)

    signals = {
        "coverage": clamp(coverage_score),
        "member_quality": clamp(quality),
        "collaboration": collaboration,
        "complementarity": complementarity,
    }

    total = clamp(
        sum(
            signals[name] * weight
            for name, weight in TEAM_WEIGHTS.items()
        )
    )

    return total, signals, coverage


def recommend(
    query,
    department=None,
    team_size="auto",
    top_people=12,
    top_teams=3,
):
    """
    Recommend one faculty member or a team using:
    - one unified overall faculty score;
    - independent faculty rankings for every extracted area;
    - weighted coverage and concentration-based complementarity.
    """
    max_areas = int(
        os.getenv("FACULTY_MATCH_MAX_AREAS", "6")
    )
    area_top_faculty = int(
        os.getenv("FACULTY_MATCH_AREA_TOP_FACULTY", "20")
    )

    max_areas = max(1, min(max_areas, 6))
    area_top_faculty = max(
        1,
        min(area_top_faculty, 20),
    )
    top_people = max(1, min(int(top_people), 50))
    top_teams = max(1, min(int(top_teams), 5))

    areas = extract_areas(
        query,
        max_areas=max_areas,
    )

    # Overall fit to the full request.
    overall = score_faculty(query=query, department=department,
                            top_n=max(area_top_faculty, top_people))
    overall_scores = {f["person_id"]: f["score"] for f in overall}
    profiles = {f["person_id"]: f for f in overall}
    best_profile = {f["person_id"]: f for f in overall}

    # Independent top-N ranking for each extracted area.
    area_rankings = []
    area_scores = defaultdict(dict)
    for area in areas:
        ranked = score_faculty(query=area["search_query"],
                               department=department,
                               top_n=max(area_top_faculty, top_people))
        area_rankings.append(ranked)
        for f in ranked:
            profiles.setdefault(f["person_id"], f)
            current = best_profile.get(f["person_id"])
            if current is None or f["score"] > current["score"]:
                best_profile[f["person_id"]] = f
            area_scores[f["person_id"]][area["name"]] = f["score"]

    # Round-robin pool so every area contributes its best faculty.
    lists = area_rankings + [overall]
    pool, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for ranked in lists:
            if i < len(ranked) and ranked[i]["person_id"] not in seen:
                seen.add(ranked[i]["person_id"])
                pool.append(ranked[i]["person_id"])

    faculty = []
    for pid in pool[:top_people]:
        scores = {x["name"]: area_scores[pid].get(x["name"], 0.0)
                  for x in areas}
        best = max([overall_scores.get(pid, 0.0), *scores.values()])
        faculty.append({
            **best_profile[pid],
            "overall_score": overall_scores.get(pid, 0.0),
            "area_scores": scores,
            "best_fit": round(best, 4),
            "score": round(best, 4),
        })
    faculty.sort(key=lambda x: (x["overall_score"], x["best_fit"]), reverse=True)

    team_pool_size = max(1, int(os.getenv("FACULTY_MATCH_TEAM_POOL", "20")))
    team_ids = set(pool[:top_people][:team_pool_size])
    team_faculty = [f for f in faculty if f["person_id"] in team_ids]

    collab = collaboration_matrix([
        candidate["person_id"]
        for candidate in team_faculty
    ])

    maximum_size = min(6, len(team_faculty))

    if team_size == "auto":
        sizes = range(1, maximum_size + 1)
    else:
        requested_size = int(team_size)

        if requested_size < 1 or requested_size > 6:
            raise ValueError(
                "team_size must be auto or 1-6"
            )

        sizes = [requested_size]

    teams = []

    for size in sizes:
        if size > maximum_size:
            continue

        for combo in itertools.combinations(
            team_faculty,
            size,
        ):
            score, signals, coverage = team_score(
                combo,
                areas,
                collab,
            )

            if team_size == "auto":
                selection_score = (
                    score - (size - 1) * 0.015
                )
            else:
                selection_score = score

            covered_weight = sum(
                area["weight"]
                for area in areas
                if coverage[area["name"]] >= 0.55
            )

            teams.append({
                "score": round(score, 4),
                "selection_score": round(
                    selection_score,
                    4,
                ),
                "signals": {
                    name: round(value, 4)
                    for name, value in signals.items()
                },
                "coverage": {
                    name: round(value, 4)
                    for name, value in coverage.items()
                },
                "members": [
                    {
                        "person_id": member["person_id"],
                        "name": member["name"],
                        "score": member["score"],
                        "area_scores": member[
                            "area_scores"
                        ],
                    }
                    for member in combo
                ],
                "size": size,
                "coverage_threshold": round(
                    covered_weight,
                    4,
                ),
            })

    teams.sort(
        key=lambda team: (
            team["selection_score"],
            team["coverage_threshold"],
        ),
        reverse=True,
    )

    chosen = teams[:top_teams]

    def top_unique(key, count=top_teams):
        ranked = sorted(
            teams,
            key=key,
            reverse=True,
        )

        output = []
        seen = set()

        for team in ranked:
            ids = tuple(sorted(
                member["person_id"]
                for member in team["members"]
            ))

            if ids in seen:
                continue

            seen.add(ids)
            output.append(team)

            if len(output) >= count:
                break

        return output

    categories = {
        "overall": top_unique(
            lambda team: (
                team["selection_score"],
                team["signals"]["coverage"],
            )
        ),
        "collaboration": top_unique(
            lambda team: (
                team["signals"]["collaboration"],
                team["selection_score"],
            )
        ),
        "complementarity": top_unique(
            lambda team: (
                team["signals"]["complementarity"],
                team["signals"]["coverage"],
                team["selection_score"],
            )
        ),
    }

    edges = []

    for (left, right), value in collab.items():
        if value["score"]:
            edges.append({
                "source": left,
                "target": right,
                **value,
            })

    return {
        "query": query,
        "scope": "UTRGV",
        "areas": areas,
        "weights": {
            "individual": INDIVIDUAL_WEIGHTS,
            "team": TEAM_WEIGHTS,
        },
        "faculty": faculty,
        "teams": chosen,
        "team_categories": categories,
        "collaboration_edges": edges,
    }
