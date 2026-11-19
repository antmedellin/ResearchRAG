"""
LightRAG-style knowledge-graph API for ResearchRAG (installed by upgrade_graph_lightrag.py).

Endpoints (all GET, JSON):
  /api/kg/meta                       node labels, relationship types, departments (with counts)
  /api/kg/overview?limit&labels&rels&dept
  /api/kg/neighbors?id&depth&limit&skip&labels&rels
  /api/kg/search?q&limit&labels
  /api/kg/node?id                    full properties + relations
  /graph-view                        the visualization page
"""
import functools
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter()
HTML_PATH = Path(__file__).with_name("graph_view.html")

CAPTION_KEYS = ["display_name", "name", "title", "full_name", "slug",
                "entity_id", "person_id"]
MAX_STR = 300
_state = {"driver": None, "id": None}


# ------------------------------------------------------------------ neo4j
def _driver():
    if _state["driver"] is None:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
        pwd = os.getenv("NEO4J_PASSWORD", "")
        _state["driver"] = GraphDatabase.driver(uri, auth=(user, pwd))
    return _state["driver"]


def _run(cypher, **params):
    with _driver().session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
        return [r.data() for r in s.run(cypher, **params)]


def _idf(var):
    """elementId() on Neo4j 5, toString(id()) on Neo4j 4."""
    if _state["id"] is None:
        try:
            _run("RETURN elementId(null) AS x")
            _state["id"] = "elementId({})"
        except Exception:
            _state["id"] = "toString(id({}))"
    return _state["id"].format(var)


def _csv(v):
    return [x for x in (v or "").split(",") if x]


LABEL_OK = "(size($labels) = 0 OR any(l IN labels({v}) WHERE l IN $labels))"
REL_OK = "(size($rels) = 0 OR type({v}) IN $rels)"


def _lok(v):
    return LABEL_OK.format(v=v)


def _rok(v):
    return REL_OK.format(v=v)


def _deg(v):
    return f"size([({v})-[_r]-(_m) WHERE {_rok('_r')} AND {_lok('_m')} | 1])"


def _plain(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return str(v)                       # neo4j Date/DateTime/Point etc.


def _short(v):
    v = _plain(v)
    if isinstance(v, str) and len(v) > MAX_STR:
        return v[:MAX_STR] + "…"
    if isinstance(v, list):
        return [_short(x) for x in v[:20]] + (["…"] if len(v) > 20 else [])
    return v


def _caption(props, labels):
    for k in CAPTION_KEYS:
        if props.get(k):
            return str(props[k])
    return labels[0] if labels else "node"


def _node(row, full=False):
    props = {k: (_plain(v) if full else _short(v)) for k, v in (row.get("props") or {}).items()
             if not isinstance(v, (bytes, bytearray))}
    labels = row.get("labels") or []
    return {"id": row["id"], "labels": labels, "label": labels[0] if labels else "Node",
            "name": _caption(props, labels), "degree": row.get("degree", 0),
            "props": props}


def _node_ret(v="n"):
    return f"{_idf(v)} AS id, labels({v}) AS labels, properties({v}) AS props, {_deg(v)} AS degree"


def _edges_among(ids, rels, labels):
    if not ids:
        return []
    rows = _run(
        f"""MATCH (a)-[r]->(b)
            WHERE {_idf('a')} IN $ids AND {_idf('b')} IN $ids AND {_rok('r')}
            RETURN {_idf('r')} AS id, {_idf('a')} AS source, {_idf('b')} AS target,
                   type(r) AS type, properties(r) AS props""",
        ids=list(ids), rels=rels, labels=labels)
    for r in rows:
        r["props"] = {k: _short(v) for k, v in (r["props"] or {}).items()}
    return rows


def _graph(node_rows, rels, labels, **extra):
    nodes = {}
    for r in node_rows:
        if r and r.get("id") and r["id"] not in nodes:
            nodes[r["id"]] = _node(r)
    return {"nodes": list(nodes.values()),
            "edges": _edges_among(nodes.keys(), rels, labels), **extra}


def _safe(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return wrap


# ------------------------------------------------------------------ routes
@router.get("/graph-view", include_in_schema=False)
def graph_view():
    return FileResponse(HTML_PATH, media_type="text/html",
                        headers={"Cache-Control": "no-store"})


@router.get("/api/kg/meta")
@_safe
def kg_meta():
    labels = _run("MATCH (n) UNWIND labels(n) AS l RETURN l AS name, count(*) AS count "
                  "ORDER BY count DESC")
    rels = _run("MATCH ()-[r]->() RETURN type(r) AS name, count(*) AS count "
                "ORDER BY count DESC")
    depts = _run("MATCH (d:Department) "
                 "RETURN coalesce(d.slug, d.name) AS slug, "
                 "coalesce(d.name, d.display_name, d.slug) AS name, "
                 "size([(d)<-[:MEMBER_OF]-() | 1]) AS members ORDER BY name")
    total = _run("MATCH (n) RETURN count(n) AS c")[0]["c"]
    return {"labels": labels, "relationships": rels, "departments": depts,
            "total_nodes": total}


@router.get("/api/kg/overview")
@_safe
def kg_overview(limit: int = Query(300, ge=10, le=5000), labels: str = "",
                rels: str = "", dept: str = ""):
    L, R = _csv(labels), _csv(rels)
    if dept:
        seeds = _run(
            f"""MATCH (d:Department) WHERE d.slug = $dept OR d.name = $dept
                OPTIONAL MATCH (d)<-[:MEMBER_OF]-(p)
                WITH collect(DISTINCT d) + collect(DISTINCT p) AS ns
                UNWIND ns AS n WITH DISTINCT n WHERE n IS NOT NULL
                RETURN {_node_ret()}""", dept=dept, labels=L, rels=R)
        seed_ids = [s["id"] for s in seeds]
        nb = _run(
            f"""MATCH (s) WHERE {_idf('s')} IN $seed
                MATCH (s)-[r]-(n) WHERE {_rok('r')} AND {_lok('n')}
                      AND NOT {_idf('n')} IN $seed
                WITH DISTINCT n
                WITH n, {_deg('n')} AS degree ORDER BY degree DESC
                WITH collect(n) AS ns_all, count(n) AS total
                UNWIND ns_all[0..$lim] AS n
                RETURN {_node_ret()}, total""",
            seed=seed_ids, lim=max(0, limit - len(seeds)), labels=L, rels=R)
        total = len(seeds) + (nb[0]["total"] if nb else 0)
        return _graph(seeds + nb, R, L, total=total, mode="department", dept=dept)

    rows = _run(
        f"""MATCH (n) WHERE {_lok('n')}
            WITH n, {_deg('n')} AS degree ORDER BY degree DESC LIMIT $limit
            RETURN {_node_ret()}""", limit=limit, labels=L, rels=R)
    total = _run(f"MATCH (n) WHERE {_lok('n')} RETURN count(n) AS c", labels=L)[0]["c"]
    return _graph(rows, R, L, total=total, mode="overview")


@router.get("/api/kg/neighbors")
@_safe
def kg_neighbors(id: str, depth: int = Query(1, ge=1, le=3),
                 limit: int = Query(500, ge=1, le=5000), skip: int = Query(0, ge=0),
                 labels: str = "", rels: str = ""):
    L, R = _csv(labels), _csv(rels)
    center = _run(f"MATCH (n) WHERE {_idf('n')} = $id RETURN {_node_ret()}",
                  id=id, labels=L, rels=R)
    if not center:
        raise HTTPException(404, "node not found")
    rows = _run(
        f"""MATCH (c) WHERE {_idf('c')} = $id
            MATCH p = (c)-[*1..{int(depth)}]-(n)
            WHERE n <> c AND all(r IN relationships(p) WHERE {_rok('r')})
                  AND all(x IN nodes(p) WHERE x = c OR {_lok('x')})
            WITH DISTINCT n
            WITH n, {_deg('n')} AS degree ORDER BY degree DESC
            WITH collect(n) AS ns_all, count(n) AS total
            UNWIND ns_all[$skip..$end] AS n
            RETURN {_node_ret()}, total""",
        id=id, skip=skip, end=skip + limit, labels=L, rels=R)
    total = rows[0]["total"] if rows else 0
    return _graph(center + rows, R, L, center=id, total=total, skip=skip,
                  returned=len(rows), has_more=skip + len(rows) < total)


@router.get("/api/kg/search")
@_safe
def kg_search(q: str, limit: int = Query(20, ge=1, le=100), labels: str = ""):
    L = _csv(labels)
    q = q.strip()
    if not q:
        return []
    rows = _run(
        f"""MATCH (n) WHERE {_lok('n')} AND any(k IN $keys
                WHERE toLower(toString(n[k])) CONTAINS toLower($q))
            WITH n, {_deg('n')} AS degree ORDER BY degree DESC LIMIT $limit
            RETURN {_node_ret()}""",
        q=q, keys=CAPTION_KEYS, limit=limit, labels=L, rels=[])
    out = [_node(r) for r in rows]
    if len(out) < 5:                     # typo-tolerant fallback
        try:
            from rapidfuzz import fuzz, process
            cand = _run(
                f"""MATCH (n) WHERE {_lok('n')}
                    RETURN {_idf('n')} AS id,
                           coalesce({', '.join('n.' + k for k in CAPTION_KEYS)}) AS name
                    LIMIT 20000""", labels=L)
            names = [c["name"] or "" for c in cand]
            seen = {o["id"] for o in out}
            ids = [cand[i]["id"] for _, s, i in
                   process.extract(q, names, scorer=fuzz.WRatio, limit=limit)
                   if s >= 70 and cand[i]["id"] not in seen]
            if ids:
                more = _run(f"MATCH (n) WHERE {_idf('n')} IN $ids RETURN {_node_ret()}",
                            ids=ids, labels=[], rels=[])
                order = {i: k for k, i in enumerate(ids)}
                out += sorted((_node(r) for r in more), key=lambda x: order[x["id"]])
        except ImportError:
            pass
    for o in out:
        o.pop("props", None)
    return out[:limit]


@router.get("/api/kg/node")
@_safe
def kg_node(id: str):
    rows = _run(f"MATCH (n) WHERE {_idf('n')} = $id RETURN {_node_ret()}",
                id=id, labels=[], rels=[])
    if not rows:
        raise HTTPException(404, "node not found")
    node = _node(rows[0], full=True)
    node["relations"] = _run(
        f"""MATCH (n)-[r]-(m) WHERE {_idf('n')} = $id
            RETURN type(r) AS type,
                   CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS dir,
                   {_idf('m')} AS id, labels(m)[0] AS label,
                   coalesce({', '.join('m.' + k for k in CAPTION_KEYS)}) AS name
            ORDER BY type, name LIMIT 500""", id=id)
    return node
