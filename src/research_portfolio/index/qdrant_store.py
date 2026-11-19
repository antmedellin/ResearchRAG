"""Qdrant collection (dense mxbai + server-side BM25) built from Neo4j read-backs.
One point per chunk; payload carries every filterable field. Point IDs are deterministic."""
import os, uuid
from collections import defaultdict
import requests
from research_portfolio.common import keep

QDRANT      = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLL        = os.environ.get("RP_COLLECTION", "rp_chunks")
EMBED_URL   = os.environ.get("DENSE_EMBEDDING_URL", "http://ollama:11434/api/embed")
EMBED_MODEL = os.environ.get("DENSE_EMBEDDING_MODEL", "mxbai-embed-large:335m")
AVG_LEN     = float(os.environ.get("BM25_AVG_LEN", 120))
BM25_OPTS   = {"avg_len": AVG_LEN, "language": "english"}       # identical at index and query time
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_WORDS, OVERLAP = 300, 50                                    # under mxbai's 512-token limit

PAYLOAD_INDEXES = {
    "keyword": ["entity_id", "entity_type", "chunk_type", "person_ids", "lead_person_ids", "department_slugs",
                "rank", "type", "status", "sponsor", "sponsor_type", "doi", "flags", "ingest_run"],
    "integer": ["year", "start_year", "end_year", "latest_pub_year", "publication_count", "grant_count"],
    "float":   ["amount"],
    "bool":    ["has_abstract", "refereed", "public", "extramural", "multi_department"],
}

def check(r):
    if not r.ok:
        raise RuntimeError(f"{r.request.method} {r.url} -> {r.status_code}: {r.text[:500]}")
    return r

def embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        embs = check(requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": part},
                                   timeout=300)).json().get("embeddings") or []
        if len(embs) != len(part):
            raise RuntimeError(f"expected {len(part)} embeddings, got {len(embs)} from {EMBED_URL}")
        out.extend(embs)
    return out

def bm25(text):
    return {"text": text, "model": "qdrant/bm25", "options": BM25_OPTS}

# ---------------- chunk builders ----------------
def split(header, body):
    words = (body or "").split()
    if not words:
        return [header]
    step = MAX_WORDS - OVERLAP
    return [f"{header}\n{' '.join(words[i:i + MAX_WORDS])}" for i in range(0, max(len(words) - OVERLAP, 1), step)]

def make(entity_id, chunk_type, texts, payload):
    return [{"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"rp:{entity_id}:{chunk_type}:{i}")), "text": t,
             "payload": {**payload, "entity_id": entity_id, "chunk_type": chunk_type,
                         "chunk_index": i, "chunk_count": len(texts), "text": t}}
            for i, t in enumerate(texts)]

def join(items, fields, sep=", "):
    return " | ".join(sep.join(str(i[f]) for f in fields if i.get(f)) for i in items)

def person_chunks(p):
    pid = p["person_id"]
    ranks = sorted({m["rank"] for m in p.get("departments", []) if m.get("rank")})
    depts = [m.get("name") or m["slug"] for m in p.get("departments", [])]
    head = f"{p.get('display_name')}, {'; '.join(ranks) or 'Faculty'} ({'; '.join(depts) or 'UTRGV'})."
    base = keep({
        "entity_type": "person", "person_ids": [pid], "person_id": pid,
        "display_name": p.get("display_name"), "family_name": p.get("family_name"), "email": p.get("email"),
        "rank": ranks, "department_slugs": p.get("department_slugs", []),
        "departments": p.get("departments"), "flags": p.get("flags"),
        "research_keywords": p.get("research_keywords"),
        "publication_count": p.get("publication_count"), "grant_count": p.get("grant_count"),
        "funded_grant_count": p.get("funded_grant_count"), "latest_pub_year": p.get("latest_pub_year"),
        "multi_department": len(p.get("department_slugs", [])) > 1,
    })
    out = []
    if p.get("bio"):
        out += make(pid, "bio", split(head, p["bio"]), base)
    interests = "; ".join(filter(None, [p.get("research_interests_text"), p.get("teaching_interests") and
                                        f"Teaching: {p['teaching_interests']}"]))
    orgs = join(p.get("org_memberships") or [], ("name", "leadership"))
    if interests or orgs:
        out += make(pid, "interests", split(f"{head} Research interests:",
                                            interests + (f". Memberships: {orgs}" if orgs else "")), base)
    if p.get("education"):
        text = join(p["education"], ("degree", "major", "support_area", "institution", "year"))
        diss = [e["dissertation"] for e in p["education"] if e.get("dissertation")]
        out += make(pid, "education", split(f"{head} Education:", text + (f". Dissertation: {'; '.join(diss)}" if diss else "")),
                    {**base, "education": p["education"]})
    if p.get("awards"):
        out += make(pid, "award", split(f"{head} Awards and honors:",
                                        join(p["awards"], ("name", "organization", "year", "description"))),
                    {**base, "awards": p["awards"]})
    if p.get("experience"):
        out += make(pid, "experience", split(f"{head} Professional experience:",
                                             join(p["experience"], ("title", "organization", "start_year",
                                                                    "end_year", "description"))),
                    {**base, "experience": p["experience"]})
    if p.get("licenses"):
        base_lic = {**base, "licenses": p["licenses"]}
        out += make(pid, "license", [f"{head} Licenses: " + join(p["licenses"], ("title", "organization", "year"))], base_lic)
    return out

ENTITY_SKIP = {"id", "kind", "contributor_names", "family_names", "last_run", "identity_keys"}

def entity_chunks(x):
    kind = x["kind"]
    payload = {k: v for k, v in x.items() if k not in ENTITY_SKIP}
    payload.update(entity_type=kind, has_abstract=bool(x.get("abstract")),
                   multi_department=len(x.get("department_slugs", [])) > 1,
                   contributor_names=[c.get("name") for c in x.get("contributors", []) if c.get("name")])
    payload = keep(payload) | {"has_abstract": payload["has_abstract"], "multi_department": payload["multi_department"]}
    yr = f" ({x['year']})" if x.get("year") else ""
    who = f" Faculty: {', '.join(x['faculty_names'])}." if x.get("faculty_names") else ""
    if kind == "publication":
        head = f"{x['title']}{yr}. {x.get('type') or 'Publication'}" + (f", {x['publisher']}" if x.get("publisher") else "") + f".{who}"
    elif kind == "grant":
        span = "-".join(str(v) for v in (x.get("start_year"), x.get("end_year")) if v)
        head = (f"Grant: {x['title']}. Sponsor: {x.get('sponsor') or 'unknown'}"
                + (f" ({span})" if span else "") + (f". Status: {x['status']}" if x.get("status") else "") + f".{who}")
    else:
        head = f"Presentation: {x['title']}{yr}" + (f". {x['event']}" if x.get("event") else "") + f".{who}"
    return make(x["id"], kind, split(head, x.get("abstract")), payload)

# ---------------- collection ops ----------------
def ensure_collection(dim, recreate=False):
    url = f"{QDRANT}/collections/{COLL}"
    if recreate:
        requests.delete(url, timeout=60)
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        check(requests.put(url, json={"vectors": {"default": {"size": dim, "distance": "Cosine"}},
                                      "sparse_vectors": {"bm25": {"modifier": "idf"}}}, timeout=60))
        print(f"Created collection {COLL}")
    else:
        params = check(r).json()["result"]["config"]["params"]
        if params["vectors"]["default"]["size"] != dim or "bm25" not in (params.get("sparse_vectors") or {}):
            raise RuntimeError(f"{COLL} layout mismatch (dim/bm25); re-run with --recreate-collection")
    for schema, fields in PAYLOAD_INDEXES.items():
        for f in fields:
            check(requests.put(f"{url}/index?wait=true", json={"field_name": f, "field_schema": schema}, timeout=60))

def upsert(chunks, run, batch=64):
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vecs = embed([c["text"] for c in part])                     # documents: no query prefix
        check(requests.put(f"{QDRANT}/collections/{COLL}/points?wait=true", json={"points": [
            {"id": c["id"], "payload": {**c["payload"], "ingest_run": run},
             "vector": {"default": v, "bm25": bm25(c["text"])}} for c, v in zip(part, vecs)]}, timeout=300))
        print(f"  upserted {min(i + batch, len(chunks))}/{len(chunks)}", flush=True)

def _by_ids(ids, size=500):
    ids = list(ids)
    for i in range(0, len(ids), size):
        yield ids[i:i + size]

def delete_stale(entity_ids, run):
    """Remove chunks of re-indexed entities that this run no longer produced."""
    for part in _by_ids(entity_ids):
        check(requests.post(f"{QDRANT}/collections/{COLL}/points/delete?wait=true", json={"filter": {
            "must": [{"key": "entity_id", "match": {"any": part}}],
            "must_not": [{"key": "ingest_run", "match": {"value": run}}]}}, timeout=300))

def apply_graph_changes(result):
    """Propagate department_slugs changes to existing points; delete removed entities/departed people."""
    by_slugs = defaultdict(list)
    for eid, slugs in result["slug_changes"]:
        by_slugs[tuple(slugs)].append(eid)
    for slugs, ids in by_slugs.items():
        for part in _by_ids(ids):
            check(requests.post(f"{QDRANT}/collections/{COLL}/points/payload?wait=true", json={
                "payload": {"department_slugs": list(slugs), "multi_department": len(slugs) > 1},
                "filter": {"must": [{"key": "entity_id", "match": {"any": part}}]}}, timeout=120))
    gone = list(result["removed_entities"]) + list(result["departed"])
    for part in _by_ids(gone):
        check(requests.post(f"{QDRANT}/collections/{COLL}/points/delete?wait=true", json={
            "filter": {"must": [{"key": "entity_id", "match": {"any": part}}]}}, timeout=120))

def count():
    return check(requests.post(f"{QDRANT}/collections/{COLL}/points/count", json={"exact": True},
                               timeout=60)).json()["result"]["count"]
