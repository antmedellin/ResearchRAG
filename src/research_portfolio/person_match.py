"""
Fuzzy faculty-name resolution for ResearchRAG (installed by apply_rag_fixes.py).

resolve_name("Hirem Moya")                 -> picker / typed-name resolution
find_people_in_question("what does hirem moya research")
canonicalize_question(q, hits)             -> rewrite misspelled span with canonical name
normalize_query_request(req)               -> used by the /api/query guard wrapper
"""
import os
import re
import time
import unicodedata

from rapidfuzz import fuzz, process

TTL_SECONDS = 300
AUTO_ACCEPT = 90
MARGIN = 5
QUESTION_MIN = 86
PERSON_FILTER_MODES = {"", "answer", "ask", "cited", "search", "evidence"}

STOP = {
    "what", "who", "does", "do", "did", "is", "are", "the", "a", "an", "of", "on",
    "in", "for", "and", "or", "with", "to", "about", "research", "grants", "grant",
    "publications", "publication", "papers", "list", "show", "me", "his", "her",
    "their", "has", "have", "work", "works", "dr", "prof", "professor", "faculty",
    "interests", "collaborators", "related", "how", "which", "where", "when",
}

_cache = {"t": 0.0, "keys": [], "owners": [], "people": {}, "last_unique": {}}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _fetch_people():
    from neo4j import GraphDatabase

    d = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        rows, _, _ = d.execute_query(
            """
            MATCH (p:Person)-[:MEMBER_OF]->(dep:Department)
            RETURN p.person_id AS person_id,
                   coalesce(p.display_name, p.name, p.full_name) AS name,
                   coalesce(p.aliases, []) AS aliases,
                   collect(DISTINCT dep.slug) AS departments
            """,
            database_=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
    finally:
        d.close()
    return [dict(r) for r in rows if r["person_id"] and r["name"]]


def _name_variants(name):
    toks = norm(name).split()
    out = {" ".join(toks)}
    if len(toks) >= 3:
        out.add(f"{toks[0]} {toks[-1]}")
        out.add(f"{toks[0]} {toks[-2]}")
        out.add(" ".join(toks[1:]))
    if len(toks) >= 2:
        out.add(f"{toks[-1]} {toks[0]}")
    return {v for v in out if v}


def build_index(people):
    keys, owners, by_id, last_count = [], [], {}, {}
    for p in people:
        by_id[p["person_id"]] = p
        variants = set()
        for n in [p["name"], *(p.get("aliases") or [])]:
            variants |= _name_variants(n)
        for v in variants:
            keys.append(v)
            owners.append(p["person_id"])
        last = norm(p["name"]).split()[-1]
        last_count.setdefault(last, set()).add(p["person_id"])
    last_unique = {k: next(iter(v)) for k, v in last_count.items()
                   if len(v) == 1 and len(k) >= 4}
    _cache.update(t=time.time(), keys=keys, owners=owners, people=by_id,
                  last_unique=last_unique)


def _ensure_index(force=False):
    if force or not _cache["keys"] or time.time() - _cache["t"] > TTL_SECONDS:
        build_index(_fetch_people())


def _ranked(query, limit=8):
    _ensure_index()
    hits = process.extract(norm(query), _cache["keys"], scorer=fuzz.WRatio,
                           limit=limit * 4)
    best = {}
    for _key, score, idx in hits:
        pid = _cache["owners"][idx]
        best[pid] = max(best.get(pid, 0), score)
    ranked = sorted(best.items(), key=lambda x: -x[1])[:limit]
    return [{"person_id": pid, "name": _cache["people"][pid]["name"],
             "departments": _cache["people"][pid]["departments"],
             "score": round(score, 1)} for pid, score in ranked]


def resolve_name(text, limit=5):
    """status: match | ambiguous | none"""
    text = (text or "").strip()
    if not text:
        return {"status": "none", "candidates": []}
    if text.startswith("dm:"):
        _ensure_index()
        p = _cache["people"].get(text)
        return {"status": "match", "person_id": text,
                "name": p["name"] if p else text, "score": 100, "candidates": []}
    cands = _ranked(text, limit)
    if not cands or cands[0]["score"] < 70:
        return {"status": "none", "candidates": cands}
    top = cands[0]
    runner = cands[1]["score"] if len(cands) > 1 else 0
    clear_winner = top["score"] >= 82 and top["score"] - runner >= 25
    if (top["score"] >= AUTO_ACCEPT and top["score"] - runner >= MARGIN) or clear_winner:
        return {"status": "match", **top, "candidates": cands}
    return {"status": "ambiguous", "candidates": cands}


def find_people_in_question(question, max_people=4):
    _ensure_index()
    raw = re.findall(r"[A-Za-zÀ-ÿ'\-\.]+", question or "")
    toks = [norm(t) for t in raw]
    found = []
    for n in (4, 3, 2):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if any(not t for t in gram) or sum(t in STOP for t in gram) >= n - 1:
                continue
            hit = process.extractOne(" ".join(gram), _cache["keys"], scorer=fuzz.ratio)
            if hit and hit[1] >= QUESTION_MIN:
                found.append({"start": i, "end": i + n,
                              "person_id": _cache["owners"][hit[2]],
                              "score": round(hit[1], 1),
                              "span": " ".join(raw[i:i + n])})
    if not found:
        for i, t in enumerate(toks):
            if t in STOP or len(t) < 4:
                continue
            hit = process.extractOne(t, list(_cache["last_unique"]), scorer=fuzz.ratio)
            if hit and hit[1] >= 88:
                found.append({"start": i, "end": i + 1,
                              "person_id": _cache["last_unique"][hit[0]],
                              "score": round(hit[1], 1), "span": raw[i]})
    found.sort(key=lambda h: (-h["score"], -(h["end"] - h["start"])))
    taken, used, out = set(), set(), []
    for h in found:
        rng = set(range(h["start"], h["end"]))
        if rng & taken or h["person_id"] in used:
            continue
        taken |= rng
        used.add(h["person_id"])
        out.append({**h, "name": _cache["people"][h["person_id"]]["name"]})
        if len(out) >= max_people:
            break
    return out


def canonicalize_question(question, hits):
    q = question
    for h in hits:
        q = re.sub(re.escape(h["span"]), h["name"], q, count=1, flags=re.IGNORECASE)
    return q


# ---------------------------------------------------------------- request glue
Q_FIELDS = ("question", "query", "q", "text")
P_FIELDS = ("people", "persons", "person_ids", "person")


def _get(req, names):
    for n in names:
        if isinstance(req, dict) and n in req:
            return n, req[n]
        if not isinstance(req, dict) and hasattr(req, n):
            return n, getattr(req, n)
    return None, None


def _set(req, name, value):
    try:
        if isinstance(req, dict):
            req[name] = value
        else:
            setattr(req, name, value)
        return True
    except Exception:
        return False


def normalize_query_request(req):
    """
    Mutates req in place (canonical question + resolved person ids).
    Returns {"question", "notes", "clarify"}; clarify is a dict or None.
    """
    out = {"question": None, "notes": [], "clarify": None}
    qf, question = _get(req, Q_FIELDS)
    pf, people = _get(req, P_FIELDS)
    _, mode = _get(req, ("mode",))
    out["question"] = question
    try:
        _ensure_index()
    except Exception as e:                       # Neo4j down: don't block queries
        out["notes"].append(f"Name matching unavailable ({type(e).__name__})")
        return out

    # a) picker / typed names
    if people:
        single = isinstance(people, str)
        items = [people] if single else list(people)
        resolved = []
        for p in items:
            m = resolve_name(str(p))
            if m["status"] != "match":
                out["clarify"] = {"term": str(p), "candidates": m["candidates"]}
                return out
            resolved.append(m["person_id"])
            if not str(p).startswith("dm:") and m["name"].lower() != str(p).lower():
                out["notes"].append(f'Interpreted "{p}" as {m["name"]}')
        _set(req, pf, resolved[0] if single else resolved)

    # b) names inside the question
    if question and qf:
        hits = find_people_in_question(question)
        if hits:
            newq = canonicalize_question(question, hits)
            if newq != question:
                _set(req, qf, newq)
                out["question"] = newq
            for h in hits:
                if norm(h["span"]) != norm(h["name"]):
                    out["notes"].append(f'Interpreted "{h["span"]}" as {h["name"]}')
            if (not people and pf and str(mode or "").lower() in PERSON_FILTER_MODES):
                ids = [h["person_id"] for h in hits]
                cur = getattr(req, pf, None) if not isinstance(req, dict) else req.get(pf)
                _set(req, pf, ids[0] if isinstance(cur, str) else ids)
    return out
