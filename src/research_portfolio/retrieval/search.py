#!/usr/bin/env python3
"""Hybrid research-portfolio search.

Features:
- Dense + BM25 retrieval from Qdrant, fused with RRF
- Batched TEI reranking (max batch 32)
- Person lookup by Digital Measures ID or display name through Neo4j
- Balanced person summaries: profile evidence first, then publications/grants
- Department-scoped faculty ranking using Neo4j MEMBER_OF relationships

Examples:
  python search.py "What does Hiram Moya research?" --person "Hiram Moya" --ask
  python search.py "research interests" --person dm:1478901 --types bio,interests
  python search.py "defense manufacturing" --faculty --dept maie
"""
import argparse
import os
from collections import defaultdict

import requests
from neo4j import GraphDatabase

from research_portfolio.index.qdrant_store import (
    QDRANT,
    COLL,
    QUERY_PREFIX,
    check,
    embed,
    bm25,
)

RERANK_URL = os.environ.get("RERANK_URL")
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL",
    "BAAI/bge-reranker-v2-m3"
)
RERANK_ROUTE_MODEL = os.environ.get(
    "RERANK_ROUTE_MODEL",
    RERANK_MODEL,
)
from research_portfolio.llm_endpoints import CHAT_URL
from research_portfolio.llm_client import post_chat
CHAT_MODEL = os.environ.get("CHAT_MODEL")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

PROFILE_TYPES = ["bio", "interests", "education", "experience", "award", "license"]
RESEARCH_TYPES = ["publication", "grant", "presentation"]
PROFILE_SET = set(PROFILE_TYPES)


def neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


def resolve_person(value):
    """Resolve dm:<id> or a faculty name to one canonical Person.

    Exact case-insensitive display-name matches win. Otherwise, a partial-name
    query is permitted only when it returns exactly one person.
    """
    value = value.strip()
    with neo4j_driver() as driver:
        if value.startswith("dm:"):
            records, _, _ = driver.execute_query(
                """
                MATCH (p:Person {person_id: $value})
                OPTIONAL MATCH (p)-[:MEMBER_OF]->(d:Department)
                RETURN p.person_id AS person_id,
                       p.display_name AS display_name,
                       collect(DISTINCT d.slug) AS department_slugs
                """,
                value=value,
                database_=NEO4J_DATABASE,
            )
        else:
            records, _, _ = driver.execute_query(
                """
                MATCH (p:Person)
                WHERE toLower(p.display_name) = toLower($value)
                OPTIONAL MATCH (p)-[:MEMBER_OF]->(d:Department)
                RETURN p.person_id AS person_id,
                       p.display_name AS display_name,
                       collect(DISTINCT d.slug) AS department_slugs
                """,
                value=value,
                database_=NEO4J_DATABASE,
            )
            if not records:
                records, _, _ = driver.execute_query(
                    """
                    MATCH (p:Person)
                    WHERE toLower(p.display_name) CONTAINS toLower($value)
                    OPTIONAL MATCH (p)-[:MEMBER_OF]->(d:Department)
                    RETURN p.person_id AS person_id,
                           p.display_name AS display_name,
                           collect(DISTINCT d.slug) AS department_slugs
                    ORDER BY p.display_name
                    LIMIT 10
                    """,
                    value=value,
                    database_=NEO4J_DATABASE,
                )

    if not records:
        raise ValueError(f"No person found for {value!r}")
    if len(records) > 1:
        options = ", ".join(
            f"{r['display_name']} ({r['person_id']})" for r in records
        )
        raise ValueError(f"Person name {value!r} is ambiguous: {options}")
    return dict(records[0])

def list_person_publications(person_id, since=None, limit=50):
    """Return publications authored by one person, newest first."""

    cypher = """
    MATCH (p:Person {person_id: $person_id})-[:AUTHORED]->(pub:Publication)
    WHERE $since IS NULL OR pub.year >= $since
    RETURN
        pub.id AS entity_id,
        pub.title AS title,
        pub.year AS year,
        pub.type AS type,
        pub.status AS status,
        pub.publisher AS publisher,
        pub.doi AS doi,
        pub.url AS url,
        pub.abstract AS abstract
    ORDER BY
        pub.year DESC,
        toLower(pub.title)
    LIMIT $limit
    """

    with neo4j_driver() as driver:
        records, _, _ = driver.execute_query(
            cypher,
            person_id=person_id,
            since=since,
            limit=limit,
            database_=NEO4J_DATABASE,
        )

    return [dict(record) for record in records]

def department_member_ids(slugs):
    """Return the authoritative set of faculty IDs in the requested departments."""
    if not slugs:
        return None
    with neo4j_driver() as driver:
        records, _, _ = driver.execute_query(
            """
            MATCH (p:Person)-[:MEMBER_OF]->(d:Department)
            WHERE d.slug IN $slugs
            RETURN DISTINCT p.person_id AS person_id
            """,
            slugs=slugs,
            database_=NEO4J_DATABASE,
        )
    return {r["person_id"] for r in records}


def build_filter(types=None, depts=None, since=None, people=None, funded_only=False):
    must = []
    if types:
        must.append({"key": "chunk_type", "match": {"any": types}})
    if depts:
        must.append({"key": "department_slugs", "match": {"any": depts}})
    if since:
        must.append({"key": "year", "range": {"gte": since}})
    if people:
        must.append({"key": "person_ids", "match": {"any": people}})
    if funded_only:
        must.append({"key": "status", "match": {"value": "Funded"}})
    return {"must": must} if must else None


def hybrid(query, k=20, prefetch=80, flt=None):
    qvec = embed([QUERY_PREFIX + query])[0]

    def pf(q, using):
        row = {"query": q, "using": using, "limit": prefetch}
        if flt:
            row["filter"] = flt
        return row

    response = check(
        requests.post(
            f"{QDRANT}/collections/{COLL}/points/query",
            json={
                "prefetch": [pf(bm25(query), "bm25"), pf(qvec, "default")],
                "query": {"fusion": "rrf"},
                "limit": k,
                "with_payload": True,
            },
            timeout=60,
        )
    )
    return response.json()["result"]["points"]


def rerank(query, points, top_n, batch_size=32):
    """Rerank an arbitrary number of points in TEI-compatible batches."""
    if not RERANK_URL or not points:
        for point in points:
            point["rerank_score"] = None
        return points[:top_n]

    ranked = []
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        response = check(
            requests.post(
                RERANK_URL,
                json={
                    "model": RERANK_ROUTE_MODEL,
                    "query": query,
                    "texts": [p["payload"]["text"] for p in batch],
                    "truncate": True,
                },
                timeout=120,
            )
        )
        for item in response.json():
            point = batch[item["index"]]
            point["rerank_score"] = float(item["score"])
            ranked.append(point)

    ranked.sort(key=lambda p: p["rerank_score"], reverse=True)
    return ranked[:top_n]


def point_score(point):
    score = point.get("rerank_score")
    return float(score if score is not None else point["score"])



def entity_key(point):
    """PROFILE_SPLIT_V1: profile chunks (bio, interests, award, ...) all share the
    person's entity_id, so treat each one as its own record. Research chunks still
    collapse to one per publication/grant/presentation."""
    payload = point["payload"]
    if payload.get("chunk_type") in PROFILE_SET:
        return f"{payload.get('entity_id') or ''}#{point['id']}"
    return payload.get("entity_id") or point["id"]


def collapse_by_entity(points, limit=None):
    """Keep the best chunk per entity so multi-chunk records do not dominate."""
    best = {}
    for point in points:
        entity_id = entity_key(point)
        if entity_id not in best or point_score(point) > point_score(best[entity_id]):
            best[entity_id] = point
    ranked = sorted(best.values(), key=point_score, reverse=True)
    return ranked[:limit] if limit else ranked


def balanced_person_evidence(question, person_id, profile_k=3, research_k=5, candidates=32, fill_to=None): 
    """Retrieve stated profile data and demonstrated research separately."""
    profile_filter = build_filter(types=PROFILE_TYPES, people=[person_id])
    research_filter = build_filter(types=RESEARCH_TYPES, people=[person_id])

    profile = rerank(
        question,
        hybrid(question, k=candidates, prefetch=max(80, candidates), flt=profile_filter),
        top_n=candidates,
    )
    research = rerank(
        question,
        hybrid(question, k=candidates, prefetch=max(80, candidates), flt=research_filter),
        top_n=candidates,
    )

    # BALANCED_FILL_V1: guaranteed quotas first, then fill with best remaining evidence
    profile_all = collapse_by_entity(profile)
    research_all = collapse_by_entity(research)
    picked = profile_all[:profile_k] + research_all[:research_k]
    if fill_to and len(picked) < fill_to:
        rest = sorted(profile_all[profile_k:] + research_all[research_k:], key=point_score, reverse=True)
        picked += rest[: fill_to - len(picked)]
    return picked


def rank_faculty(points, top=10, per_person=3, allowed_person_ids=None, query=""):
    """
    Rank faculty using the shared 0-1 faculty scoring algorithm.

    per_person is retained only for backward compatibility.
    """
    from research_portfolio.matching.faculty_match import score_faculty

    rows = score_faculty(
        query=query,
        top_n=min(max(1, int(top)), 20),
        points=points,
        allowed_person_ids=allowed_person_ids,
    )

    ranked = []

    for row in rows:
        evidence = []

        for item in row.get("evidence", []):
            evidence.append({
                "id": item.get("point_id"),
                "score": item.get("score", 0.0),
                "rerank_score": item.get("score", 0.0),
                "payload": {
                    "entity_id": item.get("entity_id"),
                    "chunk_type": item.get("chunk_type"),
                    "title": item.get("title"),
                    "year": item.get("year"),
                    "status": item.get("status"),
                    "sponsor": item.get("sponsor"),
                    "text": item.get("text"),
                    "department_slugs": item.get("departments", []),
                    "person_ids": [row["person_id"]],
                },
            })

        ranked.append((
            row["score"],
            row["person_id"],
            row["name"],
            evidence,
        ))

    return ranked

def label(payload):
    chunk_type = payload["chunk_type"]
    if payload.get("entity_type") == "person":
        return f"{chunk_type}: {payload.get('display_name')}"
    extra = payload.get("sponsor") if chunk_type == "grant" else payload.get("publisher") or payload.get("event")
    text = f"{chunk_type}: {payload.get('title')} ({payload.get('year', '?')})"
    return text + (f" - {extra}" if extra else "")


def ask(question, points, person=None):
    context = "\n\n".join(
        f"[{index}] ({label(point['payload'])})\n{point['payload']['text']}"
        for index, point in enumerate(points, 1)
    )
    person_instruction = ""
    if person:
        person_instruction = (
            f"\nThe subject is {person['display_name']}.\n"
            "For questions about this person's research, expertise, topics, or interests:\n"
            "1. Report explicitly stated research interests from interests or bio evidence.\n"
            "2. Separately summarize demonstrated research themes supported by publications, "
            "grants, and presentations.\n"
            "3. Do not describe demonstrated themes as explicitly stated interests.\n"
            "4. Use both categories when both are available.\n"
            "5. Mention representative publications, grants, or presentations that support each "
            "demonstrated theme.\n"
            "6. If one category is unavailable, explicitly say which evidence category was not found.\n"
        )
    response = check(
        post_chat(CHAT_URL,
            json={
                "model": CHAT_MODEL,
                "think": False,   # NO_THINK_V1: reasoning consumed the whole output, leaving content empty
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 8192},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer university research-portfolio questions using ONLY the numbered context.\n"
                            "Every factual claim must include one or more citations such as [1] or [2][4].\n"
                            "Use the evidence's labels and source types when interpreting it.\n"
                            "Do not treat profile statements, publications, grants, and presentations as "
                            "interchangeable evidence.\n"
                            "If evidence is insufficient, say what information is missing. Do not guess."
                            + person_instruction
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            },
            timeout=300,
        )
    )
    answer = response.json()["message"]["content"]
    return answer.split("</think>", 1)[1].strip() if "</think>" in answer else answer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--types", help="comma list of chunk types")
    parser.add_argument("--dept", help="comma list of department slugs")
    parser.add_argument("--person", help="person_id such as dm:1478901, or a display name such as Hiram Moya")
    parser.add_argument("--since", type=int)
    parser.add_argument("--funded", action="store_true")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=int(os.environ.get("RERANK_CANDIDATES", 20)))
    parser.add_argument("--profile-k", type=int, default=3)
    parser.add_argument("--research-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=float(os.environ.get("RERANK_MIN_SCORE", 0)))
    
    parser.add_argument(
        "--list-publications",
        action="store_true",
        help="List a person's publications directly from Neo4j",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum records returned by structured list commands",
    )
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--faculty", action="store_true", help="rank faculty for the query")
    mode.add_argument("--ask", action="store_true", help="answer with the chat model")
    args = parser.parse_args()

    departments = args.dept.split(",") if args.dept else None
    person = resolve_person(args.person) if args.person else None

    if args.list_publications:
        if not person:
            parser.error("--list-publications requires --person")

        publications = list_person_publications(
            person_id=person["person_id"],
            since=args.since,
            limit=args.limit,
        )

        print(
            f"{person['display_name']}: "
            f"{len(publications)} publication(s) returned"
        )

        if args.since:
            print(f"Published since {args.since}")

        print()

        for index, pub in enumerate(publications, 1):
            year = pub.get("year") or "year unknown"
            pub_type = pub.get("type") or "publication"

            print(
                f"{index:3}. {pub['title']} "
                f"({year}) [{pub_type}]"
            )

            details = []

            if pub.get("publisher"):
                details.append(pub["publisher"])

            if pub.get("doi"):
                details.append(f"DOI: {pub['doi']}")

            if details:
                print("     " + " | ".join(details))

        return 0

    if person and args.ask and not args.types:
        points = balanced_person_evidence(
            args.query,
            person["person_id"],
            profile_k=args.profile_k,
            research_k=args.research_k,
            candidates=max(args.candidates, 20),
        )
    else:
        point_filter = build_filter(
            types=args.types.split(",") if args.types else None,
            depts=departments,
            since=args.since,
            people=[person["person_id"]] if person else None,
            funded_only=args.funded,
        )
        candidates = hybrid(
            args.query,
            k=max(args.candidates, args.k),
            prefetch=max(80, args.candidates),
            flt=point_filter,
        )
        points = rerank(
            args.query,
            candidates,
            top_n=len(candidates) if args.faculty else args.k,
        )
        if RERANK_URL and args.min_score:
            points = [p for p in points if point_score(p) >= args.min_score]

    if args.faculty:
        allowed = department_member_ids(departments)
        for index, (score, pid, name, evidence) in enumerate(
            rank_faculty(points, top=args.k, allowed_person_ids=allowed), 1
        ):
            print(f"{index:2}. {name:<32} score {score:.3f}  ({pid})")
            for point in evidence:
                print(f"      - {label(point['payload'])[:110]}")
        return 0

    points = collapse_by_entity(points, args.k)
    if args.ask:
        if not points:
            print("I don't have enough information in the indexed records.")
            return 0
        print(ask(args.query, points, person=person), "\n\nSources:")

    for index, point in enumerate(points, 1):
        rs = f"rerank={point['rerank_score']:.3f} " if point.get("rerank_score") is not None else ""
        print(f"[{index}] {rs}rrf={point['score']:.3f}  {label(point['payload'])[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
