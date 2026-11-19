#!/usr/bin/env python3
"""Validate one department across ledger, Neo4j, and Qdrant. Writes validation.json; nonzero on errors."""
import argparse, json, os, requests
from collections import Counter
from research_portfolio.common import load_config, department_dir, read_json, write_atomic
from research_portfolio.graph.store import driver, DB, run_q

QDRANT=os.getenv("QDRANT_URL","http://qdrant:6333")
COLL=os.getenv("RP_COLLECTION","rp_chunks")

def qcount(drv, cypher, **p): return run_q(drv, cypher, **p)[0]["n"]
def qdrant_count(flt):
    r=requests.post(f"{QDRANT}/collections/{COLL}/points/count",json={"exact":True,"filter":flt},timeout=60)
    r.raise_for_status(); return r.json()["result"]["count"]
def scroll_ids(slug):
    offset=None; ids=set(); point_ids=set()
    flt={"must":[{"key":"department_slugs","match":{"value":slug}}]}
    while True:
        body={"filter":flt,"limit":256,"with_payload":["entity_id"],"with_vector":False}
        if offset is not None: body["offset"]=offset
        r=requests.post(f"{QDRANT}/collections/{COLL}/points/scroll",json=body,timeout=60); r.raise_for_status()
        result=r.json()["result"]
        for p in result["points"]:
            point_ids.add(str(p["id"]));
            if p.get("payload",{}).get("entity_id"): ids.add(p["payload"]["entity_id"])
        offset=result.get("next_page_offset")
        if offset is None: break
    return ids,point_ids

def check(name, ok, actual=None, expected=None, severity="error"):
    return {"name":name,"status":"pass" if ok else severity,"actual":actual,"expected":expected}

def validate(cfg):
    slug = cfg["department_slug"]
    ledger = read_json(department_dir(cfg) / "ledger.json")

    expected_people = len(ledger["people"])
    checks = []

    # ---------------------------------------------------------
    # Neo4j validation
    # ---------------------------------------------------------
    with driver() as drv:

        # 1. Department roster
        people = qcount(
            drv,
            """
            MATCH (:Person)-[:MEMBER_OF]->(:Department {slug:$slug})
            RETURN count(*) AS n
            """,
            slug=slug,
        )

        checks.append(
            check(
                "roster_membership",
                people == expected_people,
                people,
                expected_people,
            )
        )

        # 2. Research entities belonging to this department.
        # These MUST exist in Qdrant.
        research_ids = {
            r["id"]
            for r in run_q(
                drv,
                """
                MATCH (n)
                WHERE (
                    n:Publication OR
                    n:Grant OR
                    n:Presentation
                )
                AND $slug IN coalesce(n.department_slugs, [])
                RETURN n.id AS id
                """,
                slug=slug,
            )
        }

        # 3. Faculty belonging to this department.
        # People may legitimately have no Qdrant profile chunks.
        person_ids = {
            r["id"]
            for r in run_q(
                drv,
                """
                MATCH (p:Person)-[:MEMBER_OF]->(
                    :Department {slug:$slug}
                )
                RETURN p.person_id AS id
                """,
                slug=slug,
            )
        }

        # 4. Every research entity must have a contributor.
        orphan_entities = qcount(
            drv,
            """
            MATCH (n)
            WHERE (
                n:Publication OR
                n:Grant OR
                n:Presentation
            )
            AND $slug IN coalesce(n.department_slugs, [])
            AND NOT ()-[
                :AUTHORED|
                INVESTIGATOR_ON|
                PRESENTED
            ]->(n)
            RETURN count(n) AS n
            """,
            slug=slug,
        )

        checks.append(
            check(
                "graph_entities_have_contributors",
                orphan_entities == 0,
                orphan_entities,
                0,
            )
        )

        # 5. An IdentityKey must identify exactly one canonical entity.
        duplicate_keys = qcount(
            drv,
            """
            MATCH (k:IdentityKey)
            WHERE size([
                (k)-[:IDENTIFIES]->(n) | n
            ]) <> 1
            RETURN count(k) AS n
            """,
        )

        checks.append(
            check(
                "identity_key_cardinality",
                duplicate_keys == 0,
                duplicate_keys,
                0,
            )
        )

    # ---------------------------------------------------------
    # Qdrant validation
    # ---------------------------------------------------------

    qids, point_ids = scroll_ids(slug)

    # Publication / Grant / Presentation MUST exist in Qdrant.
    missing_research = research_ids - qids

    # People may or may not have profile chunks.
    missing_people = person_ids - qids

    # Every Qdrant entity must correspond to something in Neo4j.
    valid_graph_ids = research_ids | person_ids
    orphan_vectors = qids - valid_graph_ids

    checks.append(
        check(
            "research_entities_missing_qdrant",
            not missing_research,
            sorted(missing_research),
            [],
        )
    )

    checks.append(
        check(
            "qdrant_entities_missing_graph",
            not orphan_vectors,
            sorted(orphan_vectors),
            [],
        )
    )

    # ---------------------------------------------------------
    # Faculty profile coverage
    # ---------------------------------------------------------

    text_poor_names = set(
        ledger.get("quality", {}).get(
            "text_poor",
            []
        )
    )

    text_poor_ids = {
        p["person_id"]
        for p in ledger["people"]
        if p.get("display_name") in text_poor_names
    }

    # Missing people NOT already known to be text-poor are errors.
    unexpected_missing_people = (
        missing_people - text_poor_ids
    )

    # Missing people already classified as text-poor are warnings.
    expected_missing_people = (
        missing_people & text_poor_ids
    )

    checks.append(
        check(
            "searchable_people_missing_qdrant",
            not unexpected_missing_people,
            sorted(unexpected_missing_people),
            [],
        )
    )

    checks.append(
        check(
            "text_poor_people_without_qdrant",
            not expected_missing_people,
            sorted(expected_missing_people),
            [],
            "warning",
        )
    )

    checks.append(
        check(
            "text_poor_faculty",
            not text_poor_names,
            sorted(text_poor_names),
            [],
            "warning",
        )
    )

    # ---------------------------------------------------------
    # Publication data quality
    # ---------------------------------------------------------

    pubs = ledger["entities"]["publication"]

    no_abs = sum(
        not p.get("abstract")
        for p in pubs
    )

    checks.append(
        check(
            "publications_without_abstract",
            no_abs == 0,
            no_abs,
            0,
            "warning",
        )
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    passed = not any(
        c["status"] == "error"
        for c in checks
    )

    return {
        "department": slug,
        "passed": passed,
        "checks": checks,
        "summary": {
            "roster_people": len(person_ids),
            "research_entities": len(research_ids),
            "qdrant_entities": len(qids),
            "qdrant_points": len(point_ids),
            "people_without_qdrant": len(missing_people),
            "publications_without_abstract": no_abs,
        },
    }
    
    
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=load_config(a.config); result=validate(cfg); write_atomic(department_dir(cfg)/"validation.json",result)
    for c in result["checks"]: print(f"[{c['status'].upper():7}] {c['name']}: {c['actual']}")
    print("PASS" if result["passed"] else "FAIL", "->",department_dir(cfg)/"validation.json")
    return 0 if result["passed"] else 1
if __name__=="__main__": raise SystemExit(main())
