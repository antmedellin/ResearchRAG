#!/usr/bin/env python3
"""Run one department end-to-end:
  discover -> fetch profiles -> build ledger -> resolve (vs Neo4j) -> Neo4j sync -> Qdrant index.
Dry run by default (stages 1-3 + a read-only dedup preview). --apply writes Neo4j and Qdrant."""
import argparse, json, statistics, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from research_portfolio.common import load_config, department_dir, read_json, write_atomic, to_props, norm_key
from research_portfolio.ledger import discover_faculty, fetch_profiles, build_ledger
from research_portfolio.ledger.merge import merge_records

DROP_FROM_PROPS = {"id", "kind", "ledger_id", "claimed_by", "identity_keys", "department_slugs",
                   "person_ids", "faculty_names", "lead_person_ids", "last_run"}
PERSON_DROP = {"rank", "counts"}

def stage(name, fn, args):
    print(f"\n===== {name} =====", flush=True)
    t = time.time()
    rc = fn(args)
    print(f"[{name}] exit {rc} in {time.time() - t:.1f}s")
    if rc:
        raise SystemExit(f"Stage '{name}' failed (exit {rc})")

def resolve_ledger(drv, ledger):
    from research_portfolio.graph.store import load_existing
    from research_portfolio.resolve.dedup import Resolver
    entities, claims, stats = {}, {}, {}
    for kind, items in ledger["entities"].items():
        existing = load_existing(drv, kind) if drv else []
        current = {e["id"]: {k: v for k, v in e.items() if k not in DROP_FROM_PROPS | {"family_names", "contributor_names"}} | {"kind": kind}
                   for e in existing}
        res = Resolver(kind, existing)
        merged, keys, crows = {}, defaultdict(set), {}
        for x in items:
            cid, _, ks = res.resolve(x)
            base = merged.get(cid) or current.get(cid)
            merged[cid] = merge_records(base, x) if base else x
            keys[cid].update(ks)
            for owner in x.get("claimed_by", []):
                for c in x.get("contributors", []):
                    if c.get("person_id"):
                        crows[(cid, c["person_id"], owner)] = {
                            "entity_id": cid, "person_id": c["person_id"], "owner": owner, "name": c.get("name"),
                            "dm_user_id": c.get("dm_user_id"), "position": c.get("position"), "role": c.get("role")}
        rows = []
        for cid, m in merged.items():
            props = to_props({k: v for k, v in m.items() if k not in DROP_FROM_PROPS})
            props["family_names"] = sorted({norm_key(c.get("family_name")) for c in m.get("contributors", []) if c.get("family_name")})
            props["contributor_names"] = [c["name"] for c in m.get("contributors", []) if c.get("name")]
            rows.append({"id": cid, "keys": sorted(keys[cid]), "props": props,
                         "sponsor": m.get("sponsor"), "sponsor_key": norm_key(m.get("sponsor")) or None})
        entities[kind], claims[kind] = rows, list(crows.values())
        stats[kind] = {"ledger": len(items), "canonical": len(rows), **dict(res.stats)}
    return entities, claims, stats

def person_rows(ledger):
    rows = []
    for p in ledger["people"]:
        props = to_props({k: v for k, v in p.items() if k not in PERSON_DROP})
        topics = [{"id": "kw:" + norm_key(t), "name": t} for t in p.get("research_keywords", []) if norm_key(t)]
        rows.append({"person_id": p["person_id"], "rank": p.get("rank"), "props": props, "topics": topics})
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--apply", action="store_true", help="write Neo4j and Qdrant")
    ap.add_argument("--skip-discover", action="store_true", help="reuse existing faculty.json")
    ap.add_argument("--skip-fetch", action="store_true", help="reuse cached raw profiles")
    ap.add_argument("--refresh", action="store_true", help="re-download all profiles")
    ap.add_argument("--allow-partial", action="store_true", help="continue if some profiles failed")
    ap.add_argument("--no-graph-preview", action="store_true", help="dry run without connecting to Neo4j")
    ap.add_argument("--recreate-collection", action="store_true", help="drop and recreate the Qdrant collection")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = department_dir(cfg)
    c = ["--config", a.config]
    started = datetime.now(timezone.utc)
    print(f"Department: {cfg['department_name']} ({cfg['department_slug']})  ->  {out}")
    print(f"Mode: {'APPLY (writes Neo4j + Qdrant)' if a.apply else 'DRY RUN'}")

    if not a.skip_discover:
        stage("Discover faculty", discover_faculty.main, c)
    if not a.skip_fetch:
        stage("Fetch profiles", fetch_profiles.main,
              c + (["--refresh"] if a.refresh else []) + (["--allow-partial"] if a.allow_partial else []))
    stage("Build ledger", build_ledger.main, c + (["--allow-partial"] if a.allow_partial else []))

    ledger = read_json(out / "ledger.json")
    log = {"started_at": started.isoformat(), "department": cfg["department_slug"], "apply": a.apply,
           "quality": ledger["quality"]}

    drv = None
    if a.apply or not a.no_graph_preview:
        from research_portfolio.graph.store import driver
        drv = driver()
        drv.verify_connectivity()
    try:
        print("\n===== Resolve against graph =====")
        entities, claims, stats = resolve_ledger(drv, ledger)
        for kind, s in stats.items():
            print(f"  {kind:13} {s}")
        log["dedup"] = stats
        if not a.apply:
            print("\nDry run complete. Re-run with --apply to write Neo4j and Qdrant.")
            return 0

        from research_portfolio.graph.store import sync_department, read_people, read_entities
        from research_portfolio.index import qdrant_store as Q
        print("\n===== Neo4j sync =====")
        people = person_rows(ledger)
        dept = {**ledger["department"], "institution": ledger["institution"]}
        result = sync_department(drv, dept, people, entities, claims)
        print(f"  run {result['run']} | faculty {result['faculty_count']} | departed {len(result['departed'])} "
              f"| removed entities {len(result['removed_entities'])} | slug changes {len(result['slug_changes'])}")
        log["neo4j"] = {k: (v if not isinstance(v, list) else len(v)) for k, v in result.items()}

        print("\n===== Qdrant index =====")
        pids = [p["person_id"] for p in people]
        eids = [r["id"] for rows in entities.values() for r in rows]
        chunks = [ch for p in read_people(drv, pids) for ch in Q.person_chunks(p)] + \
                 [ch for x in read_entities(drv, eids) for ch in Q.entity_chunks(x)]
        avg = statistics.mean(len(ch["text"].split()) for ch in chunks) if chunks else 0
        print(f"  {len(chunks)} chunks | avg {avg:.0f} words (BM25_AVG_LEN={Q.AVG_LEN:.0f})")
        if avg and abs(avg - Q.AVG_LEN) / Q.AVG_LEN > 0.25:
            print(f"  NOTE: consider BM25_AVG_LEN={avg:.0f} in .env (keep it fixed across departments).")
        Q.ensure_collection(len(Q.embed(["dimension probe"])[0]), a.recreate_collection)
        Q.upsert(chunks, result["run"])
        Q.delete_stale(pids + eids, result["run"])
        Q.apply_graph_changes(result)
        n = Q.count()
        print(f"  collection {Q.COLL}: {n} points total")
        log["qdrant"] = {"chunks": len(chunks), "avg_words": round(avg, 1), "collection_points": n}
        return 0
    finally:
        if drv:
            drv.close()
        log["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_atomic(out / "runs" / f"run_{started:%Y%m%dT%H%M%SZ}.json", log)

if __name__ == "__main__":
    raise SystemExit(main())
