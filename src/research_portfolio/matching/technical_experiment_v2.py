"""Read-only experimental faculty matching v2: explicit evidence quality and coverage.

Place at src/research_portfolio/matching/technical_experiment_v2.py.
This module does NOT change production scoring, UI, or any database records.

The evidence-type multipliers and thresholds are experimental heuristics, not
validated probabilities of expertise. Use for controlled comparisons only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from research_portfolio.matching import faculty_match as F
from research_portfolio.matching import technical_experiment as E

# Experimental provenance weights. Deliberately kept explicit and reversible.
TYPE_FACTOR = {"publication": 1.0, "grant": 0.85, "presentation": 0.60}
MIN_MEANINGFUL_RELEVANCE = 0.20  # For reporting evidence breadth ONLY, not score cutoff.


def research_records(points):
    return sorted(
        (p for p in points if (p.get("payload") or {}).get("chunk_type") in E.RESEARCH_TYPES),
        key=F.pscore, reverse=True,
    )


def score_requirement(points):
    """Return (v1 score, v2 score, top supporting records).

    v1: 70% best + 30% mean(top 3) of transformed relevance.
    v2: same aggregation, adjusted by experimental evidence-type factors.
    """
    records = research_records(points)
    evidence = []
    for point in records:
        payload = point.get("payload") or {}
        kind = payload.get("chunk_type")
        rel = F.normalize_rerank(F.pscore(point))
        adjusted = rel * TYPE_FACTOR[kind]
        evidence.append({
            "entity_key": F.entity_key(point),
            "title": payload.get("title") or payload.get("display_name") or "(untitled)",
            "kind": kind,
            "raw": rel,
            "adjusted": adjusted,
            "text": payload.get("text") or "",
        })

    def aggregate(xs):
        values = sorted(xs, reverse=True)
        if not values:
            return 0.0
        return F.clamp(0.7 * values[0] + 0.3 * sum(values[:3]) / len(values[:3]))

    return (
        aggregate(e["raw"] for e in evidence),
        aggregate(e["adjusted"] for e in evidence),
        sorted(evidence, key=lambda x: x["adjusted"], reverse=True),
    )


def compare(query, candidates=80):
    methods, applications, unclassified = E.concept_split(query)
    areas = methods + applications
    if not areas:
        raise ValueError("No recognized requirements; expand extract_areas vocabulary first")

    members = F.member_ids()
    baseline = {x["person_id"]: x for x in F.score_faculty(query, top_n=10000)}
    profiles = E.group_by_faculty(F.relevant_points(query), members)
    per_area = {}
    for area in areas:
        grouped = E.group_by_faculty(F.relevant_points(area, candidates=candidates), members)
        per_area[area] = {
            pid: score_requirement(entities.values()) for pid, entities in grouped.items()
        }

    if methods and applications:
        method_weight, application_weight = 0.5, 0.3
    elif methods:
        method_weight, application_weight = 0.8, 0.0
    else:
        method_weight, application_weight = 0.0, 0.8

    ids = set(baseline) | set(profiles)
    for grouped in per_area.values():
        ids.update(grouped)

    results = []
    for pid in ids:
        if pid not in members:
            continue
        per = {area: per_area[area].get(pid, (0.0, 0.0, [])) for area in areas}
        def avg(group, index):
            return sum(per[a][index] for a in group) / len(group) if group else 0.0

        method_v1, app_v1 = avg(methods, 0), avg(applications, 0)
        method_v2, app_v2 = avg(methods, 1), avg(applications, 1)
        profile_points = list(profiles.get(pid, {}).values())
        profile_records = [p for p in profile_points if (p.get("payload") or {}).get("chunk_type") in E.PROFILE_TYPES]
        profile_score = E.best_profile_relevance(profile_records)
        v1 = method_weight * method_v1 + application_weight * app_v1 + .2 * profile_score
        v2 = method_weight * method_v2 + application_weight * app_v2 + .2 * profile_score

        # Count unique, meaningfully relevant items across requirements, so an item
        # reused for two requirements is counted only once for breadth/confidence.
        supported = {}
        for _, _, evidence in per.values():
            for ev in evidence:
                if ev["raw"] < MIN_MEANINGFUL_RELEVANCE:
                    continue
                key = ev["entity_key"]
                if key not in supported or supported[key]["adjusted"] < ev["adjusted"]:
                    supported[key] = ev
        count_pubs = sum(e["kind"] == "publication" for e in supported.values())
        count_grants = sum(e["kind"] == "grant" for e in supported.values())
        count_presentations = sum(e["kind"] == "presentation" for e in supported.values())

        base = baseline.get(pid)
        results.append({
            "name": members[pid]["name"],
            "old": base["signals"]["technical"] if base else None,
            "v1": round(F.clamp(v1) * 100, 1),
            "v2": round(F.clamp(v2) * 100, 1),
            "methods": round(method_v2 * 100, 1),
            "applications": round(app_v2 * 100, 1),
            "profile": round(profile_score * 100, 1),
            "profile_found": bool(profile_records),
            "evidence_count": len(supported),
            "pubs": count_pubs,
            "grants": count_grants,
            "presentations": count_presentations,
            "areas": per,
        })
    results.sort(key=lambda r: (-r["v2"], r["name"]))
    return {"methods": methods, "applications": applications, "unclassified": unclassified,
            "weights": (method_weight, application_weight, .2), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidates", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--names", default="", help="Comma-separated faculty names to inspect (can be outside top N)")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    if args.candidates < 1 or args.top_n < 1:
        parser.error("--candidates and --top-n must be positive")

    result = compare(args.query, args.candidates)
    print("Method requirements:", result["methods"])
    print("Application requirements:", result["applications"])
    print("Unclassified:", result["unclassified"])
    print("Weights (methods, applications, profile):", result["weights"])
    print("Type factors:", TYPE_FACTOR)
    print("WARNING: Experimental relevance indices, NOT calibrated expertise percentages.")
    print("Coverage counts: unique records with normalized relevance >= 0.20; these are NOT confidence probabilities.")
    print("Faculty                          Old    V1     V2     Method   App  Prof  Pub Grant Pres Profile?")
    print("-" * 109)
    rows = result["results"]
    if args.names:
        selected = {s.strip().casefold() for s in args.names.split(",") if s.strip()}
        rows = [r for r in rows if r["name"].casefold() in selected]
        missing = selected - {r["name"].casefold() for r in rows}
        if missing:
            print("Not found:", ", ".join(sorted(missing)))
    else:
        rows = rows[:args.top_n]
    for r in rows:
        old = "n/a" if r["old"] is None else f"{r['old']:.1f}"
        print(f"{r['name'][:31]:31s} {old:>5s} {r['v1']:5.1f}  {r['v2']:5.1f}  "
              f"{r['methods']:6.1f} {r['applications']:5.1f} {r['profile']:5.1f}  "
              f"{r['pubs']:3d} {r['grants']:4d} {r['presentations']:4d} "
              f"{'yes' if r['profile_found'] else 'NO'}")
        if args.details:
            for area, (raw, adjusted, evs) in r["areas"].items():
                if not evs:
                    print(f"    {area}: no evidence retrieved")
                    continue
                ev = evs[0]
                print(f"    {area} [{ev['kind']}] raw_normalized={ev['raw']:.3f} adjusted={ev['adjusted']:.3f}: {ev['title']}")
                print("       Text sample:", " ".join(ev["text"].split())[:220])


if __name__ == "__main__":
    main()
