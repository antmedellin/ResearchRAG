"""Read-only experimental faculty Technical matching, ResearchRAG.

Place at: src/research_portfolio/matching/technical_experiment.py
Run inside user_app: python -m research_portfolio.matching.technical_experiment --query '...'

Does not replace the production score, modify the databases, or change the web UI.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from research_portfolio.matching import faculty_match as F

# Labels returned by the current extract_areas() implementation.
# Add missing recognized concepts here after validating them.
METHOD_AREAS = {
    "Machine Learning", "Computer Vision", "Deep Learning",
    "Image Classification", "Anomaly Detection", "Artificial Intelligence",
    "Data Analytics", "Nondestructive Evaluation", "Materials Characterization",
    "Robotics", "Automation", "Simulation Modeling", "Queueing Theory",
}
APPLICATION_AREAS = {
    "Defect Detection", "Industrial Defect Detection", "Automated Inspection",
    "Industrial Inspection", "Quality Control", "Advanced Manufacturing",
    "Additive Manufacturing", "Digital Manufacturing", "Supply Chain",
    "Cybersecurity", "Renewable Energy", "Energy Systems", "Process Safety",
    "Healthcare", "Workforce Development", "Engineering Education",
    "Border Security",
}
RESEARCH_TYPES = {"publication", "grant", "presentation"}
PROFILE_TYPES = {"bio", "interests"}


def group_by_faculty(points, members):
    """Link deduplicated retrieved records to Neo4j faculty IDs."""
    grouped = defaultdict(dict)
    for point in points:
        payload = point.get("payload") or {}
        ids = set(payload.get("person_ids") or [])
        if payload.get("entity_type") == "person":
            pid = payload.get("person_id")
            if pid:
                ids.add(pid)
        for pid in ids:
            if pid not in members:
                continue
            key = F.entity_key(point)
            previous = grouped[pid].get(key)
            if previous is None or F.pscore(point) > F.pscore(previous):
                grouped[pid][key] = point
    return grouped


def best_research_relevance(points):
    """Emphasize strong matching research without averaging weak tail records."""
    scores = sorted(
        (F.normalize_rerank(F.pscore(p)) for p in points
         if (p.get("payload") or {}).get("chunk_type") in RESEARCH_TYPES),
        reverse=True,
    )
    if not scores:
        return 0.0
    return 0.70 * scores[0] + 0.30 * sum(scores[:3]) / len(scores[:3])


def best_profile_relevance(points):
    return max(
        (F.normalize_rerank(F.pscore(p)) for p in points
         if (p.get("payload") or {}).get("chunk_type") in PROFILE_TYPES),
        default=0.0,
    )


def concept_split(query):
    areas = F.extract_areas(query)
    methods = [a for a in areas if a in METHOD_AREAS]
    applications = [a for a in areas if a in APPLICATION_AREAS]
    unclassified = [a for a in areas if a not in METHOD_AREAS | APPLICATION_AREAS]
    return methods, applications, unclassified


def compare(query, candidates=80):
    methods, applications, unclassified = concept_split(query)
    if not (methods or applications):
        raise ValueError(
            "No recognized methodological/application requirements. "
            "Extend the concept vocabulary before scoring this opportunity."
        )

    # Production baseline, with current grant/recency/evidence signals unchanged.
    baseline = {
        row["person_id"]: row
        for row in F.score_faculty(query, top_n=10000)
    }
    members = F.member_ids()

    # The original full-query retrieval supplies the independent profile component.
    full_query = group_by_faculty(F.relevant_points(query), members)

    # Each recognized requirement gets its own Qdrant search and TEI reranking.
    per_requirement = {}
    for area in methods + applications:
        points = F.relevant_points(area, candidates=candidates)
        grouped = group_by_faculty(points, members)
        per_requirement[area] = {
            pid: (best_research_relevance(entities.values()), entities)
            for pid, entities in grouped.items()
        }

    method_weight, app_weight = (
        (0.50, 0.30) if methods and applications else
        (0.80, 0.00) if methods else
        (0.00, 0.80)
    )

    ids = set(baseline) | set(full_query)
    for grouped in per_requirement.values():
        ids.update(grouped)

    results = []
    for pid in ids:
        if pid not in members:
            continue
        req_scores = {
            area: per_requirement[area].get(pid, (0.0, {}))[0]
            for area in methods + applications
        }
        method_score = sum(req_scores[a] for a in methods) / len(methods) if methods else 0.0
        app_score = sum(req_scores[a] for a in applications) / len(applications) if applications else 0.0
        profile_score = best_profile_relevance(full_query.get(pid, {}).values())
        experimental = F.clamp(
            method_weight * method_score
            + app_weight * app_score
            + 0.20 * profile_score
        )

        evidence = {}
        for area in methods + applications:
            _, matched = per_requirement[area].get(pid, (0.0, {}))
            research = sorted(
                (p for p in matched.values()
                 if (p.get("payload") or {}).get("chunk_type") in RESEARCH_TYPES),
                key=F.pscore,
                reverse=True,
            )
            if research:
                best = research[0]
                payload = best.get("payload") or {}
                evidence[area] = {
                    "title": payload.get("title") or payload.get("display_name") or "(untitled)",
                    "type": payload.get("chunk_type"),
                    "score": F.normalize_rerank(F.pscore(best)),
                }

        base = baseline.get(pid)
        results.append({
            "name": members[pid]["name"],
            "person_id": pid,
            "baseline_technical": base["signals"]["technical"] if base else None,
            "experimental_technical": round(experimental * 100, 1),
            "methods": round(method_score * 100, 1),
            "applications": round(app_score * 100, 1),
            "profile": round(profile_score * 100, 1),
            "per_requirement": {k: round(v * 100, 1) for k, v in req_scores.items()},
            "evidence": evidence,
            "baseline_overall": base["score"] if base else None,
        })

    results.sort(key=lambda x: x["experimental_technical"], reverse=True)
    return {
        "methods": methods,
        "applications": applications,
        "unclassified": unclassified,
        "weights": {"methods": method_weight, "applications": app_weight, "profile": .20},
        "faculty": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidates", type=int, default=80,
                        help="Retrieval candidates per requirement (default: 80)")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--details", action="store_true",
                        help="Show top evidence per requirement")
    args = parser.parse_args()
    if args.candidates < 1 or args.top_n < 1:
        parser.error("--candidates and --top-n must be positive")
    result = compare(args.query, candidates=args.candidates)
    print("\nMethod requirements:", result["methods"])
    print("Application requirements:", result["applications"])
    print("Unclassified requirements:", result["unclassified"])
    print("Experimental weights:", result["weights"])
    print("\nWARNING: exploratory relevance indices, NOT calibrated expertise percentages.")
    print("Faculty                          Old-T  New-T  Method  App    Profile")
    print("-" * 82)
    for row in result["faculty"][:args.top_n]:
        old = "  n/a" if row["baseline_technical"] is None else f"{row['baseline_technical']:5.1f}"
        print(f"{row['name'][:31]:31s} {old}  {row['experimental_technical']:5.1f}  "
              f"{row['methods']:6.1f}  {row['applications']:5.1f}  {row['profile']:7.1f}")
        if args.details:
            for area in result["methods"] + result["applications"]:
                ref = row["evidence"].get(area)
                if ref:
                    print(f"    {area} [{ref['type']}]: {ref['title']} "
                          f"(relevance={ref['score']:.3f})")
                else:
                    print(f"    {area}: no research evidence among retrieved candidates")


if __name__ == "__main__":
    main()
