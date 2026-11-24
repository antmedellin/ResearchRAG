"""Read-only ResearchRAG Technical V4 profile experiment.

Place at src/research_portfolio/matching/technical_experiment_v3.py.
Requires technical_terms_dynamic.py, technical_experiment_v2.py and
ResearchRAG's faculty_match.py, retrieval/search.py.

V4: dynamic requirement extraction -> Qdrant candidate discovery -> point-id dedupe ->
per-requirement cross-encoder reranking -> per-entity evidence aggregation.
The research evidence types and coefficients remain V2's exploratory choices.
No database writes or production ranking changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from research_portfolio.matching import faculty_match as F
from research_portfolio.matching import technical_experiment as E
from research_portfolio.matching import technical_experiment_v2 as V2
from research_portfolio.matching import technical_terms_dynamic as T
from research_portfolio.retrieval import search as R

PROFILE_WEIGHT = 0.20
RELEVANCE_THRESHOLD = V2.MIN_MEANINGFUL_RELEVANCE  # diagnostic only


def technical_requirements(query):
    extracted = T.extract_concepts(query)
    methods = [x['label'] for x in extracted if x['category'] == 'method']
    applications = [x['label'] for x in extracted if x['category'] == 'application']
    if not methods and not applications:
        raise ValueError('LLM did not return any grounded research requirements')
    return methods, applications


def get_discovery_queries(query, max_queries=12, include_related=True):
    """V4 phase 1: original query plus balanced, extracted requirements.

    No unverified synonym expansion yet; include_related is retained for
    CLI compatibility but intentionally unused in this validation phase.
    """
    if max_queries < 1:
        raise ValueError('max_queries must be positive')
    extracted = T.extract_concepts(query)
    pending = {
        cat: [x for x in extracted if x['category'] == cat]
        for cat in ('method', 'application')
    }
    selected = [{'query': query, 'role': 'original', 'requirement': None}]
    seen = {query.strip().casefold()}
    while len(selected) < max_queries and any(pending.values()):
        for cat in ('method', 'application'):
            if not pending[cat] or len(selected) >= max_queries:
                continue
            item = pending[cat].pop(0)
            label = item['label']
            key = label.strip().casefold()
            if key not in seen:
                selected.append({
                    'query': label,
                    'role': 'requirement',
                    'requirement': label,
                })
                seen.add(key)
    return selected


def discover(query, candidates=40, max_queries=12, include_related=True):
    """Use hybrid only for discovery; rerank later using one query per requirement.

    Keep chunks separate by point-id; same source record may have >1 chunk.
    """
    if candidates < 1:
        raise ValueError('candidates must be positive')
    queries = get_discovery_queries(query, max_queries, include_related)
    flt = R.build_filter(types=['bio', 'interests', 'publication', 'grant', 'presentation'])
    discovered = {}
    per_query = []
    for item in queries:
        points = R.hybrid(item['query'], k=candidates, prefetch=max(100, candidates), flt=flt)
        seen_local = set()
        for point in points:
            key = str(point['id'])
            seen_local.add(key)
            if key not in discovered:
                discovered[key] = {'point': point, 'discovered_by': [], 'original': False}
            if item not in discovered[key]['discovered_by']:
                discovered[key]['discovered_by'].append(item)
            if item['role'] == 'original':
                discovered[key]['original'] = True
        per_query.append({'role': item['role'], 'query': item['query'], 'points': len(seen_local)})
    return discovered, per_query


def score_requirement_from_pool(requirement, discovered, members):
    """Score all candidate chunks against a requirement, not the original query.

    Do not mutate master records: R.rerank adds rerank_score in place.
    Group by entity per faculty *after* scoring, so best chunk wins for each
    (faculty, requirement, entity).
    """
    candidates = []
    for item in discovered.values():
        p = item['point']
        payload = p.get('payload') or {}
        if payload.get('chunk_type') in E.RESEARCH_TYPES and payload.get('text'):
            candidates.append(dict(p))
    if not candidates:
        return {}
    ranked = R.rerank(requirement, candidates, top_n=len(candidates))
    grouped = E.group_by_faculty(ranked, members)
    results = {}
    for pid, entity_points in grouped.items():
        v1, v2, evidence = V2.score_requirement(entity_points.values())
        for ev in evidence:
            # Track true retrieval provenance, not reranking relevance.
            # Find the particular chunk selected for this entity/requirement.
            source_point = entity_points.get(ev['entity_key'])
            if source_point is None:
                continue
            source = discovered.get(str(source_point['id']))
            ev['point_id'] = str(source_point['id'])
            ev['discovery_roles'] = sorted({x['role'] for x in source['discovered_by']}) if source else []
            ev['discovery_queries'] = [x['query'] for x in source['discovered_by']] if source else []
        results[pid] = {'v1': v1, 'v3': v2, 'evidence': evidence}
    return results


def compare(query, candidates=40, max_queries=12, include_related=True, with_v2=True):
    if not R.RERANK_URL:
        raise RuntimeError('V3 requires reranking; RERANK_URL is unset')
    methods, applications = technical_requirements(query)
    discovered, discovery_log = discover(query, candidates, max_queries, include_related)
    members = F.member_ids()
    area_results = {}
    for area in methods + applications:
        area_results[area] = score_requirement_from_pool(area, discovered, members)

    # Profile search has its own filtered Qdrant budget.
    # Rerank before computing profile relevance, preserving the existing
    # normalization, entity grouping, and 20% profile scoring weight.
    profile_candidates = R.hybrid(
        query,
        k=candidates,
        prefetch=max(100, candidates),
        flt=R.build_filter(types=['bio', 'interests']),
    )
    profile_ranked = (
        R.rerank(query, [dict(p) for p in profile_candidates],
                 top_n=len(profile_candidates))
        if profile_candidates else []
    )
    profiles = E.group_by_faculty(profile_ranked, members)
    profile_values = {}
    profile_found = {}
    for pid, entities in profiles.items():
        records = [p for p in entities.values()
                   if (p.get('payload') or {}).get('chunk_type') in E.PROFILE_TYPES]
        profile_found[pid] = bool(records)
        profile_values[pid] = E.best_profile_relevance(records)

    # Original production baseline comes from its own implementation.
    baseline = {row['person_id']: row for row in F.score_faculty(query, top_n=10000)}
    v2_result = V2.compare(query, candidates=candidates) if with_v2 else None
    v2_by_name = {r['name']: r for r in v2_result['results']} if v2_result else {}

    if methods and applications:
        wm, wa = .50, .30
    elif methods:
        wm, wa = .80, .0
    else:
        wm, wa = .0, .80
    ids = set(baseline) | set(profile_values)
    for group in area_results.values():
        ids.update(group)

    output = []
    for pid in ids:
        if pid not in members:
            continue
        def avg(reqs):
            return sum(area_results[a].get(pid, {}).get('v3', 0.0) for a in reqs) / len(reqs) if reqs else 0.0
        method_score = avg(methods)
        app_score = avg(applications)
        prof_score = profile_values.get(pid, 0.0)
        technical = F.clamp(wm * method_score + wa * app_score + PROFILE_WEIGHT * prof_score)
        evidence = {}
        supported = {}
        for area in methods + applications:
            e = area_results[area].get(pid, {}).get('evidence', [])
            evidence[area] = e
            for record in e:
                if record['raw'] < RELEVANCE_THRESHOLD:
                    continue
                key = record['entity_key']
                if key not in supported or supported[key]['adjusted'] < record['adjusted']:
                    supported[key] = record
        base = baseline.get(pid)
        name = members[pid]['name']
        output.append({
            'person_id': pid, 'name': name,
            'old': base['signals']['technical'] if base else None,
            'v2': v2_by_name.get(name, {}).get('v2'),
            'v4': round(100 * technical, 1),
            'method': round(method_score * 100, 1),
            'application': round(app_score * 100, 1),
            'profile': round(prof_score * 100, 1),
            'profile_found': profile_found.get(pid, False),
            'supporting_records': len(supported),
            'evidence': evidence,
        })
    output.sort(key=lambda x: (-x['v4'], x['name']))
    return {
        'methods': methods, 'applications': applications,
        'weights': (wm, wa, PROFILE_WEIGHT),
        'discovery': discovery_log,
        'pool_points': len(discovered),
        'original_points': sum(x['original'] for x in discovered.values()),
        'rows': output,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--query', required=True)
    ap.add_argument('--candidates', type=int, default=40, help='Qdrant candidates per expansion query')
    ap.add_argument('--max-queries', type=int, default=12, help='Maximum discovery queries including original')
    ap.add_argument('--top-n', type=int, default=12)
    ap.add_argument('--names', default='', help='Comma-separated names to inspect, regardless of ranking')
    ap.add_argument('--details', action='store_true')
    ap.add_argument('--no-related', action='store_true', help='Do not retrieve related discovery-only terms')
    ap.add_argument('--skip-v2', action='store_true', help='Skip expensive V2 comparison')
    args = ap.parse_args()
    if args.candidates < 1 or args.top_n < 1 or args.max_queries < 1:
        ap.error('--candidates, --top-n and --max-queries must be positive')
    data = compare(args.query, candidates=args.candidates, max_queries=args.max_queries,
                   include_related=not args.no_related, with_v2=not args.skip_v2)
    print('Method requirements:', data['methods'])
    print('Application requirements:', data['applications'])
    print('Weights (method, application, profile):', data['weights'])
    print('Discovery queries:', len(data['discovery']))
    for d in data['discovery']:
        print(f"  {d['role']:<22} {d['query']:<48} {d['points']} points")
    print(f"Unique Qdrant chunks: {data['pool_points']} (original-query chunks: {data['original_points']})")
    print('WARNING: experimental relevance indices, not calibrated expertise percentages.')
    print('Profile? NO means no profile evidence was found within the filtered top-k retrieval budget; not proof of absent expertise.')
    print('Faculty                          Old     V2     V4P Method    App  Profile Support Profile?')
    print('-' * 107)
    rows = data['rows']
    if args.names:
        names = {n.strip().casefold() for n in args.names.split(',') if n.strip()}
        rows = [r for r in rows if r['name'].casefold() in names]
        missing = names - {r['name'].casefold() for r in rows}
        if missing:
            print('Not found:', ', '.join(sorted(missing)))
    else:
        rows = rows[:args.top_n]
    for row in rows:
        old = f"{row['old']:.1f}" if row['old'] is not None else 'n/a'
        v2 = f"{row['v2']:.1f}" if row['v2'] is not None else 'n/a'
        print(f"{row['name'][:31]:31} {old:>5} {v2:>6} {row['v4']:6.1f} {row['method']:7.1f} {row['application']:6.1f} {row['profile']:8.1f} {row['supporting_records']:7} {'yes' if row['profile_found'] else 'NO':>8}")
        if args.details:
            for area, evidence in row['evidence'].items():
                if not evidence:
                    print(f'    {area}: no supporting research retrieved')
                    continue
                best = evidence[0]
                print(f"    {area}: [{best['kind']}] adjusted={best['adjusted']:.3f} raw_normalized={best['raw']:.3f}")
                print(f"       {best['title']}")
                print(f"       Found via: {', '.join(best.get('discovery_queries', []))}")


if __name__ == '__main__':
    main()
