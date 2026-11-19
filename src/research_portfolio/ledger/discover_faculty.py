#!/usr/bin/env python3
"""Stage 1 - roster discovery from the Digital Measures faculty directory (read-only).
Output: data/departments/<slug>/faculty.json and raw/directory_<ts>.json"""
import argparse, sys
from collections import Counter
from datetime import datetime, timezone
from research_portfolio.common import (load_config, department_dir, fetch_json, dm_records, index_values,
                                       as_list, clean, sha, write_atomic)

def extract(records, targets):
    wanted = {t.casefold() for t in targets}
    faculty, seen, skipped = [], set(), []
    for rec in records:
        depts = index_values(rec, "DEPARTMENT")
        if not any(d.casefold() in wanted for d in depts):
            continue
        pci = (as_list(rec.get("PCI")) or [{}])[0]
        given, middle, family, suffix = (clean(pci.get(k)) for k in ("FNAME", "MNAME", "LNAME", "SUFFIX"))
        uid, user = clean(rec.get("@userId")), clean(rec.get("@username"))
        if not uid or not (given or family):
            skipped.append(user or "?")
            continue
        pid = f"dm:{uid}"
        if pid in seen:
            continue
        seen.add(pid)
        faculty.append({
            "person_id": pid, "dm_user_id": uid, "username": user,
            "display_name": " ".join(x for x in (given, middle, family, suffix) if x),
            "given_name": given, "middle_name": middle, "family_name": family,
            "rank": (index_values(rec, "RANK") or [None])[0],
            "dm_departments": depts,
        })
    faculty.sort(key=lambda p: ((p["family_name"] or "").casefold(), (p["given_name"] or "").casefold()))
    return faculty, skipped

def main(argv=None):
    ap = argparse.ArgumentParser(description="Discover department faculty from Digital Measures")
    ap.add_argument("--config", required=True)
    ap.add_argument("--list-departments", action="store_true", help="print DEPARTMENT values with counts and exit")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    src = cfg["sources"]["faculty"]
    targets = [src.get("department_name") or cfg["department_name"], *src.get("department_aliases", [])]
    out = department_dir(cfg)

    print(f"Fetching directory: {src['api_url']}")
    raw, data = fetch_json(src["api_url"])
    records = dm_records(data)
    if not records:
        print("ERROR: directory returned no Data.Record entries", file=sys.stderr)
        return 1

    if a.list_departments:
        for name, n in sorted(Counter(d for r in records for d in index_values(r, "DEPARTMENT")).items()):
            print(f"{n:4}  {name}")
        return 0

    faculty, skipped = extract(records, targets)
    if not faculty:
        print(f"ERROR: 0 faculty matched {targets}. Run with --list-departments.", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc)
    raw_path = out / "raw" / f"directory_{ts:%Y%m%dT%H%M%SZ}.json"
    write_atomic(raw_path, raw)
    write_atomic(out / "faculty.json", {
        "schema_version": 1, "stage": "discover_faculty", "retrieved_at": ts.isoformat(),
        "institution": cfg["institution"],
        "department": {"slug": cfg["department_slug"], "name": cfg["department_name"], "matched_names": targets},
        "source": {"url": src["api_url"], "raw_snapshot": str(raw_path.relative_to(out)),
                   "sha256": sha(raw), "records_in_source": len(records)},
        "count": len(faculty), "skipped_incomplete": skipped, "faculty": faculty,
    })
    for i, p in enumerate(faculty, 1):
        print(f"{i:3}. {p['display_name']:<35} {p['rank'] or '-':<30} {p['person_id']}")
    if skipped:
        print(f"WARNING: skipped {len(skipped)} records missing name or userId: {skipped}")
    print(f"\n{len(faculty)} faculty -> {out / 'faculty.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
