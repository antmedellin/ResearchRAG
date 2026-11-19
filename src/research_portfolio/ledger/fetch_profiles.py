#!/usr/bin/env python3
"""Stage 2 - download one raw DM profile per roster member (network only).
Output: raw/profiles/<dm_user_id>.json + profiles_manifest.json"""
import argparse, sys, time, urllib.parse
from datetime import datetime, timedelta, timezone
from research_portfolio.common import (load_config, department_dir, read_json, fetch_json, write_atomic,
                                       single_record, clean, sha)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--refresh", action="store_true", help="re-download every profile")
    ap.add_argument("--max-age-days", type=float, help="re-download cached profiles older than this")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--allow-partial", action="store_true", help="exit 0 even if some profiles fail")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, require=("sources.faculty.profile_api_url",))
    out = department_dir(cfg)
    roster = read_json(out / "faculty.json")
    base_url = cfg["sources"]["faculty"]["profile_api_url"]
    prof_dir = out / "raw" / "profiles"
    now = datetime.now(timezone.utc)
    entries, failed = [], []

    for i, p in enumerate(roster["faculty"], 1):
        uid, user = p["dm_user_id"], p.get("username")
        path = prof_dir / f"{uid}.json"
        e = {"person_id": p["person_id"], "username": user, "path": str(path.relative_to(out))}
        fresh = path.exists() and (a.max_age_days is None or
                now - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < timedelta(days=a.max_age_days))
        try:
            if fresh and not a.refresh:
                raw, e["status"] = path.read_bytes(), "cached"
            else:
                if not user:
                    raise ValueError("roster entry has no username")
                raw, data = fetch_json(base_url + urllib.parse.quote(user))
                got = clean(single_record(data).get("@userId"))
                if got != uid:
                    raise ValueError(f"profile userId {got} != roster {uid}")
                write_atomic(path, raw)
                e["status"] = "fetched"
                time.sleep(a.delay)
            e["sha256"] = sha(raw)
            e["fetched_at"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        except Exception as exc:
            e.update(status="failed", error=str(exc))
            failed.append(p["display_name"])
        entries.append(e)
        print(f"[{i:3}/{len(roster['faculty'])}] {e['status']:<7} {p['display_name']}"
              + (f"  ERROR: {e['error']}" if e["status"] == "failed" else ""))

    write_atomic(out / "profiles_manifest.json", {
        "schema_version": 1, "stage": "fetch_profiles", "generated_at": now.isoformat(),
        "roster_retrieved_at": roster.get("retrieved_at"), "profiles": entries})
    print(f"\n{len(entries) - len(failed)}/{len(entries)} profiles available")
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 0 if a.allow_partial else 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
