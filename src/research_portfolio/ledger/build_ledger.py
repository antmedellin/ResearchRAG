#!/usr/bin/env python3
"""Stage 3 - build the department ledger from cached raw profiles (offline, deterministic).
Exact-key merging within the department only; cross-department dedup happens at sync time.
Output: data/departments/<slug>/ledger.json"""
import argparse, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from research_portfolio.common import (load_config, department_dir, read_json, write_atomic, decode_dm,
                                       single_record, norm_key, sha)
from research_portfolio.ledger.dm_parse import SECTIONS, records, parse_person
from research_portfolio.ledger.merge import merge_records, identity_keys

SCHEMA_VERSION = 2
ROSTER_FIELDS = ("person_id", "dm_user_id", "username", "display_name", "given_name", "middle_name",
                 "family_name", "rank", "dm_departments")

class Bucket:
    """Merge records that share any exact identity key (union semantics)."""
    def __init__(self):
        self.items, self.key_to = {}, {}
    def add(self, x):
        keys = identity_keys(x, include_doi=False)
        hits = sorted({self.key_to[k] for k in keys if k in self.key_to})
        if not hits:
            lid = len(self.items)
            self.items[lid] = x
        else:
            lid = hits[0]
            self.items[lid] = merge_records(self.items[lid], x)
            for other in hits[1:]:
                self.items[lid] = merge_records(self.items[lid], self.items.pop(other))
                for k, v in list(self.key_to.items()):
                    if v == other:
                        self.key_to[k] = lid
        for k in identity_keys(self.items[lid], include_doi=False):
            self.key_to[k] = lid

def resolve_dois(pubs):
    """Accept a DOI only if exactly one title in this department uses it; else it's a venue DOI."""
    titles = defaultdict(set)
    for x in pubs:
        for d in x.get("doi_candidates", []):
            titles[d].add(norm_key(x["title"]))
    for x in pubs:
        cands = x.pop("doi_candidates", [])
        good = [d for d in cands if len(titles[d]) == 1]
        if good:
            x["doi"] = good[0]
        venue = [d for d in cands if d not in good]
        if venue:
            x["venue_doi"] = venue[0]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="build even if some profiles are missing")
    ap.add_argument("--list-sections", action="store_true", help="print DM sections across profiles and exit")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = department_dir(cfg)
    roster_doc = read_json(out / "faculty.json")
    manifest = {e["person_id"]: e for e in read_json(out / "profiles_manifest.json")["profiles"]}
    roster = {p["person_id"]: p for p in roster_doc["faculty"]}

    def record_for(pid):
        e = manifest.get(pid)
        if not e or e.get("status") == "failed" or not (out / e["path"]).is_file():
            return None
        return single_record(decode_dm((out / e["path"]).read_bytes()))

    if a.list_sections:
        c = Counter(k for pid in roster if (r := record_for(pid)) for k in r if not k.startswith("@"))
        for k, n in c.most_common():
            print(f"{n:4}  {k}")
        return 0

    people, missing = [], []
    buckets = {kind: Bucket() for kind in SECTIONS}
    rejected = 0
    for rp in roster_doc["faculty"]:
        pid = rp["person_id"]
        rec = record_for(pid)
        base = {k: rp.get(k) for k in ROSTER_FIELDS if rp.get(k)}
        if rec is None:
            missing.append(rp["display_name"])
            people.append({**base, "flags": ["profile_missing"]})
            continue
        people.append(parse_person(base, rec))
        for kind, (section, parse) in SECTIONS.items():
            for r in records(rec, section):
                x = parse(r, roster, pid)
                if not x:
                    continue
                x["claimed_by"] = [pid]
                rejected += sum(1 for c in x["contributors"] if c.get("dm_user_id_rejected"))
                buckets[kind].add(x)

    if missing and not a.allow_partial:
        print(f"ERROR: no profile for {missing}. Re-run fetch_profiles or pass --allow-partial.", file=sys.stderr)
        return 1

    entities = {}
    for kind, b in buckets.items():
        items = list(b.items.values())
        if kind == "publication":
            resolve_dois(items)
        for x in items:
            x.pop("doi_candidates", None)
            x["ledger_id"] = f"{kind}:{sha('|'.join(sorted(identity_keys(x, False))), 12)}"
        items.sort(key=lambda x: (-(x.get("year") or 0), norm_key(x["title"])))
        entities[kind] = items

    # owner-not-in-contributors check + per-person counts
    suspicious = []
    count = defaultdict(Counter)
    for kind, items in entities.items():
        for x in items:
            linked = {c["person_id"] for c in x["contributors"] if c.get("in_roster")}
            for pid in linked:
                count[pid][kind] += 1
            for c in x["contributors"]:
                if c.get("match") in ("owner_given_name", "owner_claim"):
                    suspicious.append({"kind": kind, "title": x["title"], "owner": c["person_id"], "issue": c["match"]})
    for p in people:
        c = count[p["person_id"]]
        p["counts"] = {k: c[k] for k in SECTIONS}
        flags = set(p.get("flags", []))
        if not p.get("research_interests_text"):
            flags.add("no_research_interests")
        if not p.get("bio"):
            flags.add("no_bio")
        has_text = any(x.get("abstract") for k in ("publication", "grant") for x in entities[k]
                       if any(cc.get("person_id") == p["person_id"] for cc in x["contributors"]))
        if not has_text and {"no_bio", "no_research_interests"} <= flags:
            flags.add("text_poor")
        p["flags"] = sorted(flags)

    q = {
        "people": len(people), "profiles_missing": missing,
        "text_poor": [p["display_name"] for p in people if "text_poor" in p["flags"]],
        "rejected_dm_user_ids": rejected, "suspicious": suspicious,
        **{f"{k}s": len(v) for k, v in entities.items()},
        **{f"{k}s_with_abstract": sum(bool(x.get("abstract")) for x in v) for k, v in entities.items()},
        "publications_with_doi": sum(bool(x.get("doi")) for x in entities["publication"]),
        "grant_status": dict(Counter(x.get("status") or "unknown" for x in entities["grant"])),
    }
    write_atomic(out / "ledger.json", {
        "schema_version": SCHEMA_VERSION, "stage": "build_ledger",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "institution": cfg["institution"],
        "department": {"slug": cfg["department_slug"], "name": cfg["department_name"],
                       "aliases": cfg["sources"]["faculty"].get("department_aliases", [])},
        "inputs": {"roster_retrieved_at": roster_doc.get("retrieved_at"),
                   "profiles": {pid: e.get("sha256") for pid, e in manifest.items()}},
        "quality": q, "people": people, "entities": entities,
    })
    print(f"People {q['people']} | missing {len(missing)} | text-poor {len(q['text_poor'])}")
    for k in SECTIONS:
        print(f"{k:13} {q[k + 's']:5} | with abstract {q[k + 's_with_abstract']}")
    print(f"DOIs {q['publications_with_doi']} | rejected DM ids {rejected} | suspicious {len(suspicious)}")
    print(f"-> {out / 'ledger.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
