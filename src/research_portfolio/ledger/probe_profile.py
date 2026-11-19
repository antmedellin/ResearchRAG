#!/usr/bin/env python3
"""Parse one raw DM profile and print what the ledger would extract (no writes).
  python -m research_portfolio.ledger.probe_profile data/departments/maie/raw/profiles/1450741.json"""
import sys
from collections import Counter
from pathlib import Path
from research_portfolio.common import decode_dm, single_record, as_list, clean
from research_portfolio.ledger.dm_parse import SECTIONS, records, parse_person

def main():
    rec = single_record(decode_dm(Path(sys.argv[1]).read_bytes()))
    uid = clean(rec["@userId"]); pid = f"dm:{uid}"
    pci = (as_list(rec.get("PCI")) or [{}])[0]
    {pid: (clean(pci.get("FNAME")), clean(pci.get("LNAME")))}
    p = parse_person({"person_id": pid, "family_name": roster[pid][1]}, rec)
    print(f"{clean(pci.get('FNAME'))} {roster[pid][0]} {roster[pid][1]} ({pid})")
    print(f"  bio: {'yes' if p.get('bio') else 'no'} | keywords: {len(p.get('research_keywords', []))} | "
          f"education {len(p.get('education', []))} | awards {len(p.get('awards', []))} | "
          f"experience {len(p.get('experience', []))}")
    for kind, (section, parse) in SECTIONS.items():
        xs = [x for r in records(rec, section) if (x := parse(r, roster, pid))]
        linked = sum(any(c.get("person_id") == pid for c in x["contributors"]) for x in xs)
        rej = sum(1 for x in xs for c in x["contributors"] if c.get("dm_user_id_rejected"))
        print(f"  {kind:13} {len(xs):4} | abstract {sum('abstract' in x for x in xs):4} | owner linked {linked}/{len(xs)}"
              f" | rejected ids {rej} | status {dict(Counter(x.get('status') for x in xs))}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
