"""Resolve ledger entities to canonical IDs against what is already in Neo4j (and within the batch).
Tiers: exact keys -> exclusion rules -> fuzzy -> decision model (uncertain band only, cached) -> overrides."""
import json, os
from collections import Counter, defaultdict
from datetime import datetime, timezone
import requests
from rapidfuzz import fuzz
from research_portfolio.common import REGISTRY, norm_key, sha
from research_portfolio.ledger.merge import identity_keys
import re


AUTO_SAME, REVIEW_LOW, DOI_TITLE_MIN, LLM_MIN_CONF = 96, 85, 85, 0.8

CACHE, OVERRIDES, REVIEW = (REGISTRY / "dedup_decisions.jsonl", REGISTRY / "dedup_overrides.json",
                            REGISTRY / "review.jsonl")

from research_portfolio.llm_endpoints import DECISION_URL
DECISION_MODEL = os.environ.get("DECISION_MODEL")
SAME_MIN, REVIEW_MIN = 0.85, 0.20          # thresholds on probabilities["same"]

QUESTIONS = {
    "relation": {
        "type": "choice",
        "instructions": "How are record_a and record_b related? Records were entered by different "
                        "faculty, so capitalization, punctuation, abbreviations and small typos vary.",
        "criteria": {   # 'different' first: ties fall to the safe option
            "different": "Two unrelated items.",
            "variant": (
                "Related but distinct items: the same presentation delivered at different events; "
                "a conference paper and journal article; Part 1 versus Part 2; different reports "
                "in a series; or a grant renewal, supplement, extension, phase, or separate award."
            ),
            "same": (
                "One single real-world item entered twice. For presentations, the same talk "
                "given at two different events is NOT the same item. For grants, a renewal, "
                "supplement, extension, or separate award is NOT the same item."
            )
        },
    },
}


PART_RE = re.compile(
    r"\b(?:part|phase|volume|vol|series)\s*[:\-]?\s*"
    r"(\d+|i{1,3}|iv|v|vi{0,3}|ix|x)\b",
    re.I,
)

def sequence_marker(title):
    m = PART_RE.search(title or "")
    return m.group(1).casefold() if m else None


def family_names(x):
    if x.get("family_names"):
        return set(x["family_names"])
    return {norm_key(c.get("family_name")) for c in x.get("contributors", []) if c.get("family_name")}

def view(x, cid=None):
    names = x.get("contributor_names") or [c.get("name") for c in x.get("contributors", []) if c.get("name")]
    return {"id": cid or x.get("id"), "title": x["title"], "nt": norm_key(x["title"]), "year": x.get("year"),
            "doi": x.get("doi"), "sponsor": x.get("sponsor"), "venue": x.get("publisher") or x.get("event"),
            "type": x.get("type"), "status": x.get("status"), "fam": family_names(x), "names": names[:8],
            "abstract": " ".join((x.get("abstract") or "").split()[:60])}

def fingerprint(v):
    return sha(f"{v['nt']}|{v['year']}|{','.join(sorted(v['fam']))}", 16)

def _jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []

class Resolver:
    def __init__(self, kind, existing):
        """existing: Neo4j rows (decoded props) with 'id' and 'identity_keys'."""
        REGISTRY.mkdir(parents=True, exist_ok=True)
        self.kind, self.stats, self.pool, self.key_index, self.views = kind, Counter(), defaultdict(list), {}, {}
        for e in existing:
            self._register(view(e), e.get("identity_keys", []))
        self.cache = {d["pair"]: d for d in _jsonl(CACHE)}
        ov = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
        self.overrides = {"|".join(sorted(p)): v for v in ("same", "distinct") for p in ov.get(v, [])}

    def _register(self, v, keys):
        for k in keys:
            self.key_index.setdefault(k, v["id"])
        self.pool[v["year"]].append(v)
        self.views[v["id"]] = v

    def resolve(self, x):
        """-> (canonical_id, method, identity_keys). Registers x so later records in the batch match it."""
        keys, v = identity_keys(x), view(x)
        hits = set()
        for k in keys:
            cid = self.key_index.get(k)
            if not cid:
                continue
            if ":doi:" in k and fuzz.token_sort_ratio(v["nt"], self.views[cid]["nt"]) < DOI_TITLE_MIN:
                self._review("doi_title_mismatch", v, {"candidate": cid, "key": k})
                continue                                  # shared DOI but different title -> venue DOI
            hits.add(cid)
        hits = sorted(hits)
        if hits:
            cid, method = hits[0], "key"
            if len(hits) > 1:
                self._review("key_conflict", v, {"ids": hits})
        else:
            cid, method = self._fuzzy(v)
        if cid is None:
            cid, method = f"{self.kind}_{sha(keys[0], 16)}", "new"
        v["id"] = cid
        self._register(v, keys)
        self.stats[method] += 1
        return cid, method, keys

    def _candidates(self, v):
        if v["year"] is None:
            return self.pool.get(None, [])
        return [c for y in (v["year"] - 1, v["year"], v["year"] + 1) for c in self.pool.get(y, [])]

    def _fuzzy(self, v):
        best = None
        for c in self._candidates(v):
            seq_a = sequence_marker(v["title"])
            seq_b = sequence_marker(c["title"])

            if seq_a and seq_b and seq_a != seq_b:
                continue
            
            if v["doi"] and c["doi"] and v["doi"] != c["doi"]:
                continue
            score = fuzz.token_sort_ratio(v["nt"], c["nt"])
            if score < REVIEW_LOW:
                continue
            shared = bool(v["fam"] & c["fam"])
            if self.kind == "grant" and v["sponsor"] and c["sponsor"]:
                shared = shared and fuzz.token_set_ratio(norm_key(v["sponsor"]), norm_key(c["sponsor"])) >= 80
            if not best or score > best[0]:
                best = (score, shared, c)
        if not best:
            return None, None
        score, shared, c = best
        pair = "|".join(sorted([fingerprint(v), fingerprint(c)]))
        if pair in self.overrides:
            return (c["id"], "override") if self.overrides[pair] == "same" else (None, None)
        if score >= AUTO_SAME and shared and v["year"] == c["year"]:
            self._review("auto_fuzzy_merge", v, {
                "candidate": c["id"],
                "candidate_title": c["title"],
                "candidate_fingerprint": fingerprint(c),
                "score": score,
                "shared_family_names": sorted(v["fam"] & c["fam"]),
            })
            return c["id"], "fuzzy"
        if shared or score >= AUTO_SAME:
            return (c["id"], "llm") if self._decide(pair, v, c, score) else (None, None)
        return None, None

    def _decide(self, pair, a, b, score):
        if pair in self.cache:
            return self.cache[pair]["same"]
        cand = {"candidate": b["id"], "candidate_title": b["title"], "candidate_fingerprint": fingerprint(b)}
        if not DECISION_MODEL:
            self._review("undecided_no_model", a, {**cand, "score": score})
            return False
        show = lambda v: {k: v[k] for k in ("title", "year", "type", "status", "venue", "sponsor", "doi",
                                            "names", "abstract") if v.get(k)}
        state = {"record_type": self.kind, "record_a": show(a), "record_b": show(b)}
        try:
            r = requests.post(DECISION_URL, json={"model": DECISION_MODEL, "state": state,
                                                  "questions": QUESTIONS}, timeout=180)
            if not r.ok:
                raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
            ans = r.json()["answers"]["relation"]
            probs = {k: float(v) for k, v in ans["probabilities"].items()}
        except Exception as exc:
            self._review("decision_error", a, {**cand, "score": score, "error": str(exc)[:300]})
            return False                      # not cached, so the pair is retried on the next run

        same = ans["choice"] == "same" and probs.get("same", 0) >= SAME_MIN
        rec = {"pair": pair, "same": same, "choice": ans["choice"],
               "probabilities": {k: round(v, 4) for k, v in probs.items()},
               "confidence": ans.get("confidence"), "score": score, "model": DECISION_MODEL,
               "kind": self.kind, "a": a["title"], "b": b["title"],
               "at": datetime.now(timezone.utc).isoformat()}
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.cache[pair] = rec
        if not same and (probs.get("same", 0) >= REVIEW_MIN or ans["choice"] == "variant"):
            self._review("uncertain", a, {**cand, "choice": ans["choice"], "probabilities": rec["probabilities"]})
        return same

    def _review(self, issue, v, extra):
        with REVIEW.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": self.kind, "issue": issue, "title": v["title"], "year": v["year"],
                                "fingerprint": fingerprint(v), **extra,
                                "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
