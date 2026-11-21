"""Conservative person-name resolution with Clef and learned aliases."""
import json, os
from datetime import datetime, timezone
import requests
from rapidfuzz import fuzz
from research_portfolio.common import REGISTRY, norm_key, write_atomic

DECISION_URL=os.getenv("DECISION_URL","http://ollama:11434/v1/systemone").strip()
DECISION_MODEL=os.getenv("DECISION_MODEL","clef-flash:9b").strip()
SAME_MIN=float(os.getenv("PERSON_SAME_MIN","0.85"))
ALIASES=REGISTRY/"person_aliases.json"
DECISIONS=REGISTRY/"person_identity_decisions.jsonl"
REVIEW=REGISTRY/"person_identity_review.jsonl"
QUESTION={"identity":{"type":"choice","instructions":"Do the roster person and contributor describe the same human? The Digital Measures user ID agrees, but historical names may reverse given/family names, omit surnames, use initials, contain transliteration variants, or reflect a surname change.","criteria":{"different":"Different people; the ID is erroneously attached to a coauthor.","uncertain":"Insufficient evidence.","same":"Same person under an alternate or historical name."}}}


def clef_candidate(name, person):
    """Require some lexical evidence before Clef may override a name mismatch."""

    candidate = set(tokens(name))
    roster = set(tokens(display_name(person)))

    if not candidate or not roster:
        return False

    # At least one meaningful exact token in common.
    shared = {
        token
        for token in candidate & roster
        if len(token) >= 3
    }

    if shared:
        return True

    # Or at least one strong fuzzy token match.
    for a in candidate:
        if len(a) < 4:
            continue

        for b in roster:
            if len(b) < 4:
                continue

            if fuzz.ratio(a, b) >= 88:
                return True

    return False


def display_name(p): return p.get("display_name") or " ".join(x for x in (p.get("given_name"),p.get("family_name")) if x)
def tokens(v): return norm_key(v).split()
def deterministic_level(name,p):
    t=set(tokens(name)); fam=tokens(p.get("family_name")); given=tokens(p.get("given_name"))
    if not t:return 0
    if not fam or set(fam)<=t:return 3
    given_ok=bool(given) and (given[0] in t or given[0][0] in t)
    fam_ok=any(f in t for f in fam if len(f)>2) or any(fuzz.ratio(f,x)>=88 for f in fam for x in t if len(x)>3)
    if given_ok and fam_ok:return 2
    shared=t & set(tokens(display_name(p)))
    if len(t)>=2 and len(shared)>=2 and len(shared)/len(t)>=.8:return 2
    if given and given[0] in t:return 1
    return 0
def load_json(path,default): return json.loads(path.read_text()) if path.exists() else default
def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
def learn(person_id,name):
    d=load_json(ALIASES,{})
    if norm_key(name) not in {norm_key(x) for x in d.get(person_id,[])}:
        d.setdefault(person_id,[]).append(name); d[person_id].sort(key=norm_key); write_atomic(ALIASES,d)
def resolve_same_id(person_id, name, person):
    # 1. Strong deterministic match.
    if deterministic_level(name, person) >= 2:
        return True, "deterministic", None

    # 2. Previously learned alias.
    aliases = load_json(ALIASES, {})
    if norm_key(name) in {
        norm_key(x)
        for x in aliases.get(person_id, [])
    }:
        return True, "learned_alias", None

    pair = f"{person_id}|{norm_key(name)}"

    # 3. Previously cached Clef decision.
    cache = (
        {
            r["pair"]: r
            for r in (
                json.loads(x)
                for x in DECISIONS.read_text().splitlines()
            )
        }
        if DECISIONS.exists()
        else {}
    )

    if pair in cache:
        return (
            cache[pair]["same"],
            "clef_cache",
            cache[pair].get("probabilities"),
        )

    # 4. IMPORTANT SAFETY GATE:
    # Do not allow Clef to override a completely unrelated name.
    if not clef_candidate(name, person):
        return False, "name_conflict", None

    # 5. Ambiguous but lexically plausible. Ask Clef.
    state = {
        "roster_person": {
            "person_id": person_id,
            "display_name": display_name(person),
            "given_name": person.get("given_name"),
            "family_name": person.get("family_name"),
        },
        "contributor": {
            "digital_measures_user_id": person_id.removeprefix("dm:"),
            "name": name,
        },
    }

    try:
        r = requests.post(
            DECISION_URL,
            json={
                "model": DECISION_MODEL,
                "state": state,
                "questions": QUESTION,
                "keep_alive": "10m",
            },
            timeout=180,
        )

        r.raise_for_status()

        a = r.json()["answers"]["identity"]

        probs = {
            k: float(v)
            for k, v in a["probabilities"].items()
        }

    except Exception as e:
        append(
            REVIEW,
            {
                "pair": pair,
                "issue": "decision_error",
                "error": str(e),
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )

        return False, "decision_error", None

    same = (
        a["choice"] == "same"
        and probs.get("same", 0) >= SAME_MIN
    )

    row = {
        "pair": pair,
        "same": same,
        "choice": a["choice"],
        "probabilities": probs,
        "confidence": a.get("confidence"),
        "model": DECISION_MODEL,
        "person_id": person_id,
        "roster_name": display_name(person),
        "contributor_name": name,
        "at": datetime.now(timezone.utc).isoformat(),
    }

    append(DECISIONS, row)

    if same:
        learn(person_id, name)

    elif (
        a["choice"] == "uncertain"
        or probs.get("same", 0) >= 0.2
    ):
        append(
            REVIEW,
            row | {"issue": "uncertain"},
        )

    return same, "clef", probs
