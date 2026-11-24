"""Digital Measures profile record -> ledger entities. Field names verified against a real UTRGV profile.
Used: PCI, EDUCATION, AWARDHONOR, PASTHIST, MEMBER, LICCERT, INTELLCONT, CONGRANT, PRESENT.
Skipped (no research signal): SCHTEACH*, NCTEACH, DSL, FACDEV, SERVICE_*."""
import re
from research_portfolio.common import as_list, clean, norm_key, keep
from rapidfuzz import fuzz
from research_portfolio.resolve.person_identity import resolve_same_id, deterministic_level

MIN_ABSTRACT_WORDS = 30      # shorter DM "abstracts" are notes ("updated December 2005")
DOI_RE  = re.compile(r"10\.\d{4,9}/[^\s\"<>,;]+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

def year(*vals):
    for v in vals:
        m = YEAR_RE.search(str(v or ""))
        if m:
            return int(m.group(0))
    return None

def doi_in(v):
    m = DOI_RE.search(str(v or ""))
    return m.group(0).rstrip(".").lower() if m else None

def money(v):
    try:
        return float(re.sub(r"[^\d.]", "", str(v))) if v not in (None, "") else None
    except ValueError:
        return None

def records(rec, section):
    return [r for r in as_list(rec.get(section)) if isinstance(r, dict)]

def split_abstract(text, min_words=MIN_ABSTRACT_WORDS):
    t = clean(text)
    if not t:
        return None, None
    return (t, None) if len(t.split()) >= min_words else (None, t)

def _tokens(v):
    return norm_key(v).split()

def name_match_level(name, given, family):
    """0 = no match, 1 = weak (given name only), 2 = medium, 3 = strong.
    Handles compound/hyphenated surnames, name changes keeping one surname
    ('Orozco-Leonhardt' vs 'Orozco Lopez'), typos ('Srivastav') and initials."""
    toks = set(_tokens(name)); fam = _tokens(family)
    if not toks:
        return 0
    if not fam or set(fam) <= toks:
        return 3
    g = _tokens(given)
    given_ok = bool(g) and (g[0] in toks or g[0][0] in toks)
    fam_part = any(f in toks for f in fam if len(f) > 2)
    fam_fuzzy = any(fuzz.ratio(f, t) >= 88 for f in fam for t in toks if len(t) > 3)
    if given_ok and (fam_part or fam_fuzzy):
        return 2
    if g and g[0] in toks:
        return 1
    return 0

def contributors(items, roster, owner, role_key=None):
    """roster: {person_id: (given_name, family_name)}; owner: person_id of the profile being parsed.
    DM stamps the OWNER's userId onto external co-authors, so FACULTY_NAME is trusted only at level >= 2.
    If the owner is still unidentified, the best owner-name match is accepted (a weak given-name match
    only if unique). If the owner isn't listed at all, they're appended with listed=False."""
    entries = [x for x in as_list(items) if isinstance(x, dict)]
    out = []
    for pos, a in enumerate(entries, 1):
        raw_uid = clean(a.get("FACULTY_NAME"))
        name = " ".join(filter(None, (clean(a.get(k)) for k in ("FNAME", "MNAME", "LNAME")))) or None
        pid = f"dm:{raw_uid}" if raw_uid else None
        
        identity_method = None
        if pid and pid in roster:
            trusted, identity_method, _ = resolve_same_id(pid, name, roster[pid])
        else:
            trusted = bool(pid)
        
        out.append({"position": pos, "name": name, "family_name": clean(a.get("LNAME")),
                    "person_id": pid if trusted else None, "raw_uid": raw_uid, "identity_method": identity_method,
                    "role": clean(a.get(role_key)) if role_key else None,
                    "institution": clean(a.get("INSTITUTION")), "student_level": clean(a.get("STUDENT_LEVEL"))})
    if owner in roster and not any(c["person_id"] == owner for c in out):
        levels = [(deterministic_level(c["name"], roster[owner]), c) for c in out if not c["person_id"]]
        best = max((l for l, _ in levels), default=0)
        winners = [c for l, c in levels if l == best]
        if best >= 2 or (best == 1 and len(winners) == 1):
            winners[0]["person_id"] = owner
            winners[0]["match"] = "owner_name" if best >= 2 else "owner_given_name"
            winners[0]["identity_method"] = "owner_fallback"
            
        else:
            out.append({"position": None, "name": None, "person_id": owner, "listed": False, "match": "owner_claim","identity_method": "owner_claim"})
    result = []

    for c in out:
        pid = c.pop("person_id")
        raw = c.pop("raw_uid", None)

        result.append(keep({
            **c,
            "person_id": pid,
            "in_roster": (pid in roster) if pid else None,
            "dm_user_id": pid[3:] if pid else None,
            "dm_user_id_rejected": (
                raw
                if raw and (not pid or pid[3:] != raw)
                else None
            ),
        }))

    return result

def split_keywords(text):
    if not text:
        return []
    #remove a leading "research interests:" label, if presents
    text = re.sub(
        r"^\s*Research\s+Interests\s*:\s*",
        "",
        text,
        flags = re.IGNORECASE
    )
    seen, out = set(), []
    #split the research-interest text into individual keywords
    for p in (
        clean(x.strip(" ."))
         for x in re.split(r"[;,]", re.sub(r"\(.*?\)", "", text))
    ):
        if p and 2 <= len(p) <= 60 and (k := norm_key(p)) not in seen:
            seen.add(k)
            out.append(p)
    return out

# ---------------- person ----------------
def parse_person(rp, rec):
    pci = (records(rec, "PCI") or [{}])[0]
    ri = pci.get("PCI_RESEARCH_INTERESTS")
    interests = clean(ri.get("DESC")) if isinstance(ri, dict) else clean(ri)
    interests = interests or clean(pci.get("RESEARCH_INTERESTS"))
    return keep({
        **rp,
        "prefix": clean(pci.get("PREFIX")), "email": clean(pci.get("EMAIL")),
        "bio": clean(pci.get("BIO")),
        "research_interests_text": interests,
        "research_keywords": split_keywords(interests),
        "teaching_interests": clean(pci.get("TEACHING_INTERESTS")),
        "education": [keep({
            "degree": clean(e.get("DEG")) or clean(e.get("DEGOTHER")), "major": clean(e.get("MAJOR")),
            "support_area": clean(e.get("SUPPAREA")), "institution": clean(e.get("SCHOOL")),
            "location": clean(e.get("LOCATION")), "year": year(e.get("YR_COMP"), e.get("COMP_START")),
            "dissertation": clean(e.get("DISSTITLE")), "highest": clean(e.get("HIGHEST")) == "Yes" or None,
        }) for e in records(rec, "EDUCATION")],
        "awards": [keep({
            "name": clean(w.get("NAME")), "organization": clean(w.get("ORG")), "scope": clean(w.get("SCOPE")),
            "description": clean(w.get("DESC")), "year": year(w.get("DTY_DATE"), w.get("DATE_START")),
        }) for w in records(rec, "AWARDHONOR")],
        "experience": [keep({
            "type": clean(x.get("EXPTYPE")), "organization": clean(x.get("ORG")), "title": clean(x.get("TITLE")),
            "description": clean(x.get("DESC")), "start_year": year(x.get("DTY_START")),
            "end_year": year(x.get("DTY_END")),
        }) for x in records(rec, "PASTHIST")],
        "org_memberships": [keep({
            "name": clean(m.get("NAME")), "abbr": clean(m.get("ORGABBR")), "leadership": clean(m.get("LEADERSHIP")),
            "scope": clean(m.get("SCOPE")), "start_year": year(m.get("DTY_START")), "end_year": year(m.get("DTY_END")),
        }) for m in records(rec, "MEMBER")],
        "licenses": [keep({
            "title": clean(c.get("TITLE")), "organization": clean(c.get("ORG")), "scope": clean(c.get("SCOPE")),
            "year": year(c.get("DTY_START")),
        }) for c in records(rec, "LICCERT")],
    })

# ---------------- entities (all use "contributors") ----------------
def parse_publication(r, roster, owner):
    title = clean(r.get("TITLE"))
    if not title:
        return None
    abstract, note = split_abstract(r.get("ABSTRACT"))
    url = clean(r.get("WEB_ADDRESS"))
    return keep({
        "kind": "publication", "dm_record_ids": [clean(r.get("@id"))],
        "title": title, "title_secondary": clean(r.get("TITLE_SECONDARY")),
        "year": year(r.get("DTY_PUB"), r.get("PUB_START"), r.get("DTY_ACC"), r.get("DTY_SUB")),
        "type": clean(r.get("CONTYPE")) or clean(r.get("CONTYPEOTHER")),
        "status": clean(r.get("STATUS")), "refereed": clean(r.get("REFEREED")) == "true",
        "classification": clean(r.get("CLASSIFICATION")),
        "publisher": clean(r.get("PUBLISHER")), "volume": clean(r.get("VOLUME")),
        "issue": clean(r.get("ISSUE")), "pages": clean(r.get("PAGENUM")),
        "city": clean(r.get("PUBCTYST")), "country": clean(r.get("PUBCNTRY")),
        "funded": clean(r.get("FUNDED")), "contract_number": clean(r.get("CONTRACT_NUM")),
        # DM often holds a *venue* DOI (proceedings volume); resolved in build_ledger
        "doi_candidates": sorted({d for d in (doi_in(r.get("DOI")), doi_in(url), doi_in(r.get("VOLUME"))) if d}),
        "url": url if url and re.match(r"https?://", url, re.I) else None,
        "abstract": abstract, "note": note,
        "contributors": contributors(r.get("INTELLCONT_AUTH"), roster, owner),
        "public": clean(r.get("INCLUDE_PROFILE")) == "true",
    })

def parse_grant(r, roster, owner):
    title = clean(r.get("TITLE"))
    if not title:
        return None
    abstract, _ = split_abstract(r.get("ABSTRACT"), min_words=1)      # grant abstracts are short but real
    start = year(r.get("DTY_START"), r.get("START_START"))
    sub = year(r.get("DTY_SUB"), r.get("SUB_START"))
    return keep({
        "kind": "grant", "dm_record_ids": [clean(r.get("@id"))],
        "title": title, "type": clean(r.get("TYPE")), "status": clean(r.get("STATUS")),
        "sponsor": clean(r.get("SPONORG")), "sponsor_type": clean(r.get("AWARDORG")),
        "extramural": clean(r.get("EXTRAMURAL")) == "Yes",
        "other_agencies": clean(r.get("OTHER_AGENCIES")), "amount": money(r.get("AMOUNT")),
        "start_year": start, "end_year": year(r.get("DTY_END"), r.get("END_START")), "submitted_year": sub,
        "year": start or sub,                                         # uniform filter field
        "abstract": abstract,
        "contributors": contributors(r.get("CONGRANT_INVEST"), roster, owner, role_key="ROLE"),
        "public": clean(r.get("INCLUDE_PROFILE")) == "true",
    })

def parse_presentation(r, roster, owner):
    title = clean(r.get("TITLE"))
    if not title:
        return None
    abstract, note = split_abstract(r.get("ABSTRACT"))
    return keep({
        "kind": "presentation", "dm_record_ids": [clean(r.get("@id"))],
        "title": title.strip("“”\"' "), "type": clean(r.get("PRESENTATION_TYPE")),
        "event": clean(r.get("NAME")), "organization": clean(r.get("ORG")),
        "location": clean(r.get("LOCATION")), "scope": clean(r.get("SCOPE")),
        "refereed": clean(r.get("REFEREED")) == "true", "invited": clean(r.get("INVACC")),
        "academic": clean(r.get("ACADEMIC")), "in_proceedings": clean(r.get("PUBPROCEED")),
        "year": year(r.get("DTY_DATE"), r.get("DATE_START")),
        "abstract": abstract, "note": note,
        "contributors": contributors(r.get("PRESENT_AUTH"), roster, owner, role_key="ROLE"),
        "public": clean(r.get("INCLUDE_PROFILE")) == "true",
    })

SECTIONS = {"publication": ("INTELLCONT", parse_publication),
            "grant":       ("CONGRANT",   parse_grant),
            "presentation":("PRESENT",    parse_presentation)}
