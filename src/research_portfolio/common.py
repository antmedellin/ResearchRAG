"""Shared helpers: paths, config, text cleaning, HTTP, atomic JSON IO."""
import hashlib, html, json, os, re, time, unicodedata, urllib.error, urllib.request
from pathlib import Path
import yaml

REPO_ROOT = Path(os.getenv("RAG_REPO_ROOT", "/app")).resolve()
DATA_ROOT = Path(os.getenv("RP_DATA_ROOT", REPO_ROOT / "data" / "departments")).resolve()
REGISTRY  = Path(os.getenv("RP_REGISTRY", REPO_ROOT / "data" / "registry")).resolve()
USER_AGENT = "Mozilla/5.0 (UTRGV-ResearchRAG)"
SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?$")
_TAG = re.compile(r"<[^>]+>")

def _get(d, dotted):
    for part in dotted.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d

def load_config(path, require=()):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = ("institution", "department_name", "department_slug", "sources.faculty.type", "sources.faculty.api_url")
    missing = [k for k in (*base, *require) if not _get(cfg, k)]
    if missing:
        raise ValueError(f"{path}: missing config fields: {', '.join(missing)}")
    if not SLUG_RE.match(cfg["department_slug"]):
        raise ValueError("department_slug must be lowercase letters, digits, '-' or '_'")
    if cfg["sources"]["faculty"]["type"] != "digital_measures":
        raise ValueError("sources.faculty.type must be 'digital_measures'")
    return cfg

def department_dir(cfg):
    return DATA_ROOT / cfg["department_slug"]

def as_list(x):
    """DM JSON is XML-derived: a single child arrives as an object, not a list."""
    return [] if x is None else x if isinstance(x, list) else [x]

def clean(v):
    """Display text: unescape entities, strip HTML tags, normalize unicode, collapse whitespace."""
    if v is None or isinstance(v, (dict, list)):
        return None
    s = unicodedata.normalize("NFKC", html.unescape(str(v)))
    s = " ".join(_TAG.sub(" ", s).split())
    return s or None

def norm_key(v):
    """Comparison key only - never stored as display text."""
    return " ".join(re.sub(r"[\W_]+", " ", (clean(v) or "").casefold()).split())

def sha(text, n=None):
    h = hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()
    return h[:n] if n else h

def keep(d):
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}

def decode_dm(raw_bytes):
    data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    return json.loads(data) if isinstance(data, str) else data      # DM double-encodes

def fetch_json(url, timeout=60, retries=3, backoff=2.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            return raw, decode_dm(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)

def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run the previous stage first")
    return json.loads(path.read_text(encoding="utf-8"))

def write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def dm_records(data):
    return [r for r in as_list((data or {}).get("Data", {}).get("Record")) if isinstance(r, dict)]

def single_record(data):
    recs = dm_records(data)
    if len(recs) != 1:
        raise ValueError(f"expected 1 Data.Record, got {len(recs)}")
    return recs[0]

def index_values(rec, key):
    return [clean(e.get("@text")) for e in as_list(rec.get("dmd:IndexEntry"))
            if isinstance(e, dict) and e.get("@indexKey") == key and clean(e.get("@text"))]

# ---- Neo4j property encoding: primitives / primitive lists stored as-is, nested -> <key>_json ----
def to_props(d):
    out = {}
    for k, v in d.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, dict) or (isinstance(v, list) and any(isinstance(i, (dict, list)) for i in v)):
            out[f"{k}_json"] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            out[k] = v
    return out

def from_props(d):
    out = {}
    for k, v in d.items():
        if k.endswith("_json") and isinstance(v, str):
            out[k[:-5]] = json.loads(v)
        else:
            out[k] = v
    return out
