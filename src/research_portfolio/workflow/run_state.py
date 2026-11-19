"""Durable ingestion state. A department is active only after validation passes."""
import json
from datetime import datetime, timezone
from pathlib import Path
from research_portfolio.common import department_dir, write_atomic

STATES = ("prepared", "graph_written", "vectors_written", "validated", "failed")

def now(): return datetime.now(timezone.utc).isoformat()

def path_for(cfg, run_id): return department_dir(cfg) / "runs" / f"run_{run_id}.json"

def begin(cfg, run_id, policy):
    rec = {"run_id": run_id, "department_slug": cfg["department_slug"], "status": "prepared",
           "started_at": now(), "policy": policy, "events": []}
    save(cfg, rec); return rec

def transition(cfg, rec, status, **details):
    if status not in STATES: raise ValueError(status)
    rec["status"] = status
    rec["events"].append({"status": status, "at": now(), **details})
    if status in ("validated", "failed"): rec["finished_at"] = now()
    save(cfg, rec)

def save(cfg, rec): write_atomic(path_for(cfg, rec["run_id"]), rec)
