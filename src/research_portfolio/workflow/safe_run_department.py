#!/usr/bin/env python3
"""Stateful wrapper around run_department.py. Marks active only after cross-store validation."""
import argparse, subprocess, sys, traceback
from datetime import datetime, timezone
from research_portfolio.common import load_config
from research_portfolio.workflow.run_state import begin, transition

POLICY={"version":"1.0","auto_fuzzy_threshold":96,"clef_same_threshold":0.85,
        "decision_model_env":"DECISION_MODEL","department_ownership":"graph-derived"}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--apply",action="store_true")
    ap.add_argument("--skip-discover",action="store_true"); ap.add_argument("--skip-fetch",action="store_true")
    a=ap.parse_args(); cfg=load_config(a.config); rid=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec=begin(cfg,rid,POLICY)
    cmd=[sys.executable,"/app/src/research_portfolio/run_department.py","--config",a.config]
    if a.apply: cmd.append("--apply")
    if a.skip_discover: cmd.append("--skip-discover")
    if a.skip_fetch: cmd.append("--skip-fetch")
    try:
        subprocess.run(cmd,check=True)
        if not a.apply: return 0
        transition(cfg,rec,"vectors_written")
        subprocess.run([sys.executable,"-m","research_portfolio.workflow.validate_department","--config",a.config],check=True)
        transition(cfg,rec,"validated")
        print(f"Department {cfg['department_slug']} validated and active (run {rid})")
        return 0
    except Exception as exc:
        transition(cfg,rec,"failed",error=str(exc),traceback=traceback.format_exc())
        print(f"Run {rid} FAILED: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
