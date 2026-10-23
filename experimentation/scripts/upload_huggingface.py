#!/usr/bin/env python
"""Upload experiment artifacts to the HF dataset repo (Manaswi2006/RL_project) under core_extensions_2026/.

Token: read from $HF_TOKEN, or from --env-file (key HUGGINGFACE_TOKEN).  The token is never printed or written anywhere.
Heavy folders (runs/) are uploaded with upload_large_folder (resumable); small folders with upload_folder.
"""
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO, PREFIX = "Manaswi2006/RL_project", "core_extensions_2026"


def token(env_file):
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for l in open(env_file):
        l = l.strip()
        if l.startswith("HUGGINGFACE_TOKEN="):
            return l.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("no token")


ap = argparse.ArgumentParser()
ap.add_argument("--env-file", default="/home/colligo/DatasetDiscovery/.env")
ap.add_argument("--what", nargs="+", default=["small", "runs"])
args = ap.parse_args()
api = HfApi(token=token(args.env_file))
SMALL = ["reports", "results", "figures", "configs", "logs", "scripts", "src", "tests"] + sorted(p.name for p in ROOT.glob("jobs_*.txt")) + ["README.md", "environment.yml", "requirements.lock", "run_all.sh", "reproduce.sh", "patches", "manifests"]
IGN = ["*/__pycache__/*", "*.pyc"]
if "small" in args.what:
    for item in SMALL:
        p = ROOT / item
        if p.is_dir():
            api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(p), path_in_repo=f"{PREFIX}/{item}", ignore_patterns=IGN,
                              commit_message=f"core_extensions_2026: {item}")
        elif p.is_file():
            api.upload_file(repo_id=REPO, repo_type="dataset", path_or_fileobj=str(p), path_in_repo=f"{PREFIX}/{item}",
                            commit_message=f"core_extensions_2026: {item}")
        print("uploaded", item, flush=True)
    # mandatory top-level report copies
    for f in ["FINAL_RESEARCH_REPORT.md", "FINAL_RESEARCH_REPORT.pdf", "MASTER_RESULTS.csv", "RESEARCH_DECISION.json", "REPRODUCIBILITY.md",
              "TRANSFER_AND_CLEANUP_MANIFEST.md", "TRANSFER_AND_CLEANUP_MANIFEST.json"]:
        p = ROOT / "reports" / f
        if p.exists():
            api.upload_file(repo_id=REPO, repo_type="dataset", path_or_fileobj=str(p), path_in_repo=f"{PREFIX}/{f}", commit_message=f"core_extensions_2026: {f}")
            print("uploaded top-level", f, flush=True)
if "runs" in args.what:
    for sm in sorted((ROOT / "runs").glob("*/*")):
        if not sm.is_dir():
            continue
        rel = sm.relative_to(ROOT)
        for attempt in range(3):
            try:
                api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(sm), path_in_repo=f"{PREFIX}/{rel}",
                                  ignore_patterns=IGN + ["*/vlm_label_set/*", "vlm_label_set/*"], commit_message=f"core_extensions_2026: {rel}")
                print("uploaded", rel, flush=True); break
            except Exception as e:
                print("retry", rel, attempt, str(e)[:120], flush=True)
