#!/usr/bin/env python
"""Fetch from the HF dataset what the verifier screen needs: episode banks and the saved false-positive exploit frames of the earlier VLM study."""
import fnmatch, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
tok = [l.split("=", 1)[1].strip().strip("\"'") for l in open("/home/colligo/DatasetDiscovery/.env") if l.startswith("HUGGINGFACE_TOKEN=")][0]
api = HfApi(token=tok); P = "vlm_outcome_2026"
want = []
for e in api.list_repo_tree("Manaswi2006/RL_project", repo_type="dataset", path_in_repo=P, recursive=True):
    p = e.path
    if hasattr(e, "size") and (fnmatch.fnmatch(p, f"{P}/results/vlm/bank_*.pkl") or "/runs/stage8/fp_diag/" in p and "/fp_examples/" in p and p.endswith((".jpg", ".json"))
                                or p.endswith("results/vlm/hardening_test.json") or p.endswith("results/vlm/unseen_task_judge.json") or p.endswith("results/vlm/outcome_judge_test.json")):
        want.append(p)
print(len(want), "files")
for p in want:
    q = hf_hub_download("Manaswi2006/RL_project", p, repo_type="dataset", token=tok, local_dir="tmp/hf_screen")
    dst = p[len(P) + 1:]; os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(q, dst)
print("done")
