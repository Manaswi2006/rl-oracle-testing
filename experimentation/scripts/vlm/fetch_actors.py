#!/usr/bin/env python
"""Download final SAC actors of the earlier study from the HF dataset (core_extensions_2026/runs/...) for episode-bank generation."""
import os, sys
from huggingface_hub import hf_hub_download
tok = [l.split("=", 1)[1].strip().strip("\"'") for l in open("/home/colligo/DatasetDiscovery/.env") if l.startswith("HUGGINGFACE_TOKEN=")][0]
ARMS = [("stage0", "core"), ("stage0", "rrm_only"), ("stage0", "env_dense"), ("stage1", "psvm_oracle"), ("stage1", "facr_facr"), ("stage5", "outcome_sparse"), ("stage2", "rlf_uncertainty")]
n = 0
for task, seeds in [(t, range(6)) for t in (sys.argv[1:] or ["button", "hammer"])]:
    for stage, arm in ARMS:
        for s in seeds:
            try:
                p = hf_hub_download("Manaswi2006/RL_project", f"core_extensions_2026/runs/{stage}/{arm}/{task}_s{s}/actor.pt", repo_type="dataset", token=tok,
                                    local_dir="checkpoints/hf_actors")
                d = f"checkpoints/actors/{arm}"; os.makedirs(d, exist_ok=True); os.replace(p, f"{d}/{task}_s{s}.pt"); n += 1
            except Exception as e:
                pass
print("fetched", n)
