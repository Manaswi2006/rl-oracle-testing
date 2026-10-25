#!/usr/bin/env python
"""One row per run of stages 6-8 (VLM outcome study): results/vlm/MASTER_RESULTS_VLM.csv"""
import csv, glob, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
commit = os.environ.get("GIT_COMMIT", "uncommitted")
rows = []
for f in sorted(glob.glob(str(ROOT / "runs/stage[678]/*/*/result.json"))):
    r = json.load(open(f)); d = Path(f).parent; es = r.get("ext_stats") or {}
    p = es.get("params", {}) if isinstance(es, dict) else {}
    rows.append(dict(stage=d.parent.parent.name, arm=d.parent.name, task=r["env"].replace("metaworld_", ""), seed=r["seed"], training_steps=r["steps"], eval_episodes=50,
                     success_rate=r["final_success_rate"], reward_mean=r["final_true_return"], gpu_hours=round(r["gpu_hours"], 3), vlm_queries=es.get("n_vlm_queries", 0),
                     verdict_source=p.get("source", ""), verdict_agg=p.get("agg", ""), rnd_beta=p.get("rnd_beta", 0), reward_alpha_rrm_weight=0.0 if "noRRM" in d.parent.name or "rnd" in d.parent.name else 0.5,
                     label_acc=es.get("label_acc", ""), fp=es.get("fp", ""), fn=es.get("fn", ""), n_train_successes=r.get("n_success_train", ""), first_success_step=r.get("first_success_step", ""),
                     oracle_information_used=r.get("oracle_information_used", ""), config_path=str(d.relative_to(ROOT) / "cfg.yaml"), checkpoint_path=str(d.relative_to(ROOT) / "actor.pt"), git_commit=commit, status="completed"))
with open(ROOT / "results/vlm/MASTER_RESULTS_VLM.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "runs")
