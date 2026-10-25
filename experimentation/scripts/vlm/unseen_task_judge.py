#!/usr/bin/env python
"""Frozen outcome judge (prompt selected on Button Press / Hammer dev episodes) applied UNCHANGED to tasks it was never tuned on.
Uses all bank episodes of the task (no tuning happens here).  usage: unseen_task_judge.py dial peg"""
import json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/vlm"))
from src.vlm.vlm_client import VLM, judge_outcome
from analyze_label_quality import metrics
vlm = VLM(workers=32)
res = {}
for task in sys.argv[1:]:
    bank = pickle.load(open(ROOT / f"results/vlm/bank_{task}.pkl", "rb"))
    out = vlm.pmap(lambda e: judge_outcome(vlm, e["frames16"], task, 3), bank)
    V = np.array([o[0] for o in out]); y = np.array([e["success"] for e in bank]); q = V.mean(1)
    E = (V != y[:, None]).astype(float)
    pw = [np.corrcoef(E[:, a], E[:, b])[0, 1] for a in range(3) for b in range(a + 1, 3) if E[:, a].std() > 0 and E[:, b].std() > 0]
    r = dict(n=len(y), n_success=int(y.sum()), prevalence=float(y.mean()), majority=metrics((q > 0.5).astype(float), y), soft_vote_fraction=metrics(q, y),
             per_paraphrase=[metrics(V[:, k].astype(float), y) for k in range(3)],
             reliability=[dict(q=float(v), n=int((q == v).sum()), frac_success=float(y[q == v].mean())) for v in sorted(set(q))],
             error_corr=float(np.nanmean(pw)) if pw else None, sample_texts=[o[1][0] for o in out[:3]])
    res[task] = r; m = r["majority"]
    print(task, "n", len(y), "succ", int(y.sum()), "| majority acc", round(m["acc"], 3), "bal", round(m["balanced_acc"], 3), "prec", round(m["precision"], 3), "rec", round(m["recall"], 3), "fpr", round(m["fpr"], 3), "fnr", round(m["fnr"], 3), "| brier", round(r["soft_vote_fraction"]["brier"], 3), "ece", round(r["soft_vote_fraction"]["ece"], 3), "| err corr", r["error_corr"])
json.dump(res, open(ROOT / "results/vlm/unseen_task_judge.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
