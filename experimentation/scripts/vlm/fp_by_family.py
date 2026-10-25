#!/usr/bin/env python
"""Where does the frozen judge raise false positives? Majority verdicts on all bank episodes of a task, grouped by policy family / noise setting."""
import json, pickle, sys, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm.vlm_client import VLM, judge_outcome
vlm = VLM(workers=32); res = {}
for task in sys.argv[1:]:
    bank = pickle.load(open(ROOT / f"results/vlm/bank_{task}.pkl", "rb"))
    out = vlm.pmap(lambda e: judge_outcome(vlm, e["frames16"], task, 3), bank)
    q = np.array([np.mean(o[0]) for o in out]); y = np.array([e["success"] for e in bank]); tags = [e["tag"] for e in bank]
    g = collections.defaultdict(list)
    for i, t in enumerate(tags):
        fam = "random" if t == "random" else t.split("|")[1] + "|" + t.split("|")[2]
        g[fam].append(i)
    res[task] = {}
    for fam, idx in sorted(g.items()):
        idx = np.array(idx); neg = idx[y[idx] == 0]; pos = idx[y[idx] == 1]
        res[task][fam] = dict(n=len(idx), n_fail=len(neg), fpr=float((q[neg] > 0.5).mean()) if len(neg) else None, n_succ=len(pos), fnr=float((q[pos] <= 0.5).mean()) if len(pos) else None)
        print(task, f"{fam:12s} n={len(idx):3d} failures={len(neg):3d} FPR={res[task][fam]['fpr'] if res[task][fam]['fpr'] is None else round(res[task][fam]['fpr'],2)} successes={len(pos):3d} FNR={res[task][fam]['fnr'] if res[task][fam]['fnr'] is None else round(res[task][fam]['fnr'],2)}")
json.dump(res, open(ROOT / "results/vlm/fp_by_family.json", "w"), indent=1)
