#!/usr/bin/env python
"""Held-out TEST evaluation of the frozen outcome judge (3 paraphrases, CoT) on the odd half of the permuted bank (never used for prompt selection)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/vlm"))
from src.vlm.vlm_client import VLM, judge_outcome
from analyze_label_quality import metrics, ece
vlm = VLM(workers=32)
res = {}
for task in ("button", "hammer"):
    bank = pickle.load(open(ROOT / f"results/vlm/bank_{task}.pkl", "rb"))
    rs = np.random.RandomState(0); perm = rs.permutation(len(bank)); test = [int(i) for i in perm[len(bank) // 2:]]
    out = vlm.pmap(lambda i: judge_outcome(vlm, bank[i]["frames16"], task, 3), test)
    V = np.array([o[0] for o in out]); y = np.array([bank[i]["success"] for i in test]); q = V.mean(1)
    E = (V != y[:, None]).astype(float)
    pw = [np.corrcoef(E[:, a], E[:, b])[0, 1] for a in range(3) for b in range(a + 1, 3)]
    r = dict(n=len(y), prevalence=float(y.mean()), majority=metrics((q > 0.5).astype(float), y), soft_vote_fraction=metrics(q, y),
             per_paraphrase=[metrics(V[:, k].astype(float), y) for k in range(3)],
             reliability=[dict(q=float(v), n=int((q == v).sum()), frac_success=float(y[q == v].mean())) for v in sorted(set(q))],
             errors=dict(any_error=float((E.sum(1) > 0).mean()), all3_wrong=float((E.sum(1) == 3).mean()), mean_pairwise_error_corr=float(np.nanmean(pw)), independent_all3_wrong=float(np.prod(E.mean(0)))))
    # balanced resampled metrics (equal positives / negatives)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); m = min(len(pos), len(neg)); idx = np.concatenate([pos[:m], neg[:m]])
    r["majority_balanced_subset"] = metrics((q[idx] > 0.5).astype(float), y[idx])
    res[task] = r
    mm = r["majority"]
    print(task, "n", len(y), "prev", round(r["prevalence"], 2), "| majority acc", round(mm["acc"], 3), "bal", round(mm["balanced_acc"], 3), "prec", round(mm["precision"], 3), "rec", round(mm["recall"], 3), "fpr", round(mm["fpr"], 3), "fnr", round(mm["fnr"], 3),
          "| soft brier", round(r["soft_vote_fraction"]["brier"], 3), "ece", round(r["soft_vote_fraction"]["ece"], 3), "| err corr", round(r["errors"]["mean_pairwise_error_corr"], 2), "all3wrong", round(r["errors"]["all3_wrong"], 3), "indep", round(r["errors"]["independent_all3_wrong"], 3))
json.dump(res, open(ROOT / "results/vlm/outcome_judge_test.json", "w"), indent=1)
