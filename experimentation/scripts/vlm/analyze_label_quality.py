#!/usr/bin/env python
"""Label-quality metrics of the VLM against simulator ground truth (episode outcome + window preferences)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for i in range(bins):
        m = b == i
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def metrics(p_hat, y, thr=0.5):
    pred = (p_hat >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(acc=float((pred == y).mean()), balanced_acc=float((rec + spec) / 2), precision=float(tp / max(tp + fp, 1)), recall=float(rec), specificity=float(spec),
                fpr=float(fp / max(fp + tn, 1)), fnr=float(fn / max(fn + tp, 1)), auc=auc(p_hat, y), brier=float(((p_hat - y) ** 2).mean()), ece=ece(p_hat, y), n=int(len(y)))


def main(tag=""):
    res = {}
    for task in ("button", "hammer"):
        d = json.load(open(ROOT / f"results/vlm/label_quality_{task}{tag}.json"))
        y = np.array([t["success"] for t in d["truth"]])
        n = len(y)
        O = {(o["ep"], o["prompt"], o["n_frames"]): o for o in d["outcome"]}
        P = np.array([[O[(i, k, 16)]["p_yes"] for k in range(5)] for i in range(n)])
        V = np.array([[O[(i, k, 16)]["vote"] for k in range(5)] for i in range(n)])
        r = dict(n_episodes=n, prevalence=float(y.mean()))
        r["per_prompt_hard_vote"] = [metrics(V[:, k].astype(float), y) for k in range(5)]
        r["per_prompt_token_prob"] = [metrics(P[:, k], y) for k in range(5)]
        q_vote = V.mean(1)                                   # fraction of paraphrases voting yes
        r["majority_vote"] = metrics((q_vote > 0.5).astype(float), y)
        r["vote_fraction_soft"] = metrics(q_vote, y)
        r["mean_token_prob_soft"] = metrics(P.mean(1), y)
        r["frames8_prompt0"] = metrics(np.array([O[(i, 0, 8)]["vote"] for i in range(n)], float), y)
        # reliability of the vote fraction
        r["reliability_vote_fraction"] = [dict(q=float(v), n=int((q_vote == v).sum()), frac_success=float(y[q_vote == v].mean())) for v in sorted(set(q_vote)) if (q_vote == v).sum() > 0]
        # error correlation across paraphrases: P(all 5 wrong | at least one wrong) and pairwise phi of error indicators
        E = (V != y[:, None]).astype(float)
        any_wrong = E.sum(1) > 0
        r["errors"] = dict(frac_episodes_with_any_error=float(any_wrong.mean()), frac_all5_wrong=float((E.sum(1) == 5).mean()),
                           frac_all5_wrong_given_any=float((E.sum(1) == 5).sum() / max(any_wrong.sum(), 1)),
                           mean_pairwise_error_corr=float(np.nanmean([np.corrcoef(E[:, a], E[:, b])[0, 1] for a in range(5) for b in range(a + 1, 5)])),
                           expected_if_independent_all5_wrong=float(np.prod(E.mean(0))))
        # error vs behaviour type
        tags = np.array([t["tag"].split("/")[0] if "/" in t["tag"] else t["tag"] for t in d["truth"]])
        r["by_policy_family"] = {t: dict(n=int((tags == t).sum()), success_rate=float(y[tags == t].mean()), majority_acc=float(((q_vote[tags == t] > 0.5) == y[tags == t]).mean())) for t in sorted(set(tags))}
        # preferences
        pr = d["prefs"]
        ra = np.array([p["rew_a"] for p in pr]); rb = np.array([p["rew_b"] for p in pr])
        truth = (ra > rb).astype(int)                         # A better by environment reward (the scripted teacher of the earlier study)
        pa1 = np.array([p["p_a_first"] for p in pr]); pa2 = 1 - np.array([p["p_a_second"] for p in pr])   # P(A better) from both orders
        rel = np.abs(ra - rb) / (np.maximum(np.abs(ra), np.abs(rb)) + 1e-6)
        pref = {}
        for name, pa in (("order1", pa1), ("order2_swapped", pa2), ("avg_two_orders", (pa1 + pa2) / 2)):
            pref[name] = dict(acc=float(((pa > 0.5) == truth).mean()), auc=auc(pa, truth))
        consistent = ((pa1 > 0.5) == (pa2 > 0.5))
        pref["order_consistent_frac"] = float(consistent.mean())
        pref["acc_when_consistent"] = float(((pa1[consistent] > 0.5) == truth[consistent]).mean())
        pref["position_bias_P(A)_order1"] = float((pa1 > 0.5).mean())
        for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1.01)):
            m = (rel >= lo) & (rel < hi)
            pref[f"acc_rel_reward_diff_{lo}-{hi}"] = dict(n=int(m.sum()), acc=float(((((pa1 + pa2) / 2)[m] > 0.5) == truth[m]).mean()) if m.any() else None)
        sa = np.array([p["succ_a"] for p in pr]); sb = np.array([p["succ_b"] for p in pr]); m = sa != sb
        pref["pairs_where_exactly_one_window_contains_success"] = dict(n=int(m.sum()), acc=float(((((pa1 + pa2) / 2)[m] > 0.5) == (sa[m] > sb[m])).mean()) if m.any() else None)
        r["preference"] = pref
        res[task] = r
    json.dump(res, open(ROOT / f"results/vlm/label_quality_summary{tag}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    res = main(sys.argv[1] if len(sys.argv) > 1 else "")
    for task, r in res.items():
        print("=====", task, "n", r["n_episodes"], "prevalence", round(r["prevalence"], 3))
        for k in ("majority_vote", "vote_fraction_soft", "mean_token_prob_soft", "frames8_prompt0"):
            m = r[k]; print(f"{k:22s} acc {m['acc']:.3f} bal {m['balanced_acc']:.3f} prec {m['precision']:.3f} rec {m['recall']:.3f} spec {m['specificity']:.3f} auc {m['auc']:.3f} brier {m['brier']:.3f} ece {m['ece']:.3f}")
        print("per-prompt hard vote acc/bal:", [(round(m['acc'], 2), round(m['balanced_acc'], 2)) for m in r["per_prompt_hard_vote"]])
        print("errors:", {k: round(v, 3) for k, v in r["errors"].items()})
        print("reliability (vote fraction -> true success freq):", [(x['q'], x['n'], round(x['frac_success'], 2)) for x in r["reliability_vote_fraction"]])
        print("pref:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["preference"].items() if not isinstance(v, dict)}, {k: v for k, v in r["preference"].items() if isinstance(v, dict)})
