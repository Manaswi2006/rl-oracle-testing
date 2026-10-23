#!/usr/bin/env python
"""A2: synthetic outcome-label corruption on FIXED trajectories (stage-0 CoRe runs).

For each task, episodes from all stage-0 `core` seeds are pooled, split 60/40 (train/test, by episode).  For each
corruption scenario x label-handling method a visitation discriminator is trained on the *train* episodes with the
corrupted labels, then evaluated against the TRUE simulator outcome on held-out episodes:
  ep_auc     AUC of the mean discriminator logit of an episode for predicting its true success (test eps)
  step_sp    Spearman corr. between per-step reward lam*clip(logit) and the environment dense reward (test steps)
  label_acc  accuracy of the training labels (hard version) w.r.t. truth
Methods: oracle | hard (single corrupted hard label) | majority (k=7 votes) | soft_raw | conf_filter | soft_calibrated
Label-information assumptions are written to the output (calibration set size for soft_calibrated, etc.).
"""
import glob, json, sys, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.psvm.psvm_ext import VisitationDiscriminator
from src.psvm.label_models import VoteModel
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

TASKS = {"button_press": "button", "hammer": "hammer", "peg_insert": "peg"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_task(task_short, group="stage0", method="core"):
    eps = []
    for f in sorted(glob.glob(str(ROOT / f"runs/{group}/{method}/{task_short}_s*/trajectories.npz"))):
        d = np.load(f)
        ep, t, succ = d["ep"], d["t"], d["succ"]
        obs, act, r = d["obs"].astype(np.float32), d["act"].astype(np.float32), d["env_r"]
        for e in np.unique(ep):
            m = ep == e
            if m.sum() < 400:       # drop the truncated last episode
                continue
            s = succ[m]
            first = int(np.argmax(s)) if s.any() else 10 ** 6
            eps.append(dict(obs=obs[m], act=act[m], r=r[m], y=float(s.any()), first=first, near=float(r[m].max())))
    return eps


def scenarios(prev_target=None):
    return {
        "clean": dict(),
        "iid_flip10": dict(flip_fp=0.10, flip_fn=0.10),
        "iid_flip30": dict(flip_fp=0.30, flip_fn=0.30),
        "asym_FP20": dict(flip_fp=0.20, flip_fn=0.0),
        "asym_FN20": dict(flip_fp=0.0, flip_fn=0.20),
        "correlated_sys20": dict(sys_err=0.20),
        "correlated_sys40": dict(sys_err=0.40),
        "instance_dep": dict(instance=True),
    }


def make_labels(method, eps, sc, seed):
    rng = np.random.RandomState(seed)
    single_vote = method == "hard" and (sc.get("sys_err", 0) > 0 or sc.get("instance"))
    p = dict(mode={"oracle": "oracle_hard", "hard": "majority" if single_vote else "hard_flip", "majority": "majority", "soft_raw": "soft_raw",
                   "conf_filter": "conf_filter", "soft_calibrated": "soft_calibrated"}[method],
             k_votes=1 if single_vote else 7, tpr=0.9, tnr=0.9, flip_fp=sc.get("flip_fp", 0.0), flip_fn=sc.get("flip_fn", 0.0),
             sys_err=sc.get("sys_err", 0.0), sys_conf=0.9, conf_thresh=0.8, calib_n=30, vlm_seed=seed)
    if "flip_fp" in sc or "flip_fn" in sc:      # iid/asymmetric scenarios: vote accuracy mirrors the flip rates
        p["tpr"], p["tnr"] = 1 - sc.get("flip_fn", 0.0), 1 - sc.get("flip_fp", 0.0)
    if method == "oracle" or sc == {}:
        p["tpr"] = p["tnr"] = 1.0 if sc == {} else p["tpr"]
    vm = VoteModel(p, seed)
    q = []
    nearmax = max(e["near"] for e in eps)
    for i, e in enumerate(eps):
        if sc.get("instance"):      # failures that nearly succeed (high max env reward) are mislabeled as success more often
            nearness = e["near"] / nearmax
            pf = 0.45 * nearness ** 2 if e["y"] == 0 else 0.15 * (1 - nearness)
            if method == "oracle":
                q.append(e["y"])
            else:
                vm.tpr, vm.tnr = 1 - pf, 1 - pf
                q.append(vm(i, e["y"]))
        else:
            q.append(vm(i, e["y"]))
    return np.array(q, np.float32), vm


def train_eval(train, test, q, seed, K=1500):
    obs = np.concatenate([e["obs"] for e in train]); act = np.concatenate([e["act"] for e in train])
    qs = np.concatenate([np.full(len(e["obs"]), qi) for e, qi in zip(train, q)])
    mask = np.concatenate([(np.arange(len(e["obs"])) <= e["first"]) if e["y"] > 0 else np.ones(len(e["obs"]), bool) for e in train])
    mask &= ~np.isnan(qs)
    qs = np.nan_to_num(qs)
    x = torch.from_numpy(np.concatenate([obs, act], 1)).to(dev)
    D = VisitationDiscriminator(x.shape[1], dev, seed=seed)
    D.set_norm(x[torch.from_numpy(mask).to(dev)])
    res = D.fit(x, torch.from_numpy(qs).to(dev), torch.from_numpy(mask.astype(np.float32)).to(dev), steps=K, batch=256)
    if res is None:
        return dict(ep_auc=np.nan, step_sp=np.nan)
    with torch.no_grad():
        lg = [D.logit(torch.from_numpy(np.concatenate([e["obs"], e["act"]], 1)).to(dev)).clamp(-20, 20).cpu().numpy() for e in test]
    y = np.array([e["y"] for e in test])
    ep_score = np.array([l.mean() for l in lg])
    auc = roc_auc_score(y, ep_score) if 0 < y.sum() < len(y) else np.nan
    sp = spearmanr(np.concatenate(lg), np.concatenate([e["r"] for e in test]))[0]
    return dict(ep_auc=float(auc), step_sp=float(sp))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--scenarios", nargs="*"); ap.add_argument("--methods", nargs="*"); ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    out = {}
    for task, short in TASKS.items():
        eps = load_task(short, method=("env_dense" if short == "peg" else "core"))   # CoRe peg runs never succeeded in 200k steps
        if len(eps) < 40:
            print("not enough episodes for", task, len(eps)); continue
        ys = np.array([e["y"] for e in eps])
        print(task, "episodes", len(eps), "prevalence", ys.mean())
        res = []
        for seed in range(3):
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(eps)); ntr = int(0.6 * len(eps))
            train, test = [eps[i] for i in idx[:ntr]], [eps[i] for i in idx[ntr:]]
            for prev in (None, 0.05, 0.5):      # success prevalence of the TRAIN set (subsample)
                tr = train
                if prev is not None:
                    pos = [e for e in train if e["y"] > 0]; neg = [e for e in train if e["y"] == 0]
                    if len(pos) < 3 or len(neg) < 3: continue
                    npos = int(min(len(pos), prev / (1 - prev) * len(neg))) if prev < 0.5 else min(len(pos), len(neg))
                    nneg = len(neg) if prev < 0.5 else npos
                    tr = [pos[i] for i in rng.permutation(len(pos))[:max(npos, 2)]] + [neg[i] for i in rng.permutation(len(neg))[:nneg]]
                for scn, sc in scenarios().items():
                    if args.scenarios and scn not in args.scenarios:
                        continue
                    for method in ("oracle", "hard", "majority", "soft_raw", "conf_filter", "soft_calibrated"):
                        if args.methods and method not in args.methods:
                            continue
                        if method == "oracle" and scn != "clean":
                            continue
                        q, vm = make_labels(method, tr, sc, seed)
                        lab_acc = float(np.mean(((q > 0.5) == (np.array([e["y"] for e in tr]) > 0))[~np.isnan(q)]))
                        r = train_eval(tr, test, q, seed)
                        res.append(dict(task=task, seed=seed, prevalence=prev if prev else "natural", scenario=scn, method=method,
                                        train_pos_frac=float(np.mean([e["y"] for e in tr])), label_acc=lab_acc,
                                        n_oracle_calibration_labels=getattr(vm, "n_oracle_calibration_labels", 0),
                                        frac_discarded=float(np.isnan(q).mean()), **r))
                print(task, "seed", seed, "prev", prev, "done", len(res), flush=True)
        out[task] = res
    out_dir = ROOT / "results" / "psvm"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / f"A2_label_corruption{args.suffix}.json", "w"), indent=1)
    import csv
    rows = [r for v in out.values() for r in v]
    if rows:
        with open(out_dir / f"A2_label_corruption{args.suffix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
