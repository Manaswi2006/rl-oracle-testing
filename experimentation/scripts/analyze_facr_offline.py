#!/usr/bin/env python
"""FACR offline diagnostics B1 (oracle success-value), B2 (counterfactual validity), B4 (model-bias ablations).

Inputs : runs/stage0/<core|rrm_only|env_dense>/<task>_s<seed>/trajectories.npz  (training data, oracle success flags)
         results/facr/branching_<task>.npz  (simulator-branching ground truth; scripts/facr_branching_data.py)
Outputs: results/facr/B1_value.csv, B2_counterfactual.csv, B4_model_bias.csv  (+ json)
"""
import glob, json, sys, csv
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import rrm_scripted  # noqa (path setup)
from src.facr.facr_ext import FACRExt
import importlib

TASKS = {"button": "metaworld_button-press-topdown-v2", "hammer": "metaworld_hammer-v2", "peg": "metaworld_peg-insert-side-v2"}
DM = {"button": "core", "hammer": "core", "peg": "env_dense"}   # data/policy source per task (CoRe peg policies never succeeded within 200k steps)
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results" / "facr"; OUT.mkdir(parents=True, exist_ok=True)


def load_eps(group, method, task, seeds=None):
    eps = []
    for f in sorted(glob.glob(str(ROOT / f"runs/{group}/{method}/{task}_s*/trajectories.npz"))):
        sd = int(f.split("_s")[-1].split("/")[0])
        if seeds is not None and sd not in seeds:
            continue
        d = np.load(f)
        for e in np.unique(d["ep"]):
            m = np.flatnonzero(d["ep"] == e)
            if len(m) < 400:
                continue
            eps.append(dict(obs=d["obs"][m].astype(np.float32), act=d["act"][m].astype(np.float32), succ=d["succ"][m], seed=sd))
    return eps


def feed(ext, eps, frac=1.0, rng=None, pos_eps=None):
    """pos_eps: keep at most this many successful episodes (limited-positives experiment)."""
    if frac < 1.0:
        keep = rng.permutation(len(eps))[: max(int(frac * len(eps)), 2)]
        eps = [eps[i] for i in sorted(keep)]
    if pos_eps is not None:
        pos = [e for e in eps if e["succ"].any()]
        neg = [e for e in eps if not e["succ"].any()]
        eps = neg + pos[:pos_eps]
    step = 0
    for k, e in enumerate(eps):
        L = len(e["obs"])
        for t in range(L - 1):
            ext.on_step(k, t, e["obs"][t], e["act"][t], e["obs"][t + 1], 0.0, float(e["succ"][t]), step)
            step += 1
        ext.on_episode_end(k, float(e["succ"].any()), step)
    return ext


def make_ext(variant="facr", N=0, K=2000, n_dyn=5, H=50, seed=0):
    cfg = OmegaConf.create(dict(num_train_steps=int(N) + 10, facr=dict(variant=variant, K=K, update_every=1, n_dyn=n_dyn, H=H, batch=512)))
    return FACRExt(cfg, 39, 4, dev, seed)


def train(eps, variant="facr", frac=1.0, pos_eps=None, seed=0, K=2000, n_dyn=5, H=50):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    N = sum(len(e["obs"]) for e in eps)
    ext = make_ext(variant, N, K, n_dyn, H, seed)
    feed(ext, eps, frac, rng, pos_eps)
    ok = ext.maybe_update(10 ** 7)
    return ext if ok else None


def future_labels(eps, h):
    X, Y = [], []
    for e in eps:
        s = e["succ"]
        idx = np.flatnonzero(s)
        L = len(s)
        nxt = np.full(L, np.inf)
        if len(idx):
            pos = np.searchsorted(idx, np.arange(L)); ok = pos < len(idx)
            nxt[ok] = idx[pos[ok]] - np.arange(L)[ok]
        X.append(e["obs"][:-1]); Y.append((nxt[:-1] <= h).astype(np.float32))
    return np.concatenate(X), np.concatenate(Y)


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


@torch.no_grad()
def predict_V(ext, X, h=50):
    out = []
    for i in range(0, len(X), 8192):
        out.append(ext._V(torch.from_numpy(X[i:i + 8192]).to(dev), h).cpu().numpy())
    return np.concatenate(out)


def frm_score(task, X, G):
    """released FRM reward (geometric proximity / progress) of observations X given per-row goal positions G (hidden in the obs)."""
    mod = importlib.import_module(f"{TASKS[task].replace('-', '_').lower()}_frm")
    zero = np.zeros(4, np.float32)
    return np.array([mod.reward_function(x, zero, G[i])[0] for i, x in enumerate(X)])


def metrics(p, y):
    if y.sum() == 0 or y.sum() == len(y):
        return dict(auc=np.nan, brier=float(brier_score_loss(y, p)), ece=ece(p, y), pos_rate=float(y.mean()))
    return dict(auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)), ece=ece(p, y), pos_rate=float(y.mean()))


def run_B1(task, rows):
    train_eps = load_eps("stage0", DM[task], task, seeds={0, 1})
    tests = {f"{DM[task]}_seed2 (in-dist)": load_eps("stage0", DM[task], task, seeds={2})}
    for other in ("core", "rrm_only", "env_dense"):
        if other != DM[task]:
            tests[f"{other} (distribution shift)"] = load_eps("stage0", other, task)
    if not train_eps:
        return
    npos = sum(e["succ"].any() for e in train_eps)
    for pe in (1, 2, 4, 8, None):
        ext = train(train_eps, "value", pos_eps=pe, seed=0)
        if ext is None:
            continue
        for name, te in tests.items():
            if not te: continue
            X, y = future_labels(te, 50)
            r = metrics(predict_V(ext, X), y)
            row = dict(task=task, n_train_pos_eps=pe if pe else f"all({npos})", test=name, predictor="V(s,50) learned", **r)
            rows.append(row)
    print(task, "B1 done", flush=True)


def corr_stats(pred, emp):
    """pred, emp: [S, A]; within-state rank correlation, centered advantage correlation, top-1 agreement."""
    rs, n_used = [], 0
    for p, e in zip(pred, emp):
        if e.std() > 0.05 and p.std() > 1e-9:
            rs.append(spearmanr(p, e)[0]); n_used += 1
    adv_p = (pred - pred.mean(1, keepdims=True)).reshape(-1); adv_e = (emp - emp.mean(1, keepdims=True)).reshape(-1)
    pr = pearsonr(adv_p, adv_e)[0] if adv_p.std() > 1e-9 and adv_e.std() > 1e-9 else np.nan
    top = [float(emp[i, p.argmax()] >= emp[i].max() - 0.05) for i, p in enumerate(pred) if emp[i].std() > 0.05]
    return dict(n_states_discriminable=n_used, within_state_spearman=float(np.nanmean(rs)) if rs else np.nan,
                advantage_pearson=float(pr), top1_ok=float(np.mean(top)) if top else np.nan)


@torch.no_grad()
def q_cf(ext, S, A, true_next=None, H=50, member=None, unc=False):
    """S [N,d], A [N,na,4] -> predicted Q_cf [N,na] (and V-disagreement)."""
    N, na = A.shape[:2]
    s = torch.from_numpy(S).to(dev); out = np.zeros((N, na)); un = np.zeros((N, na))
    z = (s - ext.mu) / ext.sd
    for j in range(na):
        if true_next is not None:
            v = ext._V(torch.from_numpy(true_next[:, j]).to(dev), H - 1).cpu().numpy(); out[:, j] = v
        else:
            inp = torch.cat([z, torch.from_numpy(A[:, j]).to(dev)], 1)
            dyn = [ext.dyn[member]] if member is not None else list(ext.dyn)
            vs = torch.stack([ext._V(s + m(inp) * ext.dsd + ext.dmu, H - 1) for m in dyn])
            out[:, j] = vs.mean(0).cpu().numpy(); un[:, j] = vs.std(0).cpu().numpy() if len(dyn) > 1 else 0
    return out, un


def run_B2_B4(task, rows2, rows4):
    fs = sorted(OUT.glob(f"branching_{task}*.npz"))
    if not fs:
        print("no branching data for", task); return
    ds = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([d[k] for d in ds]).astype(np.float32)
    S, A, T, TN, G = cat("state"), cat("actions"), cat("first_succ"), cat("true_next"), cat("goal")
    T = np.concatenate([d["first_succ"] for d in ds])
    train_eps = load_eps("stage0", DM[task], task)         # note: includes the seed whose policy generated the branching states
    emp = {H: (T <= H - 1).mean(2) for H in (10, 25, 50)}   # [S, A] empirical success-within-H from first action
    full = train(train_eps, "facr", seed=0)
    if full is None:
        print("no positives to train FACR for", task); return
    H = 50
    q, un = q_cf(full, S, A, H=H)
    rows2.append(dict(task=task, predictor="FACR Q_cf (learned dyn ensemble, V horizon 49)", **corr_stats(q, emp[H]), n_states=len(S)))
    qt, _ = q_cf(full, S, A, true_next=TN, H=H)
    rows2.append(dict(task=task, predictor="V(true simulator next state, 49) = success-value baseline w/ perfect dynamics", **corr_stats(qt, emp[H]), n_states=len(S)))
    fr = np.stack([frm_score(task, TN[:, j], G) for j in range(A.shape[1])], 1)
    rows2.append(dict(task=task, predictor="FRM reward of true next state (geometric proximity)", **corr_stats(fr, emp[H]), n_states=len(S)))
    rows2.append(dict(task=task, predictor="state-only V(s) (constant over actions; cannot rank actions)", **corr_stats(np.tile(predict_V(full, S, H)[:, None], (1, A.shape[1])), emp[H]), n_states=len(S)))
    # state-level: does value/proximity track the empirical success rate of the policy's own actions?
    p_pol = emp[H][:, :4].mean(1)
    rows2.append(dict(task=task, predictor="STATE-LEVEL spearman(V(s,50), emp p_succ of policy actions)", within_state_spearman=float(spearmanr(predict_V(full, S, H), p_pol)[0]), n_states=len(S)))
    rows2.append(dict(task=task, predictor="STATE-LEVEL spearman(FRM(s), emp p_succ of policy actions)", within_state_spearman=float(spearmanr(frm_score(task, S, G), p_pol)[0]), n_states=len(S)))
    # ---- B4 model-bias ablations
    base = dict(task=task)
    rows4.append(dict(base, ablation="learned ensemble (5), H=50, all data", **corr_stats(q, emp[50])))
    qs, _ = q_cf(full, S, A, member=0, H=50)
    rows4.append(dict(base, ablation="single dynamics member", **corr_stats(qs, emp[50])))
    rows4.append(dict(base, ablation="perfect (simulator) next state", **corr_stats(qt, emp[50])))
    keep = un[:, 0] <= np.median(un[:, 0])
    rows4.append(dict(base, ablation="uncertainty filter: low-disagreement half of states", **corr_stats(q[keep], emp[50][keep])))
    rows4.append(dict(base, ablation="uncertainty filter: high-disagreement half of states", **corr_stats(q[~keep], emp[50][~keep])))
    for Hh in (10, 25):
        qh, _ = q_cf(full, S, A, H=Hh)
        rows4.append(dict(base, ablation=f"planning horizon H={Hh} (vs empirical within-{Hh})", **corr_stats(qh, emp[Hh])))
    for fr_ in (0.1, 0.3):
        e2 = train(train_eps, "facr", frac=fr_, seed=1)
        if e2 is not None:
            q2, _ = q_cf(e2, S, A, H=50)
            rows4.append(dict(base, ablation=f"transition data fraction {fr_}", **corr_stats(q2, emp[50])))
    # exploitation check: is the model's predicted gain concentrated where the *true* next state is worse?
    pm, pt = q.max(1), qt.max(1)
    rows4.append(dict(base, ablation="model exploitation: mean[max_a Q_model - max_a V(true next)]", mean_gap=float((pm - pt).mean()),
                      frac_states_model_overestimates_by_gt_0p1=float(((pm - pt) > 0.1).mean())))
    print(task, "B2/B4 done", flush=True)


def write(rows, name):
    if not rows: return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("task", "predictor", "ablation", "test", "n_train_pos_eps"), k))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    tasks = sys.argv[1:] or list(TASKS)
    r1, r2, r4 = [], [], []
    for t in tasks:
        run_B1(t, r1); run_B2_B4(t, r2, r4)
    sfx = "_" + "_".join(tasks) if len(sys.argv) > 1 else ""
    write(r1, f"B1_value{sfx}.csv"); write(r2, f"B2_counterfactual{sfx}.csv"); write(r4, f"B4_model_bias{sfx}.csv")
    json.dump(dict(B1=r1, B2=r2, B4=r4), open(OUT / f"facr_offline{sfx}.json", "w"), indent=1, default=float)
