#!/usr/bin/env python
"""Aggregate runs/**/result.json + eval.csv into results/aggregate/*.

Outputs
  MASTER_RESULTS.csv     one row per (method, task, seed, eval checkpoint) incl. the final 50-episode evaluation
  summary_final.csv      per (stage, method, task): mean/std/CI over seeds of final success, AULC, steps-to-threshold, ...
  gap_closure.csv        G = (S_new - S_RRM)/(S_CoRe - S_RRM) per new method & task (observed baselines)
"""
import csv, glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aggregate"
OUT.mkdir(parents=True, exist_ok=True)
TASK = {"metaworld_button-press-topdown-v2": "button_press", "metaworld_hammer-v2": "hammer",
        "metaworld_peg-insert-side-v2": "peg_insert", "metaworld_dial-turn-v2": "dial_turn", "metaworld_reach-v2": "reach"}


def git_commit():
    import os
    if os.environ.get('GIT_COMMIT'): return os.environ['GIT_COMMIT']
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def load_runs():
    runs = []
    for res in sorted(glob.glob(str(ROOT / "runs" / "*" / "*" / "*" / "result.json"))):
        d = Path(res).parent
        stage, method = d.parent.parent.name, d.parent.name
        r = json.load(open(res))
        r.update(stage=stage, method_name=method, run_dir=str(d.relative_to(ROOT)), task=TASK.get(r["env"], r["env"]))
        ev = d / "eval.csv"
        r["evals"] = list(csv.DictReader(open(ev))) if ev.exists() else []
        runs.append(r)
    return runs


def oracle_desc(r):
    m = r["method"]
    if m == "env_dense": return "environment dense reward (privileged)"
    if m == "env_sparse": return "simulator success flag as reward (privileged)"
    base = "RRM preference teacher = noisy scripted teacher on env reward (privileged; replaces VLM)"
    if m == "core": return base + "; FRM = released LLM-written reward on state features"
    if m in ("rrm_only", "rrm_extra"): return base
    if m == "rlf": return base + "; verifier = simulator success flag (oracle)"
    return base + "; " + str(r.get("oracle_information_used", ""))


def main():
    runs = load_runs()
    commit = git_commit()
    rows = []
    for r in runs:
        common = dict(stage=r["stage"], method=r["method_name"], task=r["task"], seed=r["seed"], gpu_hours=round(r["gpu_hours"], 3),
                      vlm_queries=0, teacher_queries=r.get("teacher_queries", 0), oracle_information_used=oracle_desc(r),
                      config_path=f'{r["run_dir"]}/cfg.yaml', checkpoint_path=f'{r["run_dir"]}/actor.pt', git_commit=commit,
                      status="completed", env_steps_total=r["steps"] + (r.get("ext_stats") or {}).get("extra_env_steps", 0)
                      if isinstance(r.get("ext_stats"), dict) else r["steps"])
        for e in r["evals"]:
            rows.append(dict(common, training_steps=int(e["step"]), eval_episodes=int(e["n_eval"]), success_rate=float(e["success_rate"]),
                             reward_mean=float(e["true_return"]), reward_std="", eval_kind="periodic"))
        rows.append(dict(common, training_steps=r["steps"], eval_episodes=50, success_rate=r["final_success_rate"],
                         reward_mean=r["final_true_return"], reward_std="", eval_kind="final"))
    if rows:
        keys = list(rows[0].keys())
        with open(OUT / "MASTER_RESULTS.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # ---- per method/task summaries
    def stats(v):
        v = np.array(v, float)
        if len(v) == 0: return (np.nan,) * 3
        m, s = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        ci = 1.96 * s / np.sqrt(len(v)) if len(v) > 1 else np.nan
        return m, s, ci

    groups = {}
    for r in runs:
        groups.setdefault((r["stage"], r["method_name"], r["task"]), []).append(r)
    summ = []
    for (st, me, ta), rs in sorted(groups.items()):
        fin = [x["final_success_rate"] for x in rs]
        aulc, t50 = [], []
        for x in rs:
            if x["evals"]:
                steps = np.array([int(e["step"]) for e in x["evals"]]); sr = np.array([float(e["success_rate"]) for e in x["evals"]])
                aulc.append(np.trapz(sr, steps) / (steps[-1] - steps[0] + 1e-9))
                hit = np.flatnonzero(sr >= 50)
                t50.append(steps[hit[0]] if len(hit) else np.nan)
        m, s, ci = stats(fin)
        summ.append(dict(stage=st, method=me, task=ta, n_seeds=len(rs), final_success_mean=round(m, 2), final_success_std=round(s, 2),
                         final_success_ci95=round(ci, 2) if ci == ci else "", per_seed=";".join(f"{v:.0f}" for v in fin),
                         aulc_mean=round(float(np.mean(aulc)), 2) if aulc else "", steps_to_50pct_mean=(round(float(np.nanmean(t50)), 0) if t50 and not np.all(np.isnan(t50)) else "NA"),
                         n_seeds_reaching_50=int(np.sum(~np.isnan(t50))) if t50 else 0,
                         gpu_hours_mean=round(float(np.mean([x["gpu_hours"] for x in rs])), 2),
                         extra_env_steps_mean=round(float(np.mean([(x.get("ext_stats") or {}).get("extra_env_steps", 0) if isinstance(x.get("ext_stats"), dict) else 0 for x in rs])), 0),
                         first_success_step_median=(float(np.nanmedian([x["first_success_step"] if x["first_success_step"] is not None else np.nan for x in rs])) if rs else ""),
                         train_successes_mean=round(float(np.mean([x["n_success_train"] for x in rs])), 1)))
    if summ:
        with open(OUT / "summary_final.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    # ---- gap closure vs observed baselines (stage0 rrm_only / core)
    base = {(x["method"], x["task"]): x["final_success_mean"] for x in summ if x["stage"] == "stage0"}
    gaps = []
    for x in summ:
        if x["stage"] == "stage0":
            continue
        s_rrm, s_core = base.get(("rrm_only", x["task"])), base.get(("core", x["task"]))
        if s_rrm is None or s_core is None:
            continue
        gap = s_core - s_rrm
        g = (x["final_success_mean"] - s_rrm) / gap if abs(gap) >= 10 else float("nan")
        gaps.append(dict(stage=x["stage"], method=x["method"], task=x["task"], S_new=x["final_success_mean"], S_rrm=s_rrm, S_core=s_core,
                         gap=round(gap, 1), G=round(g, 3) if g == g else "undefined(|gap|<10pt)"))
    if gaps:
        with open(OUT / "gap_closure.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gaps[0].keys())); w.writeheader(); w.writerows(gaps)
    print(f"{len(runs)} runs aggregated; {len(rows)} rows")


if __name__ == "__main__":
    main()
