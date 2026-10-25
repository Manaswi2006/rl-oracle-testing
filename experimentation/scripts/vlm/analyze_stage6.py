#!/usr/bin/env python
"""Final success tables for the VLM-outcome (stage 6) and exploration (stage 7) arms, with the earlier-study references and in-loop label accuracy of the VLM."""
import csv, glob, json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
rng = np.random.RandomState(0)
TASK = {"metaworld_button-press-topdown-v2": "button", "metaworld_hammer-v2": "hammer"}
res = collections.defaultdict(list); lab = collections.defaultdict(list)
for f in sorted(glob.glob(str(ROOT / "runs/stage[67]/*/*/result.json"))):
    r = json.load(open(f)); arm = f.split("/")[-3]; t = TASK.get(r["env"], r["env"])
    res[(arm, t)].append((r["seed"], r["final_success_rate"]))
    es = r.get("ext_stats") or {}
    if "label_acc" in es: lab[(arm, t)].append((es["label_acc"], es["fp"], es["fn"], es["n_episodes"], es["n_true_success"]))
ref = collections.defaultdict(list)
for r in csv.DictReader(open(ROOT / "results/vlm/earlier_master_results.csv")):
    if r["eval_kind"] == "final" and r["method"] in ("rrm_only", "core", "outcome_sparse", "psvm_oracle", "psvm_noRRM", "env_sparse", "env_dense") and r["task"] in ("button_press", "hammer"):
        ref[(r["method"], "button" if r["task"] == "button_press" else "hammer")].append((int(r["seed"]), float(r["success_rate"])))
def ms(v):
    x = np.array([s for _, s in v], float); return f"{x.mean():.1f} ± {x.std(ddof=1) if len(x) > 1 else float('nan'):.1f} (n={len(x)})"
def boot(a, b, n=5000):
    a = np.array([s for _, s in a], float); b = np.array([s for _, s in b], float)
    if len(a) < 2 or len(b) < 2: return "n/a"
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return f"{a.mean()-b.mean():+.1f} [{np.percentile(d,2.5):+.1f}, {np.percentile(d,97.5):+.1f}]"
rows = []
order = [("earlier: RRM-only (scripted teacher)", ref, "rrm_only"), ("earlier: full CoRe", ref, "core"), ("earlier: RRM + per-step oracle success flag", ref, "outcome_sparse"), ("earlier: env sparse only", ref, "env_sparse"),
         ("S6 scripted RRM + oracle EPISODE outcome", res, "ep_oracle"), ("S6 scripted RRM + VLM verdict, majority of 3", res, "ep_vlm_major"), ("S6 scripted RRM + VLM verdict, soft fraction", res, "ep_vlm_soft"),
         ("S6 oracle episode outcome only (RRM weight 0)", res, "ep_oracle_noRRM"), ("S6 VLM verdict only (RRM weight 0)", res, "ep_vlm_noRRM"),
         ("S7 RND + oracle outcome (RRM weight 0)", res, "rnd_oracle"), ("S7 RND + VLM verdict (RRM weight 0)", res, "rnd_vlm"), ("S7 RND only (no outcome signal)", res, "rnd_only")]
lines = ["| arm | Button Press | Hammer |", "|---|---|---|"]
for name, src, key in order:
    lines.append(f"| {name} | {ms(src[(key,'button')]) if (key,'button') in src else 'NOT RUN'} | {ms(src[(key,'hammer')]) if (key,'hammer') in src else 'NOT RUN'} |")
cmp_ = [("ep_vlm_major", "ep_oracle"), ("ep_vlm_soft", "ep_oracle"), ("ep_vlm_major", "ep_vlm_soft"), ("ep_vlm_noRRM", "ep_oracle_noRRM"), ("rnd_vlm", "ep_vlm_noRRM"), ("rnd_vlm", "rnd_oracle"), ("rnd_oracle", "ep_oracle_noRRM"), ("rnd_only", "ep_vlm_noRRM")]
lines += ["", "| comparison (seed means) | Button Press diff [95% bootstrap CI] | Hammer |", "|---|---|---|"]
for a, b in cmp_:
    sa = {**res}; 
    lines.append(f"| {a} - {b} | {boot(res.get((a,'button'),[]), res.get((b,'button'),[]))} | {boot(res.get((a,'hammer'),[]), res.get((b,'hammer'),[]))} |")
lines += ["", "| VLM in-loop label accuracy (all episodes of all runs of the arm) | Button Press | Hammer |", "|---|---|---|"]
for arm in ("ep_vlm_major", "ep_vlm_soft", "ep_vlm_noRRM", "rnd_vlm"):
    cells = []
    for t in ("button", "hammer"):
        L = lab.get((arm, t), [])
        if not L: cells.append("-"); continue
        n = sum(x[3] for x in L); fp = sum(x[1] for x in L); fn = sum(x[2] for x in L); ns = sum(x[4] for x in L)
        cells.append(f"acc {1-(fp+fn)/n:.3f}; FP {fp}/{n-ns} failures; FN {fn}/{ns} successes ({len(L)} runs)")
    lines.append(f"| {arm} | {cells[0]} | {cells[1]} |")
out = "\n".join(lines); (ROOT / "results/vlm/stage6_7_tables.md").write_text(out + "\n"); print(out)
per = {f"{k[0]}|{k[1]}": sorted(v) for k, v in res.items()}
json.dump(per, open(ROOT / "results/vlm/stage6_7_per_seed.json", "w"), indent=1)
