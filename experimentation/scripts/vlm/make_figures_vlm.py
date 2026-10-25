#!/usr/bin/env python
"""Figures for the VLM-outcome study: (1) in-loop false-positive rate of the VLM verdict over training, (2) final success per seed for stage 6/7 arms.  Palette: dataviz reference categorical order, light mode."""
import glob, json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
# (1) false-positive rate over training
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, task in zip(axs, ("button", "hammer")):
    for k, (arm, lab) in enumerate((("ep_vlm_major", "VLM verdict, majority of 3"), ("ep_vlm_soft", "VLM verdict, soft fraction"))):
        blocks = collections.defaultdict(lambda: [0, 0])
        per_run = []
        for f in sorted(glob.glob(str(ROOT / f"runs/stage6/{arm}/{task}_s*/episode_verdicts.json"))):
            rr = collections.defaultdict(lambda: [0, 0])
            for r in json.load(open(f)):
                if r["true_success"] == 0:
                    b = min(r["ep"] // 50, 7); rr[b][1] += 1; rr[b][0] += int(r["y"] > 0.5)
            per_run.append([rr[b][0] / rr[b][1] if rr[b][1] else np.nan for b in range(8)])
        P = np.array(per_run) if per_run else np.full((1, 8), np.nan)
        x = np.arange(8) * 50 + 25
        for y in P: ax.plot(x, y, color=CAT[k], lw=0.8, alpha=0.3)
        ax.plot(x, np.nanmean(P, 0), color=CAT[k], lw=2, label=f"{lab} (n={len(per_run)} runs)")
    ax.set_title({"button": "Button Press", "hammer": "Hammer"}[task]); ax.set_xlabel("training episode (500 steps each)"); ax.set_ylim(0, 1)
axs[0].set_ylabel("false-positive rate of the verdict\n(failed episodes judged 'success')"); axs[0].legend(frameon=False, loc="upper left")
fig.suptitle("The policy learns to fool the VLM judge: false positives rise during training (thin = single runs)", y=1.0, fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "vlm_fpr_over_training.png", dpi=160, bbox_inches="tight"); plt.close(fig)
# (2) final success per seed
per = json.load(open(ROOT / "results/vlm/stage6_7_per_seed.json"))
arms = [("ep_oracle", "oracle outcome\n+ scripted RRM"), ("ep_vlm_major", "VLM majority\n+ scripted RRM"), ("ep_vlm_soft", "VLM soft\n+ scripted RRM"), ("ep_oracle_noRRM", "oracle outcome\nonly"), ("ep_vlm_noRRM", "VLM verdict\nonly"),
        ("rnd_oracle", "RND + oracle\noutcome"), ("rnd_vlm", "RND + VLM\nverdict"), ("rnd_only", "RND only")]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for ax, task in zip(axs, ("button", "hammer")):
    for i, (arm, lab) in enumerate(arms):
        v = [s for _, s in per.get(f"{arm}|{task}", [])]
        if not v: continue
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, color=CAT[i % 8], s=34, zorder=3, edgecolor=SURF, linewidth=1.4)
        ax.plot([i - 0.28, i + 0.28], [np.mean(v)] * 2, color=INK, lw=2, zorder=4)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms], fontsize=8); ax.set_title({"button": "Button Press", "hammer": "Hammer"}[task]); ax.set_ylim(-3, 103)
axs[0].set_ylabel("final success rate (%), 50 episodes; bar = seed mean")
fig.tight_layout(); fig.savefig(FIG / "vlm_stage6_7_final.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("figures written")
