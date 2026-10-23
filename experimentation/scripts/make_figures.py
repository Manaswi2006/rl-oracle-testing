#!/usr/bin/env python
"""Learning curves (mean + individual seeds) and final-success dot plots per stage.  Palette: dataviz reference categorical order."""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TASK = {"metaworld_button-press-topdown-v2": "Button Press", "metaworld_hammer-v2": "Hammer", "metaworld_peg-insert-side-v2": "Peg Insert"}
PIN = {"core": CAT[0], "rrm_only": CAT[1]}            # entities keep their colour in every figure
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal"})


def load(stage_methods):
    data = {}
    for stage, methods in stage_methods:
        for m in methods:
            for res in sorted(glob.glob(str(ROOT / f"runs/{stage}/{m}/*/result.json"))):
                r = json.load(open(res)); d = Path(res).parent
                ev = list(csv.DictReader(open(d / "eval.csv")))
                data.setdefault((m, TASK.get(r["env"], r["env"])), []).append(
                    dict(seed=r["seed"], steps=np.array([int(e["step"]) for e in ev]), sr=np.array([float(e["success_rate"]) for e in ev]), final=r["final_success_rate"]))
    return data


def colors_for(methods):
    out, free = {}, [c for c in CAT if c not in PIN.values()]
    for m in methods:
        out[m] = PIN.get(m) or free.pop(0)
    return out


def curves(data, methods, title, fname):
    col = colors_for(methods)
    tasks = list(TASK.values())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, t in zip(axs, tasks):
        for m in methods:
            runs = data.get((m, t), [])
            if not runs: continue
            L = min(len(r["steps"]) for r in runs); st = runs[0]["steps"][:L]
            Y = np.stack([r["sr"][:L] for r in runs])
            for y in Y: ax.plot(st / 1e3, y, color=col[m], lw=0.8, alpha=0.25)
            ax.plot(st / 1e3, Y.mean(0), color=col[m], lw=2, label=f"{m} (n={len(runs)})")
        ax.set_title(t); ax.set_xlabel("environment steps (k)"); ax.set_ylim(-3, 103)
    axs[0].set_ylabel("eval success rate (%) - 10 episodes")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / f"{fname}.png", dpi=160, bbox_inches="tight"); fig.savefig(FIG / f"{fname}.pdf", bbox_inches="tight"); plt.close(fig)


def finals(data, methods, title, fname):
    col = colors_for(methods)
    tasks = list(TASK.values())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, t in zip(axs, tasks):
        for i, m in enumerate(methods):
            v = [r["final"] for r in data.get((m, t), [])]
            if not v: continue
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=col[m], s=36, zorder=3, edgecolor=SURF, linewidth=1.5)
            ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=35, ha="right"); ax.set_title(t); ax.set_ylim(-3, 103)
    axs[0].set_ylabel("final success rate (%) - 50 episodes; bar = seed mean")
    fig.suptitle(title, y=1.0, fontsize=12); fig.tight_layout()
    fig.savefig(FIG / f"{fname}.png", dpi=160, bbox_inches="tight"); fig.savefig(FIG / f"{fname}.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    G = {
        "stage0_baselines": [("stage0", ["core", "rrm_only", "env_dense", "env_sparse"])],
        "stage1_oracle": [("stage0", ["core", "rrm_only"]), ("stage1", ["psvm_oracle", "facr_progress", "facr_value", "facr_facr"])],
        "stage5_psvm_controls": [("stage0", ["core", "rrm_only"]), ("stage1", ["psvm_oracle"]), ("stage5", ["outcome_sparse", "psvm_noRRM"])],
        "stage2_rlf": [("stage0", ["core", "rrm_only"]), ("stage2", ["rrm_extra", "rlf_onpolicy", "rlf_highreward", "rlf_uncertainty", "rlf_attack"])],
    }
    for name, sm in G.items():
        data = load(sm)
        methods = [m for _, ms in sm for m in ms if any(k[0] == m for k in data)]
        if not methods: continue
        curves(data, methods, f"{name}: learning curves (thin = individual seeds)", f"curves_{name}")
        finals(data, methods, f"{name}: final success per seed", f"final_{name}")
        print("figures", name, len(methods))
