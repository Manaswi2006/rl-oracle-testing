#!/usr/bin/env python
"""Build report tables (reports/tables/*.md) and results/aggregate/numbers.json from the aggregated results.
The narrative lives in reports/FINAL_RESEARCH_REPORT.template.md; `{{include:tables/<name>.md}}` is replaced by the generated table.
usage: generate_report.py [--render]   (--render writes FINAL_RESEARCH_REPORT.md from the template)
"""
import csv, glob, json, sys, re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP, TAB = ROOT / "reports", ROOT / "reports" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
TASKS = ["button_press", "hammer", "peg_insert"]
OTHER = "dial_turn"
TN = {"button_press": "Button Press", "hammer": "Hammer", "peg_insert": "Peg Insert", "dial_turn": "Dial Turn"}
rng = np.random.RandomState(0)


def load_runs():
    out = []
    for f in sorted(glob.glob(str(ROOT / "runs/*/*/*/result.json"))):
        r = json.load(open(f)); d = Path(f).parent
        r["stage"], r["arm"], r["dir"] = d.parent.parent.name, d.parent.name, str(d.relative_to(ROOT))
        r["task"] = {"metaworld_button-press-topdown-v2": "button_press", "metaworld_hammer-v2": "hammer", "metaworld_peg-insert-side-v2": "peg_insert", "metaworld_dial-turn-v2": "dial_turn"}[r["env"]]
        out.append(r)
    return out


RUNS = load_runs()


def vals(arm, task, stage=None, seeds=None):
    return [r for r in RUNS if r["arm"] == arm and r["task"] == task and (stage is None or r["stage"] == stage) and (seeds is None or r["seed"] in seeds)
            and r["stage"] != "stage2_noanchor"]


def ms(v):
    v = np.array([x["final_success_rate"] for x in v], float)
    if len(v) == 0: return "NOT RUN"
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{v.mean():.1f} ± {sd:.1f} (n={len(v)})"


def boot_diff(a, b, n=5000):
    a = np.array([x["final_success_rate"] for x in a], float); b = np.array([x["final_success_rate"] for x in b], float)
    if len(a) < 2 or len(b) < 2: return None
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def steps(arm, task):
    r = vals(arm, task)
    if not r: return None
    ex = [(x.get("ext_stats") or {}).get("extra_env_steps", 0) if isinstance(x.get("ext_stats"), dict) else 0 for x in r]
    return int(np.mean([x["steps"] for x in r]) + np.mean(ex))


def write(name, text):
    (TAB / f"{name}.md").write_text(text + "\n")


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    N = {}
    # ---- master table
    spec = [("CoRe RRM only (`rrm_only`)", "rrm_only"), ("Full CoRe (`core`)", "core"), ("Env sparse reward", "env_sparse"), ("Env dense reward", "env_dense"),
            ("RRM + oracle SVM (`psvm_oracle`)", "psvm_oracle"), ("Control: RRM + plain sparse outcome reward (`outcome_sparse`)", "outcome_sparse"), ("Control: oracle SVM without RRM (`psvm_noRRM`)", "psvm_noRRM"), ("RRM + P-SVM (VLM/noisy labels)", None),
            ("RRM + oracle FACR counterfactual (`facr_facr`)", "facr_facr"), ("RRM + oracle success-value shaping (`facr_value`)", "facr_value"),
            ("RRM + oracle progress prediction (`facr_progress`)", "facr_progress"), ("RRM + non-oracle FACR", None)]
    rlf = [("rrm_extra", "RRM-only + 125 extra teacher labels (control)"), ("rlf_onpolicy", "RRM + RLF repair: on-policy selection"), ("rlf_highreward", "RRM + RLF repair: high-reward (hard-negative) selection"),
           ("rlf_uncertainty", "RRM + RLF repair: uncertainty selection"), ("rlf_attack", "RRM + RLF repair: falsification policy"),
           ("rlf_hybrid", "RRM + RLF repair: hybrid attack+uncertainty"), ("rlf_random", "RRM + RLF repair: random policy"), ("rlf_disagreement", "RRM + RLF repair: verifier-disagreement (oracle upper bound)")]
    rows = []
    for label, arm in spec:
        if arm is None:
            rows.append([label, "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN", "N/A", "N/A"]); continue
        r = [ms(vals(arm, t)) for t in TASKS]
        gh = np.mean([x["gpu_hours"] for x in RUNS if x["arm"] == arm and x["stage"] != "stage2_noanchor"]) if any(x["arm"] == arm for x in RUNS) else float("nan")
        st = steps(arm, "button_press")
        rows.append([label] + r + [ms(vals(arm, OTHER)).replace("NOT RUN", "NOT RUN") + " (Dial Turn)", f"{st:,}" if st else "N/A", f"{gh:.1f}"])
    for arm, label in rlf:
        r = [ms(vals(arm, t)) for t in TASKS]
        gh = np.mean([x["gpu_hours"] for x in RUNS if x["arm"] == arm and x["stage"] != "stage2_noanchor"]) if any(x["arm"] == arm for x in RUNS) else float("nan")
        st = steps(arm, "button_press")
        rows.append([label] + r + [ms(vals(arm, OTHER)) + " (Dial Turn)", f"{st:,}" if st else "N/A", f"{gh:.1f}"])
    write("master", table(["Method", "Button Press", "Hammer", "Peg Insert", "Other tasks (Sweep, Dial, Drawer, Soccer: only Dial Turn run, key arms)", "Env steps (task+extra)", "GPU-h / run*"], rows))
    # ---- reproduction table
    pub = {"button_press": (100.0, 8.7), "hammer": (98.0, 49.3), "peg_insert": (100.0, 36.0)}
    pub_env = {"env_sparse": (66.7, 60.7, 0.0), "env_dense": (100.0, 97.3, 100.0)}
    rows = []
    for t_i, t in enumerate(TASKS):
        rows.append([TN[t], f"{pub[t][0]}", ms(vals("core", t, "stage0")), f"{pub[t][1]}", ms(vals("rrm_only", t, "stage0")), f"{pub_env['env_sparse'][t_i]}", ms(vals("env_sparse", t)), f"{pub_env['env_dense'][t_i]}", ms(vals("env_dense", t))])
    write("repro", table(["Task", "CoRe published", "CoRe reproduced (this study)", "RRM-only published", "RRM-only (sim. teacher)", "Env sparse publ.", "Env sparse repr.", "Env dense publ.", "Env dense repr."], rows))
    # ---- gap closure (observed baselines), all available seeds
    base_core = {t: np.mean([x["final_success_rate"] for x in vals("core", t)]) for t in TASKS}
    base_rrm = {t: np.mean([x["final_success_rate"] for x in vals("rrm_only", t)]) for t in TASKS}
    rows, gnum = [], {}
    for arm in ["psvm_oracle", "outcome_sparse", "psvm_noRRM", "facr_progress", "facr_value", "facr_facr", "rrm_extra", "rlf_onpolicy", "rlf_highreward", "rlf_uncertainty", "rlf_attack", "rlf_hybrid", "rlf_random", "rlf_disagreement"]:
        row = [arm]
        n_pass = 0
        for t in TASKS:
            v = vals(arm, t)
            if not v: row.append("NOT RUN"); continue
            S = np.mean([x["final_success_rate"] for x in v]); gap = base_core[t] - base_rrm[t]
            if abs(gap) < 10:
                row.append(f"S={S:.1f}; G undefined (gap {gap:+.1f})"); continue
            g = (S - base_rrm[t]) / gap
            n_pass += g >= 0.5
            row.append(f"S={S:.1f}; G={g:.2f}")
            gnum[(arm, t)] = g
        row.append(f"{n_pass}/3")
        rows.append(row)
    write("gap", table(["Arm", "Button Press", "Hammer", "Peg Insert", "tasks with G>=0.5"], rows) +
          "\n\nObserved baselines (all seeds): " + "; ".join(f"{TN[t]}: RRM={base_rrm[t]:.1f}, CoRe={base_core[t]:.1f}" for t in TASKS))
    N["gap_G"] = {f"{a}|{t}": g for (a, t), g in gnum.items()}
    N["baselines"] = {"rrm": base_rrm, "core": base_core}
    # ---- bootstrap differences vs references
    rows = []
    for arm, ref in [("psvm_oracle", "rrm_only"), ("psvm_oracle", "core"), ("psvm_oracle", "outcome_sparse"), ("outcome_sparse", "rrm_only"), ("psvm_oracle", "psvm_noRRM"), ("facr_facr", "facr_value"), ("facr_value", "rrm_only"), ("facr_facr", "rrm_only"), ("rlf_uncertainty", "rrm_extra"), ("rlf_uncertainty", "rlf_onpolicy"),
                     ("rlf_highreward", "rlf_onpolicy"), ("rlf_attack", "rlf_onpolicy"), ("rlf_attack", "rrm_extra"), ("rlf_attack", "rlf_uncertainty"), ("core", "rrm_only")]:
        row = [f"{arm} - {ref}"]
        for t in ("button_press", "hammer"):
            d = boot_diff(vals(arm, t), vals(ref, t))
            row.append("NOT RUN" if d is None else f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")
        rows.append(row)
    write("diffs", table(["Comparison (seed-level means)", "Button Press: diff [95% bootstrap CI]", "Hammer: diff [95% bootstrap CI]"], rows))
    # ---- seed table
    rows = []
    arms = sorted({r["arm"] for r in RUNS if r["stage"] != "stage2_noanchor"})
    for arm in arms:
        row = [arm]
        for t in TASKS:
            v = sorted(vals(arm, t), key=lambda x: x["seed"])
            row.append(", ".join(f"s{x['seed']}:{x['final_success_rate']:.0f}" for x in v) if v else "-")
        rows.append(row)
    write("per_seed", table(["Arm"] + [TN[t] for t in TASKS], rows))
    # ---- sample efficiency
    rows = []
    for arm in arms:
        row = [arm]
        for t in TASKS:
            fs = [x["first_success_step"] for x in vals(arm, t) if x["first_success_step"] is not None]
            ns = [x["n_success_train"] for x in vals(arm, t)]
            row.append(f"first-success step median {np.median(fs)/1000:.0f}k ({len(fs)}/{len(vals(arm, t))} seeds); train successes {np.mean(ns):.0f}" if vals(arm, t) else "-")
        rows.append(row)
    write("sample_eff", table(["Arm"] + [TN[t] for t in TASKS], rows))
    # ---- A2
    f = ROOT / "results/psvm/A2_label_corruption.csv"
    if f.exists():
        d = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["prevalence"] != "natural": continue
            if r["ep_auc"] not in ("nan", ""): d[(r["scenario"], r["method"])].append(float(r["ep_auc"]))
        scn = ["clean", "iid_flip10", "iid_flip30", "asym_FP20", "asym_FN20", "correlated_sys20", "correlated_sys40", "instance_dep"]
        meth = ["oracle", "hard", "majority", "soft_raw", "conf_filter", "soft_calibrated"]
        rows = [[s] + [(f"{np.mean(d[(s, m)]):.3f}" if d.get((s, m)) else "-") for m in meth] for s in scn]
        write("A2_auc", table(["Scenario (natural prevalence)"] + meth, rows))
        d2 = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["scenario"] == "clean" and r["method"] == "oracle" and r["ep_auc"] not in ("nan", ""): d2[r["prevalence"]].append(float(r["ep_auc"]))
        rows = []
        dd = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["scenario"] in ("iid_flip10",) and r["method"] in ("hard", "soft_raw") and r["ep_auc"] not in ("nan", ""): dd[(r["prevalence"], r["method"])].append(float(r["ep_auc"]))
        for k in sorted(dd): rows.append([k[0], k[1], f"{np.mean(dd[k]):.3f}", len(dd[k])])
        write("A2_prev", table(["train success prevalence", "method (iid 10% flips)", "mean episode AUC", "n"], rows))
    json.dump(N, open(ROOT / "results/aggregate/numbers.json", "w"), indent=1, default=float)
    print("tables written:", sorted(p.name for p in TAB.glob("*.md")))


def render():
    t = (REP / "FINAL_RESEARCH_REPORT.template.md").read_text()
    t = re.sub(r"\{\{include:([^}]+)\}\}", lambda m: (REP / m.group(1)).read_text().strip(), t)
    (REP / "FINAL_RESEARCH_REPORT.md").write_text(t)
    print("rendered FINAL_RESEARCH_REPORT.md", len(t))


if __name__ == "__main__":
    main()
    if "--render" in sys.argv:
        render()
