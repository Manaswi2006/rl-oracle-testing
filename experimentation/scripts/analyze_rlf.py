#!/usr/bin/env python
"""C1/C2/C5/C6 tables from runs/stage0 (rrm_only, core) and runs/stage2 (RLF selectors, rrm_extra)."""
import csv, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "falsification"; OUT.mkdir(parents=True, exist_ok=True)
TASK = {"metaworld_button-press-topdown-v2": "button_press", "metaworld_hammer-v2": "hammer", "metaworld_peg-insert-side-v2": "peg_insert"}


def runs(pattern):
    for f in sorted(glob.glob(str(ROOT / pattern))):
        r = json.load(open(f)); r["task"] = TASK.get(r["env"], r["env"]); r["arm"] = Path(f).parent.parent.name; r["dir"] = str(Path(f).parent.relative_to(ROOT))
        yield r


def w(name, rows):
    if rows:
        keys = list(rows[0].keys())
        with open(OUT / name, "w", newline="") as f:
            ww = csv.DictWriter(f, fieldnames=keys); ww.writeheader(); ww.writerows(rows)


def main():
    allr = list(runs("runs/stage0/*/*/result.json")) + list(runs("runs/stage2/*/*/result.json"))
    for r in runs("runs/stage2_noanchor/*/*/result.json"):     # archived first attempt: repair never fired for these selectors
        r["arm"] = r["arm"] + "__NOANCHOR_no_repair"
        allr.append(r)
    # C1: learned-reward shortcut measurements on the final policy rollouts
    c1 = []
    for r in allr:
        d = r.get("rrm_diag")
        if d:
            c1.append(dict(arm=r["arm"], task=r["task"], seed=r["seed"], diag_success_rate=d["success_rate"], auc_window_R_vs_success=d["auc_window_vs_success"],
                           frac_fail_among_high_R_episodes=d["frac_fail_among_high_reward"]))
    w("C1_reward_vs_success.csv", c1)
    # C2: discovery efficiency
    c2, c5 = [], []
    for r in allr:
        s = r.get("ext_stats")
        if r["arm"].startswith("rlf_") and isinstance(s, dict):
            s = dict(s); s["selector"] = s["selector"] + ("(noanchor)" if "NOANCHOR" in r["arm"] else "")
            rounds = s["rounds"]
            nv, ncf = s["verifier_queries_for_repair"], s["n_confirmed_failures"]
            extra = s["extra_env_steps"]
            c2.append(dict(selector=s["selector"], task=r["task"], seed=r["seed"], rounds=len(rounds), verification_queries=nv, confirmed_failures=ncf,
                           confirmed_per_query=round(ncf / max(nv, 1), 3), extra_env_steps=extra,
                           confirmed_per_1k_extra_env_steps=round(1000 * ncf / extra, 3) if extra else "NA(no extra steps)",
                           anchor_queries=s.get("anchor_queries", 0), free_oracle_evals_for_selection=s["verifier_free_evals_for_selection"], pairs_added=s["n_pairs_added"],
                           verified_successes_available=(rounds[-1]["n_verified_pos_total"] if rounds else 0),
                           final_success=r["final_success_rate"]))
            nov = [x["frac_conf_cells_novel_vs_onpolicy"] for x in rounds if x["frac_conf_cells_novel_vs_onpolicy"] == x["frac_conf_cells_novel_vs_onpolicy"]]
            c5.append(dict(selector=s["selector"], task=r["task"], seed=r["seed"], mean_frac_failure_cells_not_visited_by_task_policy=round(float(np.mean(nov)), 3) if nov else "NA",
                           n_rounds_with_conf_failures=len(nov), attack_episodes=s.get("n_attack_episodes", ""), attack_success_rate=s.get("attack_success_rate", "")))
    w("C2_discovery.csv", c2); w("C5_distinctness.csv", c5)
    # C6: main comparison (final success over seeds)
    groups = {}
    for r in allr:
        groups.setdefault((r["arm"], r["task"]), []).append(r["final_success_rate"])
    c6 = []
    for (arm, task), v in sorted(groups.items()):
        v = np.array(v)
        c6.append(dict(arm=arm, task=task, n=len(v), mean=round(v.mean(), 1), std=round(v.std(ddof=1), 1) if len(v) > 1 else "", per_seed=";".join(f"{x:.0f}" for x in v)))
    w("C6_main_comparison.csv", c6)
    print("C tables:", len(c1), len(c2), len(c5), len(c6))


if __name__ == "__main__":
    main()
