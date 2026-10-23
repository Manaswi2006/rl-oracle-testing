#!/usr/bin/env python
"""Generate job lists.  usage: gen_jobs.py <stage> > jobs.txt"""
import sys
TASKS = {"button": "metaworld_button-press-topdown-v2", "hammer": "metaworld_hammer-v2", "peg": "metaworld_peg-insert-side-v2"}
SEEDS = [0, 1, 2]
stage = sys.argv[1]
out = []
def add(group, name, task, seed, ov):
    out.append(f"runs/{group}/{name}/{task}_s{seed} env={TASKS[task]} seed={seed} " + " ".join(ov))
if stage == "stage0":
    for seed in SEEDS:
        for task in TASKS:
            add("stage0", "core", task, seed, ["method=core"])
            add("stage0", "rrm_only", task, seed, ["method=rrm_only"])
    for seed in SEEDS:
        for task in TASKS:
            add("stage0", "env_dense", task, seed, ["method=env_dense"])
            add("stage0", "env_sparse", task, seed, ["method=env_sparse"])
if stage == "stage1":   # oracle feasibility: RRM + learned structural reward replacing FRM
    for seed in SEEDS:
        for task in TASKS:
            add("stage1", "psvm_oracle", task, seed, ["method=ext", "ext_name=psvm", "psvm.mode=oracle_hard"])
            for v in ("progress", "value", "facr"):
                add("stage1", f"facr_{v}", task, seed, ["method=ext", "ext_name=facr", f"facr.variant={v}"])
if stage == "stage2":   # RLF: RRM + verified counterexample repair (oracle verifier), matched Q=5 verifications / round
    for seed in SEEDS:
        for task in TASKS:
            add("stage2", "rrm_extra", task, seed, ["method=rrm_extra", "RRM_batch=25", "max_feedback=625"])
            for sel in ("onpolicy", "highreward", "uncertainty", "attack"):
                add("stage2", f"rlf_{sel}", task, seed, ["method=rlf", f"rlf.selector={sel}", "rlf.extra_capacity=125", "rlf.Q=5"])
    for seed in (0, 1):
        for task in TASKS:
            for sel in ("random", "disagreement", "hybrid"):
                add("stage2", f"rlf_{sel}", task, seed, ["method=rlf", f"rlf.selector={sel}", "rlf.extra_capacity=125", "rlf.Q=5"])
if stage == "stage3":   # extra seeds for the key arms on the two tasks where any method succeeded (power: stage0-2 had n=3 and binary 0/100 outcomes)
    SEEDS3 = [3, 4, 5]
    arms = [("core", ["method=core"]), ("rrm_only", ["method=rrm_only"]),
            ("psvm_oracle", ["method=ext", "ext_name=psvm", "psvm.mode=oracle_hard"]),
            ("facr_value", ["method=ext", "ext_name=facr", "facr.variant=value"]),
            ("facr_facr", ["method=ext", "ext_name=facr", "facr.variant=facr"]),
            ("rlf_uncertainty", ["method=rlf", "rlf.selector=uncertainty", "rlf.extra_capacity=125", "rlf.Q=5"]),
            ("rlf_onpolicy", ["method=rlf", "rlf.selector=onpolicy", "rlf.extra_capacity=125", "rlf.Q=5"]),
            ("rlf_highreward", ["method=rlf", "rlf.selector=highreward", "rlf.extra_capacity=125", "rlf.Q=5"]),
            ("rlf_attack", ["method=rlf", "rlf.selector=attack", "rlf.extra_capacity=125", "rlf.Q=5"]),
            ("rrm_extra", ["method=rrm_extra", "RRM_batch=25", "max_feedback=625"])]
    for seed in SEEDS3:
        for task in ("button", "hammer"):
            for name, ov in arms:
                group = "stage0" if name in ("core", "rrm_only") else ("stage1" if name.startswith(("psvm", "facr")) else "stage2")
                add(group, name, task, seed, ov)
if stage == "stage4":   # Dial Turn, key arms
    T = "metaworld_dial-turn-v2"
    for seed in (0, 1, 2):
        for g, n, ov in [("stage0", "core", ["method=core"]), ("stage0", "rrm_only", ["method=rrm_only"]), ("stage1", "psvm_oracle", ["method=ext", "ext_name=psvm", "psvm.mode=oracle_hard"]),
                         ("stage1", "facr_facr", ["method=ext", "ext_name=facr", "facr.variant=facr"]), ("stage2", "rlf_uncertainty", ["method=rlf", "rlf.selector=uncertainty", "rlf.extra_capacity=125", "rlf.Q=5"])]:
            out.append(f"runs/{g}/{n}/dial_s{seed} env={T} seed={seed} " + " ".join(ov))
if stage == "stage5":   # mechanism controls for P-SVM: plain sparse outcome reward in the FRM slot; SVM without RRM (reward_alpha=0)
    T5 = dict(TASKS, dial="metaworld_dial-turn-v2")
    for seed in range(6):
        for t in ("button", "hammer"):
            out.append(f"runs/stage5/outcome_sparse/{t}_s{seed} env={T5[t]} seed={seed} method=ext ext_name=outcome")
    for seed in range(3):
        for t in ("peg", "dial"):
            out.append(f"runs/stage5/outcome_sparse/{t}_s{seed} env={T5[t]} seed={seed} method=ext ext_name=outcome")
        for t in ("button", "hammer"):
            out.append(f"runs/stage5/psvm_noRRM/{t}_s{seed} env={T5[t]} seed={seed} method=ext ext_name=psvm psvm.mode=oracle_hard reward_alpha=0.0")
print("\n".join(out))
