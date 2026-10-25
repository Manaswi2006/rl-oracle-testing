#!/usr/bin/env python
"""Query the VLM on the ground-truth episode bank: 5 paraphrased outcome questions (16 frames), the first paraphrase also with 8 frames,
and 800 window pairs (both orders) for the preference question.  Raw outputs -> results/vlm/label_quality_<task>.json"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm.vlm_client import VLM

ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--ports", default="8000"); ap.add_argument("--n_pairs", type=int, default=800)
ap.add_argument("--workers", type=int, default=24); ap.add_argument("--tag", default="")
a = ap.parse_args()
bank = pickle.load(open(ROOT / f"results/vlm/bank_{a.task}.pkl", "rb"))
vlm = VLM(ports=[int(p) for p in a.ports.split(",")], workers=a.workers)
t0 = time.time()
jobs = [(i, k, 16) for i in range(len(bank)) for k in range(5)] + [(i, 0, 8) for i in range(len(bank))]


def run_o(j):
    i, k, nf = j
    fr = bank[i]["frames16"] if nf == 16 else bank[i]["frames16"][1::2]
    return dict(ep=i, prompt=k, n_frames=nf, **vlm.outcome(fr, a.task, k))


out = vlm.pmap(run_o, jobs)
print("outcome queries", len(out), f"{time.time()-t0:.0f}s", flush=True)
rng = np.random.RandomState(0)
pairs = []
while len(pairs) < a.n_pairs:
    i, j = rng.randint(len(bank), size=2)
    if i == j: continue
    si, sj = rng.randint(8), rng.randint(8)
    pairs.append((i, si, j, sj))


def run_p(pr):
    i, si, j, sj = pr
    A, B = bank[i]["segments"][si], bank[j]["segments"][sj]
    r1 = vlm.preference(A["frames"], B["frames"], a.task)
    r2 = vlm.preference(B["frames"], A["frames"], a.task)
    return dict(ep_a=i, seg_a=si, ep_b=j, seg_b=sj, p_a_first=r1["p_a"], p_a_second=r2["p_a"],
                rew_a=A["env_reward_sum"], rew_b=B["env_reward_sum"], succ_a=A["success_in"], succ_b=B["success_in"])


prefs = vlm.pmap(run_p, pairs)
print("pref pairs", len(prefs), f"{time.time()-t0:.0f}s", flush=True)
truth = [dict(ep=i, success=e["success"], first_success=e["first_success"], tag=e["tag"], ret=e["ret"]) for i, e in enumerate(bank)]
json.dump(dict(task=a.task, truth=truth, outcome=out, prefs=prefs, secs=time.time() - t0), open(ROOT / f"results/vlm/label_quality_{a.task}{a.tag}.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved")
