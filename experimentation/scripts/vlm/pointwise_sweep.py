#!/usr/bin/env python
"""Pointwise progress scoring (position-bias free): the VLM rates ONE image 0-100 for task completion; windows are compared by the score of their last frame.
Evaluated on the same dev window pairs as pref_sweep (truth = larger summed environment reward, |rel diff| >= 0.3)."""
import argparse, json, pickle, re, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm.vlm_client import VLM, TASK_DESC, b64
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=160); ap.add_argument("--ports", default="8000"); ap.add_argument("--model", default="qwen3vl8b"); ap.add_argument("--tag", default="")
ap.add_argument("--bank", default="bank")
a = ap.parse_args()
vlm = VLM(ports=[int(p) for p in a.ports.split(",")], model=a.model, workers=32)
Q = ("Task for the robot: {task}. The image shows the current scene. In one or two short sentences describe the state of the objects that matter for the task, then rate how close the task is to "
     "being COMPLETED on a scale from 0 (nothing done yet, initial scene) to 100 (task fully completed). Finish with a final line exactly 'SCORE: <number>'.")
res = {}
for task in ("button", "hammer"):
    bank = pickle.load(open(ROOT / f"results/vlm/{a.bank}_{task}.pkl", "rb"))
    rs = np.random.RandomState(1); pairs = []
    while len(pairs) < a.n:
        i, j = rs.randint(len(bank), size=2); si, sj = rs.randint(8), rs.randint(8)
        if i == j or (i + j) % 2 == 1: continue
        A, B = bank[i]["segments"][si], bank[j]["segments"][sj]
        if abs(A["env_reward_sum"] - B["env_reward_sum"]) / (max(abs(A["env_reward_sum"]), abs(B["env_reward_sum"])) + 1e-6) < 0.3: continue
        pairs.append((A, B))
    def score(jpg):
        content = [{"type": "image_url", "image_url": {"url": b64(jpg)}}, {"type": "text", "text": Q.format(task=TASK_DESC[task])}]
        ch = vlm._chat(content, max_tokens=140, logprobs=False)
        m = re.findall(r"SCORE:\s*(\d+(?:\.\d+)?)", ch.message.content or "")
        return float(m[-1]) if m else float("nan")
    t0 = time.time()
    last = [score(x) for x in []]
    items = [(k, w, pr[w]["frames"][-1]) for k, pr in enumerate(pairs) for w in (0, 1)]
    sc = vlm.pmap(lambda it: score(it[2]), items)
    S = np.array(sc).reshape(-1, 2); t = np.array([int(pr[0]["env_reward_sum"] > pr[1]["env_reward_sum"]) for pr in pairs])
    ok = ~np.isnan(S).any(1); tie = ok & (S[:, 0] == S[:, 1])
    pred = (S[:, 0] > S[:, 1]).astype(int)
    res[task] = dict(n=len(pairs), parsed=float(ok.mean()), tie_frac=float(tie.mean()), acc_incl_ties_as_wrong=float(((pred == t) & ok & ~tie).mean()), acc_excl_ties=float((pred[ok & ~tie] == t[ok & ~tie]).mean()),
                     score_mean=float(np.nanmean(S)), secs=round(time.time() - t0, 1))
    print(task, {k: round(v, 3) for k, v in res[task].items()}, flush=True)
json.dump(res, open(ROOT / f"results/vlm/pointwise_sweep{a.tag}.json", "w"), indent=1)
