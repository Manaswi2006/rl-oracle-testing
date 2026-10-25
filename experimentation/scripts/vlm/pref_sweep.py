#!/usr/bin/env python
"""Preference-question variants on dev window pairs: direct letter vs chain-of-thought, both presentation orders.
Ground truth = window with the larger summed environment reward (the scripted teacher of the earlier study); pairs with relative difference < 0.3 are skipped."""
import argparse, json, pickle, re, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm.vlm_client import VLM, TASK_DESC, b64
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=160); ap.add_argument("--ports", default="8000"); ap.add_argument("--model", default="qwen3vl8b"); ap.add_argument("--tag", default="")
a = ap.parse_args()
vlm = VLM(ports=[int(p) for p in a.ports.split(",")], model=a.model, workers=32)
COT = ("A robot arm has to do this task: {task}. Clip A ({n} frames, time order) and Clip B ({n} frames, time order) are shown above. First describe in one short sentence per clip how far the "
       "task has progressed (position/state of the objects relevant to the task, at the end of each clip). Then decide which clip is CLOSER to a completed task. Finish with a final line exactly 'VERDICT: A' or 'VERDICT: B'.")
res = {}
for task in ("button", "hammer"):
    bank = pickle.load(open(ROOT / f"results/vlm/bank_{task}.pkl", "rb"))
    rs = np.random.RandomState(1); pairs = []
    while len(pairs) < a.n:
        i, j = rs.randint(len(bank), size=2); si, sj = rs.randint(8), rs.randint(8)
        if i == j or (i + j) % 2 == 1: continue          # dev pairs only (even/even or odd/odd mix, distinct from the test pairs of the full study)
        A, B = bank[i]["segments"][si], bank[j]["segments"][sj]
        if abs(A["env_reward_sum"] - B["env_reward_sum"]) / (max(abs(A["env_reward_sum"]), abs(B["env_reward_sum"])) + 1e-6) < 0.3: continue
        pairs.append((A, B))
    for mode in ("direct", "cot"):
        def run(pr):
            A, B = pr; outs = []
            for first, second in ((A, B), (B, A)):
                content = [{"type": "text", "text": "Clip A:"}] + [{"type": "image_url", "image_url": {"url": b64(f)}} for f in first["frames"]] + [{"type": "text", "text": "Clip B:"}] + \
                          [{"type": "image_url", "image_url": {"url": b64(f)}} for f in second["frames"]]
                if mode == "direct":
                    content.append({"type": "text", "text": f"Task: {TASK_DESC[task]}. Which clip is closer to a completed task? Answer A or B."}); ch = vlm._chat(content)
                    p, tok = vlm._p(ch, "a", "b"); outs.append(int(tok.startswith("a")))
                else:
                    content.append({"type": "text", "text": COT.format(task=TASK_DESC[task], n=len(first["frames"]))}); ch = vlm._chat(content, max_tokens=200, logprobs=False)
                    m = re.findall(r"VERDICT:\s*([AB])", ch.message.content or ""); outs.append(1 if (m and m[-1] == "A") else (0 if m else -1))
            # outs[0]: 1 if clip A (=first window) chosen in order1 ; outs[1]: 1 if first-position clip chosen in order2, i.e. window B chosen
            o1 = outs[0]; o2 = (1 - outs[1]) if outs[1] >= 0 else -1          # prob. that original A is better
            return o1, o2, int(pr[0]["env_reward_sum"] > pr[1]["env_reward_sum"])
        t0 = time.time(); r = vlm.pmap(run, pairs)
        o1 = np.array([x[0] for x in r]); o2 = np.array([x[1] for x in r]); t = np.array([x[2] for x in r])
        ok = (o1 >= 0) & (o2 >= 0); cons = ok & (o1 == o2)
        res[f"{task}|{mode}"] = dict(n=len(r), acc_order1=float((o1 == t).mean()), acc_order2=float((o2 == t).mean()), frac_order_consistent=float(cons.mean()),
                                    acc_when_consistent=float((o1[cons] == t[cons]).mean()) if cons.any() else None, P_choose_first_position=float(o1.mean()), secs=round(time.time() - t0, 1))
        print(task, mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"{task}|{mode}"].items()}, flush=True)
json.dump(res, open(ROOT / f"results/vlm/pref_sweep{a.tag}.json", "w"), indent=1)
