#!/usr/bin/env python
"""Offline screen of candidate verifiers against REAL exploits collected during training (false positives the 8B judge accepted), without spending policy-learning compute.
Pass rule fixed in advance: reject >= 70% of the exploits AND accept >= 85% of true successes (per task).
Candidates: C0 frozen judge (3 paraphrases, majority), C1 unanimity of the 3, C2 per-frame sustained check (each of the last 4 frames judged against the first; >= 3 of 4 must say Yes),
C3 skeptical-inspector prompt, C4 = C3 AND C0, C5 = C2 AND C0.  Same input sets as the earlier 8B hardening test (results/vlm/hardening_test.json) for a like-for-like comparison.
usage: screen_verifier.py <model-name> <port> <tag>"""
import glob, json, pickle, re, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/vlm"))
from src.vlm.vlm_client import VLM, TASK_DESC, b64, judge_outcome
model, port, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
vlm = VLM(ports=[port], model=model, workers=24)
SKEPTIC = ("The robot's instruction was: {task}. Image 1 is the first frame; images 2-5 are the last four frames of its attempt, in time order. Act as a skeptical inspector. First list in two short sentences any visible evidence "
           "that the instruction has NOT been completely carried out (objects still in their starting state, objects not in the required final position, the robot not in contact with the relevant object, ...). "
           "Then answer 'VERDICT: Yes' only if you found no such evidence, otherwise 'VERDICT: No'. The last line must be exactly 'VERDICT: Yes' or 'VERDICT: No'.")
FRAME_Q = ("Instruction for the robot: {task}. Image 1 is the scene before the attempt. Image 2 is one frame near the end of the attempt. In two short sentences describe the state of the objects that matter in image 2 compared with image 1, "
           "then decide whether the instruction is completely carried out IN IMAGE 2. Final line exactly 'VERDICT: Yes' or 'VERDICT: No'.")


def verdict(content, max_tokens=220):
    t = vlm._chat(content, max_tokens=max_tokens, logprobs=False).message.content or ""
    m = re.findall(r"VERDICT:\s*(Yes|No)", t, re.I)
    return int(bool(m) and m[-1].lower() == "yes")


def img(f): return {"type": "image_url", "image_url": {"url": b64(f)}}


def candidates(frames, task):
    sel = [frames[0]] + frames[-4:]
    votes = judge_outcome(vlm, frames, task, 3)[0]
    c0 = int(np.mean(votes) > 0.5); c1 = int(np.mean(votes) == 1.0)
    sk = verdict([img(f) for f in sel] + [{"type": "text", "text": SKEPTIC.format(task=TASK_DESC[task])}])
    fr = [verdict([img(frames[0]), img(f), {"type": "text", "text": FRAME_Q.format(task=TASK_DESC[task])}]) for f in frames[-4:]]
    c2 = int(sum(fr) >= 3)
    return dict(C0_frozen_majority=c0, C1_unanimous=c1, C2_sustained_3of4_frames=c2, C3_skeptic=sk, C4_skeptic_and_C0=int(sk and c0), C5_sustained_and_C0=int(c2 and c0))


res = {}
t0 = time.time()
for task in ("hammer", "button"):
    fp = []
    for d in sorted(glob.glob(str(ROOT / f"runs/stage8/fp_diag/{task}_s2*/fp_examples"))):
        for ep in sorted({Path(f).name.split("_")[0] for f in glob.glob(d + "/ep*_judge0.jpg")}):
            fp.append([open(f"{d}/{ep}_judge{k}.jpg", "rb").read() for k in range(5)])
    bank = pickle.load(open(ROOT / f"results/vlm/bank_{task}.pkl", "rb"))
    pos = [e["frames16"] for e in bank if e["success"]][:80]; neg = [e["frames16"] for e in bank if not e["success"] and e["tag"] != "random"][:80]
    out = {}
    for name, grp in (("fp_training_exploits", fp), ("bank_successes", pos), ("bank_failures", neg)):
        r = vlm.pmap(lambda fr: candidates(fr, task), grp)
        out[name] = dict(n=len(grp), accept_rate={k: float(np.mean([x[k] for x in r])) for k in r[0]})
        print(task, name, out[name]["n"], {k: round(v, 2) for k, v in out[name]["accept_rate"].items()}, f"{time.time()-t0:.0f}s", flush=True)
    res[task] = out
    passes = {}
    for k in out["fp_training_exploits"]["accept_rate"]:
        rej = 1 - out["fp_training_exploits"]["accept_rate"][k]; acc = out["bank_successes"]["accept_rate"][k]
        passes[k] = dict(exploit_rejection=rej, success_acceptance=acc, passes_rule=bool(rej >= 0.70 and acc >= 0.85))
    res[task]["pass_rule"] = passes
    print(task, "PASS RULE:", {k: (round(v["exploit_rejection"], 2), round(v["success_acceptance"], 2), v["passes_rule"]) for k, v in passes.items()}, flush=True)
# scale effect of the frozen judge on all bank episodes (incl. unseen tasks): accuracy / precision / recall
from analyze_label_quality import metrics
res["frozen_judge_full_banks"] = {}
for task in ("hammer", "button", "dial", "peg"):
    bank = pickle.load(open(ROOT / f"results/vlm/bank_{task}.pkl", "rb"))
    out = vlm.pmap(lambda e: judge_outcome(vlm, e["frames16"], task, 3)[0], bank)
    V = np.array(out); y = np.array([e["success"] for e in bank]); q = V.mean(1)
    m = metrics((q > 0.5).astype(float), y); m["prevalence"] = float(y.mean()); res["frozen_judge_full_banks"][task] = m
    print(task, "full bank", "n", len(y), "acc", round(m["acc"], 3), "bal", round(m["balanced_acc"], 3), "prec", round(m["precision"], 3), "rec", round(m["recall"], 3), "fpr", round(m["fpr"], 3), f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(ROOT / f"results/vlm/screen_{tag}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved")
