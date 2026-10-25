#!/usr/bin/env python
"""Small, documented prompt/input sweep on a DEV half of the bank (even episode ids); test half (odd ids) is never used for selection.
variants: frame subset (all16 / first+last / last4+first), answer mode (direct yes/no, chain-of-thought then verdict)."""
import argparse, json, re, sys, time, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm.vlm_client import VLM, TASK_DESC, b64

ap = argparse.ArgumentParser(); ap.add_argument("--bank", default="bank"); ap.add_argument("--split", default="dev"); ap.add_argument("--n", type=int, default=160); ap.add_argument("--ports", default="8000")
ap.add_argument("--model", default="qwen3vl8b"); ap.add_argument("--tag", default="")
a = ap.parse_args()
vlm = VLM(ports=[int(p) for p in a.ports.split(",")], model=a.model, workers=32)
GEN = ("The robot's instruction was: {task}. {frames_desc} First describe in two short sentences what is visible in the first and in the last frame "
       "(position of the objects relevant to the instruction), then decide whether the instruction has been completely carried out by the end. "
       "Finish with a final line exactly 'VERDICT: Yes' or 'VERDICT: No'.")
VARIANTS = {
    "cot_all16": dict(frames="all", mode="cot", q=GEN.replace("{frames_desc}", "You see {n} frames in time order.")),
    "cot_last8": dict(frames="l8", mode="cot", q=GEN.replace("{frames_desc}", "Image 1 is the first frame; images 2-8 are the last seven frames in time order.")),
    "cot_first_last": dict(frames="fl", mode="cot", q=GEN.replace("{frames_desc}", "Image 1 is the first frame and image 2 the last frame.")),
    "cot_first_last4": dict(frames="fl4", mode="cot", q=GEN.replace("{frames_desc}", "Image 1 is the first frame; images 2-5 are the last four frames in time order.")),
}
out = {}
for task in ("button", "hammer"):
    bank = pickle.load(open(ROOT / f"results/vlm/{a.bank}_{task}.pkl", "rb"))
    rs = np.random.RandomState(0); perm = rs.permutation(len(bank)); half = len(bank) // 2
    pool = perm[:half] if a.split == "dev" else perm[half:]
    pos = [int(i) for i in pool if bank[i]["success"]][: a.n // 2]; neg = [int(i) for i in pool if not bank[i]["success"]][: a.n // 2]
    ids = pos + neg                       # class-balanced evaluation set (labels select the evaluation set only)
    for vn, v in VARIANTS.items():
        def run(i):
            fr = bank[i]["frames16"]
            sel = {"all": fr, "fl": [fr[0], fr[-1]], "fl4": [fr[0]] + fr[-4:], "l8": [fr[0]] + fr[-7:]}[v["frames"]]
            content = [{"type": "image_url", "image_url": {"url": b64(f)}} for f in sel] + [{"type": "text", "text": v["q"].format(task=TASK_DESC[task], n=len(sel))}]
            ch = vlm._chat(content, max_tokens=(2 if v["mode"] == "direct" else 220), logprobs=(v["mode"] == "direct"))
            if v["mode"] == "direct":
                p, tok = vlm._p(ch, "yes", "no"); return dict(i=i, vote=int(tok.startswith("yes")), p=p)
            txt = ch.message.content or ""; m = re.findall(r"VERDICT:\s*(Yes|No)", txt, re.I)
            return dict(i=i, vote=int(bool(m) and m[-1].lower() == "yes"), p=float(bool(m) and m[-1].lower() == "yes"), text=txt[-160:])
        t0 = time.time(); res = vlm.pmap(run, ids)
        y = np.array([bank[r["i"]]["success"] for r in res]); v_ = np.array([r["vote"] for r in res])
        tp = ((v_ == 1) & (y == 1)).sum(); tn = ((v_ == 0) & (y == 0)).sum(); rec = tp / max(y.sum(), 1); spec = tn / max((1 - y).sum(), 1)
        out[f"{task}|{vn}"] = dict(n=len(y), prevalence=float(y.mean()), acc=float((v_ == y).mean()), recall=float(rec), specificity=float(spec), balanced_acc=float((rec + spec) / 2), secs=round(time.time() - t0, 1), examples=[r.get("text") for r in res[:2]])
        print(f"{task:7s} {vn:20s} n={len(y)} acc {out[f'{task}|{vn}']['acc']:.3f} bal {out[f'{task}|{vn}']['balanced_acc']:.3f} recall {rec:.2f} spec {spec:.2f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(ROOT / f"results/vlm/prompt_sweep_{a.split}{a.tag}.json", "w"), indent=1)
