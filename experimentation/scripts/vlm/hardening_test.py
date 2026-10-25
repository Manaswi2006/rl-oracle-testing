#!/usr/bin/env python
"""Can a more skeptical, task-agnostic judge reject the exploits the policy found (false positives collected during training), without losing true successes?
Variants are fixed a priori (not tuned): H0 frozen judge (majority of 3), H1 unanimity of the 3 paraphrases, H2 'skeptic' prompt (look for evidence AGAINST success),
H3 'change check' prompt (describe the change first->last and whether it is exactly what the instruction asks), H4 skeptic prompt + unanimity with H0.
Evaluated on (a) false-positive training episodes (all true failures that the frozen judge accepted), (b) bank successes and (c) bank failures.  usage: hardening_test.py hammer button"""
import glob, json, pickle, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm.vlm_client import VLM, TASK_DESC, b64, COT_OUTCOME, judge_outcome
vlm = VLM(workers=32)
SKEPTIC = ("The robot's instruction was: {task}. Image 1 is the first frame; images 2-5 are the last four frames of its attempt, in time order. Act as a skeptical inspector. First list in two short sentences any visible evidence "
           "that the instruction has NOT been completely carried out (objects still in their starting state, objects not in the required final position, the robot not in contact with the relevant object, ...). "
           "Then answer 'VERDICT: Yes' only if you found no such evidence, otherwise 'VERDICT: No'. The last line must be exactly 'VERDICT: Yes' or 'VERDICT: No'.")
CHANGE = ("Instruction for the robot: {task}. Image 1 is the scene before the attempt; images 2-5 are the last four frames of the attempt. In two sentences describe exactly which objects changed between image 1 and image 5, "
          "then state whether that change is exactly what the instruction asks for. Final line exactly 'VERDICT: Yes' (it is exactly what is asked) or 'VERDICT: No'.")


def ask(frames, task, q):
    sel = [frames[0]] + frames[-4:]
    content = [{"type": "image_url", "image_url": {"url": b64(f)}} for f in sel] + [{"type": "text", "text": q.format(task=TASK_DESC[task])}]
    t = vlm._chat(content, max_tokens=220, logprobs=False).message.content or ""
    m = re.findall(r"VERDICT:\s*(Yes|No)", t, re.I)
    return int(bool(m) and m[-1].lower() == "yes")


def variants(frames, task):
    base = judge_outcome(vlm, frames, task, 3)[0]
    sk, ch = ask(frames, task, SKEPTIC), ask(frames, task, CHANGE)
    return dict(H0_majority=int(np.mean(base) > 0.5), H1_unanimous3=int(np.mean(base) == 1.0), H2_skeptic=sk, H3_change=ch, H4_skeptic_and_H0=int(sk and np.mean(base) > 0.5), H5_skeptic_and_change_and_H0=int(sk and ch and np.mean(base) > 0.5))


res = {}
for task in sys.argv[1:]:
    fp = []
    for d in sorted(glob.glob(str(ROOT / f"runs/stage8/fp_diag/{task}_s2*/fp_examples"))):
        for ep in sorted({Path(f).name.split("_")[0] for f in glob.glob(d + "/ep*_judge0.jpg")}):
            fp.append([open(f"{d}/{ep}_judge{k}.jpg", "rb").read() for k in range(5)])
    bank = pickle.load(open(ROOT / f"results/vlm/bank_{task}.pkl", "rb"))
    pos = [e["frames16"] for e in bank if e["success"]][:80]; neg = [e["frames16"] for e in bank if not e["success"] and e["tag"] != "random"][:80]
    def run(group):
        # fp examples have exactly 5 frames; bank groups have 16 -> variants() uses first + last four in both cases
        return vlm.pmap(lambda fr: variants(fr if len(fr) == 5 else fr, task), group)
    out = {}
    for name, grp in (("fp_training_exploits", fp), ("bank_successes", pos), ("bank_failures", neg)):
        r = run(grp) if grp else []
        out[name] = dict(n=len(grp), accept_rate={k: float(np.mean([x[k] for x in r])) for k in (r[0] if r else {})})
    res[task] = out
    print(task, "n fp exploits", len(fp))
    for name, o in out.items(): print("  ", name, o["n"], {k: round(v, 2) for k, v in o["accept_rate"].items()})
json.dump(res, open(ROOT / "results/vlm/hardening_test.json", "w"), indent=1)
