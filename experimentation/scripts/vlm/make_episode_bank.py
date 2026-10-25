#!/usr/bin/env python
"""Episode bank with ground truth for VLM label-quality tests.

For every actor checkpoint (earlier study) and each exploration-noise level, roll out episodes; store
  * 16 uniformly spaced frames per episode (JPEG, 256 px, same camera/flip as CoRe) for the OUTCOME question,
  * 8 random 20-step windows per episode, 6 frames each, with the summed environment reward and success flags (for the PREFERENCE question),
  * ground truth: episode success (any), first-success step, return.
Ground truth is used only to *evaluate* the VLM labels (and, in the policy-learning study, never shown to the training pipeline).
usage: make_episode_bank.py --task button|hammer --out results/vlm/bank_<task>.pkl
"""
import argparse, glob, io, os, pickle, sys
from pathlib import Path
import numpy as np, torch, cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import rrm_scripted  # noqa (path setup)
from src.common.core_train import make_agent, step_env, reset_env
import utils
from omegaconf import OmegaConf

TASKS = {"button": "metaworld_button-press-topdown-v2", "hammer": "metaworld_hammer-v2", "dial": "metaworld_dial-turn-v2", "peg": "metaworld_peg-insert-side-v2"}
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--eps_per_cfg", type=int, default=3); ap.add_argument("--n_random", type=int, default=60); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--size", type=int, default=256); ap.add_argument("--n_segs", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(1)
cfg = OmegaConf.create({"env": TASKS[a.task], "seed": a.seed, "device": "cpu"})
env = utils.make_metaworld_env(cfg)
agent = make_agent(cfg, env)
rng = np.random.RandomState(a.seed)
S = a.size


def jpeg(img):
    img = cv2.resize(img[::-1, :, :], (S, S))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def rollout(policy_fn, tag):
    o = reset_env(env)
    T = env._max_episode_steps
    frame_steps = set(np.linspace(0, T - 1, 16).round().astype(int).tolist())
    segs = [int(x) for x in rng.randint(0, T - 20, size=a.n_segs)]
    seg_steps = {s0: [s0 + int(k) for k in np.linspace(0, 19, 6).round()] for s0 in segs}
    need = set(frame_steps)
    for v in seg_steps.values(): need |= set(v)
    frames, rews, succ = {}, np.zeros(T, np.float32), np.zeros(T, np.int8)
    for t in range(T):
        if t in need:
            frames[t] = jpeg(env.render())
        o, r, d, info = step_env(env, policy_fn(o))
        rews[t], succ[t] = r, info["success"]
        if d: break
    L = t + 1
    first = int(np.argmax(succ)) if succ.any() else -1
    ep = dict(tag=tag, task=a.task, length=L, success=int(succ.any()), first_success=first, ret=float(rews.sum()),
              frames16=[frames[t] for t in sorted(frame_steps) if t in frames], frame_steps=sorted(frame_steps),
              segments=[dict(start=s0, frames=[frames[t] for t in seg_steps[s0] if t in frames], env_reward_sum=float(rews[s0:s0 + 20].sum()),
                             success_in=int(succ[s0:s0 + 20].any()), succ_before=int(succ[:s0].any())) for s0 in segs])
    return ep


bank = []
files = sorted(glob.glob(str(ROOT / f"checkpoints/actors/*/{a.task}_s*.pt")))
for f in files:
    agent.actor.load_state_dict(torch.load(f, map_location="cpu")); agent.actor.eval()
    name = f"{Path(f).parent.name}/{Path(f).stem}"

    def mk(sig, sample):
        def pol(o):
            with torch.no_grad():
                d = agent.actor(torch.from_numpy(o.astype(np.float32))[None])
                x = d.sample() if sample else d.mean
            return np.clip(x[0].numpy() + sig * rng.randn(4), -1, 1).astype(np.float32)
        return pol
    for sig, sample in ((0.0, False), (0.0, True), (0.4, False), (0.8, False)):
        for _ in range(a.eps_per_cfg):
            bank.append(rollout(mk(sig, sample), f"{name}|sig{sig}|{'samp' if sample else 'mean'}"))
    print(a.task, name, len(bank), "succ frac", np.mean([e["success"] for e in bank]).round(2), flush=True)
for _ in range(a.n_random):
    bank.append(rollout(lambda o: env.action_space.sample().astype(np.float32), "random"))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pickle.dump(bank, open(a.out, "wb"))
print("saved", a.out, len(bank), "episodes; success rate", np.mean([e["success"] for e in bank]))
