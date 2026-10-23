#!/usr/bin/env python
"""B2 ground truth: empirical success probabilities of alternative first actions via simulator branching.

For each task, load the final actor of a stage-0 CoRe run, roll it out, snapshot the simulator at random steps, and for each of
7 candidate first actions (policy mean, 3 policy samples, 2 Gaussian-perturbed, 1 opposite) run K=10 independent continuations
(first action, then the stochastic policy) for `cont_len` steps.  Stores first-success time per continuation (inf if none), the true
next observation after the first action, and the state.  Never infers a counterfactual value from a single rollout.
"""
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import rrm_scripted  # noqa  (path setup)
from src.common.core_train import make_agent, step_env, reset_env
from src.common.branching import snapshot, restore
import utils
from omegaconf import OmegaConf

TASKS = {"button": "metaworld_button-press-topdown-v2", "hammer": "metaworld_hammer-v2", "peg": "metaworld_peg-insert-side-v2"}
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--run", default="runs/stage0/core/{t}_s0")
ap.add_argument("--n_states", type=int, default=60); ap.add_argument("--K", type=int, default=10)
ap.add_argument("--cont_len", type=int, default=60); ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(1)
env_name = TASKS[a.task]
cfg = OmegaConf.create({"env": env_name, "seed": a.seed, "device": "cpu"})
env = utils.make_metaworld_env(cfg)
agent = make_agent(cfg, env)
agent.actor.load_state_dict(torch.load(ROOT / a.run.format(t=a.task) / "actor.pt", map_location="cpu"))
agent.actor.eval()
rng = np.random.RandomState(a.seed)


def pol(o, sample):
    with torch.no_grad():
        d = agent.actor(torch.from_numpy(o.astype(np.float32))[None])
        x = d.sample() if sample else d.mean
    return x.clamp(-1, 1)[0].numpy()


S, A, T, NOBS, EP, GOAL = [], [], [], [], [], []
tries = 0
while len(S) < a.n_states and tries < 30 * a.n_states:
    tries += 1
    o = reset_env(env)
    t_branch = int(rng.randint(10, 300))
    sig = float(rng.choice([0.0, 0.3, 0.6, 1.0]))          # exploration noise in the pre-branch rollout -> diverse, partly misaligned states
    for _ in range(t_branch):
        o, r, d, info = step_env(env, np.clip(pol(o, True) + sig * rng.randn(4), -1, 1).astype(np.float32))
        if d or info["success"]: break
    if d or info["success"]:           # only PRE-success states: success persists once reached, which would make every action look good
        continue
    snap = snapshot(env)
    goal = np.array(env.unwrapped._target_pos).copy()
    pa = pol(o, False)
    cands = [pa] + [pol(o, True) for _ in range(3)] + [np.clip(pa + rng.randn(4) * 0.7, -1, 1) for _ in range(2)] + [-pa]
    ts = np.full((len(cands), a.K), np.inf, np.float32); nobs = []
    for ci, act0 in enumerate(cands):
        for k in range(a.K):
            restore(env, snap)
            o1, _, d, info = step_env(env, act0.astype(np.float32))
            if k == 0: nobs.append(o1.copy())
            first = 0 if info["success"] else np.inf
            oo = o1
            for t in range(1, a.cont_len):
                if np.isfinite(first) or d: break
                oo, _, d, info = step_env(env, pol(oo, True))
                if info["success"]: first = t
            ts[ci, k] = first
    S.append(o.copy()); A.append(np.stack(cands)); T.append(ts); NOBS.append(np.stack(nobs)); EP.append(t_branch); GOAL.append(goal)
    print(a.task, len(S), "emp p_succ(a_pi)=", float(np.isfinite(ts[0]).mean()), "range over actions", np.isfinite(ts).mean(1).round(2), flush=True)
out = a.out or str(ROOT / "results" / "facr" / f"branching_{a.task}.npz")
Path(out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, state=np.stack(S), actions=np.stack(A), first_succ=np.stack(T), true_next=np.stack(NOBS), t_branch=np.array(EP), goal=np.stack(GOAL), cont_len=a.cont_len)
print("saved", out)
