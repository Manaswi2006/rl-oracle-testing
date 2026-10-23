"""CoRe training loop (derived from upstream/CoRe/train_CoRe.py @ d663cab).

Changes w.r.t. upstream, all documented in REPRODUCIBILITY.md:
  * no hydra / no LLM client: FRM is the released final FRM (envs/frm_code), static;
  * RRM preferences come from a scripted+noisy teacher (src/common/rrm_scripted.py);
  * `method` selects the arm: core | rrm_only | env_dense | env_sparse | ext
    (ext = learned structural reward module replacing FRM, mixed like CoRe: (1-a)*ext + a*RRM);
  * separate evaluation env, final 50-episode evaluation, CSV/JSON logging, run-level resume;
  * RRM images are rendered at `image_size` (default 128, upstream 300) to fit 12+ parallel runs.
"""
import argparse
import importlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.common.rrm_scripted import ROOT, UP, ScriptedRewardModel  # noqa: F401  (sets sys.path)

ROOT, UP = Path(ROOT), Path(UP)
import utils  # upstream
import hydra
from replay_buffer import ReplayBuffer


def load_cfg(path, overrides):
    cfg = OmegaConf.create(yaml.safe_load(open(path)))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def make_agent(cfg, env):
    sac = OmegaConf.create(yaml.safe_load(open(UP / "config" / "agent" / "sac.yaml")))
    base = OmegaConf.create({"device": cfg.device})
    full = OmegaConf.merge(base, sac)
    for k in list(full.agent.params.keys()):          # YAML 1.1 parses '3e-4' as str unless omegaconf's resolver is active
        if k.endswith('_lr') and isinstance(full.agent.params[k], str):
            full.agent.params[k] = float(full.agent.params[k])
    full.agent.params.obs_dim = env.observation_space.shape[0]
    full.agent.params.action_dim = env.action_space.shape[0]
    full.agent.params.action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]
    return hydra.utils.instantiate(full.agent)


def step_env(env, action):
    out = env.step(action)
    if len(out) == 4:
        obs, r, done, info = out
    else:
        obs, r, term, trunc, info = out
        done = term or trunc
    return obs, r, done, info


def reset_env(env):
    o = env.reset()
    return o[0] if isinstance(o, tuple) else o


def goal_of(env):
    return np.array(env.unwrapped._target_pos).copy()


class Runner:
    def __init__(self, cfg, ext_factory=None):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        utils.set_seed_everywhere(cfg.seed)
        self.t_start = time.time()
        self.env = utils.make_metaworld_env(cfg)
        ecfg = OmegaConf.create({"env": cfg.env, "seed": cfg.seed + 1000})
        self.eval_env = utils.make_metaworld_env(ecfg)
        self.max_steps = self.env._max_episode_steps
        self.agent = make_agent(cfg, self.env)
        self.method = cfg.method
        self.rrm_alone = cfg.method in ("rrm_only", "rlf", "rrm_extra")
        self.use_rrm = cfg.method in ("core", "ext") or self.rrm_alone
        self.render = self.use_rrm
        S = cfg.image_size
        self.S = S
        self.replay = ReplayBuffer(self.env.observation_space.shape, self.env.action_space.shape,
                                   int(cfg.replay_buffer_capacity), cfg.device,
                                   image_size=S if self.render else 0, reward_alpha=0.0)
        if not self.render:  # keep the buffer API happy
            self.replay.images = None
        self.RRM = None
        if self.use_rrm:
            self.RRM = ScriptedRewardModel(
                self.env.observation_space.shape[0], self.env.action_space.shape[0],
                ensemble_size=cfg.ensemble_size, size_segment=cfg.segment, activation=cfg.activation,
                lr=cfg.RRM_lr, mb_size=cfg.RRM_batch, large_batch=cfg.large_batch,
                capacity=cfg.max_feedback + int(cfg.get("rlf", {}).get("extra_capacity", 0) if cfg.method == "rlf" else 0), env_name=cfg.env, log_dir=str(self.run_dir),
                image_height=S, image_width=S, resize_factor=1, resnet=0,
                conv_kernel_sizes=list(cfg.conv_kernel_sizes), conv_strides=list(cfg.conv_strides),
                conv_n_channels=list(cfg.conv_n_channels), eps_mistake=cfg.teacher_eps_mistake,
                rng_seed=cfg.seed)
        # structural reward -------------------------------------------------
        self.FRM = None
        self.ext = None
        if cfg.method == "core":
            mod = importlib.import_module(f"{cfg.env.replace('-', '_').lower()}_frm")
            self.FRM = mod.reward_function
        elif cfg.method == "ext":
            self.ext = ext_factory(cfg, self.env.observation_space.shape[0],
                                   self.env.action_space.shape[0], cfg.device, cfg.seed)
        self.rlf = None
        if cfg.method == "rlf":
            from src.falsification.rlf_ext import RLFModule
            self.rlf = RLFModule(self, cfg)
        if self.ext is not None and hasattr(self.ext, "bind"):
            self.ext.bind(self.agent)
        self.alpha = 0.0
        self.step = 0
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.train_rows, self.eval_rows = [], []
        self.n_success_train = 0
        self.first_success_episode = None
        self.first_success_step = None
        self.n_episodes = 0
        self.ext_relabel_time = 0.0
        N = int(cfg.num_train_steps) + 2
        D, A = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.traj = dict(obs=np.zeros((N, D), np.float32), act=np.zeros((N, A), np.float32),
                         env_r=np.zeros(N, np.float32), succ=np.zeros(N, np.int8),
                         ep=np.zeros(N, np.int32), t=np.zeros(N, np.int16), goal=np.zeros((N, 3), np.float32))
        OmegaConf.save(cfg, str(self.run_dir / "cfg.yaml"))

    # ------------------------------------------------------------------ eval
    def evaluate(self, n=None, seed_offset=0):
        n = n or self.cfg.num_eval_episodes
        succ, ret = 0.0, 0.0
        for ep in range(n):
            obs = reset_env(self.eval_env)
            done, es = False, 0.0
            while not done:
                with utils.eval_mode(self.agent):
                    a = self.agent.act(obs, sample=False)
                obs, r, done, info = step_env(self.eval_env, a)
                ret += r
                es = max(es, info["success"])
            succ += es
        return 100.0 * succ / n, ret / n

    # ------------------------------------------------------------ RRM stuff
    def learn_rrm(self):
        self.RRM.uniform_sampling()
        self.total_feedback += self.RRM.mb_size
        self.labeled_feedback = self.RRM.n_labels_kept
        acc = 0.0
        if self.RRM.buffer_index > 0 or self.RRM.buffer_full:
            for _ in range(self.cfg.RRM_update):
                self.RRM.train()
                acc = float(np.mean(self.RRM.train_reward()))
                if acc > 0.97:
                    break
        return acc

    def relabel_rrm(self):
        self.RRM.eval()
        self.replay.relabel_with_predictor(self.RRM, self.step)
        self.RRM.train()

    def relabel_ext(self):
        t0 = time.time()
        n = len(self.replay)
        if getattr(self.ext, "static_reward", False):      # per-step reward stored at collection time; only re-mix with the RRM
            a = self.alpha
            self.replay.rewards[:n] = (1 - a) * self.replay.rewards_frm[:n] + a * self.replay.rewards_rrm[:n]
            return
        B = 8192
        for i in range(0, n, B):
            j = min(i + B, n)
            r = self.ext.reward_batch(self.replay.obses[i:j], self.replay.actions[i:j], self.replay.next_obses[i:j])
            self.replay.rewards_frm[i:j, 0] = np.clip(r, -1, 1)
        a = self.alpha
        self.replay.rewards[:n] = (1 - a) * self.replay.rewards_frm[:n] + a * self.replay.rewards_rrm[:n]
        self.ext_relabel_time += time.time() - t0

    # ------------------------------------------------------------------ main
    def run(self):
        cfg = self.cfg
        done, ep_succ, ep_ret, ep_id, ep_t = True, 0.0, 0.0, -1, 0
        pre_end = cfg.num_seed_steps + cfg.num_pre_steps
        interact_count = 0
        rrm_acc = 0.0
        next_eval = cfg.eval_frequency
        goal = None
        ep_rr = deque(maxlen=10)
        while self.step <= cfg.num_train_steps:
            if done:
                if ep_id >= 0:
                    if ep_succ > 0 and self.first_success_episode is None:
                        self.first_success_episode, self.first_success_step = ep_id, self.step
                    self.n_success_train += int(ep_succ > 0)
                    self.n_episodes += 1
                    if self.rlf is not None:
                        self.rlf.on_task_episode_end(ep_id, ep_t, ep_succ)
                    if self.ext is not None:
                        self.ext.on_episode_end(ep_id, ep_succ, self.step)
                    self.train_rows.append(dict(step=self.step, episode=ep_id, true_return=ep_ret, success=ep_succ,
                                                labels=self.labeled_feedback, rrm_acc=rrm_acc))
                if self.step >= next_eval:
                    sr, rr = self.evaluate()
                    self.eval_rows.append(dict(step=self.step, success_rate=sr, true_return=rr, n_eval=cfg.num_eval_episodes,
                                               wall=time.time() - self.t_start))
                    print(f"[{cfg.run_id}] step {self.step} eval SR {sr:.1f} ret {rr:.1f} "
                          f"sps {self.step / (time.time() - self.t_start):.1f}", flush=True)
                    self._dump()
                    next_eval += cfg.eval_frequency
                obs = reset_env(self.env)
                goal = goal_of(self.env)
                done, ep_succ, ep_ret, ep_t = False, 0.0, 0.0, 0
                ep_id += 1

            if self.step < cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # ---- RRM / reward-model schedule (as upstream) ----------------
            if self.use_rrm and self.step == pre_end:
                self.alpha = 1.0 if self.rrm_alone else cfg.reward_alpha
                self.replay.reward_alpha = self.alpha
                rrm_acc = self.learn_rrm()
                self.relabel_rrm()
                if self.ext is not None:
                    self.relabel_ext()
                self.agent.reset_critic()
                self.agent.update_after_reset(self.replay, _NullLogger(), self.step,
                                              gradient_update=cfg.reset_update, policy_update=True)
                interact_count = 0
            elif self.use_rrm and self.step > pre_end and self.total_feedback < cfg.max_feedback \
                    and interact_count == cfg.num_interact:
                if self.RRM.mb_size + self.total_feedback > cfg.max_feedback:
                    self.RRM.set_batch(cfg.max_feedback - self.total_feedback)
                if self.rlf is not None:
                    self.rlf.repair(self.step)
                rrm_acc = self.learn_rrm()
                self.relabel_rrm()
                interact_count = 0

            # ---- structural-reward module update --------------------------
            if self.ext is not None and self.step >= cfg.num_seed_steps and self.ext.maybe_update(self.step):
                self.relabel_ext()

            if self.step >= cfg.num_seed_steps:
                self.agent.update(self.replay, _NullLogger(), self.step, 1)

            next_obs, env_r, done, info = step_env(self.env, action)
            success = float(info["success"])
            img = None
            if self.render:
                rgb = self.env.render()[::-1, :, :]
                if "drawer" in cfg.env or "sweep" in cfg.env:
                    rgb = rgb[100:400, 100:400, :]
                import cv2
                img = cv2.resize(rgb, (self.S, self.S))

            # ---- reward assembly -----------------------------------------
            r_frm, r_rrm = 0.0, 0.0
            if cfg.method == "env_dense":
                reward_hat = float(env_r)
            elif cfg.method == "env_sparse":
                reward_hat = success
            else:
                if self.method == "core":
                    r_frm = float(np.clip(self.FRM(next_obs, action, goal)[0], -1, 1))
                elif self.method == "ext":
                    if getattr(self.ext, "static_reward", False):
                        r_frm = float(self.ext.step_reward(success))
                    else:
                        r_frm = float(np.clip(self.ext.reward_batch(obs[None], action[None], next_obs[None])[0], -1, 1))
                if self.step >= pre_end:
                    im = img.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
                    self.RRM.eval()
                    r_rrm = float(self.RRM.r_hat(im))
                    self.RRM.train()
                r_frm_w, r_rrm_w = (1 - self.alpha) * r_frm, self.alpha * r_rrm
                reward_hat = r_frm_w + r_rrm_w
            ep_ret += env_r
            ep_succ = max(ep_succ, success)

            done_f = float(done)
            done_no_max = 0 if ep_t + 1 == self.max_steps else done_f
            if cfg.method in ("env_dense", "env_sparse"):
                self._add_noimg(obs, action, reward_hat, next_obs, done_f, done_no_max)
            else:
                self.replay.add(obs, action, reward_hat, (r_frm if getattr(self.ext, "static_reward", False) else r_frm_w), r_rrm_w, next_obs, done_f, done_no_max, image=img)
                if self.total_feedback < cfg.max_feedback:
                    self.RRM.add_data(obs, action, env_r, done, img=img)
            if self.rlf is not None:
                self.rlf.on_task_step(ep_id, self.step)
                self.rlf.maybe_attack(self.step)
            tr = self.traj
            tr["obs"][self.step], tr["act"][self.step], tr["env_r"][self.step] = obs, action, env_r
            tr["succ"][self.step], tr["ep"][self.step], tr["t"][self.step] = int(success), ep_id, ep_t
            tr["goal"][self.step] = goal
            if self.ext is not None:
                self.ext.on_step(ep_id, ep_t, obs, action, next_obs, env_r, success, self.step)

            obs = next_obs
            ep_t += 1
            self.step += 1
            interact_count += 1

        sr50, rr50 = self.final_eval()
        self._dump(final=dict(final_success_rate=sr50, final_true_return=rr50))
        return sr50

    def _add_noimg(self, obs, action, r, next_obs, done, dnm):
        rb = self.replay
        np.copyto(rb.obses[rb.idx], obs)
        np.copyto(rb.actions[rb.idx], action)
        rb.rewards[rb.idx] = r
        np.copyto(rb.next_obses[rb.idx], next_obs)
        rb.not_dones[rb.idx] = not done
        rb.not_dones_no_max[rb.idx] = not dnm
        rb.idx = (rb.idx + 1) % rb.capacity
        rb.full = rb.full or rb.idx == 0

    def final_eval(self):
        out = self.evaluate(n=self.cfg.num_final_eval_episodes)
        self.rrm_diag = self.diag_rrm() if (self.use_rrm and self.cfg.diag_episodes > 0) else None
        return out

    @torch.no_grad()
    def diag_rrm(self):
        """C1: how well does the learned reward separate real success from failure under the final policy?"""
        import cv2
        n, S, rows = self.cfg.diag_episodes, self.S, []
        self.RRM.eval()
        for _ in range(n):
            obs = reset_env(self.eval_env)
            imgs, succ, done = [], 0.0, False
            while not done:
                with utils.eval_mode(self.agent):
                    a = self.agent.act(obs, sample=False)
                obs, _, done, info = step_env(self.eval_env, a)
                succ = max(succ, float(info["success"]))
                rgb = self.eval_env.render()[::-1, :, :]
                if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                    rgb = rgb[100:400, 100:400, :]
                imgs.append(cv2.resize(rgb, (S, S)))
            x = np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            sc = np.concatenate([self.RRM.r_hat_batch(x[i:i + 250]).reshape(-1) for i in range(0, len(x), 250)])
            cs = np.concatenate([[0.0], np.cumsum(sc)])
            win = (cs[20:] - cs[:-20]).max() if len(sc) >= 20 else float(sc.sum())
            rows.append(dict(success=succ, mean_R=float(sc.mean()), max_window_R=float(win)))
        self.RRM.train()
        y = np.array([r["success"] for r in rows]); w = np.array([r["max_window_R"] for r in rows])
        auc = None
        if 0 < y.sum() < len(y):
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, w))
        hi = w >= np.median(w)
        return dict(episodes=rows, auc_window_vs_success=auc, frac_fail_among_high_reward=float((y[hi] == 0).mean()),
                    success_rate=float(y.mean()))

    # ------------------------------------------------------------- logging
    def _dump(self, final=None):
        import csv
        for name, rows in (("train.csv", self.train_rows), ("eval.csv", self.eval_rows)):
            if rows:
                with open(self.run_dir / name, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
        if final is not None:
            wall = time.time() - self.t_start
            res = dict(run_id=self.cfg.run_id, method=self.cfg.method, ext=self.cfg.get("ext_name", None), env=self.cfg.env,
                       seed=self.cfg.seed, steps=self.step - 1, wall_hours=wall / 3600.0,
                       gpu_hours=wall / 3600.0, n_success_train=self.n_success_train, n_train_episodes=self.n_episodes,
                       first_success_episode=self.first_success_episode, first_success_step=self.first_success_step,
                       teacher_queries=(self.RRM.n_queries if self.RRM else 0),
                       labels_kept=(self.RRM.n_labels_kept if self.RRM else 0),
                       teacher_label_flip_rate=(self.RRM.n_flipped / max(self.RRM.n_labels_kept, 1) if self.RRM else None),
                       ext_stats=(self.ext.stats() if self.ext else (self.rlf.stats() if self.rlf else None)), ext_relabel_seconds=self.ext_relabel_time, rrm_diag=getattr(self, "rrm_diag", None), **final)
            if self.ext:
                res["oracle_information_used"] = self.ext.oracle_information_used
            json.dump(res, open(self.run_dir / "result.json", "w"), indent=1, default=float)
            n = self.step
            np.savez_compressed(self.run_dir / "trajectories.npz", obs=self.traj["obs"][:n],
                                act=self.traj["act"][:n], env_r=self.traj["env_r"][:n],
                                succ=self.traj["succ"][:n], ep=self.traj["ep"][:n], t=self.traj["t"][:n], goal=self.traj["goal"][:n])
            torch.save(self.agent.actor.state_dict(), self.run_dir / "actor.pt")
            torch.save(self.agent.critic.state_dict(), self.run_dir / "critic.pt")
            if self.RRM:
                for m in range(self.RRM.de):
                    torch.save(self.RRM.ensemble[m].state_dict(), self.run_dir / f"rrm_{m}.pt")
                if self.RRM.label_log:
                    np.savez_compressed(self.run_dir / "preference_labels.npz",
                                        sa1=np.concatenate([x[0] for x in self.RRM.label_log]).astype(np.float16),
                                        sa2=np.concatenate([x[1] for x in self.RRM.label_log]).astype(np.float16),
                                        label=np.concatenate([x[2] for x in self.RRM.label_log]),
                                        clean=np.concatenate([x[3] for x in self.RRM.label_log]))
            if self.ext:
                self.ext.save(str(self.run_dir))
            (self.run_dir / "DONE").write_text(time.strftime("%F %T"))


class _NullLogger:
    def log(self, *a, **k):
        pass


def main(argv=None, ext_registry=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "common.yaml"))
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.cfg, args.overrides)
    if (Path(cfg.run_dir) / "DONE").exists():
        print("already done:", cfg.run_dir)
        return
    factory = None
    if cfg.method == "ext":
        from src.common.registry import get_ext
        factory = get_ext(cfg.ext_name)
    Runner(cfg, factory).run()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
