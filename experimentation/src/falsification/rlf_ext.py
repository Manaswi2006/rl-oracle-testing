"""Reward Learning through Falsification (RLF) + matched counterexample-selection baselines.

Setting: the learned reward is CoRe's RRM (image reward ensemble trained on simulated-teacher preferences).
Every RRM round (every `num_interact` task steps) a *selector* proposes Q candidate episodes for verification
by the (oracle: simulator success flag) verifier.  A verified failure that the reward model scores in the
top half of the task pool is a *confirmed reward failure* (the reward considers a 20-step window good,
but the task was not completed).  Repair: a preference pair  (window ending at first success of a verified
successful episode)  >  (window of the confirmed failure with the highest predicted reward)  is added to
the RRM preference buffer, which is then retrained as in CoRe.

selector:
  onpolicy     Q random episodes from the last `pool` task-policy episodes (ordinary on-policy correction)
  random       Q episodes of a uniformly random policy (extra env steps Q*500 per round)
  uncertainty  Q task-pool episodes with the largest RRM ensemble disagreement on their top window
  highreward   Q task-pool episodes with the highest predicted reward window (== standard hard-negative
               mining when combined with the verifier: the highest-reward episodes are checked)
  disagreement Q task-pool episodes maximising R_theta-window-score among those the verifier rejects
               (verifier evaluated on the WHOLE pool for free: oracle upper bound, reported as such)
  attack       falsification policy pi_attack (separate SAC, own env instance) maximising
               R_theta(s) - lam*C(s) per step, C = oracle success flag; Q top-scoring attack episodes verified
  hybrid       attack + beta*ensemble-disagreement bonus in the attack reward

Accounting (reported in result.json): extra env steps (attack/random), verifier evaluations used for training
the attack policy (attack arm: one per attack env step, free oracle) and queried for repair (Q per round).
"""
import numpy as np
import torch

PAD = 20


def cells(obs, cs=0.03):
    p = np.concatenate([obs[:, 0:3], obs[:, 4:7]], 1)
    return set(map(tuple, np.floor(p / cs).astype(np.int32)))


class RLFModule:
    def __init__(self, runner, cfg):
        from src.common.core_train import make_agent
        self.r = runner
        self.cfg = cfg
        p = dict(selector="onpolicy", Q=5, pool=60, attack_ratio=0.5, lam=1.0, beta_unc=1.0, att_pool=8,
                 att_capacity=100000, att_seed_steps=1000)
        p.update(dict(cfg.get("rlf", {}) or {}))
        self.p = p
        self.sel = p["selector"]
        self.rng = np.random.RandomState(cfg.seed + 77)
        self.ep_start, self.ep_len, self.ep_succ = {}, {}, {}
        self.verified_pos = []         # (kind, start, len, first_succ_t) of verified successful episodes
        self.log = []
        self.extra_env_steps = 0
        self.verifier_free_evals = 0
        self.verifier_queries = 0
        self.anchor_queries = 0
        self.n_conf_fail = 0
        self.n_pairs = 0
        self.attack_stats = []
        self.cur_ep = None
        self.is_attack = self.sel in ("attack", "hybrid")
        if self.is_attack or self.sel == "random":
            import utils
            from omegaconf import OmegaConf
            self.env = utils.make_metaworld_env(OmegaConf.create({"env": cfg.env, "seed": cfg.seed + 2000}))
        if self.is_attack:
            from replay_buffer import ReplayBuffer
            self.agent = make_agent(cfg, self.env)
            S = cfg.image_size
            self.rb = ReplayBuffer(self.env.observation_space.shape, self.env.action_space.shape, p["att_capacity"], cfg.device,
                                   image_size=S, reward_alpha=1.0)
            self.att = dict(obs=None, done=True, steps=0, ep=-1, start=0, t=0, succ=0.0, ret_r=0.0, n_eps=0)
            self.att_eps = []          # (start, len, succ, mean_R)
            self.att_pen = np.zeros(p["att_capacity"], np.float32)
        self.acc = 0.0

    # ---------------------------------------------------------------- task-policy bookkeeping
    def on_task_step(self, ep_id, step_idx):
        if ep_id not in self.ep_start:
            self.ep_start[ep_id] = step_idx

    def on_task_episode_end(self, ep_id, length, succ):
        self.ep_len[ep_id], self.ep_succ[ep_id] = length, succ

    # ---------------------------------------------------------------- reward-model access
    @torch.no_grad()
    def member_scores(self, imgs):
        """imgs uint8 [L,H,W,3] -> [3,L] per-member predicted reward."""
        rrm = self.r.RRM
        rrm.eval()
        out = [[] for _ in range(rrm.de)]
        for i in range(0, len(imgs), 250):
            x = imgs[i:i + 250].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            for m in range(rrm.de):
                out[m].append(rrm.r_hat_member(x, member=m).detach().cpu().numpy().reshape(-1))
        rrm.train()
        return np.stack([np.concatenate(o) for o in out])

    @staticmethod
    def best_window(sc):
        """sc [3,L] -> (start, window score mean-over-members, disagreement at that window)."""
        L = sc.shape[1]
        if L < PAD:
            return 0, float(sc.mean() * PAD), 0.0
        cs = np.concatenate([np.zeros((sc.shape[0], 1)), np.cumsum(sc, 1)], 1)
        win = cs[:, PAD:] - cs[:, :-PAD]                      # [3, L-PAD+1]
        wm = win.mean(0)
        k = int(wm.argmax())
        return k, float(wm[k]), float(win[:, k].std())

    def task_imgs(self, ep):
        s, L = self.ep_start[ep], self.ep_len[ep]
        return self.r.replay.images[s:s + L]

    def pool_info(self, eps, imgs_fn):
        info = []
        for e in eps:
            sc = self.member_scores(imgs_fn(e))
            k, w, u = self.best_window(sc)
            info.append(dict(ep=e, start=k, score=w, unc=u))
        return info

    # ---------------------------------------------------------------- attack policy
    def attack_step(self, step):
        from src.common.core_train import reset_env, step_env, _NullLogger
        import cv2
        a, rb, S = self.att, self.rb, self.cfg.image_size
        if a["done"]:
            if a["ep"] >= 0:
                self.att_eps.append((a["start"], a["t"], a["succ"], a["ret_r"] / max(a["t"], 1)))
            a.update(obs=reset_env(self.env), done=False, ep=a["ep"] + 1, start=rb.idx, t=0, succ=0.0, ret_r=0.0)
        if a["steps"] < self.p["att_seed_steps"]:
            act = self.env.action_space.sample()
        else:
            import utils
            with utils.eval_mode(self.agent):
                act = self.agent.act(a["obs"], sample=True)
            self.agent.update(rb, _NullLogger(), a["steps"], 1)
        nobs, r_env, done, info = step_env(self.env, act)
        succ = float(info["success"])
        rgb = self.env.render()[::-1, :, :]
        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
            rgb = rgb[100:400, 100:400, :]
        img = cv2.resize(rgb, (S, S))
        sc = self.member_scores(img[None])[:, 0]
        R, unc = float(sc.mean()), float(sc.std())
        pen = self.p["lam"] * succ
        bonus = self.p["beta_unc"] * unc if self.sel == "hybrid" else 0.0
        reward = R - pen + bonus
        done_f = float(done)
        dnm = 0 if a["t"] + 1 == self.env._max_episode_steps else done_f
        self.att_pen[rb.idx] = -pen
        rb.add(a["obs"], act, reward, -pen + bonus, R, nobs, done_f, dnm, image=img)
        self.verifier_free_evals += 1
        self.extra_env_steps += 1
        a.update(obs=nobs, done=bool(done), steps=a["steps"] + 1, t=a["t"] + 1, succ=max(a["succ"], succ), ret_r=a["ret_r"] + R)

    def maybe_attack(self, step):
        if not self.is_attack or step < self.r.cfg.num_seed_steps + self.r.cfg.num_pre_steps:
            return
        self.acc += self.p["attack_ratio"]
        while self.acc >= 1.0:
            self.acc -= 1.0
            self.attack_step(step)

    @torch.no_grad()
    def relabel_attack(self):
        rb = self.rb
        n = len(rb)
        for i in range(0, n, 500):
            j = min(i + 500, n)
            sc = self.member_scores(rb.images[i:j])
            R = sc.mean(0)
            bonus = self.p["beta_unc"] * sc.std(0) if self.sel == "hybrid" else 0.0
            rb.rewards_rrm[i:j, 0] = R
            rb.rewards[i:j, 0] = R + self.att_pen[i:j] + bonus

    # ---------------------------------------------------------------- selection + repair
    def pool_episodes(self):
        done = sorted(self.ep_len.keys())
        return done[-self.p["pool"]:]

    def repair(self, step):
        r, p = self.r, self.p
        Q = p["Q"]
        task_pool = self.pool_episodes()
        if len(task_pool) < 5:
            return
        tinfo = self.pool_info(task_pool, self.task_imgs)
        thr = float(np.median([i["score"] for i in tinfo]))       # common reference threshold for 'predicted good'
        cands = []                                                # (kind, info, imgs_fn, succ_fn)
        if self.sel == "onpolicy":
            pick = self.rng.choice(len(tinfo), size=min(Q, len(tinfo)), replace=False)
            cands = [("task", tinfo[i]) for i in pick]
        elif self.sel == "highreward":
            order = np.argsort([-i["score"] for i in tinfo])[:Q]
            cands = [("task", tinfo[i]) for i in order]
        elif self.sel == "uncertainty":
            order = np.argsort([-i["unc"] for i in tinfo])[:Q]
            cands = [("task", tinfo[i]) for i in order]
        elif self.sel == "disagreement":
            self.verifier_free_evals += len(tinfo)
            fails = [i for i in tinfo if self.ep_succ[i["ep"]] <= 0]
            fails.sort(key=lambda i: -i["score"])
            cands = [("task", i) for i in fails[:Q]]
        elif self.sel == "random":
            cands = self._random_episodes(Q)
        elif self.is_attack:
            self.relabel_attack()
            neps = self.att_eps[-p["att_pool"]:]
            if len(neps) >= 1:
                ainfo = []
                for k, (s0, L, succ, _) in enumerate(neps):
                    sc = self.member_scores(self.rb.images[s0:s0 + L])
                    kk, w, u = self.best_window(sc)
                    ainfo.append(dict(ep=("att", s0, L), start=kk, score=w, unc=u, succ=succ, s0=s0, L=L))
                ainfo.sort(key=lambda i: -(i["score"] + (p["beta_unc"] * i["unc"] if self.sel == "hybrid" else 0)))
                cands = [("att", i) for i in ainfo[:Q]]
        # ---- verify ------------------------------------------------------------------
        n_ver, n_conf, pairs = 0, 0, []
        conf_cells = set()
        for kind, inf in cands:
            n_ver += 1
            if kind == "task":
                succ = self.ep_succ[inf["ep"]]
                imgs = self.task_imgs(inf["ep"])
                s0, L = self.ep_start[inf["ep"]], self.ep_len[inf["ep"]]
                obs = r.replay.obses[s0:s0 + L]
            elif kind == "att":
                succ = inf["succ"]; imgs = self.rb.images[inf["s0"]:inf["s0"] + inf["L"]]
                obs = self.rb.obses[inf["s0"]:inf["s0"] + inf["L"]]
            else:                                                   # random episode (imgs held in inf)
                succ = inf["succ"]; imgs = inf["imgs"]; obs = inf["obs"]
            if succ > 0:
                tr_succ = r.traj["succ"]
                if kind == "task":
                    seg = np.flatnonzero(tr_succ[s0:s0 + L])
                    fs = int(seg[0]) if len(seg) else L - 1
                    self.verified_pos.append((s0, L, fs))
                continue
            if inf["score"] >= thr:
                n_conf += 1
                st = inf["start"]
                pairs.append(imgs[st:st + PAD])
                conf_cells |= cells(obs[st:st + PAD])
        # Anchor queries: selectors that do not verify on-policy episodes themselves (attack/hybrid/random/disagreement) would never
        # obtain a verified-success reference and therefore never repair anything.  They get Qpos extra verification queries per round on
        # the highest-scoring on-policy episodes; these queries are counted (verifier_queries) and reported separately (anchor_queries).
        if self.sel in ("attack", "hybrid", "random", "disagreement"):
            known = {v[0] for v in self.verified_pos}
            order = np.argsort([-i["score"] for i in tinfo])
            used = 0
            for k in order:
                e = tinfo[k]["ep"]
                if used >= p.get("Qpos", 2):
                    break
                if self.ep_start[e] in known:
                    continue
                used += 1
                n_ver += 1
                self.anchor_queries += 1
                if self.ep_succ[e] > 0:
                    seg = np.flatnonzero(r.traj["succ"][self.ep_start[e]:self.ep_start[e] + self.ep_len[e]])
                    self.verified_pos.append((self.ep_start[e], self.ep_len[e], int(seg[0]) if len(seg) else self.ep_len[e] - 1))
        self.verifier_queries += n_ver
        self.n_conf_fail += n_conf
        # ---- build preference pairs vs verified successes ------------------------------------
        n_added = 0
        if pairs and self.verified_pos:
            seg1, seg2, labels = [], [], []
            for neg in pairs:
                s0, L, fs = self.verified_pos[self.rng.randint(len(self.verified_pos))]
                st = max(0, fs - PAD + 1)
                pos = r.replay.images[s0 + st:s0 + st + PAD]
                if len(pos) < PAD or len(neg) < PAD:
                    continue
                if self.rng.rand() < 0.5:
                    seg1.append(pos); seg2.append(neg); labels.append(0)    # label 1 => seg2 preferred; here seg1 (pos) preferred
                else:
                    seg1.append(neg); seg2.append(pos); labels.append(1)
            if seg1:
                n = len(seg1)
                ds = r.RRM.ds + r.RRM.da
                r.RRM.put_queries(np.stack(seg1), np.stack(seg2), np.array(labels, np.float32).reshape(-1, 1),
                                  np.zeros((n, PAD, ds), np.float32), np.zeros((n, PAD, ds), np.float32),
                                  np.zeros((n, 2, PAD), np.float32))
                n_added = n
        self.n_pairs += n_added
        # ---- distinctness (C5): occupancy overlap of confirmed-failure windows with on-policy states ---------
        task_cells = set()
        for e in task_pool[-20:]:
            s0, L = self.ep_start[e], self.ep_len[e]
            task_cells |= cells(r.replay.obses[s0:s0 + L])
        novel = (len(conf_cells - task_cells) / max(len(conf_cells), 1)) if conf_cells else float("nan")
        self.log.append(dict(step=step, selector=self.sel, n_pool=len(tinfo), n_verified=n_ver, n_conf_fail=n_conf,
                             n_pairs_added=n_added, n_verified_pos_total=len(self.verified_pos), thr=thr,
                             frac_conf_cells_novel_vs_onpolicy=novel, extra_env_steps=self.extra_env_steps,
                             task_pool_success_rate=float(np.mean([self.ep_succ[e] > 0 for e in task_pool]))))

    def _random_episodes(self, Q):
        from src.common.core_train import reset_env, step_env
        import cv2
        out, S = [], self.cfg.image_size
        for _ in range(Q):
            obs = reset_env(self.env)
            imgs, obss, succ, done = [], [], 0.0, False
            while not done:
                a = self.env.action_space.sample()
                obss.append(obs)
                obs, _, done, info = step_env(self.env, a)
                succ = max(succ, float(info["success"]))
                rgb = self.env.render()[::-1, :, :]
                if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                    rgb = rgb[100:400, 100:400, :]
                imgs.append(cv2.resize(rgb, (S, S)))
            self.extra_env_steps += len(imgs)
            imgs = np.stack(imgs)
            k, w, u = self.best_window(self.member_scores(imgs))
            out.append(("rand", dict(ep=None, start=k, score=w, unc=u, succ=succ, imgs=imgs, obs=np.stack(obss))))
        return out

    def stats(self):
        att = {}
        if self.is_attack:
            e = self.att_eps
            att = dict(n_attack_episodes=len(e), attack_success_rate=float(np.mean([x[2] > 0 for x in e])) if e else None,
                       attack_mean_R=float(np.mean([x[3] for x in e])) if e else None, attack_steps=self.att["steps"])
        return dict(selector=self.sel, Q=self.p["Q"], extra_env_steps=self.extra_env_steps,
                    verifier_queries_for_repair=self.verifier_queries, anchor_queries=self.anchor_queries, verifier_free_evals_for_selection=self.verifier_free_evals,
                    n_confirmed_failures=self.n_conf_fail, n_pairs_added=self.n_pairs, rounds=self.log, **att)
