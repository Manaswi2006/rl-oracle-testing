"""FACR (feasibility-aware counterfactual reward) and its simpler comparators, replacing CoRe's FRM.

All variants are *oracle-supervised diagnostics*: the per-step simulator success flag defines the labels
(future success within h steps / time-to-success).  State-based (39-d MetaWorld observation), not pixels.

variant:
  progress : TimeRewarder-style ordinary progress prediction. phi(s) regresses t/t_success on steps of
             successful episodes (truncated at first success); r = std(phi(s') - phi(s)).
  value    : success-value shaping. V(s,h)=P(success within h steps | s, behaviour policy), BCE on future-success
             labels; r = std(gamma*V(s',H) - V(s,H)).
  facr     : counterfactual advantage. learned dynamics ensemble p_phi(s'|s,a) (E members, predicts delta s),
             Q_cf(s,a)=mean_e V(s_hat'_e, H-1),  A_cf = Q_cf(s,a) - mean_{a'~pi} Q_cf(s,a'),  r = std(A_cf),
             optional uncertainty filter (V-disagreement across the ensemble > quantile => 0).
  Normalisation `std(.)`: r = clip(scale * x / sigma_x, -1, 1), sigma_x measured over the replay buffer at the last
  relabel (same rule for the three variants; scale=0.5).
NOT claimed: A_cf is not an unbiased causal advantage nor a policy-invariant shaping term; the V(s,h) in the
definition is policy-dependent and the dynamics model is learned.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.ext_base import RewardExt
from src.psvm.psvm_ext import mlp

HS = (10, 25, 50, 100)
HMAX = 100.0


class FACRExt(RewardExt):
    name = "facr"
    oracle_information_used = "simulator per-step success flag -> future-success labels / time-to-success (oracle outcome supervision)"

    def __init__(self, cfg, obs_dim, act_dim, device, seed):
        super().__init__(cfg, obs_dim, act_dim, device, seed)
        p = dict(variant="facr", H=50, gamma=0.99, hidden=256, lr=3e-4, K=200, batch=512, update_every=1000,
                 n_dyn=5, n_alt=8, scale=0.5, unc_filter=False, unc_quantile=0.9, min_pos_steps=5, min_pos_eps=1,
                 train_dyn_every_update=True, dyn_frac=1.0)
        p.update(dict(cfg.get("facr", {}) or {}))
        self.p = p
        torch.manual_seed(seed)
        N = int(cfg.num_train_steps) + 2
        self.obs = np.zeros((N, obs_dim), np.float32)
        self.nobs = np.zeros((N, obs_dim), np.float32)
        self.act = np.zeros((N, act_dim), np.float32)
        self.succ = np.zeros(N, np.int8)
        self.ep = np.full(N, -1, np.int32)
        self.t = np.zeros(N, np.int16)
        self.n = 0
        self.done_ep = set()
        self.agent = None
        dev = device
        h = p["hidden"]
        self.V = mlp(obs_dim + 1, h, 1, 3).to(dev)
        self.phi = mlp(obs_dim, h, 1, 3).to(dev)
        self.dyn = nn.ModuleList([mlp(obs_dim + act_dim, h, obs_dim, 3) for _ in range(p["n_dyn"])]).to(dev)
        self.optV = torch.optim.Adam(self.V.parameters(), lr=p["lr"])
        self.optP = torch.optim.Adam(self.phi.parameters(), lr=p["lr"])
        self.optD = torch.optim.Adam(self.dyn.parameters(), lr=p["lr"])
        self.mu = torch.zeros(obs_dim, device=dev); self.sd = torch.ones(obs_dim, device=dev)
        self.dmu = torch.zeros(obs_dim, device=dev); self.dsd = torch.ones(obs_dim, device=dev)
        self.sigma = 1.0
        self.last = {}
        self.last_update = -10 ** 9

    def bind(self, agent):
        self.agent = agent

    # ---- data ------------------------------------------------------------------
    def on_step(self, ep_id, t, obs, act, next_obs, env_r, success, step):
        i = self.n
        self.obs[i], self.nobs[i], self.act[i] = obs, next_obs, act
        self.succ[i], self.ep[i], self.t[i] = int(success), ep_id, t
        self.n += 1

    def on_episode_end(self, ep_id, ep_success, step):
        self.done_ep.add(ep_id)

    def _labels(self, n):
        """time-to-next-success per step (inf if none in the rest of the episode); only completed episodes usable."""
        ep, succ = self.ep[:n], self.succ[:n]
        ttn = np.full(n, np.inf, np.float32)
        usable = np.isin(ep, list(self.done_ep))
        starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
        ends = np.r_[starts[1:], n]
        first_succ_t = np.full(n, np.inf, np.float32)
        for s0, e0 in zip(starts, ends):
            sl = slice(s0, e0)
            s_idx = np.flatnonzero(succ[sl])
            if len(s_idx):
                nxt = np.full(e0 - s0, np.inf, np.float32)
                # next success index >= i via searchsorted
                pos = np.searchsorted(s_idx, np.arange(e0 - s0))
                ok = pos < len(s_idx)
                nxt[ok] = s_idx[pos[ok]] - np.arange(e0 - s0)[ok]
                ttn[sl] = nxt
                first_succ_t[sl] = s_idx[0]
        return ttn, first_succ_t, usable

    # ---- learning ----------------------------------------------------------------
    def maybe_update(self, step):
        if step - self.last_update < self.p["update_every"]:
            return False
        n = self.n
        if n < 2000:
            return False
        ttn, fst, usable = self._labels(n)
        npos = int(((ttn <= HS[-1]) & usable).sum())
        if npos < self.p["min_pos_steps"]:
            return False
        self.last_update = step
        dev, p = self.device, self.p
        X = torch.from_numpy(self.obs[:n]).to(dev); NX = torch.from_numpy(self.nobs[:n]).to(dev)
        A = torch.from_numpy(self.act[:n]).to(dev)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-3
        d = NX - X
        self.dmu, self.dsd = d.mean(0), d.std(0) + 1e-4
        idx_u = torch.from_numpy(np.flatnonzero(usable)).to(dev)
        TT = torch.from_numpy(ttn).to(dev)
        hs = torch.tensor(HS, device=dev, dtype=torch.float32)
        v = self.p["variant"]
        res = {}
        if v in ("value", "facr"):
            tot = 0.0
            for _ in range(p["K"]):
                b = idx_u[torch.randint(len(idx_u), (p["batch"],), device=dev)]
                hh = hs[torch.randint(len(hs), (p["batch"],), device=dev)]
                y = (TT[b] <= hh).float()
                lg = self.V(torch.cat([(X[b] - self.mu) / self.sd, (hh / HMAX).unsqueeze(1)], 1)).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(lg, y)
                self.optV.zero_grad(); loss.backward(); self.optV.step(); tot += loss.item()
            res["V_bce"] = tot / p["K"]
        if v == "progress":
            FST = torch.from_numpy(fst).to(dev)
            tt = torch.from_numpy(self.t[:n].astype(np.float32)).to(dev)
            okp = torch.from_numpy(usable & np.isfinite(fst) & (self.t[:n] <= np.nan_to_num(fst, posinf=0))).to(dev)
            idx_p = okp.nonzero().squeeze(1)
            if len(idx_p) > 0:
                tot = 0.0
                for _ in range(p["K"]):
                    b = idx_p[torch.randint(len(idx_p), (p["batch"],), device=dev)]
                    y = (tt[b] / FST[b].clamp_min(1.0)).clamp(0, 1)
                    loss = F.mse_loss(self.phi((X[b] - self.mu) / self.sd).squeeze(1), y)
                    self.optP.zero_grad(); loss.backward(); self.optP.step(); tot += loss.item()
                res["phi_mse"] = tot / p["K"]
        if v == "facr":
            tot = 0.0
            idx_all = torch.arange(n, device=dev)
            for _ in range(p["K"]):
                b = idx_all[torch.randint(n, (p["batch"],), device=dev)]
                inp = torch.cat([(X[b] - self.mu) / self.sd, A[b]], 1)
                tgt = ((NX[b] - X[b]) - self.dmu) / self.dsd
                loss = sum(F.mse_loss(m(inp), tgt) for m in self.dyn)
                self.optD.zero_grad(); loss.backward(); self.optD.step(); tot += loss.item() / len(self.dyn)
            res["dyn_mse"] = tot / p["K"]
        # reward scale from the buffer
        with torch.no_grad():
            sub = torch.from_numpy(np.random.randint(0, n, size=min(n, 20000))).to(dev)
            raw = self._raw(X[sub], A[sub], NX[sub])
            self.sigma = float(raw.std().clamp_min(1e-4))
            if self.p["unc_filter"] and v == "facr":
                self.unc_thr = float(torch.quantile(self._last_u, self.p["unc_quantile"]))
        res["sigma"] = self.sigma
        res["n_pos_steps"] = npos
        self.last = res
        self.n_updates += 1
        self.ready = True
        return True

    # ---- reward ------------------------------------------------------------------
    def _V(self, s, h):
        hh = torch.full((len(s), 1), h / HMAX, device=self.device)
        return torch.sigmoid(self.V(torch.cat([(s - self.mu) / self.sd, hh], 1)).squeeze(1))

    @torch.no_grad()
    def _raw(self, s, a, ns):
        v, p = self.p["variant"], self.p
        if v == "progress":
            return self.phi((ns - self.mu) / self.sd).squeeze(1) - self.phi((s - self.mu) / self.sd).squeeze(1)
        if v == "value":
            return p["gamma"] * self._V(ns, p["H"]) - self._V(s, p["H"])
        # facr
        H1 = p["H"] - 1
        M = p["n_alt"]
        acts = [a]
        if self.agent is not None:
            for _ in range(M):
                acts.append(self.agent.actor(s).sample().clamp(-1, 1))
        else:
            acts += [torch.rand_like(a) * 2 - 1 for _ in range(M)]
        Q, U = [], []
        z = (s - self.mu) / self.sd
        for aa in acts:
            inp = torch.cat([z, aa], 1)
            preds = torch.stack([s + m(inp) * self.dsd + self.dmu for m in self.dyn])   # [E,B,d]
            vs = torch.stack([self._V(pp, H1) for pp in preds])                          # [E,B]
            Q.append(vs.mean(0)); U.append(vs.std(0))
        Qs = torch.stack(Q)                 # [1+M, B]
        adv = Qs[0] - Qs[1:].mean(0)
        self._last_u = U[0]
        if p["unc_filter"] and hasattr(self, "unc_thr"):
            adv = torch.where(U[0] > self.unc_thr, torch.zeros_like(adv), adv)
        return adv

    @torch.no_grad()
    def reward_batch(self, obs, act, next_obs):
        if not self.ready:
            return np.zeros(len(obs), np.float32)
        dev = self.device
        s = torch.from_numpy(np.asarray(obs, np.float32)).to(dev)
        a = torch.from_numpy(np.asarray(act, np.float32)).to(dev)
        ns = torch.from_numpy(np.asarray(next_obs, np.float32)).to(dev)
        out = []
        for i in range(0, len(s), 4096):
            out.append(self._raw(s[i:i + 4096], a[i:i + 4096], ns[i:i + 4096]))
        r = torch.cat(out)
        r = (self.p["scale"] * r / self.sigma).clamp(-1, 1)
        return r.cpu().numpy().astype(np.float32)

    def stats(self):
        return dict(n_updates=self.n_updates, last=self.last, params=self.p)

    def save(self, path):
        torch.save(dict(V=self.V.state_dict(), phi=self.phi.state_dict(), dyn=self.dyn.state_dict(), mu=self.mu, sd=self.sd,
                        dmu=self.dmu, dsd=self.dsd, sigma=self.sigma), f"{path}/facr_models.pt")
