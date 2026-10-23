"""(P-)SVM structural reward replacing CoRe's FRM.

Authentic SVM ingredients (Tsao, Wagenmaker, Levine, arXiv 2606.23640, Sec. 4-5, App. D):
  * D+ = steps of successful episodes (truncated at the first success: the MDP is terminal there),
    D- = steps of unsuccessful episodes;
  * ONE discriminator f(s,a) for all time steps; MLP 3x256 ReLU; Adam;
  * the classification objective E_{D+}[log f] + E_{D-}[log(1-f)] weights the two classes equally, so
    logit(f) = log( w+(s,a) / w-(s,a) ) is the ratio of *normalised* visitation densities (class prior
    correction is implicit; here it is made explicit by class-balanced sampling);
  * r_svm = lambda * clip_beta(logit f(s,a)), lambda=0.05, beta=20 -> range [-1,1], same as the FRM.
Documented deviations: discriminator trained on state vectors (not images); K=200 gradient steps of batch
256 every T=1000 env steps (paper: K=32, batch 64 on images); no r_out term in the mixture (CoRe's
RRM/outcome handling replaces it; `use_outcome_term` ablation adds it); trained from scratch (no N0
pretrained-policy rollouts).

P-SVM extension (this study): every episode i carries a soft success weight q_i in [0,1]. The loss is
    L = 1/2 * sum_n q_n * (-log f_n) / sum_n q_n  +  1/2 * sum_n (1-q_n) * (-log(1-f_n)) / sum_n (1-q_n)
i.e. each class is a q-weighted empirical distribution normalised to unit mass; this is the class-prior
corrected objective whose population minimiser has logit = log(w+/w-) for the q-weighted mixtures.
Implemented by sampling positives with prob ∝ q_n and negatives ∝ (1-q_n) (equal-size halves).
q_i = {0,1} gives the hard-label SVM exactly.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.ext_base import RewardExt


def mlp(i, h, o, depth=3):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class VisitationDiscriminator:
    """f(s,a) trained with the q-weighted, class-balanced BCE described above."""

    def __init__(self, in_dim, device, hidden=256, depth=3, lr=3e-4, arch="mlp", seed=0):
        torch.manual_seed(seed)
        self.device = device
        self.arch = arch
        if arch == "mlp":
            self.net = mlp(in_dim, hidden, 1, depth).to(device)
        elif arch == "mlp_small":
            self.net = mlp(in_dim, 64, 1, 2).to(device)
        elif arch == "mlp_wide":
            self.net = mlp(in_dim, 512, 1, 3).to(device)
        else:
            raise ValueError(arch)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.mu = torch.zeros(in_dim, device=device)
        self.sd = torch.ones(in_dim, device=device)

    def set_norm(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-3

    def logit(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)

    def fit(self, x, q, mask, steps=200, batch=256):
        """x: [N,d] float tensor (device); q: [N] soft success weight; mask: [N] bool (step usable)."""
        wp = (q * mask).clamp_min(0)
        wn = ((1 - q) * mask).clamp_min(0)
        if wp.sum() <= 0 or wn.sum() <= 0:
            return None
        hb = batch // 2
        loss_v, acc_v = 0.0, 0.0
        for _ in range(steps):
            ip = torch.multinomial(wp, hb, replacement=True)
            ineg = torch.multinomial(wn, hb, replacement=True)
            lp, ln = self.logit(x[ip]), self.logit(x[ineg])
            loss = 0.5 * F.softplus(-lp).mean() + 0.5 * F.softplus(ln).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_v += loss.item()
            acc_v += 0.5 * ((lp > 0).float().mean().item() + (ln < 0).float().mean().item())
        return loss_v / steps, acc_v / steps


class PSVMExt(RewardExt):
    name = "psvm"

    def __init__(self, cfg, obs_dim, act_dim, device, seed):
        super().__init__(cfg, obs_dim, act_dim, device, seed)
        p = dict(lam=0.05, beta=20.0, hidden=256, depth=3, lr=3e-4, K=200, batch=256, update_every=1000,
                 mode="oracle_hard", truncate=True, use_outcome_term=False, min_pos_eps=2, arch="mlp",
                 # label corruption (only for non-oracle modes; see src/psvm/label_models.py)
                 flip_fp=0.0, flip_fn=0.0, conf_thresh=0.8, vlm_seed=0)
        p.update(dict(cfg.get("psvm", {}) or {}))
        self.p = p
        self.oracle_information_used = {
            "oracle_hard": "episode success flag (simulator) used directly as hard label; per-step first-success time used to truncate positives",
        }.get(p["mode"], f"simulator success used only to synthesise corrupted labels (mode={p['mode']})")
        N = int(cfg.num_train_steps) + 2
        self.obs = np.zeros((N, obs_dim), np.float32)
        self.act = np.zeros((N, act_dim), np.float32)
        self.ep = np.full(N, -1, np.int32)
        self.t = np.zeros(N, np.int16)
        self.n = 0
        self.max_ep = int(N // 50) + 100
        self.ep_q = np.full(self.max_ep, np.nan, np.float32)      # soft label used for training
        self.ep_true = np.full(self.max_ep, np.nan, np.float32)   # simulator outcome (diagnostics only)
        self.first_succ_t = np.full(self.max_ep, 10 ** 6, np.int32)
        self.D = VisitationDiscriminator(obs_dim + act_dim, device, p["hidden"], p["depth"], p["lr"], p["arch"], seed)
        from src.psvm.label_models import make_label_model
        self.label_model = make_label_model(p, seed)
        self.last_update = -10 ** 9
        self.last_fit = None
        self.n_pos_eps = self.n_neg_eps = 0

    def on_step(self, ep_id, t, obs, act, next_obs, env_r, success, step):
        i = self.n
        self.obs[i], self.act[i], self.ep[i], self.t[i] = obs, act, ep_id, t
        if success > 0 and self.first_succ_t[ep_id] > t:
            self.first_succ_t[ep_id] = t
        self.n += 1

    def on_episode_end(self, ep_id, ep_success, step):
        self.ep_true[ep_id] = ep_success
        self.ep_q[ep_id] = self.label_model(ep_id, ep_success)

    def maybe_update(self, step):
        if step - self.last_update < self.p["update_every"]:
            return False
        done_eps = ~np.isnan(self.ep_q)
        pos = np.nansum(self.ep_q > 0.5)
        neg = np.nansum(self.ep_q <= 0.5)
        self.n_pos_eps, self.n_neg_eps = int(pos), int(neg)
        if pos < self.p["min_pos_eps"] or neg < 1:
            return False
        self.last_update = step
        n = self.n
        ep = self.ep[:n]
        q = self.ep_q[ep]                       # nan for unfinished episodes
        mask = ~np.isnan(q)
        if self.p["truncate"]:
            mask &= ~((self.ep_true[ep] > 0) & (self.t[:n] > self.first_succ_t[ep]))
        q = np.nan_to_num(q)
        x = torch.from_numpy(np.concatenate([self.obs[:n], self.act[:n]], 1)).to(self.device)
        self.D.set_norm(x[torch.from_numpy(mask).to(self.device)])
        res = self.D.fit(x, torch.from_numpy(q).to(self.device), torch.from_numpy(mask.astype(np.float32)).to(self.device),
                         steps=self.p["K"], batch=self.p["batch"])
        if res is None:
            return False
        self.last_fit = res
        self.n_updates += 1
        self.ready = True
        return True

    @torch.no_grad()
    def reward_batch(self, obs, act, next_obs):
        if not self.ready:
            return np.zeros(len(obs), np.float32)
        x = torch.from_numpy(np.concatenate([obs, act], 1).astype(np.float32)).to(self.device)
        r = self.p["lam"] * self.logit_clip(self.D.logit(x))
        return r.cpu().numpy().astype(np.float32)

    def logit_clip(self, z):
        return z.clamp(-self.p["beta"], self.p["beta"])

    def stats(self):
        return dict(n_updates=self.n_updates, n_pos_eps=self.n_pos_eps, n_neg_eps=self.n_neg_eps,
                    last_loss_acc=self.last_fit, params=self.p)

    def save(self, path):
        torch.save({"net": self.D.net.state_dict(), "mu": self.D.mu, "sd": self.D.sd}, f"{path}/psvm_discriminator.pt")
        np.savez_compressed(f"{path}/psvm_episode_labels.npz", q=self.ep_q, true=self.ep_true)
