"""Random Network Distillation intrinsic reward on the policy's OWN observation vector (no privileged information, no task knowledge).
Novelty = prediction error of a trained predictor against a fixed random target network, divided by its running RMS (so ~1 on average)."""
import numpy as np
import torch
import torch.nn as nn


def _net(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class RND:
    def __init__(self, obs_dim, seed=0, hidden=256, out=64, lr=1e-3, update_every=10, batch=256, capacity=200000):
        g = torch.Generator().manual_seed(1234 + seed)
        torch.manual_seed(1234 + seed)
        self.target, self.pred = _net(obs_dim, hidden, out), _net(obs_dim, hidden, out)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)
        self.mu, self.var, self.n = np.zeros(obs_dim, np.float64), np.ones(obs_dim, np.float64), 1e-4     # running obs statistics
        self.err_ms = 1.0                                                                                  # running mean-square of the error
        self.buf = np.zeros((capacity, obs_dim), np.float32); self.cnt = 0
        self.update_every, self.batch, self.cap = update_every, batch, capacity
        self.rng = np.random.RandomState(seed)

    def _norm(self, o):
        return np.clip((o - self.mu) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)

    def _update_stats(self, o):
        self.n += 1
        d = o - self.mu
        self.mu = self.mu + d / self.n
        self.var = self.var + (d * (o - self.mu) - self.var) / self.n

    @torch.no_grad()
    def novelty(self, o):
        x = torch.from_numpy(self._norm(np.asarray(o, np.float64))).unsqueeze(0)
        return float(((self.pred(x) - self.target(x)) ** 2).mean())

    def step(self, o):
        """call once per environment step with the new observation; returns the normalised novelty of o."""
        o = np.asarray(o, np.float64)
        self._update_stats(o)
        e = self.novelty(o)
        self.err_ms = 0.999 * self.err_ms + 0.001 * e * e
        self.buf[self.cnt % self.cap] = o; self.cnt += 1
        if self.cnt % self.update_every == 0 and self.cnt >= self.batch:
            idx = self.rng.randint(0, min(self.cnt, self.cap), size=self.batch)
            x = torch.from_numpy(self._norm(self.buf[idx].astype(np.float64)))
            loss = ((self.pred(x) - self.target(x)) ** 2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return e / (np.sqrt(self.err_ms) + 1e-8)
