import numpy as np, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omegaconf import OmegaConf
from src.facr.facr_ext import FACRExt


def _toy(variant):
    """2 relevant dims: s0 moves with a0 (success when s0>1); episodes of 60 steps; random policy."""
    cfg = OmegaConf.create(dict(num_train_steps=40000, facr=dict(variant=variant, update_every=1, K=400, H=20, n_alt=6, min_pos_steps=3)))
    ext = FACRExt(cfg, 4, 2, "cpu", 0)
    rng = np.random.RandomState(0); step = 0
    for ep in range(300):
        s = np.zeros(4, np.float32); s[1:] = rng.randn(3) * 0.1
        for t in range(60):
            a = rng.uniform(-1, 1, 2).astype(np.float32)
            ns = s.copy(); ns[0] += 0.08 * a[0]; ns[1:] += 0.01 * rng.randn(3)
            ext.on_step(ep, t, s, a, ns, 0.0, float(ns[0] > 1.0), step); step += 1
            s = ns
        ext.on_episode_end(ep, 0.0, step)
    return ext


def test_facr_counterfactual_advantage_prefers_progress_actions():
    ext = _toy("facr")
    assert ext.maybe_update(10 ** 6)
    s = np.zeros((200, 4), np.float32); s[:, 0] = np.random.RandomState(1).uniform(0.4, 0.9, 200)
    good = np.tile(np.array([[1.0, 0.0]], np.float32), (200, 1)); bad = -good
    ns_g = s.copy(); ns_g[:, 0] += 0.08; ns_b = s.copy(); ns_b[:, 0] -= 0.08
    rg, rb = ext.reward_batch(s, good, ns_g), ext.reward_batch(s, bad, ns_b)
    assert rg.mean() > rb.mean() + 0.1, (rg.mean(), rb.mean())
    assert np.abs(rg).max() <= 1.0


def test_value_and_progress_variants_run():
    for v in ("value", "progress"):
        ext = _toy(v)
        assert ext.maybe_update(10 ** 6)
        s = np.zeros((50, 4), np.float32); a = np.zeros((50, 2), np.float32)
        r = ext.reward_batch(s, a, s + 0.05)
        assert r.shape == (50,) and np.isfinite(r).all()
