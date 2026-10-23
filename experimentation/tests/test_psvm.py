import numpy as np, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.psvm.psvm_ext import VisitationDiscriminator
from src.psvm.label_models import make_label_model


def test_density_ratio_under_class_imbalance():
    """Gaussian toy: w+ = N(1,1) (1% of data), w- = N(-1,1) (99%). Balanced objective => logit = 2x, NOT shifted by log prior."""
    rng = np.random.RandomState(0)
    n_pos, n_neg = 200, 20000
    x = np.concatenate([rng.randn(n_pos, 1) + 1, rng.randn(n_neg, 1) - 1]).astype(np.float32)
    q = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)
    D = VisitationDiscriminator(1, "cpu", hidden=64, depth=2, lr=3e-3, seed=0)
    X = torch.from_numpy(x); D.set_norm(X)
    D.fit(X, torch.from_numpy(q), torch.ones(len(x)), steps=1500, batch=512)
    g = torch.linspace(-1.5, 1.5, 13).unsqueeze(1)
    est = D.logit(g).detach().numpy(); true = 2 * g.squeeze().numpy()
    assert np.abs(est - true).mean() < 0.35, (est, true)


def test_soft_weights_equal_hard_when_q_binary_and_mixture_otherwise():
    """q=0.5 everywhere => both classes identical => logit ~ 0."""
    rng = np.random.RandomState(1)
    x = rng.randn(5000, 2).astype(np.float32)
    D = VisitationDiscriminator(2, "cpu", hidden=32, depth=2, lr=3e-3, seed=0)
    X = torch.from_numpy(x); D.set_norm(X)
    D.fit(X, torch.full((5000,), 0.5), torch.ones(5000), steps=600, batch=256)
    assert D.logit(X).abs().mean().item() < 0.3


def test_vote_model_systematic_errors_break_calibration():
    p = dict(mode="soft_raw", k_votes=7, tpr=0.95, tnr=0.95, sys_err=0.3, sys_conf=0.9, vlm_seed=0)
    m = make_label_model(p, 0)
    ys = (np.random.RandomState(2).rand(4000) < 0.5).astype(float)
    qs = np.array([m(i, y) for i, y in enumerate(ys)])
    hi = qs >= 6 / 7 - 1e-6
    assert hi.sum() > 100
    # empirical P(y=1 | q>=6/7) is far from the 'probability' 6/7..1.0 implied by the votes
    assert ys[hi].mean() < 0.9


def test_conf_filter_discards():
    m = make_label_model(dict(mode="conf_filter", k_votes=7, tpr=0.7, tnr=0.7, conf_thresh=0.9), 0)
    out = [m(i, 1.0) for i in range(300)]
    assert any(np.isnan(o) for o in out)


def test_ext_end_to_end_synthetic():
    from omegaconf import OmegaConf
    from src.psvm.psvm_ext import PSVMExt
    cfg = OmegaConf.create(dict(num_train_steps=20000, psvm=dict(update_every=1, K=300)))
    ext = PSVMExt(cfg, 5, 2, "cpu", 0)
    rng = np.random.RandomState(0)
    step = 0
    for ep in range(30):
        succ = ep % 3 == 0
        for t in range(100):
            mu = 1.0 if succ else -1.0
            o = rng.randn(5).astype(np.float32) * 0.5 + mu
            ext.on_step(ep, t, o, rng.randn(2).astype(np.float32), o, 0.0, float(succ and t == 60), step); step += 1
        ext.on_episode_end(ep, float(succ), step)
    assert ext.maybe_update(step)
    o_pos = (rng.randn(200, 5) * 0.5 + 1).astype(np.float32); o_neg = (rng.randn(200, 5) * 0.5 - 1).astype(np.float32)
    a = rng.randn(200, 2).astype(np.float32)
    rp, rn = ext.reward_batch(o_pos, a, o_pos), ext.reward_batch(o_neg, a, o_neg)
    assert rp.mean() > rn.mean() + 0.2 and np.abs(rp).max() <= 1.0
    assert (ext.t[: ext.n][(ext.ep[: ext.n] % 3 == 0)] > 60).any()   # post-success steps exist in storage ...
