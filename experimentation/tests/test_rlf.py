import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.falsification.rlf_ext import RLFModule, cells, PAD


def test_best_window_finds_peak():
    sc = np.zeros((3, 100)); sc[:, 40:60] = 1.0
    sc[0, 45] += 0.1
    k, w, u = RLFModule.best_window(sc)
    assert 38 <= k <= 42 and w > 19.5 and u < 0.2


def test_best_window_short_episode():
    k, w, u = RLFModule.best_window(np.ones((3, 5)))
    assert k == 0


def test_cells_overlap():
    a = np.zeros((10, 39)); b = np.zeros((10, 39)); b[:, 0] = 0.5
    assert len(cells(a) & cells(b)) == 0 and len(cells(a)) == 1


def test_anchor_queries_give_positives_and_pairs():
    """attack selector must obtain verified positives via anchor queries and then add repair pairs (regression test for the zero-pair flaw)."""
    import types
    m = RLFModule.__new__(RLFModule)
    m.p = dict(selector="attack", Q=2, pool=10, Qpos=2, beta_unc=1.0, att_pool=4)
    m.sel, m.is_attack = "attack", True
    m.rng = np.random.RandomState(0)
    L = 60
    m.ep_start = {e: e * L for e in range(8)}; m.ep_len = {e: L for e in range(8)}
    m.ep_succ = {e: float(e in (3, 6)) for e in range(8)}
    m.verified_pos, m.log = [], []
    m.verifier_queries = m.anchor_queries = m.n_conf_fail = m.n_pairs = m.verifier_free_evals = m.extra_env_steps = 0
    m.cfg = None
    imgs = np.zeros((8 * L, 8, 8, 3), np.uint8)
    traj_succ = np.zeros(8 * L, np.int8); traj_succ[3 * L + 30] = 1; traj_succ[6 * L + 40] = 1
    added = []
    rrm = types.SimpleNamespace(ds=39, da=4, put_queries=lambda *a: added.append(a[0].shape))
    obs = np.zeros((8 * L, 39), np.float32)
    m.r = types.SimpleNamespace(replay=types.SimpleNamespace(images=imgs, obses=obs), traj=dict(succ=traj_succ), RRM=rrm)
    m.pool_episodes = lambda: list(range(8))
    # task-pool scores: successful episodes score highest -> anchors pick them
    m.pool_info = lambda eps, fn: [dict(ep=e, start=0, score=10.0 if e in (3, 6) else float(e), unc=0.0) for e in eps]
    m.relabel_attack = lambda: None
    m.rb = types.SimpleNamespace(images=np.zeros((200, 8, 8, 3), np.uint8), obses=np.zeros((200, 39), np.float32))
    m.att_eps = [(0, 100, 0.0, 1.0), (100, 100, 0.0, 0.9)]          # two failing attack episodes
    m.member_scores = lambda im: np.ones((3, len(im)))
    m.repair(step=1000)
    assert m.anchor_queries == 2 and len(m.verified_pos) == 2
    assert m.n_pairs >= 1 and added and added[0][1] == 20
