import numpy as np, sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omegaconf import OmegaConf
from src.vlm.episode_outcome_ext import EpisodeOutcomeExt, FRAME_STEPS


def _run(source, verdict_votes=None, agg="majority"):
    cfg = OmegaConf.create(dict(env="metaworld_button-press-topdown-v2", ep=dict(source=source, agg=agg, k=3)))
    ext = EpisodeOutcomeExt.__new__(EpisodeOutcomeExt)
    # minimal manual init (no VLM server needed)
    from src.common.ext_base import RewardExt
    RewardExt.__init__(ext, cfg, 39, 4, "cpu", 0)
    ext.p = dict(source=source, k=3, agg=agg, ports=[8000], task="button", model="m", weight=1.0)
    ext.frames, ext.ep_start, ext.ep_end, ext.records, ext.n_queries, ext.vlm, ext.task = [], {}, {}, [], 0, None, "button"
    ext.rnd, ext.r_int, ext._n = None, np.zeros(5000, np.float32), 0
    N = 1200
    rb = types.SimpleNamespace(rewards_frm=np.zeros((N, 1), np.float32), rewards=np.zeros((N, 1), np.float32), rewards_rrm=np.full((N, 1), 0.2, np.float32))
    ext.runner = types.SimpleNamespace(replay=rb, alpha=0.5)
    if verdict_votes is not None:
        ext._verdict = lambda frames, ts: (float(np.mean(verdict_votes) > 0.5) if agg == "majority" else float(np.mean(verdict_votes)), verdict_votes)
    for t in range(500):
        if t in FRAME_STEPS: ext.on_frame(t, np.zeros((480, 480, 3), np.uint8))
        ext.on_step(0, t, None, None, None, 0.0, 0.0, 100 + t)
    ext.on_episode_end(0, 1.0, 600)
    return ext, rb


def test_oracle_episode_reward_written_to_all_steps_and_mixed():
    ext, rb = _run("oracle")
    assert np.allclose(rb.rewards_frm[100:600, 0], 1.0) and np.allclose(rb.rewards[100:600, 0], 0.5 * 1.0 + 0.5 * 0.2)
    assert rb.rewards_frm[:100].sum() == 0 and rb.rewards_frm[600:].sum() == 0


def test_vlm_majority_and_soft_never_use_true_success():
    ext, rb = _run("vlm", verdict_votes=[0, 0, 1])           # true success = 1 but VLM says mostly No -> reward must follow the VLM
    assert np.allclose(rb.rewards_frm[100:600, 0], 0.0)
    ext, rb = _run("vlm", verdict_votes=[1, 1, 0], agg="soft")
    assert np.allclose(rb.rewards_frm[100:600, 0], 2 / 3, atol=1e-6)
    assert ext.records[0]["votes"] == [1, 1, 0]


def test_sixteen_frames_collected():
    ext = EpisodeOutcomeExt.__new__(EpisodeOutcomeExt); ext.frames = []
    for t in range(500): ext.on_frame(t, np.zeros((480, 480, 3), np.uint8))
    assert len(ext.frames) == 16


def test_rnd_part_is_kept_and_verdict_added():
    ext, rb = _run("oracle")
    ext.rnd = object(); ext.r_int[100:600] = 0.25                       # pretend an intrinsic bonus was collected for the episode
    ext.on_episode_end(0, 1.0, 600)
    assert np.allclose(rb.rewards_frm[100:600, 0], 1.25)
