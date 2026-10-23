"""Interface for the learned structural-reward modules that replace CoRe's FRM."""
import numpy as np


class RewardExt:
    """A structural reward learned online from the agent's own experience.

    The trainer feeds every transition (with *oracle* success flags; each concrete
    module decides whether it may use them) and asks for rewards for arbitrary
    (s, a, s') batches.  `reward_batch` must return values already scaled to the
    range of CoRe's FRM, i.e. [-1, 1].
    """
    name = "none"
    oracle_information_used = "none"

    def __init__(self, cfg, obs_dim, act_dim, device, seed):
        self.cfg, self.obs_dim, self.act_dim, self.device, self.seed = cfg, obs_dim, act_dim, device, seed
        self.n_updates = 0
        self.ready = False          # reward is 0 until the module has something to say

    # data hooks -----------------------------------------------------------
    def on_step(self, ep_id, t, obs, act, next_obs, env_r, success, step):
        pass

    def on_episode_end(self, ep_id, ep_success, step):
        pass

    # learning hook: return True when rewards changed and the buffer must be relabelled
    def maybe_update(self, step):
        return False

    def reward_batch(self, obs, act, next_obs):
        return np.zeros(len(obs), dtype=np.float32)

    def stats(self):
        return {}

    def save(self, path):
        pass
