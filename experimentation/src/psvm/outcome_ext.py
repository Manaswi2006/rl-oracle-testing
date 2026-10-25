"""Control arm: the plain sparse OUTCOME reward (simulator success flag, r_out of the SVM paper) used in place of CoRe's FRM,
mixed exactly like the other structural rewards: (1-a)*r_out + a*RRM (FRM-slot reward before the RRM kicks in).
Isolates the information in the success flag from the visitation-matching (dense process reward) mechanism of P-SVM."""
import numpy as np
from src.common.ext_base import RewardExt


class OutcomeExt(RewardExt):
    name = "outcome"
    static_reward = True            # per-step reward, never relabelled
    oracle_information_used = "simulator success flag used directly as a sparse per-step reward (oracle outcome reward)"

    def __init__(self, cfg, obs_dim, act_dim, device, seed):
        super().__init__(cfg, obs_dim, act_dim, device, seed)
        self.ready = True

    def step_reward(self, success, next_obs=None):
        return float(success)

    def reward_batch(self, obs, act, next_obs):
        return np.zeros(len(obs), np.float32)
