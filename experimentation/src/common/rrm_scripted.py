"""CoRe RRM with a *simulated* preference teacher (no VLM API available).

Deviation from CoRe (documented in REPRODUCIBILITY.md): CoRe's RRM is trained on
video-level preferences produced by a VLM (gemini-2.5-flash-lite) together with
per-frame importance weights.  No VLM credentials were authorised for this study,
so segment-pair preferences are produced by a scripted teacher on the privileged
environment reward (PEBBLE-style), with symmetric label noise `eps_mistake` to
mimic an imperfect VLM.  The frame-importance KL term is disabled (no weights).
The reward network, image pipeline, ensemble, query sampling and training loop
are CoRe's own (upstream/CoRe/RRM/reward_model.py).
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UP = os.path.join(ROOT, "upstream", "CoRe")
for p in (UP, os.path.join(UP, "envs", "frm_code"), os.path.join(UP, "envs", "env_info")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MY_API_KEY", "UNUSED_NO_API_CALLS_ARE_MADE")  # upstream modules read it at import

# --- pure speed patches (no numerical effect) --------------------------------------------
import gymnasium.spaces.box as _gbox  # noqa: E402
_gbox._short_repr = lambda arr: ""  # MetaWorld builds a Box every step; repr formatting cost ~2 ms/step
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
_Adam = torch.optim.Adam
torch.optim.Adam = lambda params, *a, **k: _Adam(params, *a, foreach=True, **k)

import RRM.reward_model as _rm  # noqa: E402


class _NoVLM:
    """Stand-in for upstream PreAgent: never touches the network or the goal images."""
    def __init__(self, *a, **k):
        pass


_rm.PreAgent = _NoVLM


class ScriptedRewardModel(_rm.RewardModel):
    def __init__(self, *args, eps_mistake=0.0, tie_eps=1e-6, rng_seed=0, **kwargs):
        kwargs["kl_weight"] = 0
        super().__init__(*args, **kwargs)
        self.eps_mistake = eps_mistake
        self.tie_eps = tie_eps
        self.rng = np.random.RandomState(rng_seed)
        self.n_queries = 0           # queries issued to the (simulated) teacher
        self.n_labels_kept = 0
        self.n_flipped = 0
        self.last_label_acc = 1.0
        self.label_log = []          # (sa_t_1, sa_t_2, label, clean_label) for later release

    # -- teacher -----------------------------------------------------------
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1=None, img_t_2=None):
        s1 = np.sum(r_t_1, axis=1).reshape(-1)
        s2 = np.sum(r_t_2, axis=1).reshape(-1)
        self.n_queries += len(s1)
        keep = np.abs(s1 - s2) > self.tie_eps
        clean = (s1 < s2).astype(np.int64)
        lab = clean.copy()
        flip = self.rng.rand(len(lab)) < self.eps_mistake
        lab[flip] = 1 - lab[flip]
        sa_t_1, sa_t_2 = sa_t_1[keep], sa_t_2[keep]
        r_t_1, r_t_2 = r_t_1[keep], r_t_2[keep]
        img_t_1, img_t_2 = img_t_1[keep], img_t_2[keep]
        clean, lab, flip = clean[keep], lab[keep], flip[keep]
        self.n_labels_kept += len(lab)
        self.n_flipped += int(flip.sum())
        self.last_label_acc = float((lab == clean).mean()) if len(lab) else 1.0
        self.label_log.append((sa_t_1.astype(np.float32), sa_t_2.astype(np.float32),
                               lab.astype(np.int8), clean.astype(np.int8)))
        weights = np.zeros((len(lab), 2, sa_t_1.shape[1]), dtype=np.float32)
        return (sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2,
                clean.reshape(-1, 1), lab.reshape(-1, 1), weights)

    # -- identical to upstream train_reward minus the (disabled) KL term ----
    def train_reward(self):
        device = _rm.device
        ensemble_losses = np.zeros(self.de)
        ensemble_acc = np.zeros(self.de)
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = [np.random.permutation(max_len) for _ in range(self.de)]
        num_iters = int(np.ceil(max_len / self.train_batch_size))
        total = 0
        for iter_id in range(num_iters):
            self.opt.zero_grad()
            total_loss = 0.0
            last_index = min((iter_id + 1) * self.train_batch_size, max_len)
            for member in range(self.de):
                idxs = total_batch_index[member][iter_id * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = torch.from_numpy(self.buffer_label[idxs].flatten()).long().to(device)
                if member == 0:
                    total += labels.size(0)
                batch, time_step, H, W, C = sa_t_1.shape
                sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3)).reshape(-1, C, H, W).astype(np.float32) / 255.0
                sa_t_2 = np.transpose(sa_t_2, (0, 1, 4, 2, 3)).reshape(-1, C, H, W).astype(np.float32) / 255.0
                r1 = self.r_hat_member(sa_t_1, member=member).view(batch, time_step, 1).sum(axis=1)
                r2 = self.r_hat_member(sa_t_2, member=member).view(batch, time_step, 1).sum(axis=1)
                r_hat_ce = torch.cat([r1, r2], axis=-1)
                loss = self.CEloss(r_hat_ce, labels)
                total_loss += loss
                ensemble_acc[member] += (r_hat_ce.argmax(1) == labels).sum().item()
                ensemble_losses[member] += loss.item()
            total_loss.backward()
            self.opt.step()
        torch.cuda.empty_cache()
        return ensemble_acc / max(total, 1)
