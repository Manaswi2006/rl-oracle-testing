"""Synthetic episode-outcome label models used for P-SVM (A2/A4).

Every model maps (episode id, true simulator outcome y) -> q in [0,1] (soft weight), or NaN (= episode
discarded).  They emulate k paraphrased VLM judgements per episode, INCLUDING correlated errors: an episode
can be 'systematically' misjudged, in which case most paraphrases err together (so 6/7 positive votes is NOT
86% probability of success).  The simulator outcome y is only used to synthesise the corrupted label and, for
`soft_calibrated`, to fit a small calibration set (counted in `n_oracle_calibration_labels`).
"""
import numpy as np


class VoteModel:
    def __init__(self, p, seed):
        self.p = p
        self.rng = np.random.RandomState(10_000 + 97 * seed + int(p.get("vlm_seed", 0)))
        self.k = int(p.get("k_votes", 7))
        self.tpr = float(p.get("tpr", 0.9))
        self.tnr = float(p.get("tnr", 0.9))
        self.sys_err = float(p.get("sys_err", 0.0))
        self.sys_conf = float(p.get("sys_conf", 0.9))     # prob. each vote repeats the systematic mistake
        self.mode = p["mode"]
        self.calib_n = int(p.get("calib_n", 30))
        self.calib = []                                   # (raw_q, y) for the calibration episodes
        self.n_oracle_calibration_labels = 0

    def votes(self, y):
        if self.rng.rand() < self.sys_err:                # correlated, confident, wrong
            wrong = self.rng.rand(self.k) < self.sys_conf
            return np.where(wrong, 1 - y, y).astype(float)
        acc = self.tpr if y > 0 else self.tnr
        correct = self.rng.rand(self.k) < acc
        return np.where(correct, y, 1 - y).astype(float)

    def __call__(self, ep_id, y):
        y = float(y > 0)
        m = self.mode
        if m == "oracle_hard":
            return y
        if m == "hard_flip":                              # one noisy hard label
            fp, fn = float(self.p.get("flip_fp", 0.0)), float(self.p.get("flip_fn", 0.0))
            flip = self.rng.rand() < (fn if y > 0 else fp)
            return 1.0 - y if flip else y
        v = self.votes(y)
        q = float(v.mean())
        if m == "majority":
            return float(q > 0.5)
        if m == "soft_raw":
            return q
        if m == "conf_filter":
            th = float(self.p.get("conf_thresh", 0.8))
            return float(q > 0.5) if max(q, 1 - q) >= th else np.nan
        if m == "soft_calibrated":
            if len(self.calib) < self.calib_n:            # first episodes: use the (oracle) calibration labels
                self.calib.append((q, y)); self.n_oracle_calibration_labels += 1
                return y
            return self._calibrate(q)
        raise ValueError(m)

    def _calibrate(self, q):
        """histogram-binning calibrator on the calibration set (k+1 bins, Laplace smoothed)."""
        qs = np.array([c[0] for c in self.calib]); ys = np.array([c[1] for c in self.calib])
        b = np.isclose(qs, q)
        return float((ys[b].sum() + 0.5) / (b.sum() + 1.0))


def make_label_model(p, seed):
    return VoteModel(p, seed)
