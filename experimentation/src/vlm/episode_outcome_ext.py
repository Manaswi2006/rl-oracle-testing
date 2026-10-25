"""Episode-level outcome reward in CoRe's FRM slot.  After every episode a verdict y in [0,1] is obtained and written to the reward of ALL
steps of that episode (FRM slot; mixed with the RRM exactly like the other structural rewards: (1-a)*y + a*RRM):

  source=oracle : y = the simulator's episode success flag                       (privileged upper bound, episode-level)
  source=vlm    : y = aggregate of k paraphrased, chain-of-thought VLM verdicts on 5 frames (first + last four of 16 uniformly spaced frames)
                  agg=majority -> y in {0,1};  agg=soft -> y = fraction of 'Yes' votes.
In vlm mode the simulator success flag is NEVER used by the pipeline (it is only logged to score the VLM's accuracy).
"""
import io, json, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from src.common.ext_base import RewardExt

T_EP = 500
FRAME_STEPS = set(np.linspace(0, T_EP - 1, 16).round().astype(int).tolist())


def jpeg256(rgb):
    img = cv2.resize(rgb, (256, 256))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


class EpisodeOutcomeExt(RewardExt):
    name = "ep_outcome"
    static_reward = True            # rewards are written per episode at its end; relabelling only re-mixes with the RRM
    wants_frames = True

    def __init__(self, cfg, obs_dim, act_dim, device, seed):
        super().__init__(cfg, obs_dim, act_dim, device, seed)
        p = dict(source="oracle", k=3, agg="majority", ports=[8000], task="button", model="qwen3vl8b", weight=1.0, rnd_beta=0.0)
        p.update(dict(cfg.get("ep", {}) or {}))
        if isinstance(p["ports"], (int, str)): p["ports"] = [int(p["ports"])]
        self.p = p
        self.ready = True
        self.oracle_information_used = ("episode success flag (simulator) as episode-level reward [oracle]" if p["source"] == "oracle" else
                                        "none from the simulator: VLM verdicts on rendered frames (simulator success only logged to score label accuracy)")
        self.frames, self.ep_start, self.ep_end = [], {}, {}
        self.records = []
        self.runner = None
        self.n_queries = 0
        self.vlm = None
        if p["source"] == "vlm":
            from src.vlm.vlm_client import VLM
            self.vlm = VLM(ports=p["ports"], model=p["model"], workers=p["k"])
        self.rnd = None
        self.r_int = np.zeros(int(cfg.num_train_steps) + 5, np.float32)
        self._n = 0
        if float(p["rnd_beta"]) > 0:
            from src.vlm.rnd import RND
            self.rnd = RND(obs_dim, seed=seed)
        self.task = str(cfg.env).replace("metaworld_", "").split("-")[0]
        self.task = {"button": "button", "hammer": "hammer"}.get(self.task, self.task)

    def bind_runner(self, runner):
        self.runner = runner

    def step_reward(self, success, next_obs=None):
        """per-step part of the reward: the RND novelty bonus (collection-time value, as in standard RND); the episode verdict is added at the episode end."""
        r = 0.0
        if self.rnd is not None and next_obs is not None:
            r = float(self.p["rnd_beta"]) * min(self.rnd.step(next_obs), 10.0)
        if self._n < len(self.r_int):
            self.r_int[self._n] = r
        self._n += 1
        return r

    def on_frame(self, t, rgb):
        if t in FRAME_STEPS:
            self.frames.append(jpeg256(rgb))

    def on_step(self, ep_id, t, obs, act, next_obs, env_r, success, step):
        if t == 0:
            self.ep_start[ep_id] = step
        self.ep_end[ep_id] = step + 1

    def _verdict(self, frames, true_success):
        if self.p["source"] == "oracle":
            return float(true_success > 0), []
        from src.vlm.vlm_client import COT_OUTCOME, TASK_DESC, b64
        import re
        sel = [frames[0]] + frames[-4:]
        def one(q):
            content = [{"type": "image_url", "image_url": {"url": b64(f)}} for f in sel] + [{"type": "text", "text": q.format(task=TASK_DESC[self.task])}]
            for attempt in range(10):
                try:
                    ch = self.vlm._chat(content, max_tokens=220, logprobs=False)
                    txt = ch.message.content or ""; m = re.findall(r"VERDICT:\s*(Yes|No)", txt, re.I)
                    return int(bool(m) and m[-1].lower() == "yes"), txt[-160:]
                except TypeError:
                    raise
                except Exception as e:
                    time.sleep(3 + 3 * attempt); err = e
            raise err
        with ThreadPoolExecutor(self.p["k"]) as ex:
            rs = list(ex.map(one, COT_OUTCOME[: self.p["k"]]))
        votes = [r[0] for r in rs]; self.n_queries += len(votes)
        y = float(np.mean(votes)) if self.p["agg"] == "soft" else float(np.mean(votes) > 0.5)
        self._last_texts = [r[1] for r in rs]
        return y, votes

    def on_episode_end(self, ep_id, ep_success, step):
        frames, self.frames = self.frames, []
        if len(frames) < 16:                      # should not happen (500-step episodes)
            frames = (frames * 16)[:16] if frames else None
        y, votes = self._verdict(frames, ep_success)
        y = y * self.p["weight"]
        if self.p["source"] == "vlm" and y > 0.5 and ep_success == 0 and getattr(self, "_fp_saved", 0) < 30 and frames:
            import os
            d = os.path.join(str(self.runner.run_dir), "fp_examples"); os.makedirs(d, exist_ok=True)
            for nm, jb in (("first", frames[0]), ("last", frames[-1]), ("mid", frames[-3])):
                open(os.path.join(d, f"ep{ep_id}_{nm}.jpg"), "wb").write(jb)
            for k, jb in enumerate([frames[0]] + frames[-4:]):               # the exact 5 frames the judge saw
                open(os.path.join(d, f"ep{ep_id}_judge{k}.jpg"), "wb").write(jb)
            open(os.path.join(d, f"ep{ep_id}_reasoning.json"), "w").write(json.dumps(getattr(self, "_last_texts", [])))
            self._fp_saved = getattr(self, "_fp_saved", 0) + 1
        s, e = self.ep_start[ep_id], self.ep_end[ep_id]
        rb = self.runner.replay
        a = self.runner.alpha
        part = self.r_int[s:e] if self.rnd is not None else 0.0
        rb.rewards_frm[s:e, 0] = y + part
        rb.rewards[s:e, 0] = (1 - a) * (y + part) + a * rb.rewards_rrm[s:e, 0]
        self.records.append(dict(ep=int(ep_id), step=int(step), y=float(y), votes=votes, true_success=float(ep_success)))

    def reward_batch(self, obs, act, next_obs):
        return np.zeros(len(obs), np.float32)

    def stats(self):
        r = self.records
        out = dict(params=self.p, n_episodes=len(r), n_vlm_queries=self.n_queries)
        if r and self.p["source"] == "vlm":
            y = np.array([x["y"] > 0.5 for x in r]); t = np.array([x["true_success"] > 0 for x in r])
            out.update(label_acc=float((y == t).mean()), fp=int((y & ~t).sum()), fn=int((~y & t).sum()), n_true_success=int(t.sum()))
        return out

    def save(self, path):
        json.dump(self.records, open(f"{path}/episode_verdicts.json", "w"))
