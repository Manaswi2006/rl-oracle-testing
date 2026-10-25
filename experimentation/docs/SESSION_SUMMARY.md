# Session summary: everything we tried, what worked, what did not, and where to go next

*Written 2026-10-25 during the session. Part 2 (the VLM outcome study) was still running when this was written; its final numbers are in `reports/VLM_OUTCOME_REPORT.md` once that exists. Everything marked **[verified]** was measured in this session and is backed by files in the repository / Hugging Face dataset. Everything marked **[not verified]** is my reasoning or a literature claim that I only read at abstract level.*

---

## 0. How to read this document

1. Section 1 is the **overall goal** and the chronology of what you asked and what I did.
2. Part 1 (Sections 2-8) is the **original assignment**: reproduce CoRe and test three ideas (P-SVM, FACR, RLF). It is finished, pushed to GitHub, uploaded to Hugging Face, and the local folder was deleted.
3. Part 2 (Sections 9-13) is the **follow-up study you asked for**: can a small open-weight VLM replace the privileged signals so that the reward pipeline works on unseen tasks? It is in progress.
4. Section 14 lists **every idea** that came up (yours and mine), with a verdict.
5. Section 15 is **what works / what does not**, consolidated.
6. Section 16 is **what I think we should do next** (with decision criteria).
7. Sections 17-18 are the **inventory of artifacts** and an **honesty log** (mistakes I made and corrected, things I could not verify).

The short version of the whole session:

* **None of the three original ideas (P-SVM, FACR, RLF) is supported by the evidence** against its simplest control. The most striking result was a *control* (RRM + plain success reward) that beat all three ideas and full CoRe.
* That control showed the thing that matters is a **reliable success signal**. This led to your goal: get that signal without the simulator, from a VLM, so the method can work on unseen tasks.
* A small open VLM (Qwen3-VL-8B) **can judge success from rendered frames about 85-90% of the time** if it is allowed to reason before answering, but its **pairwise preferences and progress scores are at chance**. So the success signal is replaceable; the dense guidance toward the goal is not (yet).
* The remaining gap is therefore **guidance without privileged information**. Candidate replacements (exploration bonus, VLM subgoals, self-discovered milestones, perception-grounded rewards) are discussed in Section 14; most have prior art, and our own related oracle experiments suggest that anything learned only from successes will fail on the cold start.

---

## 1. Goal and chronology

**Your stated goal (late in the session):** *"remove privileged rewards coming from LLMs so our way of defining rewards can do well on new unseen tasks too."*

### 1.1 Chronology of requests

| # | What you asked | What I did |
|---|---|---|
| 1 | "make a folder named experimentation in working/manaswi" | Created `/home/colligo/working/manaswi/experimentation`. |
| 2 | The long assignment (autonomous RL research on CoRe; three ideas P-SVM, FACR, RLF; reproduce CoRe; GitHub + Hugging Face upload; delete the folder at the end only after verified transfer) | Executed end to end (Part 1). |
| 3 | You pointed me to the HF token in `DatasetDiscovery/.env`, told me to use all 8 GPUs and not stop for permission | Used the token at upload time only (never stored in the workspace); used all 8 GPUs. |
| 4 | "will you be able to do everything by yourself while I sleep?" / "keep going, run full" | Kept the session alive, ran everything, finished with GitHub + HF + cleanup. |
| 5 | "should you see any tougher tasks?" | Explained Peg Insert was already the hard task (all zeros); added Dial Turn for key arms instead. |
| 6 | "so do you have any idea what hypothesis I should test?" | Proposed H1-H3 (VLM success verdict + learned reward; correlated-error aggregation; unseen-task transfer). |
| 7 | "which VLM is best?" | Surveyed Hugging Face; recommended Qwen3-VL (8B first, 32B as backup). |
| 8 | "does CoRe take the yes/no signal from the VLM?" | Answered: no (see Section 4.1); CoRe's code uses the simulator success flag for its LLM-reward refinement loop. |
| 9 | "all nos sounds weird" | Showed the model's reasoning text and explained the first-token bias (Section 10). |
| 10 | "what about frames instead of video / image LLMs?" | Clarified we already use still frames; listed untested options. |
| 11 | "what can we replace the guidance with ... think please" | Listed options 1-4 with evidence (Section 14). |
| 12 | "option 2 already used/not novel? look at option 1 too; state the full hypothesis" | Searched arXiv; wrote the full hypothesis (Section 12). |
| 13 | "option 2: are we using anything privileged?" | Answered (Section 14.3). |
| 14 | "what if subgoals are discovered with RL instead of asked from an LLM?" | Discussed (Section 14.4). |
| 15 | This document. | |

---

## PART 1 - The original assignment (CoRe, P-SVM, FACR, RLF)

## 2. Background and setup

### 2.1 The base paper: CoRe

CoRe (arXiv 2607.01721, ICML 2026, code https://github.com/nihx9919/CoRe, commit `d663cabc38e6d6f39ac5e1e4327d94f17f84fa59`) combines two reward components:

* **FRM (formal reward module):** an LLM writes executable reward code for each task using task-specific state features (object positions, goal position, ...). This is the privileged part you want to remove.
* **RRM (residual reward module):** a CNN reward model trained on video-level preferences produced by a VLM ("which of these two clips is closer to the task being done?"), with per-frame importance weights.

Reward fed to SAC: `(1-a)*FRM + a*RRM`, a = 0.5 after step 20k (FRM only before that).

**Published results [verified against the paper's Table 2]:** RRM-only vs full CoRe: Sweep Into 94.0 -> 97.3, Button Press 8.7 -> 100, Dial Turn 36.0 -> 98.0, Hammer 49.3 -> 98.0, Peg Insert 36.0 -> 100. Each from **three seeds**, 500k steps, with online FRM refinement and a Gemini-based RRM.

### 2.2 Related work used

* **SVM** (Success Visitation Matching, arXiv 2606.23640): a discriminator between state-action pairs of successful and unsuccessful episodes; reward `r = r_out + lambda * clip_beta(logit)`, lambda = 0.05, beta = 20, one discriminator for all time steps. I read this paper in depth.
* RL-VLM-F, PEBBLE, RIME, TREND, T2-VLM, ReLAM, TimeRewarder etc.: handled by a literature-review sub-agent (`reports/prior_art_review.md`). I re-checked 7 ids/titles myself. Most 2025-26 papers were read at abstract/method level only.

### 2.3 Hardware and environment

* 8 x NVIDIA A100-SXM4-80GB, 96 vCPU Xeon 8275CL, 1.1 TB RAM, ~60 GB free disk at the start.
* Python 3.9 venv (via `uv`), torch 1.13.1+cu117, mujoco 3.1.1, the MetaWorld copy bundled with CoRe, headless rendering via EGL.
* **No upstream file was modified.** All changes live in `src/`.

### 2.4 Deviations from the CoRe setup (all uniform across arms)

1. **No VLM was authorised in Part 1.** The RRM's preference labels came from a *scripted teacher on the privileged environment reward* with 15% symmetric label noise (`src/common/rrm_scripted.py`). So "RRM" in Part 1 is not CoRe's RRM. It is PEBBLE with images and a noisy reward teacher.
2. FRM = the released final FRM files, **static** (no online LLM refinement, which would call an LLM).
3. 200k steps instead of 500k; RRM images 128 px instead of 300 px.
4. State-based modules (39-d observation, goal hidden), not pixels, for P-SVM/FACR.
5. 10-episode evaluations every 10k steps; **final evaluation 50 episodes per seed** on a separate environment instance.
6. 3 seeds per cell (screening scale), 6 seeds for the key arms on Button Press and Hammer.

### 2.5 Seeds, tasks, budgets

* Main tasks: Button Press (`button-press-topdown-v2`), Hammer, Peg Insert. Dial Turn for key arms (3 seeds).
* Not run: Sweep Into, Drawer Open, Soccer, SoftGym tasks.
* **255 training runs** in total (21 of them are an archived flawed first attempt, Section 18).
* Persistence: runs are idempotent (`DONE` marker) with a job queue (`scripts/run_queue.py`); there are **no mid-run checkpoints**, so interrupted runs restart from step 0.

---

## 3. The three ideas as implemented

### 3.1 Idea A: P-SVM-CoRe (`src/psvm/`)

*Question:* can CoRe's LLM-written reward be replaced by a dense reward learned from success-vs-failure visitation, and can that work with noisy (VLM-style) outcome labels?

Implementation:

* One MLP discriminator `f(s,a)` over state-action pairs of all steps; positives are the steps of successful episodes **truncated at the first success** (terminal convention of the SVM paper), negatives are all steps of failed episodes.
* Class-prior handling: **class-balanced sampling**, so the logit equals the log-ratio of the *normalised* visitation densities (unit-tested on a 1%/99% Gaussian toy: logit within 0.35 of 2x).
* Reward `0.05 * clip(logit, -20, 20)` in [-1, 1]. Mixed like CoRe: `(1-a)*r_svm + a*RRM`.
* **P-SVM extension** (the new part): each episode carries a soft weight q_i and the loss is the q-weighted, class-normalised BCE. q in {0,1} recovers hard-label SVM.
* Deviations from the SVM paper: state-vector discriminator (no images), 200 gradient steps of batch 256 every 1000 env steps, trained from scratch (no pretrained policy rollouts), no `r_out` term in the main arm.

### 3.2 Idea B: FACR, feasibility-aware counterfactual reward (`src/facr/`)

*Question:* can a learned "probability of success from here" plus a learned dynamics model give an action-level signal that beats plain progress shaping?

* `V(s,h) = P(success within h steps | s)`, trained with BCE on future-success labels from simulator success flags (horizons 10/25/50/100).
* A 5-member dynamics ensemble `p(s'|s,a)` (predicts delta s).
* `Q_cf(s,a) = E_e V(s_hat'_e, H-1)`; `A_cf = Q_cf(s,a) - mean_{a' ~ pi} Q_cf(s,a')`; reward `clip(0.5 * A_cf / sigma, -1, 1)`.
* Comparators: `facr_value` (gamma V(s') - V(s) shaping) and `facr_progress` (a TimeRewarder-style progress regressor).
* **No claim** that `A_cf` is an unbiased causal advantage or policy-invariant shaping; I explicitly listed it as a hypothesis.

### 3.3 Idea C: RLF, reward learning through falsification (`src/falsification/`)

*Question:* does an adversarial "attack" policy that finds states where the learned reward is high but the task is not done give better counterexamples than normal sampling, and does repairing the reward with them help?

* The learned reward is CoRe's RRM. Every RRM round (every 4k task steps) a **selector** proposes Q = 5 candidate episodes; the oracle verifier (simulator success flag) labels them; a verified failure that the RRM scores in the top half of the recent pool is a **confirmed reward failure**.
* **Repair:** add the preference pair "window ending at the first success of a verified successful episode > window of the confirmed failure" to the RRM's buffer and retrain.
* Selectors: `onpolicy` (ordinary on-policy correction), `highreward` (hard-negative mining), `uncertainty` (ensemble disagreement), `random`, `disagreement` (oracle upper bound), `attack` (a separate SAC maximising `R_theta(s) - lambda * success`), `hybrid` (attack + uncertainty bonus).
* Control `rrm_extra`: same number of extra labels (125) from the ordinary teacher, so label count is not confounded with verification.

---

## 4. Findings from the original study

### 4.1 Things I learned about CoRe's code that matter

* **CoRe does not use a VLM yes/no verdict.** The VLM gives pairwise preferences between video clips (plus frame importance weights); the LLM writes reward code.
* **CoRe's code uses the simulator's success flag in the FRM refinement loop.** In `train_CoRe.py`, for MetaWorld, the per-episode `episode_success` is logged as the score fed back to the LLM that refines the reward code. This is a privileged signal in CoRe itself. In our port the refinement was disabled.
* CoRe's task descriptions are hand-written (e.g. "to press the red button down completely from top to bottom") and CoRe also supplies initial/goal images for its VLM prompts (we did not use goal images in Part 2).
* **Upstream replay-buffer quirk:** the (1-alpha) weight is applied at collection time and again at relabel time, so FRM rewards of post-phase samples end up with an effective weight 0.25. Our port of `core` keeps this behaviour; our learned-reward arms apply the weight once.
* `FRMCode.__init__` *rewrites the environment source file* at runtime to inject `self.target_pos_frm`. (I first misreported this as a missing attribute bug; it is a runtime source injection. I retracted that. Our port reads `env._target_pos` directly.)

### 4.2 Baseline reproduction [verified]

Final success %, mean over seeds (n = 6 for Button Press / Hammer, 3 for Peg Insert and Dial Turn):

| Method | Button Press | Hammer | Peg Insert | Dial Turn |
|---|---|---|---|---|
| Env dense reward | 100 (n=3) | 66.7 (n=3) | 100 (n=3) | - |
| Env sparse reward | 0 (n=3) | 66.7 (n=3) | 0 (n=3) | - |
| RRM only (simulated teacher) | 16.7 | 16.7 | 0 | 33.3 |
| Full CoRe (our reproduction) | 66.7 | 66.7 | **0** | **0** |
| CoRe published | 100 | 98.0 | 100 | 98.0 |

* Environment baselines reproduce (the dense Hammer miss is one failed seed).
* **Full CoRe reproduces only partially**: 0% on Peg Insert and Dial Turn where the paper reports 100% and 98%. Likely causes (untested): 200k vs 500k steps, no online FRM refinement, our simulated-teacher RRM, 128-px images.
* All "gap closure" numbers use **our observed baselines**.

### 4.2.1 Learning-curve properties

Per-seed outcomes are almost always **0% or 100%** after a 50-episode evaluation. With 3-6 seeds the standard error of a mean is 20-33 points and 95% bootstrap intervals are about +-50 points. Only the largest effects are distinguishable from noise. This statistical weakness is the single biggest limitation of the whole study.

### 4.3 Idea A results (P-SVM)

| Arm | Button Press | Hammer | Peg Insert | Dial Turn |
|---|---|---|---|---|
| RRM + oracle SVM (`psvm_oracle`) | 83.3 | 66.7 | 0 | 66.7 |
| **Control: RRM + plain sparse success reward (`outcome_sparse`)** | **100** | **100** | 0 | 66.7 |
| Control: oracle SVM without RRM (`psvm_noRRM`, n=3) | 0 | 33.3 | - | - |

* **Provisional screening rule** (close >= 50% of the RRM->CoRe gap on >= 2 of 3 difficult tasks): `psvm_oracle` gives G = 1.33 on Button Press, 1.00 on Hammer; Peg Insert undefined (both baselines 0%). **Formally passes.**
* **But the control beats it:** `outcome_sparse - psvm_oracle` = +16.7 (CI [0, +50]) on Button Press and +33.3 (CI [0, +66.7]) on Hammer (95% bootstrap over seeds). So the gain over RRM-only is explained by the information in the **oracle success flag**, not by the visitation-matching mechanism. SVM alone cannot replace the learned reward (0% / 33%).
* The success flag in the control is a **per-step** flag; the same signal given only at episode level is tested in Part 2.

**A2: synthetic label corruption on fixed trajectories** (episode AUC of the discriminator's mean logit for the true outcome of held-out episodes, natural prevalence; 3 seeds x 3 tasks):

| Scenario | oracle | hard | majority | soft_raw | conf_filter | soft_calibrated |
|---|---|---|---|---|---|---|
| clean | 0.962 | 0.962 | 0.962 | 0.962 | 0.962 | 0.982 |
| iid flips 10% | - | 0.892 | 0.960 | 0.974 | 0.957 | 0.983 |
| iid flips 30% | - | 0.837 | 0.920 | 0.966 | 0.947 | 0.956 |
| asymmetric FP 20% | - | 0.883 | 0.947 | 0.982 | 0.963 | 0.984 |
| asymmetric FN 20% | - | 0.953 | 0.962 | 0.961 | 0.957 | 0.984 |
| correlated errors (20% systematic) | - | 0.820 | 0.905 | 0.919 | 0.861 | 0.934 |
| correlated errors (40% systematic) | - | 0.685 | **0.807** | 0.749 | 0.734 | 0.696 |
| instance-dependent | - | 0.947 | 0.960 | 0.977 | 0.960 | 0.981 |

Interpretation: soft vote fractions are best under independent noise, but **under correlated errors the advantage vanishes and calibrating on 30 clean labels does not rescue it**. At 5% success prevalence everything degrades (hard 0.62, soft 0.77). "Repeated votes are calibrated probabilities" is false when errors are correlated; this later turned out to be the exact failure of the VLM's direct answers (Section 10.1).

Not run: A3 (real VLM labels), A4 (mixing coefficient / architecture / online noisy-label ablations).

### 4.4 Idea B results (FACR)

**B1 oracle success value** (trained on seeds 0-1, tested on seed 2 / other methods' data, h = 50): AUC 0.95-0.98 in distribution, ECE 0.01-0.04; with a *single* successful training episode AUC is only 0.38-0.41 (Button/Hammer); ranking becomes useful from ~4-8 successful episodes. This is the cold-start problem.

**B2 counterfactual validity** (simulator branching: 360 pre-success states per task, 7 candidate first actions each, 10 independent continuations per action; only 14 / 33 / 5 states had action-discriminable outcomes, which is a finding in itself: one action rarely changes 50-step success):

| predictor (H = 50) | Button Spearman | Hammer | Peg |
|---|---|---|---|
| FACR Q_cf (learned dynamics) | 0.03 | 0.19 | 0.31 |
| V(true simulator next state) (perfect dynamics) | -0.03 | 0.20 | 0.42 |
| FRM reward of true next state (geometric proximity) | -0.09 | 0.26 | 0.12 |
| **state level:** Spearman(V(s,50), empirical success) | **0.85** | **0.69** | **0.67** |
| state level: Spearman(FRM(s), empirical success) | -0.18 | 0.56 | 0.19 |

* The learned **success value is a much better predictor of real success than geometric proximity** (supports the "feasibility vs closeness" intuition).
* The **action-counterfactual part adds nothing** over V of the true next state. Shorter horizons are far more action-sensitive (H = 10: Spearman 0.54 / 0.41 / 0.49).
* B4 model bias: single member vs ensemble, perfect dynamics, data fraction 10%/30%: no consistent differences. Model *exploitation* was small (overestimation > 0.1 in 1.4-3.1% of states), so the failure is weak signal, not exploitation.

**B3 policy learning** (oracle): `facr_facr` 50.0 / 33.3 / 0 vs `facr_value` 33.3 / 0 / 0 and `facr_progress` 0 / 33.3 / 0 (Button / Hammer / Peg); RRM-only 16.7 / 16.7 / 0; wall-clock cost 1.3x. FACR beats `facr_value` on Hammer (+33.3, CI [0, +66.7]) but not RRM-only, and is far below the simple outcome control.

Not run: B5 (non-oracle FACR), pixel representations.

### 4.5 Idea C results (RLF)

**Final success % (Button Press / Hammer, n = 6 each):**

| arm | Button | Hammer | extra env steps |
|---|---|---|---|
| RRM-only | 16.7 | 16.7 | 0 |
| RRM-only + 125 extra teacher labels (control) | 16.7 | 0 | 0 |
| RLF repair, on-policy selection | 50.0 | 16.7 | 0 |
| RLF repair, high-reward (hard-negative) | 66.7 | 16.7 | 0 |
| RLF repair, uncertainty | 66.7 | 16.7 | 0 |
| RLF repair, falsification policy (attack) | 66.7 | 0 | +90,000 |
| full CoRe | 66.7 | 66.7 | 0 |
| RRM + plain sparse success reward | 100 | 100 | 0 |

(hybrid / random / disagreement have n = 2: Button 100 / 50 / 100, Hammer 0 / 0 / 0.)

**C2 / C5 discovery and distinctness:**

| selector | confirmed failures per verification query | extra env steps | fraction of confirmed-failure cells not visited by the task policy |
|---|---|---|---|
| onpolicy | 0.447 | 0 | 0.385 |
| highreward | 0.778 | 0 | 0.423 |
| uncertainty | 0.611 | 0 | 0.413 |
| random policy | 0.205 | 60,000 | 0.703 |
| attack (falsification) | 0.376 (0.53 excluding anchor queries) | 90,000 | 0.754 |

* **45% of random on-policy episodes are confirmed reward failures**: the learned reward frequently ranks failing behaviour as good.
* The attack policy finds *different* regions (75% novel cells), but so does a random policy (70%), and it confirms fewer failures per query than cheap selectors. "Different" does not imply "useful".
* **Decision:** the distinct falsification-policy mechanism is rejected. The weaker claim "verified counterexample repair helps" is supported only on Button Press (uncertainty - control: +50, CI [0, +100]).

### 4.6 Final decision of Part 1 (`RESEARCH_DECISION.json`)

* Recommended idea: **NONE** as framed. Scores (1-5): P-SVM empirical 1 / novelty 1 / difficulty 2; FACR 1 / 2 / 4; RLF 2 / 3 / 4.
* Novelty classes (from the literature review): P-SVM *Incremental*, FACR *Incremental* (the success-value part is eVTA0 arXiv 2609.33653 and BSRS 2501.00989), RLF *Potentially novel in setting* (attack-then-repair is Adv-RM 2504.06141 / REFORM 2507.06419 for LLM reward models) but unsupported empirically.

---

## 5. What I learned about running this kind of study (infrastructure notes)

These are practical lessons worth keeping:

* **A single EGL device problem:** MuJoCo's EGL renderer ignores `CUDA_VISIBLE_DEVICES` and puts every process's render context on **GPU 0** unless `MUJOCO_EGL_DEVICE_ID` is set. This made GPU 0 the slowest GPU in Part 1 (unnoticed) and caused segfaults in Part 2 once the VLM server filled GPU 0's memory. Fixed in `scripts/run_queue.py` (sets `MUJOCO_EGL_DEVICE_ID` to the job's GPU).
* Throughput was **GPU-time-slice bound** (~66 processes on 8 GPUs, 15-45 steps/s each). CUDA MPS gave a modest improvement for the jobs that used it.
* Memory per image-based run grows to ~16 GB (replay buffer of images), so concurrency was limited to ~56-66 runs.
* Queue bugs found and fixed: duplicate launches (pgrep-based duplicate detection), all jobs landing on one GPU (placement lag), and my own `pkill -f` patterns killing my shell.
* **No mid-run checkpoint/resume** was implemented; ten runs that were starved on an overloaded GPU were killed and restarted from step 0.

---

## PART 2 - The VLM outcome study

## 9. Motivation and design

The control in Part 1 (RRM + plain success reward, 100% / 100%) told us that **a reliable success signal is extremely valuable**. But that signal came from the simulator. Your goal is to make the reward pipeline work on **unseen tasks without privileged information**. So the natural next question: *can a VLM supply the success signal?* and *what else in the pipeline is privileged?*

Privileged pieces in the Part 1 pipeline:

| piece | where it is privileged |
|---|---|
| FRM (CoRe) | LLM writes reward code from simulator state features |
| RRM teacher (our port) | scripted teacher on the environment reward |
| success flag (our control) | simulator success flag |
| CoRe's FRM refinement | uses the simulator success flag as feedback |

### 9.1 Choice of VLM

Surveyed Hugging Face (2026-10): Qwen3-VL (2B/4B/8B/30B-A3B/32B/235B), Qwen2.5-VL, InternVL3/3.5, Gemma-4, MiniCPM-V, Molmo2. I chose **`Qwen/Qwen3-VL-8B-Instruct`** (Apache-2.0, ungated, accepts multiple images with text, many sizes available for later scaling). 32B was kept as a backup but needs ~65 GB disk; we had ~60 GB free.

### 9.2 Serving

vLLM 0.31.0 (separate `.venv_vlm`, managed Python 3.10 because the system Python lacks headers), model cached inside the workspace, one server on GPU 0 (`--max-model-len 20000`, up to 28 images per prompt, video disabled), `VLLM_USE_FLASHINFER_SAMPLER=0` (the system `nvcc` cannot build FlashInfer kernels). The server was never the bottleneck.

## 10. VLM label-quality study [verified]

### 10.1 Ground-truth episode banks

For Button Press and Hammer I rolled out actors from the earlier study (66 checkpoints downloaded from Hugging Face) with four noise settings each plus random-action episodes, **456 episodes per task** (Button Press 61% success, Hammer 43%). Per episode: 16 uniformly spaced frames (JPEG, 256 px, CoRe's camera and flip), 8 random 20-step windows with 6 frames each and the summed environment reward, true success flag and first-success step. Ground truth is used only to *evaluate* the VLM.

I looked at sample frames. On Button Press **the arm is never visible** (the camera shows only the button and its box); success shows the button flush with the box. On Hammer the hammer and the nail are visible, but the nail depth is a subtle cue.

### 10.2 Finding 1: direct one-word answers are useless

Asking "Did the robot complete the task? Answer Yes or No." with 5 paraphrases (16 frames):

| | Button Press | Hammer |
|---|---|---|
| accuracy (majority of 5) | 0.436 | 0.579 |
| recall of true successes | **0.082** | **0.026** |
| mean pairwise error correlation across paraphrases | **0.91** | **0.95** |
| all 5 paraphrases wrong | 53% of episodes | 38% |

The model says "No" almost always, with **errors that are almost perfectly correlated across paraphrases** (exactly the correlated-error scenario of the A2 experiment). The token-probability version was even *inverted* (AUC 0.23 / 0.14). Direct preferences were also degenerate (always choosing clip B; Button Press AUC 0.32-0.37).

### 10.3 Finding 2: reasoning before the verdict fixes the outcome question

Prompt sweep on a **dev half** of the bank (class-balanced random sample; the test half was never used for selection):

| variant (256 px) | Button bal. acc | Hammer bal. acc |
|---|---|---|
| reason + verdict, all 16 frames | 0.875 | 0.713 |
| reason + verdict, first + last 7 frames | 0.856 | 0.631 |
| reason + verdict, first + last frame | 0.500 (always yes) | 0.594 |
| **reason + verdict, first + last 4 frames** | **0.925** | 0.694 |

At 448 px the numbers were similar (Button Press 0.84-0.91, Hammer 0.66-0.74), so **resolution is not the limiter**. I froze one prompt for both tasks (selected on dev by mean balanced accuracy across the two tasks): **first frame + last 4 frames, short reasoning, then `VERDICT: Yes/No`, 3 paraphrased prompts**.

Examples of the model's reasoning: *"In the last frame, the button is fully depressed, sitting flush with the yellow surface of the base"* (correct, Button Press); *"the nail is still only partially embedded, indicating the task is incomplete"* (Hammer).

### 10.4 Finding 3: held-out TEST performance of the frozen judge

(228 test episodes per task, natural prevalence, majority of 3 paraphrases):

| | Button Press | Hammer |
|---|---|---|
| accuracy | 0.899 | 0.846 |
| balanced accuracy | 0.892 | 0.865 |
| precision | 0.908 | 0.737 |
| recall | 0.928 | 0.956 |
| false-positive rate | 0.144 | 0.226 |
| false-negative rate | 0.072 | 0.044 |
| Brier / ECE of the vote fraction | 0.077 / 0.031 | 0.123 / 0.079 |
| mean pairwise error correlation | 0.23 | 0.19 |

The judge is decent and the vote fraction is reasonably calibrated; errors are far less correlated than with direct answers. **Hammer has many false positives** (23%).

### 10.5 Finding 4: preferences and progress scores are at chance

| question | Button Press | Hammer |
|---|---|---|
| pairwise preference, direct (both orders) | always picks "B"; consistent 0-18% | always picks "B"; consistent 0.6-38% |
| pairwise preference with reasoning | accuracy 0.43-0.46, order-consistent 30% | accuracy 0.51-0.54, consistent 27.5% |
| pointwise 0-100 progress score of the window's last frame | 93% ties (all scores ~1); accuracy 0.55 excluding ties | 49% ties; 0.51 excluding ties |

Ground truth for preferences = larger summed environment reward in the window (|relative difference| >= 0.3). So with an open 8B VLM and 6-frame windows, **the VLM cannot supply the dense guidance that CoRe's RRM relies on** (consistent with CoRe's own RRM-only result on Button Press of 8.7%). On Button Press this is unsurprising: the arm is invisible, so there is little to compare.

## 11. The policy-learning experiment (stage 6)

### 11.1 Design

* New reward module `src/vlm/episode_outcome_ext.py`: after each episode a **verdict y** is obtained and written as the reward of **all steps of that episode** (FRM slot; mixed with the RRM as `(1-a)*y + a*RRM`).
  * `source=oracle`: y = simulator episode success (episode-level upper bound).
  * `source=vlm`: y = aggregate of the 3 paraphrased reasoning verdicts on 5 frames. `agg=majority` gives {0,1}; `agg=soft` gives the fraction of Yes votes. **The simulator success flag is never used in this mode** (unit test `tests/test_episode_outcome.py` checks that the reward follows the VLM even when the true flag disagrees).
* Frames: 16 uniformly spaced frames are rendered from the same render call used for the RRM, JPEG-encoded at 256 px.
* The RRM is still trained by the **scripted teacher** (privileged) in the main arms. The `noRRM` arms set the RRM weight to 0 (`reward_alpha=0.0`), so the policy sees only the verdict (the RRM is still trained in the background but has no influence).

### 11.2 Arms (48 runs; 200k steps each)

| arm | tasks x seeds | reward the policy sees |
|---|---|---|
| `ep_oracle` | Button, Hammer x 6 | scripted RRM + oracle episode outcome |
| `ep_vlm_major` | x 6 | scripted RRM + VLM majority verdict |
| `ep_vlm_soft` | x 6 | scripted RRM + VLM vote fraction |
| `ep_oracle_noRRM` | x 3 | oracle episode outcome only |
| `ep_vlm_noRRM` | x 3 | VLM verdict only (no environment reward reaches the policy) |

References from Part 1 (same code): RRM-only 16.7 / 16.7, CoRe 66.7 / 66.7, per-step oracle success reward 100 / 100.

### 11.3 Status and provisional read (not final)

At about 45-80k steps (10-episode evaluations, still noisy): some VLM-judged runs already reach 100% (one Button Press seed in each VLM arm; two Hammer seeds in the soft-vote arm); the `noRRM` arms are at 0% except one Hammer seed. **Final numbers: see `reports/VLM_OUTCOME_REPORT.md` once the runs finish** (expected ~08:45 on 2026-10-25).

My expectation (stated before the results, so it can be checked): replacing the oracle verdict with the VLM verdict costs a moderate amount (limited by the 14-23% false-positive rate); the `noRRM` arms stay low because there is no guidance toward the first success.

### 11.4 Incidents in Part 2

* The first outcome-label save crashed (numpy int64 not JSON-serialisable) after the queries had already run; re-run (a few minutes).
* The old `openai==1.3.5` pinned by CoRe does not accept `logprobs` for chat completions; the client builds its arguments conditionally. A retry loop initially hid that error for 7 minutes; non-transient errors now re-raise.
* **EGL on GPU 0 full -> segfaults** for the last 12 jobs (Section 5). Fixed and relaunched; no further failures.

---

## 12. The hypothesis in full (as of the end of the discussion)

**Overall question (empirical, not a new algorithm):** *Can a CoRe-style reward pipeline run with no privileged information (no simulator state, no environment reward, no simulator success flag), by replacing each privileged piece with a task-agnostic component, and what does each replacement cost? And does the frozen pipeline transfer to tasks it was never tuned on?*

| piece | privileged version | replacement | status |
|---|---|---|---|
| success signal | simulator success flag | VLM verdict (frozen, instruction-only prompt, reason-then-verdict) | **running (stage 6)**; judge validated offline |
| preference teacher | scripted teacher on environment reward | VLM pairwise preferences | **tested offline: at chance (8B)** |
| dense guidance | LLM-written reward from simulator state | exploration bonus / VLM subgoals / self-discovered milestones / perception-grounded reward | **proposed, not built** |
| unseen tasks | tuned on Button Press + Hammer | same frozen prompts and settings on new tasks | **proposed** (cheap offline test first) |

Sub-hypotheses:

* **H1:** a VLM verdict can replace the simulator success flag at small cost. *(running)*
* **H2:** guidance, not the verdict, is the bottleneck: an open 8B VLM gives no usable preferences, so removing the privileged teacher fails. *(supported by the offline preference tests; the policy-level check is the `noRRM` arms)*
* **H3:** a task-agnostic exploration bonus plus the VLM verdict can fill the guidance gap. *(untested)*
* **H4:** error *structure* matters more than error rate (correlated false positives; majority vs soft voting). *(partly addressed: A2 offline, `ep_vlm_major` vs `ep_vlm_soft`)*

**Honest novelty statement:** every component has prior art (below). The credible contribution would be a **measured audit** of which privileged parts a small open-weight VLM can replace, at what cost, and whether the frozen pipeline transfers to unseen tasks.

---

## 13. Prior-art checks done in Part 2 (abstract level) [verified ids, not deep reads]

| item | closest work | note |
|---|---|---|
| VLM yes/no success detection | arXiv 2303.07280 "Vision-Language Models as Success Detectors" (SuccessVQA, Flamingo) | established; survey 2609.09250 "No Free Checker" covers ~150 verifiers |
| subgoals checked by a VLM | 2307.09668 "Towards A Unified Agent with Foundation Models"; 2507.04789 T2-VLM; ReLAM | established |
| VLM reward + exploration | 2308.12270 LAMP (VLM-derived exploration rewards for pretraining); RND is standard | the exact "VLM verdict + exploration bonus" combination not found, but incremental |

---

## 14. Every idea that came up, with a verdict

### 14.1 P-SVM, FACR, RLF (original three)
See Section 4. All three **not supported** against their simplest control; P-SVM and FACR are incremental over existing work; RLF's attack policy adds nothing over cheap selectors.

### 14.2 "RRM + plain outcome reward" (the control that won)
Not a hypothesis we planned, but the strongest empirical result: with the learned reward as guidance, adding a success signal gives 100% / 100% on Button Press and Hammer. It is the basis of Part 2. Caveat: it used the **simulator's success flag**.

### 14.3 Option 2: VLM-checked subgoals
*Is anything privileged?* No simulator information is used. But: (a) the subgoal list comes from an LLM's task knowledge (the same kind of LLM-provided structure CoRe relies on), (b) task wording and camera views are human-chosen (CoRe uses a fixed camera per task; on Button Press the arm is invisible), and (c) the prompts must be tuned on a dev set, never on the simulator success flag, to avoid leakage. **Prior art exists** (Section 13). Weakness: per-frame checks need the key objects to be visible and large. Verdict: **possible comparator, not a novel contribution.**

### 14.4 "Learn subgoals with RL instead of asking an LLM" (your idea)
Concrete version: the agent explores; the VLM verdict marks successes; milestones are extracted from successful trajectories (e.g. temporal clustering of states) and rewarded in order, refreshed as more successes arrive (a self-bootstrapping curriculum). Task-agnostic by construction, which fits the unseen-task goal.
* **Main obstacle: cold start.** Milestones come from successes; with no successes there is nothing to learn from. Our own oracle experiments are close cousins and **failed to bootstrap** (oracle progress predictor 0% / 33%, oracle discriminator alone 0% / 33%).
* **Prior art:** skill/option discovery (DIAYN etc.), hierarchical RL with learned subgoals, Go-Explore, reverse curriculum, TimeRewarder (2509.26627). **[not verified in detail]**
* Verdict: **plausible only combined with something that produces the first successes** (exploration). Test the exploration version first.

### 14.5 Option 1: exploration bonus (e.g. RND) + VLM verdict
Cheapest guidance replacement and uses only the policy's own observations. For: sparse success alone sometimes works (Hammer 67%). Against: bonuses often fail in manipulation; tiny button. Prior art: LAMP (2308.12270), RND. **Proposed as the next experiment.**

### 14.6 Perception-grounded rewards
Replace "state features from the simulator" by objects *perceived* from images (VLM/detector localises "the red hammer", "the nail"), then reward written over perceived coordinates. Attacks the actual privileged dependency, but is the most work, noise-sensitive, and impossible on Button Press (arm invisible). **Interesting for unseen tasks; try only if exploration + milestones show promise.**

### 14.7 Stronger VLM / better visual questions
32B model; per-frame single-fact checks ("is the button flush?"); voting over several final frames to cut Hammer false positives. **Cheap offline tests**, not yet run. Resolution (448 px) and frame counts showed no consistent effect.

### 14.8 "Use video input instead of frames"
We already feed still frames as multiple images (the server even disables video). Native video input was not tested. **[not verified]**

### 14.9 Tougher tasks
Peg Insert was already the hard task and gave all zeros for every learned reward (no information). Better: tasks with a large RRM-vs-CoRe gap in the paper (Dial Turn, Sweep Into). Dial Turn was added for key arms; our CoRe got 0% there (a reproduction gap).

---

## 15. What works and what does not (consolidated)

**Works / supported [verified]:**
* A reliable **success signal** (even episode-level) combined with a learned guidance reward is very strong (100% / 100% with the simulator flag).
* A **small open VLM can judge success from frames** if allowed to reason first (90% / 85% accuracy held-out; calibrated vote fractions; modest error correlation).
* The learned **success value V(s,h)** predicts real success much better than geometric proximity (state-level Spearman 0.67-0.85 vs -0.18 to 0.56).
* Verified-counterexample **repair** with cheap selectors improved Button Press over a matched label-budget control (borderline significance).
* Soft vote fractions beat hard labels under **independent** label noise.

**Does not work / unsupported [verified]:**
* P-SVM's visitation-matching reward (beaten by a plain outcome reward; cannot replace the learned reward).
* FACR's counterfactual action advantage (adds nothing over V of the true next state; weak signal; 1.3x cost).
* The RLF falsification policy (no better than cheap selectors; +45% environment steps).
* **Direct yes/no VLM answers** (all "No"; errors perfectly correlated across paraphrases).
* **VLM pairwise preferences and progress scores** with an 8B model on 6-frame windows (chance), so no dense guidance from the VLM.
* Soft weighting and calibration under **correlated** errors.
* Peg Insert: all learned-reward arms 0% within 200k steps.
* Our CoRe baseline on Peg Insert and Dial Turn (0%) does not match the paper.

**Unknown:**
* How much the VLM's false positives hurt policy learning (stage 6 in progress).
* Whether any guidance replacement (exploration, subgoals, milestones, perception) works without privileged information.
* Whether anything here transfers to unseen tasks (not yet tested).

---

## 16. Recommended direction

1. **Finish stage 6 and decide from data.** If `ep_vlm_*` is within a few points of `ep_oracle`, H1 is supported: the *success signal* can be privilege-free. If `noRRM` arms stay near 0%, H2 is supported in practice.
2. **Cheap offline unseen-task test (about 15 minutes of VLM time):** generate ground-truth episode banks for Dial Turn and Peg Insert from the existing actors, run the **frozen** judge, and report accuracy/FPR/ECE. This tests the unseen-task claim directly, and tells us whether the frozen prompt is task-general before spending policy-learning compute.
3. **Exploration bonus + VLM verdict (option 1)** on Button Press and Hammer, no scripted teacher. If it beats the `noRRM` arms, guidance can be replaced by exploration; if not, exploration alone is not enough.
4. **Only then** try self-discovered milestones (14.4) on top of the exploration version, and the perception-grounded reward (14.6) on Hammer-like tasks with visible objects.
5. **Evaluation hygiene for every new experiment:** >= 10 seeds or a graded metric (return / time to success), because 0/100 outcomes with 3-6 seeds cannot separate methods; pick tasks with a clear baseline gap; freeze all prompts on a dev set before policy runs; report false-positive and false-negative rates of the verdicts alongside success rates.
6. **If the full pipeline is the goal for a paper:** the defensible claim is the audit, not a new algorithm. It would need the unseen-task transfer result plus a careful failure analysis (which privileged component costs how much).

Decision I need from you: whether to run steps 2 and 3 after stage 6 finishes (I asked; no answer yet at the time of writing).

---

## 17. Inventory of artifacts

| what | where |
|---|---|
| Part 1 code, results, final report | GitHub `Manaswi2006/rl-oracle-testing`, folder `experimentation/` (commits `86276e6`, `f28153b`, `e33e3e9`) |
| Part 1 data, models, logs, 255 runs | Hugging Face `Manaswi2006/RL_project` folder `core_extensions_2026/` (revision `2f59fb7`) |
| Part 1 final report | `experimentation/reports/FINAL_RESEARCH_REPORT.md` / `.pdf`; `RESEARCH_DECISION.json`; `MASTER_RESULTS.csv`; `REPRODUCIBILITY.md` |
| Prior-art review (sub-agent draft) | `experimentation/reports/prior_art_review.md` |
| Part 2 code | `src/vlm/`, `scripts/vlm/` (bank generation, label-quality queries, prompt/preference/pointwise sweeps, frozen-judge test), `tests/test_episode_outcome.py` |
| Part 2 raw VLM outputs | `results/vlm/*.json` (label quality, sweeps, judge test), `results/vlm/bank*.pkl` (JPEG frames + truth; 256-px banks ~0.6 GB, plus 448-px banks) |
| Stage 6 runs | `runs/stage6/<arm>/<task>_s<seed>/` (each with `episode_verdicts.json`: per-episode votes, reasoning tails, true success) |
| This document | `docs/SESSION_SUMMARY.md` |

Timeline of deletion: the Part 1 workspace was deleted after verified transfer (sibling `DATASET_GENERATION` untouched; listing hash identical before/after); the workspace was recreated for Part 2 and will again be deleted only after GitHub + Hugging Face uploads (new HF folder `vlm_outcome_2026/`) are verified.

---

## 18. Honesty log: mistakes, corrections, and things not verified

**Mistakes I made and corrected:**
1. I first said CoRe's `target_pos_frm` was an undefined attribute (bug). It is injected at runtime into the environment source by `FRMCode.__init__`. Retracted.
2. **RLF zero-pair flaw:** attack / hybrid / random / disagreement arms added **zero** repair pairs because they never verified a successful episode. Found from the tables, fixed with anchor queries (+ regression test), rerun; the original runs are archived (`runs/stage2_noanchor/`) and reported.
3. **A2 "hard label" bug:** in the correlated-error and instance-dependent scenarios the hard-label method silently used uncorrupted labels. Fixed and rerun.
4. I let a screening rule ("P-SVM passes G >= 0.5") stand until a missing control (plain outcome reward) showed it was misleading.
5. Queue/GPU placement bugs (ten runs restarted), duplicate launch, shell-killing `pkill -f`.
6. **The EGL-on-GPU-0 effect** was unnoticed in Part 1; it affected speed there and caused the Part 2 crashes.
7. First-estimate runtimes were too short (the real throughput was limited by GPU time-slicing).
8. The first VLM test (all "No") looked like a bug; it was a real first-token bias, but I should have shown the reasoning outputs sooner.

**Not verified / limits:**
* The literature review was a sub-agent draft; I verified 7 ids/titles and read the SVM paper in depth, but most other claims are abstract-level. Absence of a paper in a search is **not** evidence of novelty.
* Everything in Part 1 uses a **simulated teacher** (privileged), not a VLM, so none of Part 1 says anything about VLM-labelled CoRe.
* The VLM judge was tuned on dev episodes of the same two tasks and tested on held-out episodes of those tasks; **cross-task generality is untested**.
* Statistical power is low (binary outcomes, 3-6 seeds): no significance is claimed for differences under ~50 points.
* The preference tests used the environment-reward sum as "ground truth", which is itself an imperfect notion of "closer to done".
* GPU hours are wall-clock hours of runs sharing a GPU with ~8 other runs, so only relative overheads are meaningful.
* Peg Insert and Dial Turn have 3 seeds; Sweep Into, Drawer Open, Soccer and SoftGym were not run.
