# Prior-art / novelty review: P-SVM (A), FACR (B), RLF (C)

Date of review: 2026-10-22. Method: arXiv API metadata check (id, title, first authors, year) for every cited paper. For most 2025-2026 papers that matter for novelty I also downloaded the PDF and read the abstract and method. For classical papers I verified metadata and read the arXiv abstract only. This is flagged per paper in section 1. Search coverage is limited to arXiv API queries plus web search. "No paper found" never means "does not exist".

Scope caveat: the CoRe and SVM papers were only skimmed here (abstract level; SVM details are taken from the requester's description). The classical references (HER, MBPO, Dreamer, QRL, GVL, Eureka, Text2Reward, VLAC, SURF, PEBBLE, RL-VLM-F, TREND, RoboReward, TOPReward, LRM, T2-VLM, REDS) were checked at abstract level only.

## 1. Verified papers and what they do

Read = abstract plus method text read from PDF. Abs = metadata and abstract only.

### Base and direct neighbours
- **CoRe**, arXiv 2607.01721 (Ni, Lu, Cai; ICML 2026) [Read: abstract/intro]. https://arxiv.org/abs/2607.01721. FRM: VLM iteratively designs and optimizes a formal (code) reward from task knowledge plus preference feedback. RRM: residual reward learned from VLM-generated video-level preference labels. No human in the loop.
- **SVM**, arXiv 2606.23640 (Tsao, Wagenmaker, Levine; 2026) [Abs + requester's reading]. https://arxiv.org/abs/2606.23640. Discriminator between past successful and unsuccessful episodes. Its logit is used as a dense process reward added to the outcome reward. Authors claim the optimal policy is preserved.
- **SASR**, arXiv 2408.03029 (Ma, Luo, Vo, ...; ICLR 2025) [Read]. https://arxiv.org/abs/2408.03029. "Highly Efficient Self-Adaptive Reward Shaping". The shaped reward is a state's success rate (its presence in successful trajectories divided by its total occurrences). This is estimated with KDE and random Fourier features. A Beta distribution is sampled to model uncertainty that shrinks with data. It is a count-based precursor of SVM. It uses no VLM and no dynamics model.
- **eVTA0 / RLER**, arXiv 2609.33653 (Wu, Wang, Lu, ...; Sep 2026) [Read method]. https://arxiv.org/abs/2609.33653. Defines the dense reward as r_t = P(S=1 | history prefix). Theorems 1-2 show this is a martingale and that one-step TD bootstrapping recovers it from terminal binary outcomes only. A VLM with a reward head is trained on mixed-quality policy rollouts. RLER refreshes the model with new rollouts as the policy evolves. No dynamics model, no action counterfactuals, no horizon H.
- **Bootstrapped Reward Shaping (BSRS)**, arXiv 2501.00989 (Adamczyk, Makarenko, Tiomkin; 2025) [Abs + intro]. https://arxiv.org/abs/2501.00989. The agent's current state-value estimate is used as the potential in potential-based reward shaping. There are tabular convergence proofs and Atari results.
- **DenseReward**, arXiv 2607.13033 (Fang, Dong, ...; Jul 2026) [Read abstract/intro]. https://arxiv.org/abs/2607.13033. Dense frame-level VLM reward model trained on 27k simulator-synthesized success and failure episodes (collision, miss, drop, recover). Used for MPC and PPO.

### VLM / learned robot reward methods
- **RL-VLM-F**, 2402.03681 (Wang, Sun, Zhang, Xian, ...; ICML 2024) [Abs]. https://arxiv.org/abs/2402.03681. The VLM gives preferences over pairs of image observations given a goal text. A reward is learned from those preference labels, not from raw VLM scores.
- **PEBBLE**, 2106.05091 (Lee, Smith, Abbeel; 2021) [Abs]. https://arxiv.org/abs/2106.05091. Off-policy interactive preference-based RL with unsupervised pre-training and experience relabeling after each reward update.
- **RIME**, 2402.17257 (Cheng, Xiong, Dai, ...; ICML 2024) [Read abstract]. https://arxiv.org/abs/2402.17257. Robust PbRL with noisy preferences. A sample-selection discriminator filters noisy labels, plus a warm-start for the reward model.
- **TREND**, 2505.06079 (Huang, Levy, Gupta, ...; 2025) [Abs]. https://arxiv.org/abs/2505.06079. Tri-teaching: three reward models each pass their small-loss preference pairs to a peer. Adds a few expert demos.
- **SURF**, 2203.10050 (Park, Seo, Shin, ...; 2022) [Abs]. https://arxiv.org/abs/2203.10050. Pseudo-labels unlabeled segment pairs with the current reward model, plus temporal data augmentation.
- **T2-VLM**, 2507.04789 (Zhao, Yuan, Xu, ...; ICCV 2025) [Read abstract via search + metadata]. https://arxiv.org/abs/2507.04789. Training-free. One VLM query per episode yields subgoals. A Bayesian particle filter tracks subgoal completion. Reward comes from changes in estimated completion.
- **ReLAM**, 2509.22402 (Tang, Pang, Li, ...; 2025) [Read abstract]. https://arxiv.org/abs/2509.22402. Learns an anticipation model proposing keypoint subgoals from action-free video, then trains a goal-conditioned low-level policy in an HRL setup. The "anticipation" is of subgoals, not of action outcomes.
- **RoboReward**, 2601.00675 (Lee, Wagenmaker, Pertsch, ...; 2026) [Abs]. https://arxiv.org/abs/2601.00675. VLM reward dataset and models (4B/8B). Negative examples are produced by counterfactual relabeling (instruction relabeling) and temporal clipping.
- **TimeRewarder**, 2509.26627 (Liu, Wen, Hu, ...; ICML 2026) [Read abstract]. https://arxiv.org/abs/2509.26627. Learns frame-pair temporal distance from passive videos. Its step-wise progress is used as dense reward.
- **GVL**, 2411.04549 (Ma, Hejna, Wahid, ...; 2024) [Abs]. https://arxiv.org/abs/2411.04549. Zero-shot VLM value (progress) estimation, posed as ordering shuffled frames.
- **VLAC**, 2509.15937 (Zhai, Zhang, Zhang, ...; 2025) [Abs]. https://arxiv.org/abs/2509.15937. InternVL-based critic producing pairwise progress delta and a done signal, with robustness to irrelevant prompts.
- **TOPReward**, 2602.19313 (Chen, Harrison, Lee, ...; 2026) [Abs]. https://arxiv.org/abs/2602.19313. Training-free: the VLM's token probability that the task is complete is used as a dense progress reward. It is a probability, not a repeated-vote fraction.
- **Large Reward Models**, 2603.16065 (Wu, Yuan, Licon, ...; 2026) [Abs]. https://arxiv.org/abs/2603.16065. Fine-tuned VLM giving frame-level progress, completion and temporal-contrastive rewards.
- **RARM**, 2606.22027 (Yang, Wang, Jing, ...; 2026) [Read abstract]. https://arxiv.org/abs/2606.22027. A single successful demo is turned into a dense reward via a contrastive comparator. Only confident forward progress is rewarded, to suppress false-positive reward.
- **VLMeg**, 2610.15401 (Liu, Jiang, Su, ...; Oct 2026) [Read abstract]. https://arxiv.org/abs/2610.15401. Uses a VLM as a success detector (not a regressor). A key-observation gate decides whether an episode is worth judging; the episodic reward comes from the VLM verdict.
- **ARS**, 2609.34484 (Hu, Lu, Zeng, ...; Sep 2026) [Read abstract]. https://arxiv.org/abs/2609.34484. Inference-time agentic VLM that proposes and verifies an event timeline. It can take terminal outcome labels and can audit other reward models.
- **REDS**, 2502.20630 (Kim, Heo, Lee, ...; ICLR 2025) [Abs via web search; arXiv id verified in metadata check]. https://arxiv.org/abs/2502.20630. Subtask-segmented demos are used to train a dense reward with an EPIC-distance objective and contrastive subtask alignment.
- **Eureka**, 2310.12931 (Ma, Liang, Wang, ...; 2023) [Abs]. https://arxiv.org/abs/2310.12931. Evolutionary search over LLM-written reward code, with reflection on training statistics.
- **Text2Reward**, 2309.11489 (Xie, Zhao, Wu, ...; 2023) [Abs]. https://arxiv.org/abs/2309.11489. LLM writes dense reward programs grounded in an environment abstraction; refinement with human feedback.

### RL foundations
- **HER**, 1707.01495 (Andrychowicz, Wolski, Ray, ...; 2017) [Abs]. https://arxiv.org/abs/1707.01495. Relabels goals of failed episodes with achieved states.
- **MBPO**, 1906.08253 (Janner, Fu, Zhang, Levine; 2019) [Abs]. https://arxiv.org/abs/1906.08253. Short model-generated rollouts branched from real data, with a monotonic-improvement analysis.
- **Dreamer**, 1912.01603 (Hafner, Lillicrap, ...; 2019) [Abs]. https://arxiv.org/abs/1912.01603. Backpropagates analytic gradients of learned values through imagined latent trajectories.
- **QRL**, 2304.01203 (Wang, Torralba, Isola, ...; 2023) [Abs]. https://arxiv.org/abs/2304.01203. Quasimetric models of optimal goal-reaching distance with recovery guarantees.
- **GAIL**, 1606.03476 (Ho, Ermon; 2016) [Abs only, metadata verified]. https://arxiv.org/abs/1606.03476. A discriminator between expert and policy occupancy is used as the reward.
- **Christiano et al.**, 1706.03741 (2017) [metadata only]. https://arxiv.org/abs/1706.03741. Deep RL from human preferences.

### Reward hacking, adversarial reward training, noisy verifiers
- **Adv-RM**, 2504.06141 (Bukharin, Qian, Sun, ...; 2025) [Read]. https://arxiv.org/abs/2504.06141. An RL-trained adversarial policy finds responses a target RM scores high but that are OOD and low quality. These are added as rejected examples to retrain the RM, and the process is repeated. LLM RLHF only. They report that uncertainty estimates from ensembles are not reliable indicators of quality for in-distribution responses.
- **REFORM**, 2507.06419 (Pathmanathan, Huang; v3 Jun 2026) [Read abstract/intro]. https://arxiv.org/abs/2507.06419. Reward-guided controlled decoding generates falsely scored responses. The training data is augmented with them to patch the RM. LLM only.
- **Fuzzing RLVR Verifiers**, 2606.01066 (Ray; 2026) [Abs]. https://arxiv.org/abs/2606.01066. Generates adversarial completions to compare buggy and strict verifiers; reports false-positive and exploit rates.
- **RLVR under imperfect verifiers**, 2510.00915 (Cai, Wang, ...; 2025) [Abs]. https://arxiv.org/abs/2510.00915. Models verifier error as a stochastic reward channel with asymmetric FP/FN rates and derives backward/forward corrections for unbiased gradients. This is the closest theory for "noisy outcome labels as rewards", but in LLM RLVR.
- **Verifier Errors in RLVR**, 2609.35677 (Moya, Thornley, ...; 2026) [Abs]. https://arxiv.org/abs/2609.35677. Shows observations during RLVR are insufficient to detect accepted errors; corrects with audit feedback.
- **Multimodal Reward Hacking in RL**, 2607.09492 (Yao, Wang, Zhang, ...; 2026) [Abs]. https://arxiv.org/abs/2607.09492. VLM QA setting. Visual-evidence rewards help only when the verifier is reliable.
- **Skalse et al.**, 2209.13085 (2022) [metadata only]. Formal definition of reward hacking. https://arxiv.org/abs/2209.13085
- **Gao et al.**, 2210.10760 (2022) [metadata only]. Scaling laws for reward model overoptimization. https://arxiv.org/abs/2210.10760
- **Coste et al.**, 2310.02743 (2023) [metadata only]. Reward-model ensembles to mitigate overoptimization. https://arxiv.org/abs/2310.02743
- **Noise-aware PRM**, 2601.12748 (Xie, Xu, ...; 2026) [Abs]. https://arxiv.org/abs/2601.12748. LLM process reward noise; label correction and self-confidence refinement.
- **Human-Corrected Labels**, 2511.09063 (Li, Chen, ...; 2025) [Abs]. https://arxiv.org/abs/2511.09063. Risk-consistent estimator for VLM noisy labels in image classification.

## 2. Nearest-prior-art tables

### A) P-SVM: visitation discriminator on probabilistic VLM outcome labels

| Closest paper | What it established | Exact difference from proposal | Needs new objective / estimator / theorem / procedure? | Simple existing baseline that could explain results | Likely reviewer criticism |
|---|---|---|---|---|---|
| SVM 2606.23640 | Success-vs-unsuccessful (s,a) discriminator as dense reward; r = r_out + lambda*clip(logit); policy-preservation theorem under deterministic transitions with count-based estimates | Labels come from a VLM with probability q_i instead of a ground-truth bit; weighted BCE; class-prior correction | Weighted BCE is a trivial objective change. A theorem on policy preservation under label noise would be new but looks hard because the original theorem relies on counts. | SVM with hard thresholded VLM labels (q>0.5); SVM with real success labels (upper bound) | "Soft-label BCE is textbook; this is SVM plus a noisy label source" |
| SASR 2408.03029 | State success-rate via KDE/RFF with Beta uncertainty (probabilistic but over visitation counts, not label noise) | SASR is uncertainty about success rate from finite counts; P-SVM is uncertainty about the label itself | No | SASR-style counts using VLM hard labels | Same as above |
| RL-VLM-F 2402.03681; CoRe 2607.01721 | VLM-generated preferences or rewards drive learning; CoRe's RRM learns residual reward from VLM preference labels | Outcome-level probabilistic labels feeding a visitation discriminator rather than pairwise preferences | No | RRM-style preference learning; majority-vote filter on VLM labels | "Why not simply use the VLM's outcome verdict as the sparse reward?" |
| RIME 2402.17257; TREND 2505.06079; SURF 2203.10050 | Noise filtering (discriminator, small-loss co-teaching, pseudo-labeling) in preference-based reward learning | These handle noisy pairwise preferences; P-SVM handles noisy episode outcomes | Not necessarily; could reuse their filters | Confidence filtering or small-loss selection on outcome labels | "Class-prior correction and soft labels are standard label-noise tools" |
| eVTA0 2609.33653 | Reward is P(success | history), learned from terminal binary labels via TD; one VLM reward model per task suite | eVTA0 learns a value from true outcomes; P-SVM learns an occupancy discriminator from VLM probabilities | No | eVTA0-style value reward with the same VLM labels | "Success-probability reward from outcomes is already done; VLM labels are the only novelty" |
| VLMeg 2610.15401; TOPReward 2602.19313; ARS 2609.34484 | VLM as success detector or probability source; confidence gating; terminal-label-aware auditing | These use VLM confidence directly as reward or gate; P-SVM turns it into labels for a discriminator | No | Use the VLM success probability (TOPReward style) directly as reward | "Why does the discriminator add anything over using the VLM probability?" |
| RLVR under imperfect verifiers 2510.00915 | Asymmetric FP/FN noise in outcome reward; backward/forward corrections with unbiased-gradient guarantee | LLM domain, scalar outcome reward, not visitation discriminator | A noise-rate correction for the discriminator would be the right analog, but it is not given in the proposal | Backward correction applied to the sparse outcome reward alone | "Noise-rate-based correction is better justified than repeated-vote calibration" |

Calibration question: I found no paper establishing that fractions of repeated VLM votes are calibrated success probabilities. TOPReward uses token probabilities, and the LLM-calibration search result (arXiv 2601.13284, snippet only, not verified here) suggests decision-token probabilities are near 1 regardless of correctness. Repeated sampling is self-consistency, which is correlated across samples (same image, same bias). This needs an empirical reliability diagram and ECE against ground truth. The majority-vote, confidence-filtering and hard-label baselines must be run, since they are the obvious explanations for any gain.

### B) FACR: feasibility-aware counterfactual reward

| Closest paper | What it established | Exact difference from proposal | Needs new objective / estimator / theorem / procedure? | Simple existing baseline that could explain results | Likely reviewer criticism |
|---|---|---|---|---|---|
| eVTA0 2609.33653 | Reward = P(success | prefix) learned by TD from terminal outcomes; martingale property and bootstrap-consistency theorems; refreshed online | V(z,g,H) is the same object, plus a horizon H and goal conditioning | A fixed horizon H changes the target to a finite-horizon success probability. This is a modest objective change. | Plain eVTA0 value reward; V difference as potential | "The success-value part is eVTA0 under another name" |
| SASR 2408.03029; SVM 2606.23640 | State/occupancy success rate as shaping reward | Learned V instead of counts or discriminator | No | SASR-style state success rate | Same |
| BSRS 2501.00989 | Current value estimate as the PBRS potential; tabular convergence | FACR uses success value, not return value; adds action counterfactual | Possibly not | V(s') - V(s) shaping (potential-based) | "A_cf is PBRS with a model" |
| MBPO 1906.08253; Dreamer 1912.01603 | Learned dynamics produce imagined rollouts; value gradients through the model | FACR uses the model for a one-step counterfactual Q_cf(z,a) = E[V(z')] and A_cf = Q_cf - E_pi Q_cf as reward | Not clearly: Q-from-V-via-model minus on-policy baseline is the standard advantage under a model | MBPO/Dreamer critic advantage on a success-only reward | "This is model-based advantage estimation relabeled as a reward" |
| QRL 2304.01203 | Quasimetric distance gives an optimal goal-reaching value with theory | FACR value is a probability of success within H, not a distance | Only if a theorem links them | Goal-conditioned QRL value as reward | "Why not a QRL or GCRL value?" |
| HER 1707.01495 | Relabeling failures as successes for other goals | Not directly comparable; FACR's counterfactual is action-based | No | HER with sparse outcome | -- |
| TimeRewarder 2509.26627; GVL 2411.04549; TOPReward 2602.19313; VLAC 2509.15937 | Progress or value from VLMs or passive video as dense reward | Those are progress-based and action-agnostic; FACR claims action-sensitive credit | Distinct claim only if action-conditioned signals beat progress deltas | Progress-delta reward | "Progress-delta already captures it" |
| ReLAM 2509.22402; T2-VLM 2507.04789 | VLM or learned subgoals as structured reward | Subgoals vs. learned counterfactual value | No | Subgoal reward | -- |

Note: I found no paper that uses learned dynamics to compute an action-conditioned counterfactual success advantage as a dense reward for a robot VLM-reward pipeline. This is not proof of absence. Also, A_cf would need robust action-conditioned latent dynamics from visual observations; errors there are probably the dominant failure mode.

### C) RLF: reward learning through falsification

| Closest paper | What it established | Exact difference from proposal | Needs new objective / estimator / theorem / procedure? | Simple existing baseline that could explain results | Likely reviewer criticism |
|---|---|---|---|---|---|
| Adv-RM 2504.06141 | RL attack policy finds high-reward-but-bad samples; added as hard negatives; iterated; LLM RLHF | Robot trajectories; attack objective includes an independent verifier term -lambda*C; "verified" counterexamples | The verifier-penalized objective and verification gate are small but real changes; no theorem needed | Adv-RM transplanted, with or without verifier | "Adv-RM for robots with a verifier" |
| REFORM 2507.06419 | RM-guided generation of falsely scored data; patch the RM | Same, plus an explicit verifier | Same | Same | Same |
| PEBBLE 2106.05091; SURF 2203.10050; RIME 2402.17257; RL-VLM-F 2402.03681 | Query selection by ensemble disagreement; semi-supervised or noise-robust reward learning | Attack policy actively seeks reward-high/evidence-low trajectories; those methods sample from existing data | Procedure differs; no new estimator | Ensemble-disagreement uncertainty sampling; hard-negative mining from replay | "Uncertainty-based query selection does the same job more cheaply" |
| GAIL 1606.03476 | Adversarial discriminator vs. policy, jointly | GAIL discriminates policy vs. expert, not reward vs. verifier | No | GAIL-style discriminator reward | "Adversarial IRL already does this" |
| Fuzzing RLVR Verifiers 2606.01066; Verifier errors 2609.35677; Imperfect verifiers 2510.00915 | Adversarial or audited testing of verifiers; noise-channel correction | LLM domain | -- | Audit-feedback correction | -- |
| Multimodal Reward Hacking 2607.09492; ARS 2609.34484 | VLM reward hacking in QA; ARS audits other robot reward models | ARS audits offline, with no attack policy or repair loop | -- | VLM-as-judge post-hoc filter | -- |
| Skalse 2209.13085; Gao 2210.10760; Coste 2310.02743 | Formalization and measurement of reward hacking; ensembles slow overoptimization | -- | -- | Ensemble of reward models | "A reward-model ensemble was not compared" |

Key point: the "independent verifier" C(tau,g) is a second noisy detector. If it is a VLM with its own failure modes, "verified counterexamples" may be wrong, and the repair step can inject errors. The attack may also drive the policy to trajectories where the verifier is also fooled (correlated failure). I did not search CPS falsification literature in depth (see section 4).

## 3. Novelty classification

**A) P-SVM: Incremental.** The mechanism (discriminator on success vs. failure episodes) is SVM, and its count-based precursor is SASR. The change is using soft VLM labels with weighted BCE and class-prior correction, which are standard label-noise tools. No theorem or estimator is new yet. A policy-preservation result under label noise (analogous to the asymmetric noise correction of 2510.00915) would raise it. The empirical question that could differentiate it is whether repeated-vote fractions are calibrated; that is unestablished and must be shown against hard-label, majority-vote and confidence-filtered baselines.

**B) FACR: Incremental.** The success-value component is Already established in spirit: eVTA0 (2609.33653) defines and learns P(success | history) via TD with theory, SASR and SVM give occupancy-based success shaping, and BSRS shapes with a value potential. The remaining element, action-conditioned counterfactual advantage through a learned dynamics model, is a model-based advantage in the MBPO/Dreamer tradition used as reward. I found no direct prior for that exact use in VLM reward pipelines, but that is a search result, not a proof. The horizon-H success target is a small modification. Needs ablation against V-difference (potential) shaping and eVTA0.

**C) RLF: Potentially novel.** Adv-RM and REFORM already establish the attack-then-repair loop, but in LLM RLHF, and I found no robot-manipulation VLM reward instance within my search limits. The distinct piece is the verifier-penalized attack objective and verified-counterexample repair for embodied rewards. The risks are that ensemble-disagreement active querying or hard-negative mining explains the gains, and that the verifier is not independent. This is "Potentially novel" rather than "Supported distinct mechanism" because no experiment here supports that the mechanism, rather than any extra negative data, drives improvement.

No idea currently qualifies as "Supported distinct mechanism"; that label needs experiments that isolate the mechanism against the baselines in the tables.

## 4. Items not verified or not done

- SVM details beyond the requester's description (lambda, beta, theorem assumptions) and the full CoRe method were not re-read here.
- Abstract-only for most classical and VLM-reward papers (see tags in section 1); method sections not read for HER, MBPO, Dreamer, QRL, GVL, VLAC, SURF, PEBBLE, RL-VLM-F, TREND, RoboReward, TOPReward, LRM, T2-VLM, REDS, Eureka, Text2Reward.
- RIME and Adv-RM: details beyond abstracts are partial (text extraction was imperfect for some PDFs).
- "SASR, Adamczyk" Q-function shaping: SASR (Ma et al.) is a success-rate, not Q-function, shaping method; BSRS (Adamczyk et al.) is the value-as-potential method. I did not find a paper called SASR that uses a learned Q function.
- arXiv 2601.13284 (calibration of decision-token probabilities, LLM) was seen only in a search snippet; not verified.
- CounterAlign and ContactGuard (seen in a citation index snippet) were not located on arXiv and are not cited as established.
- Not searched in depth: CPS falsification literature (S-TaLiRo, Breach, adversarial scenario generation), active reward learning with disagreement (e.g., Biyik), positive-unlabeled learning, RoboCLIP, conformal calibration of VLM judges, MCTS/learned-dynamics reward papers in 2026 beyond the web-search hits. Coverage of arXiv Oct 2026 is limited to what search surfaced.
- RoboReward, DenseReward, and Large Reward Models contain failure-augmentation or counterfactual-relabeling ideas that may overlap with A or C; only abstracts were read.
