# Does a bigger VLM (Qwen3-VL-32B) fix the exploited verifier? Offline screen

*Part 3 of the project (follows `VLM_OUTCOME_REPORT.md`). Took about 1.5 hours of compute: the model (bf16, 66.7 GB) was served with vLLM on two A100s (tensor parallel 2); no paid API was used. Raw results: `results/vlm/screen_32b.json`; code: `scripts/vlm/screen_verifier.py`.*

## 1. Question and design

In the previous study the policy learned to fool the 8B judge: the false-positive rate on failed training episodes rose from 27% to 86% (Hammer) and from 6% to 58% (Button Press). Would a larger model, or a stricter judging procedure, reject those exploits without rejecting real successes?

**Test set** (identical to the earlier 8B hardening test, so the comparison is like for like):
* **Exploits:** failed episodes that the 8B majority judge accepted as successes during training: 60 on Hammer, 15 on Button Press (the exact five frames the judge saw).
* **True successes:** the first 80 successful episodes of the episode bank per task. **True failures:** the first 80 non-random failures per task.

**Candidates** (all with the 32B model; prompts fixed before running, nothing tuned on these sets):
* **C0** the frozen judge (3 paraphrased reasoning prompts, majority), the same prompts as the 8B.
* **C1** unanimity of the 3 paraphrases.
* **C2** *sustained check*: each of the last four frames is judged separately against the first frame; at least 3 of 4 must say "success" (success must hold, not just appear at the end).
* **C3** "skeptical inspector" prompt (look for evidence *against* success).
* **C4** C3 and C0 must agree; **C5** C2 and C0 must agree.

**Pass rule, fixed in advance:** reject at least 70% of the exploits **and** accept at least 85% of the true successes.

## 2. Results

Share of inputs **accepted** as "success" (lower is better for exploits and failures, higher is better for successes):

| task | candidate | exploits accepted (n) | true successes accepted (n=80) | true failures accepted (n=80) | pass rule |
|---|---|---|---|---|---|
| Hammer | 8B frozen judge (earlier test) | 0.97 (60) | 0.97 | 0.29 | no |
| Hammer | **32B C0 frozen judge** | **0.03** | **0.51** | 0.01 | no (accepts too few successes) |
| Hammer | 32B C1 unanimous | 0.00 | 0.33 | 0.00 | no |
| Hammer | 32B C2 sustained 3-of-4 frames | 0.00 | 0.26 | 0.01 | no |
| Hammer | 32B C3 skeptic / C4 | 0.00 | 0.06 | 0.00 | no |
| Hammer | 32B C5 sustained and C0 | 0.00 | 0.26 | 0.00 | no |
| Button Press | 8B frozen judge (earlier test) | 0.93 (15) | 0.99 | 0.16 | no |
| Button Press | 32B C0 frozen judge | 0.53 | 0.99 | 0.26 | no (accepts too many exploits) |
| Button Press | 32B C1 unanimous | 0.53 | 0.99 | 0.21 | no |
| Button Press | **32B C2 sustained 3-of-4 frames** | **0.13** | **0.99** | 0.15 | **yes** |
| Button Press | 32B C3 skeptic / C4 | 0.13 | 0.84 | 0.04 | no (accepts 84% of successes) |
| Button Press | **32B C5 sustained and C0** | **0.07** | **0.99** | 0.15 | **yes** |

Frozen judge on the **full** episode banks (majority of 3), 32B vs the 8B (8B numbers from the earlier held-out test; Dial Turn and Peg Insert are unseen tasks):

| task (prevalence) | model | accuracy | balanced acc. | precision | recall | false-positive rate |
|---|---|---|---|---|---|---|
| Hammer (43%) | 8B | 0.846 | 0.865 | 0.737 | 0.956 | 0.226 |
| Hammer | **32B** | 0.877 | 0.859 | **0.979** | 0.730 | **0.012** |
| Button Press (61%) | 8B | 0.899 | 0.892 | 0.908 | 0.928 | 0.144 |
| Button Press | 32B | 0.917 | 0.901 | 0.900 | 0.971 | 0.169 |
| Dial Turn (46%) | 8B | 0.816 | 0.809 | 0.847 | 0.730 | 0.111 |
| Dial Turn | 32B | 0.693 | 0.664 | 1.000 | **0.327** | 0.000 |
| Peg Insert (12%) | 8B | 0.884 | 0.934 | 0.505 | 1.000 | 0.132 |
| Peg Insert | 32B | 0.924 | 0.754 | **0.758** | 0.532 | 0.023 |

## 3. What this says

1. **A bigger model is not simply a better judge; it is a more conservative one.** On Hammer the 32B rejects 97% of the exploits the 8B fell for and has a 1.2% false-positive rate on the whole bank (precision 0.98 vs 0.74), but it also misses much more of the real successes (recall 0.73 vs 0.96; only 51% of the first 80 successes accepted). On the unseen tasks the same shift lowers balanced accuracy (Dial Turn 0.66 vs 0.81, recall 0.33; Peg Insert 0.75 vs 0.93) while raising precision (Peg Insert 0.76 vs 0.51). Balanced accuracy on Button Press and Hammer is about unchanged.
2. **Strict pre-registered rule: passed only on Button Press** (C2 and C5, rejecting 87-93% of exploits while accepting 99% of successes; only 15 exploits, so this is uncertain). **No candidate passes on Hammer**: the best (C0) rejects 97% of exploits but accepts only 51% of successes.
3. **The "sustained over several frames" check is the most useful procedural idea** on Button Press (it removed most exploits at no cost to real successes), but on Hammer it makes the judge stricter without separating exploits from successes.
4. **My pass rule may be too strict for reinforcement learning.** A missed success (false negative) only delays or weakens the reward; a false positive creates a reward that can be hacked. A judge with 98% precision and 73% recall (the 32B on Hammer) could still train a policy, and the 8B's near-perfect recall is what the policy exploited. This is an argument, **not a result**: it has to be tested in training.
5. **The exploit sets are exploits of the 8B judge** (collected while a policy optimised against it). A policy trained against the 32B would find *its own* blind spots, so this screen cannot prove robustness. It only shows that the specific exploits found so far are mostly rejected.
6. **For the paid-API question:** the evidence points to scale mostly changing the precision/recall trade-off rather than removing blind spots; a larger model should help, but I would not expect it to make reward hacking disappear. This is an inference from one open model family, not a test of any paid model.

## 4. Limitations

* Small exploit sets (60 Hammer, 15 Button Press); one open model family; one prompt family; bf16 with tensor parallel 2.
* The test inputs are the five frames the 8B judge saw (first frame + last four); judges with different frame choices were not evaluated on exploits (frames were not saved).
* C2, C3 and C5 were new prompts, not tuned, so better versions probably exist.
* Offline only: no policy was trained with the 32B verdict.

## 5. Recommended next step

Run the **32B verdict (and the sustained 3-of-4-frame check, C2) in the training loop** on Hammer, Button Press and Dial Turn, comparing against the 8B judge and the oracle verdict, with at least 10 seeds and the in-loop false-positive and false-negative curves as the main outputs. Expect slower learning from the lower recall; the question is whether the false-positive rate stays low.
