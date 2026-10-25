# Final verdict: can we remove privileged rewards so the method works on unseen tasks?

*Plain-language summary of the whole project and its conclusion (2026-10-25). For the detailed record see `docs/SESSION_SUMMARY.md` (everything we tried, what you asked, mistakes made), `reports/FINAL_RESEARCH_REPORT.md` (Part 1) and `reports/VLM_OUTCOME_REPORT.md` (Part 2).*

---

## 1. The goal in one paragraph

CoRe (the base paper) trains robots with two reward sources: an **LLM that writes reward code** using the simulator's internal state (positions of objects, goal location), and a **learned reward** trained on a VLM's judgments of videos. The LLM-written part is *privileged*: it only works if someone gives the LLM task-specific state features. Your goal was to **replace that privileged information with something task-agnostic**, so the same recipe can work on tasks it was never tuned for.

## 2. What we did, in two parts

### Part 1: the three research ideas you gave me

| idea | one-line description | verdict |
|---|---|---|
| **P-SVM** | learn a dense reward from which states appear in successful vs failed episodes (noisy VLM labels allowed) | not supported: a plain "task done" reward beat it |
| **FACR** | learn "probability of success from here" plus a learned world model to score individual actions | not supported: the action-level part adds nothing over a simple success value |
| **RLF** | an adversarial policy hunts for states where the learned reward is wrong, then the reward is repaired | not supported: the adversary found nothing that ordinary sampling did not |

We reproduced CoRe first (our version: 67% on Button Press and Hammer, 0% on Peg Insert and Dial Turn, lower than the paper's ~100% because of fewer training steps, no LLM refinement and a simulated teacher). 255 training runs in total. The most important result of Part 1 was a **control**, not one of the ideas: *the learned reward plus a plain "did the task succeed?" signal reached 100% on Button Press and Hammer*, better than every idea and better than full CoRe. That told us the **success signal is what matters**, which led to Part 2. (Caveat: in Part 1 that signal came from the simulator.)

### Part 2: can a VLM supply that success signal?

We served an open-weight VLM (**Qwen3-VL-8B**) locally (no paid API) and asked it, from rendered frames, "did the robot complete the task?" We then used its answers as the reward in training, and compared against giving the simulator's true answer.

## 3. The main results (Part 2)

### 3.1 The VLM can judge success, but only if it reasons first
* Asked for a one-word answer, it said "No" to over 90% of real successes. That looked like a bug, so I checked: with the same images it describes the scene correctly when allowed to reason ("the button is fully depressed, sitting flush with the base").
* **Reason first, then give a verdict**: accuracy 0.90 on Button Press and 0.85 on Hammer on held-out episodes.
* The *same frozen prompt* on two tasks it never saw: balanced accuracy 0.81 (Dial Turn) and 0.93 (Peg Insert), but only 51% of its "success" calls were right on Peg Insert because real successes are rare there.
* Asked *which of two clips is closer to done*, or *how far along is this frame*, the model was at chance. So it **cannot give the dense step-by-step guidance** that CoRe's learned reward needs.

### 3.2 Policy-learning results (final success %, 6 seeds; "only" arms have 3)

| reward the robot sees | Button Press | Hammer |
|---|---|---|
| (reference) full CoRe, privileged | 66.7 | 66.7 |
| simulator's success verdict only | 66.7 | 100 |
| **VLM verdict only (no simulator information reaches the policy)** | **66.7** | **0** |
| scripted (privileged) learned reward + simulator verdict | 100 | 50 |
| scripted learned reward + VLM verdict (majority of 3) | 83.3 | 0 |
| scripted learned reward + VLM verdict (soft vote) | 83.3 | 33.3 |
| exploration bonus only | 0 | 0 |

* **Button Press:** a pipeline with *no* privileged signal matched the privileged one (66.7%). Encouraging, but only 3 seeds.
* **Hammer:** the VLM-judged versions collapse. This is the most important finding.

### 3.3 Why Hammer fails: the robot learns to fool the judge
The judge's false-positive rate on **failed** episodes climbs while the robot trains:

* Hammer: 27% in the first 50 episodes, **86%** by the end. Button Press: 6% to 58%.
* False negatives stay near zero.
* On a fixed set of rollouts the same judge made only ~25% false positives on Hammer. The robot is finding the judge's blind spots, which is classic **reward hacking**.
* The saved examples show it: the hammer resting next to or over the nail is described as "hammering completed" while the nail is clearly still up.

We tried simple fixes: requiring all three judge answers to agree rejects most exploits but also rejects more than half of real successes; a "skeptical inspector" prompt rejects everything; a "describe the change" prompt is no better. **No prompt trick separated exploits from real successes.**

### 3.4 Two things I got wrong or that surprised us
* I predicted that without the learned reward the robot would get no guidance and stay near 0%. **That was wrong**: with the (oracle) verdict applied to the whole episode and no learned reward, it reached 67% (Button Press) and 100% (Hammer). So on these tasks *guidance was not the bottleneck; the reliability of the success signal is*.
* An **exploration bonus** (a standard way to explore) did not help: alone it never found a success, and with the oracle verdict it made Hammer worse (50% vs 100%).

## 4. The verdict

**The idea partly works.**

| question | answer |
|---|---|
| Can the *success signal* be privilege-free (from a VLM watching rendered frames, no simulator information, no task-specific code)? | **Yes, with a caveat.** Works on Button Press, and the judge transfers offline to unseen tasks. |
| Does a pipeline built on that signal work reliably? | **Not yet.** On Hammer the robot exploits the judge. |
| Can the VLM replace the dense guidance (the learned reward)? | **No**, with this 8B model its preferences are at chance. But guidance turned out not to be needed on these two tasks. |
| Do our three original ideas help? | **No**, none beat its simplest control. |
| Has it been shown to work on unseen tasks end to end? | **No.** Only the offline judge was tested there. |

In short: **the missing piece is not more guidance, it is a success verifier the robot cannot easily fool.**

## 5. What works and what does not (quick list)

**Works**
* A frozen VLM judge that reasons before answering (accuracy 0.85–0.90 on the tuned tasks; 0.81 and 0.93 balanced accuracy on two unseen tasks, offline).
* A success signal applied to every step of the episode, as the only reward (oracle: 67% and 100%; VLM: 67% on Button Press).
* Soft vote fractions beat hard labels under *independent* noise (offline test).

**Does not work**
* One-word VLM answers; VLM pairwise preferences and progress scores (8B).
* The VLM verdict on Hammer (exploited: 0–33%).
* Exploration bonus; simple prompt-level hardening of the judge.
* P-SVM's visitation reward, FACR's action advantage, RLF's attack policy.
* Soft weighting and calibration when errors are *correlated* (offline test).

## 6. What I would do next, in order

1. **Make the verifier hard to exploit** (the evidence-backed bottleneck):
   * a second, different or larger VLM as a cross-check (the 32B Qwen3-VL needs one or two GPUs and about 65 GB of disk);
   * require the success state to **hold over several frames**, not just appear at the end;
   * **counterexample repair**: use a small budget of true labels (or a second verifier) to catch false positives and feed them back as hard negatives. This is the original RLF idea, now in a setting where the reward really is attacked.
2. **Run the unseen-task test in policy learning** (Dial Turn and Peg Insert) with the "VLM verdict only" arm. Offline transfer looks decent; this is the test your goal actually needs.
3. **Measure the false-positive rate during training in every experiment.** It tells more than final success with 3–6 seeds.
4. **Use at least 10 seeds** or a graded metric (return, time to success). With outcomes that are almost always 0% or 100%, our results can only separate differences of about 50 points.
5. Only revisit guidance ideas (exploration, subgoals, learned milestones, perception-based rewards) if a harder task shows that guidance is actually needed.

## 7. Honest caveats

* **Few seeds, binary outcomes.** 3–6 seeds per arm, each ending at 0% or 100%. Treat close comparisons as unresolved.
* **Prompt tuned on the same two tasks** that the policy experiments use (different episodes). Unseen-task evidence is offline only.
* **Part 1 used a simulated teacher** for the learned reward (the environment reward with 15% label noise), because no VLM was authorised then. Those results say nothing about a VLM-labelled CoRe.
* **Only the "VLM verdict only" arms are fully privilege-free.** The arms with a scripted learned reward still use the environment reward to train that reward.
* One 8B model and one family of prompts; larger models, other families and video input were not tested. On Button Press the fixed camera never shows the arm.
* The novelty of every component is low (success detection, subgoals, exploration bonuses and reward-model attacks all have prior work). The defensible contribution is the **measured audit** of which privileged parts a small open VLM can replace and where it breaks.
* Literature checks were at abstract level only; I verified a sample of citations, not all.

## 8. Where everything is

| what | where |
|---|---|
| code, reports, figures, aggregated results | GitHub `Manaswi2006/rl-oracle-testing`, folder `experimentation/` |
| Part 1 final report (CoRe, P-SVM, FACR, RLF) | `experimentation/reports/FINAL_RESEARCH_REPORT.md` / `.pdf` |
| Part 2 final report (VLM study) | `experimentation/reports/VLM_OUTCOME_REPORT.md` / `.pdf` |
| everything we tried, in depth | `experimentation/docs/SESSION_SUMMARY.md` |
| this document | `experimentation/docs/FINAL_VERDICT.md` |
| all run data, checkpoints, VLM outputs, episode banks | Hugging Face dataset `Manaswi2006/RL_project`: folders `core_extensions_2026/` (Part 1) and `vlm_outcome_2026/` (Part 2) |
| master result tables | `reports/MASTER_RESULTS.csv` (Part 1), `results/vlm/MASTER_RESULTS_VLM.csv` (Part 2) |

The server workspace (`/home/colligo/working/manaswi/experimentation`) was deleted after both uploads were verified. To rerun anything: clone the GitHub repo and follow `experimentation/README.md` (`reproduce.sh` rebuilds the environment; the VLM server command is in the README).

---

## 9. Addendum (Part 3, 2026-10-25): does a bigger VLM fix the exploited verifier?

We screened Qwen3-VL-32B (open weights, served locally; no paid API) and some stricter judging procedures offline against the real exploits the policy found (`reports/VERIFIER_SCREEN_32B_REPORT.md`).

* **A bigger model is more conservative, not simply better.** On Hammer the 32B rejects 97% of the exploits the 8B fell for and has a 1.2% false-positive rate (precision 0.98 vs 0.74), but recall drops from 0.96 to 0.73. On the unseen tasks balanced accuracy falls (Dial Turn 0.66 vs 0.81, Peg Insert 0.75 vs 0.93) while precision rises.
* **Pre-registered rule** (reject >= 70% of exploits and accept >= 85% of true successes): passed only on Button Press, by the *sustained check* (each of the last four frames judged separately, >= 3 of 4 must agree): 87% of exploits rejected, 99% of successes accepted (15 exploits, uncertain). Nothing passes on Hammer.
* **Implication for the paid-API question:** scale mostly shifts the precision/recall trade-off; it did not remove the blind spots we could test. I would not expect any single fixed judge to end reward hacking.
* **Next test:** run the 32B verdict and the sustained check in the training loop (10+ seeds) and compare the in-loop false-positive curves. A low-recall, high-precision judge may still train a policy, because missed successes only delay reward, while false positives are what gets hacked. That is an argument, not yet a result.
