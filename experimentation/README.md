# CoRe extensions 2026: P-SVM, FACR and Reward-Learning-through-Falsification (RLF)

Autonomous feasibility study of three ways to replace / strengthen CoRe's LLM-written formal reward module (FRM).
**Read `reports/FINAL_RESEARCH_REPORT.md` first** (results, decision, caveats), then `reports/REPRODUCIBILITY.md`.

Base paper: CoRe (arXiv 2607.01721; code https://github.com/nihx9919/CoRe @ d663cabc38e6d6f39ac5e1e4327d94f17f84fa59).
Upstream code is fetched by `reproduce.sh` (not vendored); **no upstream file was modified** - all changes live in `src/`.

## Layout
| path | content |
|---|---|
| `src/common/` | CoRe training loop port (`core_train.py`), simulated-teacher RRM (`rrm_scripted.py`), reward-module interface, env snapshot/branching |
| `src/psvm/` | P-SVM: visitation discriminator, soft-label model, synthetic label corruption |
| `src/facr/` | FACR + progress / success-value comparators |
| `src/falsification/` | RLF: counterexample selectors, falsification (attack) policy, reward repair |
| `scripts/` | job generation/queue (`gen_jobs.py`, `run_queue.py`), offline analyses, aggregation, figures, report, upload/verify |
| `configs/` | `common.yaml` (all hyper-parameters; deviations from upstream are commented) |
| `tests/` | unit tests (`pytest tests`) |
| `results/`, `figures/`, `reports/` | aggregated results, plots, final report + machine-readable decision |

## Key deviations from the CoRe paper setup (details in REPRODUCIBILITY.md)
1. **No VLM API was authorised**: RRM preferences come from a noisy scripted teacher on the privileged environment reward (label noise 15%). This is NOT an exact CoRe reproduction.
2. FRM = the released final FRM files, static (no online LLM refinement); 200k instead of 500k steps; RRM images 128x128 instead of 300x300.
3. 3 seeds per cell (screening scale), final evaluation 50 episodes per seed.

## Quick start
```bash
bash reproduce.sh            # env + upstream clone + unit tests + smoke test
python scripts/gen_jobs.py stage0 > jobs_stage0.txt && python scripts/run_queue.py jobs_stage0.txt --concurrency 8 --gpus 0
python scripts/aggregate_results.py && python scripts/make_figures.py
```
