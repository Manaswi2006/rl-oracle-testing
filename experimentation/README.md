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

## Part 2: VLM-outcome study (follow-up)
Can a small open-weight VLM (Qwen3-VL-8B, served locally with vLLM) replace the privileged success signal? See `reports/VLM_OUTCOME_REPORT.md` (results), `docs/SESSION_SUMMARY.md` (everything tried, what works and what does not, recommended direction) and `src/vlm/`, `scripts/vlm/` (code). Job lists: `jobs_stage6.txt` (VLM verdict arms), `jobs_stage7.txt` (exploration bonus), `jobs_stage8*.txt` (false-positive diagnostics). Start the VLM server first:
```bash
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 .venv_vlm/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3vl8b --port 8000 --max-model-len 20000 --limit-mm-per-prompt '{"image":28,"video":0}'
python scripts/run_queue.py jobs_stage6.txt --concurrency 48 --gpus 1,2,3,4,5,6,7   # training jobs render on their own GPU (MUJOCO_EGL_DEVICE_ID)
```
