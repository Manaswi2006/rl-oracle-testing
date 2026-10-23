#!/bin/bash
# Full campaign as executed (see REPRODUCIBILITY.md).  Adjust --gpus / --concurrency to your hardware (8xA100-80GB, ~1.1 TB RAM used here).
set -e
cd "$(dirname "$0")"; source .venv/bin/activate
for s in stage0 stage1 stage2 stage3; do python scripts/gen_jobs.py $s > jobs_$s.txt; done
python scripts/run_queue.py jobs_stage0.txt --concurrency 36 --gpus 0,1,2,3,4,5,6,7 &
python scripts/run_queue.py jobs_stage1.txt --concurrency 36 --gpus 0,1,2,3,4,5,6,7 &
python scripts/run_queue.py jobs_stage2.txt --concurrency 60 --max_total 66 --min_avail_gb 330 --gpus 0,1,2,3,4,5,6,7 &
wait
python scripts/run_queue.py jobs_stage3.txt --concurrency 48 --gpus 0,1,2,3,4,5,6,7
python scripts/aggregate_results.py; python scripts/make_figures.py; python scripts/analyze_rlf.py
python scripts/run_psvm_label_corruption.py
for t in button hammer peg; do python scripts/facr_branching_data.py --task $t; done
python scripts/analyze_facr_offline.py
