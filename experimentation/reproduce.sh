#!/bin/bash
# Reproduce the environment, fetch upstream CoRe at the pinned commit, run unit tests and a tiny smoke run.
set -e
cd "$(dirname "$0")"
export UV_CACHE_DIR=$PWD/cache/uv TMPDIR=$PWD/tmp; mkdir -p tmp cache upstream
command -v uv >/dev/null || pip install uv
[ -d .venv ] || uv venv --python 3.9 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 torchvision==0.14.1
grep -vE "^(torch|torchvision|nvidia)" requirements.lock | uv pip install -r /dev/stdin
[ -d upstream/CoRe ] || git clone https://github.com/nihx9919/CoRe upstream/CoRe
git -C upstream/CoRe checkout d663cabc38e6d6f39ac5e1e4327d94f17f84fa59
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$PWD
python -m pytest -q tests
python -m src.common.core_train method=core run_id=smoke run_dir=runs/smoke num_train_steps=3000 num_seed_steps=500 num_pre_steps=1000 num_interact=500 eval_frequency=1500 num_final_eval_episodes=2 max_feedback=60 diag_episodes=2
