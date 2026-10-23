#!/bin/bash
# quick wiring test for every RLF selector on a tiny budget (reach task, which succeeds early)
cd "$(dirname "$0")/.."; source .venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$PWD PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=${GPU:-7}
for sel in "$@"; do
  timeout 1500 python -m src.common.core_train method=rlf env=metaworld_reach-v2 rlf.selector=$sel rlf.extra_capacity=30 rlf.pool=10 rlf.att_seed_steps=200 run_id=sm_$sel run_dir=runs/sm_$sel \
    num_train_steps=9000 num_seed_steps=300 num_pre_steps=1700 num_interact=1500 eval_frequency=4500 num_final_eval_episodes=2 diag_episodes=3 max_feedback=80 2>&1 | grep "eval SR\|Error\|Trace\|File" | tail -6
  python - <<PY
import json
r=json.load(open('runs/sm_$sel/result.json')); s=r['ext_stats']; print('$sel', {k:v for k,v in s.items() if k!='rounds'}, 'rounds', len(s['rounds']), 'train succ', r['n_success_train'], 'diag', (r['rrm_diag'] or {}).get('auc_window_vs_success'))
PY
done
