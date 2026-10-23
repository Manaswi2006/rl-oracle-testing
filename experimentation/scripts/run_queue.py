#!/usr/bin/env python
"""Run a list of training jobs with bounded concurrency, round-robin over GPUs.

jobs file: one job per line:  <run_dir> <overrides...>   (relative to experimentation root)
Skips runs that already have a DONE marker; retries a failed run once.
"""
import os, subprocess, sys, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("jobs")
ap.add_argument("--concurrency", type=int, default=18)
ap.add_argument("--gpus", default="0,1,2")
ap.add_argument("--max_total", type=int, default=10**9, help="max core_train processes system-wide (memory guard)")
ap.add_argument("--mps", action="store_true", help="connect to the CUDA MPS daemon (pipe dir tmp/mps_pipe)")
ap.add_argument("--min_avail_gb", type=float, default=0, help="do not start a job while MemAvailable is below this")
ap.add_argument("--module", default="src.common.core_train")
args = ap.parse_args()
gpus = args.gpus.split(",")
jobs = [l.split() for l in open(args.jobs) if l.strip() and not l.startswith("#")]
def avail_gb():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1e6


def least_loaded_gpu(cands, own=()):
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id', '--format=csv,noheader'], capture_output=True, text=True).stdout.split()
    ids = subprocess.run(['nvidia-smi', '--query-gpu=index,gpu_bus_id', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().split('\n')
    bus = {l.split(',')[1].strip(): l.split(',')[0].strip() for l in ids}
    cnt = {g: 0 for g in cands}
    for b in out:
        g = bus.get(b)
        if g in cnt:
            cnt[g] += 1
    for g in cands:
        cnt[g] = max(cnt[g], sum(1 for x in own if x == g))   # freshly started jobs may not be visible to nvidia-smi yet
    return min(cands, key=lambda g: cnt[g])


def n_total():
    out = subprocess.run(['pgrep', '-fc', 'python -m src.common.core_trai[n]'], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


running, gi, pending, retries = [], 0, list(jobs), {}
env0 = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", PYTHONPATH=str(ROOT),
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCH_NUM_THREADS="2", PYTHONWARNINGS="ignore")
py = str(ROOT / ".venv/bin/python")
while pending or running:
    for p, job, f in list(running):
        if p.poll() is not None:
            running.remove((p, job, f)); f.close()
            ok = (ROOT / job[0] / "DONE").exists()
            print(time.strftime("%T"), "finished", job[0], "rc", p.returncode, "ok" if ok else "FAILED", flush=True)
            if not ok and retries.get(job[0], 0) < 1:
                retries[job[0]] = 1; pending.append(job)
    while pending and len(running) < args.concurrency and n_total() < args.max_total and avail_gb() >= args.min_avail_gb:
        job = pending.pop(0)
        rd = ROOT / job[0]
        if (rd / "DONE").exists():
            continue
        if subprocess.run(["pgrep", "-f", f"run_dir={rd} "], capture_output=True).returncode == 0:   # already running (other queue instance)
            continue
        rd.mkdir(parents=True, exist_ok=True)
        f = open(rd / "stdout.log", "a")
        env = dict(env0, CUDA_VISIBLE_DEVICES=least_loaded_gpu(gpus, [j[0].gpu for j in running if hasattr(j[0], 'gpu')])); gi += 1
        if args.mps:
            env.update(CUDA_MPS_PIPE_DIRECTORY=str(ROOT / 'tmp/mps_pipe'), CUDA_MPS_LOG_DIRECTORY=str(ROOT / 'tmp/mps_log'))
        p = subprocess.Popen([py, "-m", args.module, f"run_dir={rd}", f"run_id={rd.name}"] + job[1:], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        p.gpu = env['CUDA_VISIBLE_DEVICES']
        time.sleep(30)    # let the CUDA context appear in nvidia-smi before choosing the next GPU
        running.append((p, job, f))
        print(time.strftime("%T"), "started", job[0], "gpu", env["CUDA_VISIBLE_DEVICES"], flush=True)
    time.sleep(10)
print("queue done")
