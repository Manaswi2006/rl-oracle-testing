#!/usr/bin/env python
"""Copy authored code / docs / small result tables into <repo>/experimentation (no run data, weights, upstream clone, secrets).  usage: prepare_github.py <repo_dir>"""
import fnmatch, os, re, shutil, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
DST = Path(sys.argv[1]) / "experimentation"
FILES = ["README.md", "environment.yml", "requirements.lock", "run_all.sh", "reproduce.sh"] + [p.name for p in SRC.glob("jobs_stage*.txt")]
DIRS = ["src", "scripts", "configs", "tests", "patches", "reports", "figures", "manifests", "results/aggregate", "results/psvm", "results/facr", "results/falsification"]
EXCL = ["__pycache__", "*.pyc", "*.npz", "*.pt", ".env", "*.tmp", "old"]
SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AZURE_API_KEY\s*=\s*\S+)")
if DST.exists():
    shutil.rmtree(DST)
n = 0
def cp(s, d):
    global n
    d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(s, d); n += 1
for f in FILES:
    if (SRC / f).exists(): cp(SRC / f, DST / f)
for d in DIRS:
    for dp, dn, fn in os.walk(SRC / d):
        dn[:] = [x for x in dn if not any(fnmatch.fnmatch(x, e) for e in EXCL)]
        for f in fn:
            if any(fnmatch.fnmatch(f, e) for e in EXCL): continue
            p = Path(dp) / f
            if p.stat().st_size > 5_000_000: print("skip large", p); continue
            cp(p, DST / p.relative_to(SRC))
bad = []
for p in DST.rglob("*"):
    if p.is_file() and p.suffix in (".py", ".md", ".txt", ".yml", ".yaml", ".sh", ".json", ".csv", ".lock"):
        if SECRET.search(p.read_text(errors="ignore")): bad.append(str(p))
print("copied", n, "files;", "SECRET-LIKE STRINGS IN:" if bad else "no secret-like strings", bad)
size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()); print("total MB", size / 1e6)
