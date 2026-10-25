#!/usr/bin/env python
"""Verify that every file of the upload set exists on the HF dataset with identical size and content hash.

Upload set = same folders/files as upload_huggingface.py (code, configs, results, reports, figures, logs, runs, jobs_*.txt, ...).
Integrity: LFS files -> sha256 from the HF LFS pointer metadata; regular files -> git blob id (sha1 of 'blob <n>\\0'+content).
Additionally downloads a sample of files and compares their sha256.  Writes tmp/verify_report.json and prints a summary.
usage: verify_transfer.py [--env-file ...] [--sample 25]
"""
import argparse, hashlib, json, os, random, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO, PREFIX = "Manaswi2006/RL_project", os.environ.get("HF_PREFIX", "core_extensions_2026")
TOP = [t for t in os.environ.get("HF_TOP", "").split(";") if t]
INC_DIRS = ["reports", "results", "figures", "configs", "logs", "scripts", "src", "tests", "runs", "patches", "manifests", "docs"]
INC_FILES = ["README.md", "environment.yml", "requirements.lock", "run_all.sh", "reproduce.sh"] + [p.name for p in ROOT.glob("jobs_*.txt")]
EXCL = ("__pycache__", ".pyc", "/vlm_label_set/", "logs/upload_hf.log", "logs/upload_hf2.log", "TRANSFER_AND_CLEANUP_MANIFEST", "manifests/", "logs/upload_")


def token(env_file):
    if os.environ.get("HF_TOKEN"): return os.environ["HF_TOKEN"]
    for l in open(env_file):
        if l.startswith("HUGGINGFACE_TOKEN="): return l.split("=", 1)[1].strip().strip("\"'")


def local_files():
    out = {}
    for d in INC_DIRS:
        for dp, dn, fn in os.walk(ROOT / d):
            for f in fn:
                p = Path(dp) / f; rel = str(p.relative_to(ROOT))
                if any(e in "/" + rel or rel.endswith(e) for e in EXCL): continue
                out[rel] = p
    for f in INC_FILES:
        if (ROOT / f).exists(): out[f] = ROOT / f
    tops = TOP or [f"reports/{f}" for f in ["FINAL_RESEARCH_REPORT.md", "FINAL_RESEARCH_REPORT.pdf", "MASTER_RESULTS.csv", "RESEARCH_DECISION.json", "REPRODUCIBILITY.md", "TRANSFER_AND_CLEANUP_MANIFEST.md", "TRANSFER_AND_CLEANUP_MANIFEST.json"]]
    for rel in tops:
        if (ROOT / rel).exists(): out["__top__/" + Path(rel).name] = ROOT / rel
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()


def gitblob(p):
    h = hashlib.sha1(); n = p.stat().st_size; h.update(f"blob {n}\0".encode())
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--env-file", default="/home/colligo/DatasetDiscovery/.env"); ap.add_argument("--sample", type=int, default=25)
    a = ap.parse_args()
    api = HfApi(token=token(a.env_file))
    info = api.dataset_info(REPO)
    remote = {}
    for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=PREFIX, recursive=True):
        if hasattr(e, "size") and hasattr(e, "blob_id"):
            remote[e.path[len(PREFIX) + 1:]] = e
    loc = local_files()
    missing, size_bad, hash_bad, ok = [], [], [], 0
    for rel, p in sorted(loc.items()):
        key = rel.replace("__top__/", "")
        e = remote.get(key)
        if e is None: missing.append(rel); continue
        if e.size != p.stat().st_size: size_bad.append((rel, p.stat().st_size, e.size)); continue
        if e.lfs is not None:
            good = (e.lfs.sha256 if hasattr(e.lfs, "sha256") else e.lfs["sha256"]) == sha256(p)
        else:
            good = e.blob_id == gitblob(p)
        if good: ok += 1
        else: hash_bad.append(rel)
    random.seed(0)
    sample = random.sample(sorted(loc), min(a.sample, len(loc)))
    dl_bad = []
    for rel in sample:
        key = rel.replace("__top__/", "")
        try:
            q = hf_hub_download(REPO, f"{PREFIX}/{key}", repo_type="dataset", token=api.token, local_dir=str(ROOT / "tmp" / "hf_dl"))
            if sha256(q) != sha256(loc[rel]): dl_bad.append(rel)
        except Exception as ex:
            dl_bad.append(f"{rel}: {str(ex)[:80]}")
    rep = dict(revision=info.sha, local_files=len(loc), remote_files_under_prefix=len(remote), verified_ok=ok, missing=missing, size_mismatch=size_bad, hash_mismatch=hash_bad,
               download_sample=len(sample), download_sample_mismatch=dl_bad, local_bytes=sum(p.stat().st_size for p in loc.values()))
    json.dump(rep, open(ROOT / "tmp" / "verify_report.json", "w"), indent=1)
    print({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()})
    sys.exit(0 if not (missing or size_bad or hash_bad or dl_bad) else 1)
