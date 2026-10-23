#!/usr/bin/env python
"""Inventory of the experimentation workspace: relative path, size, sha256 (parallel).  usage: make_inventory.py <out.json> [--exclude pat ...]"""
import hashlib, json, os, sys, fnmatch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCL_DEFAULT = [".venv/*", "cache/*", "tmp/*", "upstream/CoRe/.git/*", "*/__pycache__/*", "*.pyc"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(excl=EXCL_DEFAULT):
    files = []
    for dp, dn, fn in os.walk(ROOT):
        for f in fn:
            p = Path(dp) / f
            rel = str(p.relative_to(ROOT))
            if p.is_symlink() or any(fnmatch.fnmatch(rel, e) for e in excl):
                continue
            files.append((rel, p))
    with ThreadPoolExecutor(16) as ex:
        hashes = list(ex.map(lambda t: sha256(t[1]), files))
    return [dict(path=r, size=p.stat().st_size, sha256=h) for (r, p), h in zip(files, hashes)]


if __name__ == "__main__":
    out = sys.argv[1]
    inv = inventory()
    json.dump(dict(root=str(ROOT), n_files=len(inv), total_bytes=sum(i["size"] for i in inv), files=inv), open(out, "w"), indent=0)
    print(len(inv), "files", sum(i["size"] for i in inv) / 1e9, "GB ->", out)
