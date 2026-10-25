#!/usr/bin/env python
"""Write manifests/local_inventory.json (path,size,sha256 of the upload set) and reports/TRANSFER_AND_CLEANUP_MANIFEST.{md,json}.
usage: make_manifest.py --github-commit A --hf-revision R [--cleanup-state pending|done]"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_transfer import local_files, sha256   # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--github-commit", required=True); ap.add_argument("--hf-revision", required=True); ap.add_argument("--verify-report", default=str(ROOT / "tmp/verify_report.json")); ap.add_argument("--suffix", default=""); ap.add_argument("--prefix", default="core_extensions_2026"); ap.add_argument("--inventory-name", default="local_inventory.json")
a = ap.parse_args()
loc = {k: v for k, v in local_files().items() if not "TRANSFER_AND_CLEANUP_MANIFEST" not in k and not k.startswith("manifests/")}
items = sorted(loc.items())
with ThreadPoolExecutor(16) as ex:
    hs = list(ex.map(lambda kv: sha256(kv[1]), items))
inv = [dict(path=k, size=p.stat().st_size, sha256=h) for (k, p), h in zip(items, hs)]
(ROOT / "manifests").mkdir(exist_ok=True)
json.dump(dict(n_files=len(inv), total_bytes=sum(i["size"] for i in inv), files=inv), open(ROOT / "manifests" / a.inventory_name, "w"))
ver = json.load(open(a.verify_report)) if Path(a.verify_report).exists() else {}
total = sum(i["size"] for i in inv)
folders = {}
for i in inv:
    top = i["path"].split("/")[0]; folders.setdefault(top, [0, 0]); folders[top][0] += 1; folders[top][1] += i["size"]
M = dict(
    generated=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    server_workspace="/home/colligo/working/manaswi/experimentation",
    github=dict(repo="https://github.com/Manaswi2006/rl-oracle-testing", branch="main", code_commit=a.github_commit, folder="experimentation/",
                note="code_commit = commit that contains all code/configs/tests/results/report used here; a later commit adds this manifest, the commit-stamped MASTER_RESULTS.csv and the cleanup record"),
    huggingface=dict(repo="https://huggingface.co/datasets/Manaswi2006/RL_project", folder=a.prefix + "/", data_revision=a.hf_revision,
                     note="revision at which the data upload was fully verified; the manifests and the final cleanup record are added by later commits"),
    inventory=dict(n_files=len(inv), total_bytes=total, per_top_level={k: dict(files=v[0], bytes=v[1]) for k, v in sorted(folders.items())}, file=f"manifests/{a.inventory_name} (path, size, sha256)"),
    verification=dict(method="size + content hash per file (LFS sha256 or git blob id) against the HF tree listing, plus download-and-sha256 of a random sample", report=ver),
    not_transferred=[dict(path="upstream/CoRe", reason="third-party clone (68 MB), pinned commit d663cabc38e6d6f39ac5e1e4327d94f17f84fa59, no local modification; recreated by reproduce.sh"),
                     dict(path="cache/ (24 GB: Qwen/Qwen3-VL-8B-Instruct weights, uv/HF caches)", reason="public model weights, re-downloadable from Hugging Face (Qwen/Qwen3-VL-8B-Instruct, Apache-2.0)"),
                     dict(path=".venv, .venv_vlm, .venv_report, tmp/", reason="regenerable environments / scratch; requirements.lock + reproduce.sh recreate them"),
                     dict(path="checkpoints/actors, checkpoints/hf_actors", reason="copies of actors already stored in the same HF dataset under core_extensions_2026/runs/**/actor.pt"),
                     dict(path="credentials", reason="never stored in the workspace; the HF token was read from /home/colligo/DatasetDiscovery/.env at upload time only")],
    cleanup=dict(planned_deletion="/home/colligo/working/manaswi/experimentation (entire folder), only if verification above has zero missing / mismatching files",
                 safety_checks=["canonical path must equal /home/colligo/working/manaswi/experimentation", "must not be a symlink", "parent /home/colligo/working/manaswi and sibling DATASET_GENERATION untouched",
                                "no project process running"], result="recorded in CLEANUP_RESULT_VLM.json (uploaded to HF and GitHub after deletion)"))
json.dump(M, open(ROOT / f"reports/TRANSFER_AND_CLEANUP_MANIFEST{a.suffix}.json", "w"), indent=1)
md = f"""# Transfer and cleanup manifest

Generated {M['generated']}.

| item | value |
|---|---|
| server workspace (provenance) | `{M['server_workspace']}` |
| GitHub | {M['github']['repo']} branch `main`, folder `experimentation/`, **code commit `{a.github_commit}`** (a later commit adds this manifest, the commit-stamped `MASTER_RESULTS.csv` and the cleanup record) |
| Hugging Face | {M['huggingface']['repo']} folder `core_extensions_2026/`, **data revision `{a.hf_revision}`** (verified); later commits add manifests and the cleanup record |
| files in inventory | {len(inv)} files, {total/1e9:.2f} GB (`manifests/local_inventory.json`: path, size, sha256) |
| verification | {ver.get('verified_ok','?')} of {ver.get('local_files','?')} files verified by size + content hash; missing {len(ver.get('missing',[]))}; size mismatches {len(ver.get('size_mismatch',[]))}; hash mismatches {len(ver.get('hash_mismatch',[]))}; random download sample {ver.get('download_sample','?')} files, mismatches {len(ver.get('download_sample_mismatch',[]))} |

Size mismatches listed by the verification report (if any) are files edited after the upload (the verification script itself); they are re-uploaded and re-verified before cleanup.

## Inventory by top-level folder
| folder | files | bytes |
|---|---|---|
""" + "\n".join(f"| `{k}` | {v[0]} | {v[1]:,} |" for k, v in sorted(folders.items())) + """

## Not transferred (and why)
""" + "\n".join(f"- `{x['path']}`: {x['reason']}" for x in M["not_transferred"]) + """

## Cleanup procedure
- Planned deletion: the entire folder `/home/colligo/working/manaswi/experimentation`, only if the verification above shows zero missing / mismatching files.
- Safety checks before deletion: canonical path equals the workspace path; not a symlink; the parent `/home/colligo/working/manaswi` and the pre-existing sibling `DATASET_GENERATION` are never touched; no project process is running.
- The result (deleted / blocked, sibling integrity) is recorded in `CLEANUP_RESULT.json`, uploaded to both remotes after deletion (the folder cannot hold its own deletion record).
"""
(ROOT / f"reports/TRANSFER_AND_CLEANUP_MANIFEST{a.suffix}.md").write_text(md)
print("manifest written;", len(inv), "files", total / 1e9, "GB")
