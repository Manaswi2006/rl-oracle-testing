# Transfer and cleanup manifest

Generated 2026-10-25 12:24:14 PDT.

| item | value |
|---|---|
| server workspace (provenance) | `/home/colligo/working/manaswi/experimentation` |
| GitHub | https://github.com/Manaswi2006/rl-oracle-testing branch `main`, folder `experimentation/`, **code commit `36890aa27acca8391b4d51e3f49b5c16344d380e`** (a later commit adds this manifest, the commit-stamped `MASTER_RESULTS.csv` and the cleanup record) |
| Hugging Face | https://huggingface.co/datasets/Manaswi2006/RL_project folder `core_extensions_2026/`, **data revision `9422c56418067baaf561f88ed5e3de068a196115`** (verified); later commits add manifests and the cleanup record |
| files in inventory | 2416 files, 2.34 GB (`manifests/local_inventory.json`: path, size, sha256) |
| verification | 2416 of 2416 files verified by size + content hash; missing 0; size mismatches 0; hash mismatches 0; random download sample 40 files, mismatches 0 |

Size mismatches listed by the verification report (if any) are files edited after the upload (the verification script itself); they are re-uploaded and re-verified before cleanup.

## Inventory by top-level folder
| folder | files | bytes |
|---|---|---|
| `README.md` | 1 | 3,040 |
| `__top__` | 4 | 1,858,484 |
| `configs` | 1 | 1,172 |
| `docs` | 1 | 49,873 |
| `environment.yml` | 1 | 271 |
| `figures` | 19 | 3,114,164 |
| `jobs_stage0.txt` | 1 | 3,030 |
| `jobs_stage0_rerun.txt` | 1 | 230 |
| `jobs_stage1.txt` | 1 | 4,200 |
| `jobs_stage2.txt` | 1 | 8,484 |
| `jobs_stage3.txt` | 1 | 10,215 |
| `jobs_stage4.txt` | 1 | 1,521 |
| `jobs_stage5.txt` | 1 | 2,643 |
| `jobs_stage6.txt` | 1 | 6,996 |
| `jobs_stage7.txt` | 1 | 5,172 |
| `jobs_stage8.txt` | 1 | 840 |
| `jobs_stage8b.txt` | 1 | 840 |
| `logs` | 16 | 6,970,281 |
| `patches` | 1 | 53 |
| `reports` | 18 | 5,665,062 |
| `reproduce.sh` | 1 | 994 |
| `requirements.lock` | 1 | 1,786 |
| `results` | 50 | 1,279,222,558 |
| `run_all.sh` | 1 | 946 |
| `runs` | 2235 | 1,039,404,869 |
| `scripts` | 30 | 135,668 |
| `src` | 20 | 86,743 |
| `tests` | 5 | 10,960 |

## Not transferred (and why)
- `upstream/CoRe`: third-party clone (68 MB), pinned commit d663cabc38e6d6f39ac5e1e4327d94f17f84fa59, no local modification; recreated by reproduce.sh
- `cache/ (24 GB: Qwen/Qwen3-VL-8B-Instruct weights, uv/HF caches)`: public model weights, re-downloadable from Hugging Face (Qwen/Qwen3-VL-8B-Instruct, Apache-2.0)
- `.venv, .venv_vlm, .venv_report, tmp/`: regenerable environments / scratch; requirements.lock + reproduce.sh recreate them
- `checkpoints/actors, checkpoints/hf_actors`: copies of actors already stored in the same HF dataset under core_extensions_2026/runs/**/actor.pt
- `credentials`: never stored in the workspace; the HF token was read from /home/colligo/DatasetDiscovery/.env at upload time only

## Cleanup procedure
- Planned deletion: the entire folder `/home/colligo/working/manaswi/experimentation`, only if the verification above shows zero missing / mismatching files.
- Safety checks before deletion: canonical path equals the workspace path; not a symlink; the parent `/home/colligo/working/manaswi` and the pre-existing sibling `DATASET_GENERATION` are never touched; no project process is running.
- The result (deleted / blocked, sibling integrity) is recorded in `CLEANUP_RESULT.json`, uploaded to both remotes after deletion (the folder cannot hold its own deletion record).
