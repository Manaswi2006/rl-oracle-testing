# Transfer and cleanup manifest

Generated 2026-10-23 15:59:54 PDT.

| item | value |
|---|---|
| server workspace (provenance) | `/home/colligo/working/manaswi/experimentation` |
| GitHub | https://github.com/Manaswi2006/rl-oracle-testing branch `main`, folder `experimentation/`, **code commit `86276e64eca46f249d5a8fdada79c7490db631d7`** (a later commit adds this manifest, the commit-stamped `MASTER_RESULTS.csv` and the cleanup record) |
| Hugging Face | https://huggingface.co/datasets/Manaswi2006/RL_project folder `core_extensions_2026/`, **data revision `0ff0d495d9fba4f5049e8a3673c84b67fd5cd68f`** (verified); later commits add manifests and the cleanup record |
| files in inventory | 3490 files, 3.13 GB (`manifests/local_inventory.json`: path, size, sha256) |
| verification | 3492 of 3492 files verified by size + content hash; missing 0; size mismatches 0; hash mismatches 0; random download sample 40 files, mismatches 0 |

Size mismatches listed by the verification report (if any) are files edited after the upload (the verification script itself); they are re-uploaded and re-verified before cleanup.

## Inventory by top-level folder
| folder | files | bytes |
|---|---|---|
| `README.md` | 1 | 2,168 |
| `__top__` | 5 | 3,814,542 |
| `configs` | 1 | 1,172 |
| `environment.yml` | 1 | 271 |
| `figures` | 16 | 1,846,822 |
| `jobs_rerun2.txt` | 1 | 860 |
| `jobs_stage0.txt` | 1 | 3,030 |
| `jobs_stage0_rerun.txt` | 1 | 230 |
| `jobs_stage1.txt` | 1 | 4,200 |
| `jobs_stage2.txt` | 1 | 8,484 |
| `jobs_stage3.txt` | 1 | 10,215 |
| `jobs_stage4.txt` | 1 | 1,521 |
| `jobs_stage5.txt` | 1 | 2,643 |
| `logs` | 27 | 131,781 |
| `patches` | 1 | 53 |
| `reports` | 15 | 3,885,683 |
| `reproduce.sh` | 1 | 994 |
| `requirements.lock` | 1 | 1,786 |
| `results` | 44 | 3,705,461 |
| `run_all.sh` | 1 | 946 |
| `runs` | 3333 | 3,114,652,510 |
| `scripts` | 16 | 86,697 |
| `src` | 16 | 69,984 |
| `tests` | 3 | 7,549 |

## Not transferred (and why)
- `upstream/CoRe`: third-party clone (68 MB), pinned commit d663cabc38e6d6f39ac5e1e4327d94f17f84fa59, no local modification; recreated by reproduce.sh
- `.venv, .venv_report, cache/, tmp/`: regenerable environments / caches / scratch; requirements.lock + reproduce.sh recreate them
- `runs/**/vlm_label_set/`: empty directories (no VLM labels exist; teacher labels are in preference_labels.npz)
- `credentials`: never stored in the workspace; HF token was read from /home/colligo/DatasetDiscovery/.env at upload time only

## Cleanup procedure
- Planned deletion: the entire folder `/home/colligo/working/manaswi/experimentation`, only if the verification above shows zero missing / mismatching files.
- Safety checks before deletion: canonical path equals the workspace path; not a symlink; the parent `/home/colligo/working/manaswi` and the pre-existing sibling `DATASET_GENERATION` are never touched; no project process is running.
- The result (deleted / blocked, sibling integrity) is recorded in `CLEANUP_RESULT.json`, uploaded to both remotes after deletion (the folder cannot hold its own deletion record).
