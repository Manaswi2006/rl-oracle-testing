| Method | Button Press | Hammer | Peg Insert | Other tasks (Sweep, Dial, Drawer, Soccer: only Dial Turn run, key arms) | Env steps (task+extra) | GPU-h / run* |
|---|---|---|---|---|---|---|
| CoRe RRM only (`rrm_only`) | 16.7 ± 40.8 (n=6) | 16.7 ± 40.8 (n=6) | 0.0 ± 0.0 (n=3) | 33.3 ± 57.7 (n=3) (Dial Turn) | 200,000 | 3.5 |
| Full CoRe (`core`) | 66.7 ± 51.6 (n=6) | 66.7 ± 51.6 (n=6) | 0.0 ± 0.0 (n=3) | 0.0 ± 0.0 (n=3) (Dial Turn) | 200,000 | 3.4 |
| Env sparse reward | 0.0 ± 0.0 (n=3) | 66.7 ± 57.7 (n=3) | 0.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 200,000 | 2.4 |
| Env dense reward | 100.0 ± 0.0 (n=3) | 66.7 ± 57.7 (n=3) | 100.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 200,000 | 3.6 |
| RRM + oracle SVM (`psvm_oracle`) | 83.3 ± 40.8 (n=6) | 66.7 ± 51.6 (n=6) | 0.0 ± 0.0 (n=3) | 66.7 ± 57.7 (n=3) (Dial Turn) | 200,000 | 3.6 |
| Control: RRM + plain sparse outcome reward (`outcome_sparse`) | 100.0 ± 0.0 (n=6) | 100.0 ± 0.0 (n=6) | 0.0 ± 0.0 (n=3) | 66.7 ± 57.7 (n=3) (Dial Turn) | 200,000 | 1.9 |
| Control: oracle SVM without RRM (`psvm_noRRM`) | 0.0 ± 0.0 (n=3) | 33.3 ± 57.7 (n=3) | NOT RUN | NOT RUN (Dial Turn) | 200,000 | 2.0 |
| RRM + P-SVM (VLM/noisy labels) | NOT RUN | NOT RUN | NOT RUN | NOT RUN | N/A | N/A |
| RRM + oracle FACR counterfactual (`facr_facr`) | 50.0 ± 54.8 (n=6) | 33.3 ± 51.6 (n=6) | 0.0 ± 0.0 (n=3) | 66.7 ± 57.7 (n=3) (Dial Turn) | 200,000 | 5.1 |
| RRM + oracle success-value shaping (`facr_value`) | 33.3 ± 51.6 (n=6) | 0.0 ± 0.0 (n=6) | 0.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 200,000 | 3.9 |
| RRM + oracle progress prediction (`facr_progress`) | 0.0 ± 0.0 (n=3) | 33.3 ± 57.7 (n=3) | 0.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 200,000 | 4.4 |
| RRM + non-oracle FACR | NOT RUN | NOT RUN | NOT RUN | NOT RUN | N/A | N/A |
| RRM-only + 125 extra teacher labels (control) | 16.7 ± 40.8 (n=6) | 0.0 ± 0.0 (n=6) | 0.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 200,000 | 3.6 |
| RRM + RLF repair: on-policy selection | 50.0 ± 54.8 (n=6) | 16.7 ± 40.8 (n=6) | 0.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 200,000 | 3.6 |
| RRM + RLF repair: high-reward (hard-negative) selection | 66.7 ± 51.6 (n=6) | 16.7 ± 40.8 (n=6) | 0.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 200,000 | 3.7 |
| RRM + RLF repair: uncertainty selection | 66.7 ± 51.6 (n=6) | 16.7 ± 40.8 (n=6) | 0.0 ± 0.0 (n=3) | 100.0 ± 0.0 (n=3) (Dial Turn) | 200,000 | 3.5 |
| RRM + RLF repair: falsification policy | 66.7 ± 51.6 (n=6) | 0.0 ± 0.0 (n=6) | 0.0 ± 0.0 (n=3) | NOT RUN (Dial Turn) | 290,000 | 4.9 |
| RRM + RLF repair: hybrid attack+uncertainty | 100.0 ± 0.0 (n=2) | 0.0 ± 0.0 (n=2) | NOT RUN | NOT RUN (Dial Turn) | 290,000 | 4.3 |
| RRM + RLF repair: random policy | 50.0 ± 70.7 (n=2) | 0.0 ± 0.0 (n=2) | NOT RUN | NOT RUN (Dial Turn) | 260,000 | 3.6 |
| RRM + RLF repair: verifier-disagreement (oracle upper bound) | 100.0 ± 0.0 (n=2) | 0.0 ± 0.0 (n=2) | NOT RUN | NOT RUN (Dial Turn) | 200,000 | 4.1 |
