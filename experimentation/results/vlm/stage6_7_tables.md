| arm | Button Press | Hammer |
|---|---|---|
| earlier: RRM-only (scripted teacher) | 16.7 ± 40.8 (n=6) | 16.7 ± 40.8 (n=6) |
| earlier: full CoRe | 66.7 ± 51.6 (n=6) | 66.7 ± 51.6 (n=6) |
| earlier: RRM + per-step oracle success flag | 100.0 ± 0.0 (n=6) | 100.0 ± 0.0 (n=6) |
| earlier: env sparse only | 0.0 ± 0.0 (n=3) | 66.7 ± 57.7 (n=3) |
| S6 scripted RRM + oracle EPISODE outcome | 100.0 ± 0.0 (n=6) | 50.0 ± 54.8 (n=6) |
| S6 scripted RRM + VLM verdict, majority of 3 | 83.3 ± 40.8 (n=6) | 0.0 ± 0.0 (n=6) |
| S6 scripted RRM + VLM verdict, soft fraction | 83.3 ± 40.8 (n=6) | 33.3 ± 51.6 (n=6) |
| S6 oracle episode outcome only (RRM weight 0) | 66.7 ± 57.7 (n=3) | 100.0 ± 0.0 (n=3) |
| S6 VLM verdict only (RRM weight 0) | 66.7 ± 57.7 (n=3) | 0.0 ± 0.0 (n=3) |
| S7 RND + oracle outcome (RRM weight 0) | 50.0 ± 54.8 (n=6) | 50.0 ± 54.8 (n=6) |
| S7 RND + VLM verdict (RRM weight 0) | 66.7 ± 51.6 (n=6) | 16.7 ± 40.8 (n=6) |
| S7 RND only (no outcome signal) | 0.0 ± 0.0 (n=3) | 0.0 ± 0.0 (n=3) |

| comparison (seed means) | Button Press diff [95% bootstrap CI] | Hammer |
|---|---|---|
| ep_vlm_major - ep_oracle | -16.7 [-50.0, +0.0] | -50.0 [-83.3, -16.7] |
| ep_vlm_soft - ep_oracle | -16.7 [-50.0, +0.0] | -16.7 [-66.7, +33.3] |
| ep_vlm_major - ep_vlm_soft | +0.0 [-50.0, +33.3] | -33.3 [-66.7, +0.0] |
| ep_vlm_noRRM - ep_oracle_noRRM | +0.0 [-66.7, +66.7] | -100.0 [-100.0, -100.0] |
| rnd_vlm - ep_vlm_noRRM | +0.0 [-66.7, +66.7] | +16.7 [+0.0, +50.0] |
| rnd_vlm - rnd_oracle | +16.7 [-33.3, +66.7] | -33.3 [-83.3, +16.7] |
| rnd_oracle - ep_oracle_noRRM | -16.7 [-83.3, +50.0] | -50.0 [-83.3, -16.7] |
| rnd_only - ep_vlm_noRRM | -66.7 [-100.0, +0.0] | +0.0 [+0.0, +0.0] |

| VLM in-loop label accuracy (all episodes of all runs of the arm) | Button Press | Hammer |
|---|---|---|
| ep_vlm_major | acc 0.821; FP 306/1225 failures; FN 123/1175 successes (6 runs) | acc 0.440; FP 1335/2008 failures; FN 9/392 successes (6 runs) |
| ep_vlm_soft | acc 0.915; FP 79/1369 failures; FN 126/1031 successes (6 runs) | acc 0.680; FP 749/1758 failures; FN 18/642 successes (6 runs) |
| ep_vlm_noRRM | acc 0.932; FP 25/607 failures; FN 56/593 successes (3 runs) | acc 0.615; FP 460/1076 failures; FN 2/124 successes (3 runs) |
| rnd_vlm | acc 0.910; FP 65/1357 failures; FN 151/1043 successes (6 runs) | acc 0.532; FP 1110/1988 failures; FN 14/412 successes (6 runs) |
