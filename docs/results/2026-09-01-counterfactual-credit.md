# Counterfactual credit: hindsight bounds vs measured leave-one-out Δ

Sample: 30 `sh` + 30 DEV classical expressions, 1095 applications sampled (1095 at ordinal < B=100, seed 0x5eedc0dec0ffee01). wall_clock_ceiling_hit = false. git d7a29b2744c7338d8795691b74d2b870f1fd0d00.

Δ distribution: zero 1012/1095 (92.4%), positive 61/1095 (5.6%), negative 22/1095 (2.0%).

| bound | Pearson (pooled) | Spearman (pooled) | Pearson (sh) | Spearman (sh) | Pearson (dev) | Spearman (dev) | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| loose | null | null | null | null | null | null | 1095 |
| tight | null | null | null | null | null | null | 1095 |
| strict | 0.036477 | 0.389273 | 0.044512 | 0.414912 | 0.054949 | 0.352117 | 1095 |
| strict_by_output_class | -0.020099 | 0.004593 | -0.004961 | 0.014839 | -0.032104 | 0.015221 | 1095 |

11 expressions had fewer than 20 state-changing applications and contributed all available instead:

- dev_b01_f02_00035 (7 state-changing applications < 20 target)
- dev_b09_f00_00237 (6 state-changing applications < 20 target)
- dev_b16_f00_00347 (15 state-changing applications < 20 target)
- dev_b01_f01_00019 (5 state-changing applications < 20 target)
- dev_b01_f05_00077 (8 state-changing applications < 20 target)
- dev_b06_f07_00219 (16 state-changing applications < 20 target)
- dev_b33_f00_00683 (18 state-changing applications < 20 target)
- dev_b09_f06_00315 (10 state-changing applications < 20 target)
- dev_b22_f01_00469 (11 state-changing applications < 20 target)
- dev_b09_f06_00309 (15 state-changing applications < 20 target)
- dev_b01_f06_00090 (4 state-changing applications < 20 target)
