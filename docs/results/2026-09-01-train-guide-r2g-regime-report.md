# Guide return-to-go training report

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Objective: **return-mse**. Regression target: **centered** at B = **100**.

Training regime (docs/plans/2026-09-01-guide-return-to-go.md §2b.3): **expr_node_count in [101, 1000], application_ordinal < 100**.

TRAIN: 18565784 records read, 17283072 dropped by the regime, 1282712 kept, 1282712 labelled samples.

DEV (held out, never trained on): 3544477 records read, 3294877 dropped by the regime, 249600 kept, 249600 labelled samples.

**TRAIN final-epoch mean loss**: 1.302296.

**Zero-predictor floor** (constant `f ≡ 0` on the same DEV samples): MSE = 1.133602. A model at or above this floor has learned nothing.

**Held-out (DEV) regression quality**: MSE = 1.132686, Spearman(predicted return, realized return) = 0.2375.

## Training curve

| epoch | lr | train mean loss | DEV MSE | DEV Spearman |
|---:|---:|---:|---:|---:|
| 0 | 0.00010 | 1.302619 | 1.130877 | 0.2326 |
| 1 | 0.00010 | 1.302529 |  |  |
| 2 | 0.00009 | 1.302614 |  |  |
| 3 | 0.00009 | 1.302571 |  |  |
| 4 | 0.00008 | 1.302598 |  |  |
| 5 | 0.00008 | 1.302587 | 1.133614 | 0.2376 |
| 6 | 0.00008 | 1.302592 |  |  |
| 7 | 0.00007 | 1.302562 |  |  |
| 8 | 0.00007 | 1.302562 |  |  |
| 9 | 0.00007 | 1.302517 |  |  |
| 10 | 0.00007 | 1.302450 | 1.133196 | 0.2379 |
| 11 | 0.00006 | 1.302529 |  |  |
| 12 | 0.00006 | 1.302431 |  |  |
| 13 | 0.00006 | 1.302430 |  |  |
| 14 | 0.00006 | 1.302501 |  |  |
| 15 | 0.00006 | 1.302414 | 1.131843 | 0.2378 |
| 16 | 0.00006 | 1.302440 |  |  |
| 17 | 0.00005 | 1.302433 |  |  |
| 18 | 0.00005 | 1.302471 |  |  |
| 19 | 0.00005 | 1.302394 |  |  |
| 20 | 0.00005 | 1.302398 | 1.133695 | 0.2387 |
| 21 | 0.00005 | 1.302401 |  |  |
| 22 | 0.00005 | 1.302345 |  |  |
| 23 | 0.00005 | 1.302402 |  |  |
| 24 | 0.00005 | 1.302339 |  |  |
| 25 | 0.00004 | 1.302356 | 1.132418 | 0.2372 |
| 26 | 0.00004 | 1.302394 |  |  |
| 27 | 0.00004 | 1.302371 |  |  |
| 28 | 0.00004 | 1.302360 |  |  |
| 29 | 0.00004 | 1.302296 | 1.132686 | 0.2375 |

## Per-rule: DEV mean predicted return vs mean realized return

| rule | idx | DEV fired | mean predicted return | mean realized return |
|---|---:|---:|---:|---:|
| halving | 21 | 66 | 0.09483 | 0.41051 |
| reverse-associative | 26 | 2078 | 0.01876 | 0.18234 |
| distribute | 18 | 1352 | 0.01129 | 0.14642 |
| involution | 1 | 19043 | 0.00172 | 0.07665 |
| commutative | 11 | 5257 | -0.00527 | 0.10379 |
| reverse-associative | 28 | 356 | -0.00919 | 0.03715 |
| commutative | 12 | 5114 | -0.01042 | 0.07383 |
| commutative | 9 | 62980 | -0.01368 | 0.08535 |
| idempotent | 16 | 150 | -0.02127 | 0.06044 |
| factor | 19 | 2370 | -0.03843 | 0.12754 |
| associative | 22 | 2389 | -0.04002 | 0.04592 |
| associative | 24 | 266 | -0.04064 | 0.08309 |
| associative | 25 | 309 | -0.04128 | 0.02680 |
| commutative | 10 | 22338 | -0.04611 | 0.00141 |
| odd-negation | 31 | 309 | -0.04990 | -0.00627 |
| identity | 13 | 4 | -0.05677 | 0.11239 |
| reverse-associative | 27 | 3394 | -0.05756 | 0.01348 |
| idempotent | 17 | 254 | -0.06448 | -0.05194 |
| constant-fold | 8 | 13795 | -0.06609 | -0.00456 |
| canonicalize | 0 | 1852 | -0.06793 | 0.02976 |
| log2-exp2-cancel | 44 | 38 | -0.06873 | 0.16802 |
| associative | 23 | 3320 | -0.07095 | 0.02391 |
| reverse-associative | 29 | 222 | -0.07236 | 0.13233 |
| canonicalize | 4 | 1373 | -0.07496 | 0.08620 |
| odd-negation | 30 | 517 | -0.07499 | -0.03093 |
| ln-exp-cancel | 42 | 25 | -0.07942 | -0.00010 |
| power-recip | 52 | 9001 | -0.09915 | -0.06948 |
| exp2-log2-cancel | 43 | 233 | -0.09941 | 0.02203 |
| exp-ln-cancel | 41 | 191 | -0.10027 | -0.06603 |
| even-negation | 35 | 51639 | -0.10111 | -0.09307 |
| even-negation | 34 | 5953 | -0.10136 | -0.10378 |
| doubling | 20 | 731 | -0.10369 | -0.04771 |
| power-rsqrt | 53 | 3339 | -0.10629 | -0.07250 |
| cos-angle-addition | 37 | 163 | -0.10954 | -0.09573 |
| power-sqrt | 51 | 12404 | -0.11312 | -0.09266 |
| sin-angle-addition | 36 | 109 | -0.11401 | 0.10522 |
| fma-fusion | 59 | 11968 | -0.12040 | -0.10099 |
| recip-sqrt | 60 | 2247 | -0.12570 | -0.08968 |
| power-combine | 47 | 2041 | -0.12925 | -0.09038 |
| exp-homomorphism | 45 | 135 | -0.13099 | -0.02249 |
| inverse-annihilation | 3 | 125 | -0.13481 | -0.01654 |
| reverse-angle-addition | 38 | 150 | -0.15528 | -0.07903 |

Checkpoint written to `pixelflow-pipeline/data/guide_checkpoint_r2g_regime_v1.json`.
