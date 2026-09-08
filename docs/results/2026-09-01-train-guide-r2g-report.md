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

TRAIN: 971972 records, 971972 labelled samples.

DEV (held out, never trained on): 932261 records, 932261 labelled samples.

**TRAIN final-epoch mean loss**: 0.704730.

**Held-out (DEV) regression quality**: MSE = 0.494610, Spearman(predicted return, realized return) = 0.0990.

## Training curve

| epoch | lr | train mean loss | DEV MSE | DEV Spearman |
|---:|---:|---:|---:|---:|
| 0 | 0.00010 | 0.704875 | 0.494782 | 0.0567 |
| 1 | 0.00010 | 0.704799 |  |  |
| 2 | 0.00009 | 0.704853 |  |  |
| 3 | 0.00009 | 0.704782 |  |  |
| 4 | 0.00008 | 0.704816 |  |  |
| 5 | 0.00008 | 0.704801 | 0.494790 | 0.1037 |
| 6 | 0.00008 | 0.704775 |  |  |
| 7 | 0.00007 | 0.704808 |  |  |
| 8 | 0.00007 | 0.704790 |  |  |
| 9 | 0.00007 | 0.704816 |  |  |
| 10 | 0.00007 | 0.704797 | 0.494587 | 0.0732 |
| 11 | 0.00006 | 0.704810 |  |  |
| 12 | 0.00006 | 0.704746 |  |  |
| 13 | 0.00006 | 0.704793 |  |  |
| 14 | 0.00006 | 0.704750 |  |  |
| 15 | 0.00006 | 0.704780 | 0.494520 | 0.0996 |
| 16 | 0.00006 | 0.704752 |  |  |
| 17 | 0.00005 | 0.704749 |  |  |
| 18 | 0.00005 | 0.704752 |  |  |
| 19 | 0.00005 | 0.704739 |  |  |
| 20 | 0.00005 | 0.704769 | 0.494680 | 0.0440 |
| 21 | 0.00005 | 0.704743 |  |  |
| 22 | 0.00005 | 0.704709 |  |  |
| 23 | 0.00005 | 0.704765 |  |  |
| 24 | 0.00005 | 0.704710 |  |  |
| 25 | 0.00004 | 0.704768 | 0.494559 | 0.0547 |
| 26 | 0.00004 | 0.704744 |  |  |
| 27 | 0.00004 | 0.704736 |  |  |
| 28 | 0.00004 | 0.704726 |  |  |
| 29 | 0.00004 | 0.704730 | 0.494610 | 0.0990 |

## Per-rule: DEV mean predicted return vs mean realized return

| rule | idx | DEV fired | mean predicted return | mean realized return |
|---|---:|---:|---:|---:|
| identity | 14 | 45 | 0.02025 | 0.00002 |
| reverse-associative | 28 | 1223 | 0.01472 | -0.01268 |
| power-zero | 48 | 38 | 0.01101 | 0.00007 |
| associative | 24 | 1216 | 0.00562 | -0.02503 |
| inverse-annihilation | 7 | 1 | 0.00506 | 0.00000 |
| idempotent | 16 | 578 | 0.00264 | -0.01684 |
| odd-negation | 31 | 2311 | 0.00190 | -0.00968 |
| commutative | 9 | 188988 | -0.00152 | 0.02976 |
| commutative | 11 | 11937 | -0.00240 | -0.02158 |
| involution | 1 | 128549 | -0.00379 | 0.00578 |
| exp2-log2-cancel | 43 | 467 | -0.00589 | 0.05085 |
| commutative | 10 | 104384 | -0.00914 | -0.00656 |
| constant-fold | 8 | 55828 | -0.01045 | 0.00559 |
| factor | 19 | 26327 | -0.01081 | -0.01415 |
| power-combine | 47 | 5222 | -0.01145 | -0.02933 |
| odd-negation | 30 | 4342 | -0.01167 | -0.00427 |
| cos-angle-addition | 37 | 1943 | -0.01231 | 0.00142 |
| distribute | 18 | 25889 | -0.01498 | -0.01447 |
| reverse-associative | 26 | 24846 | -0.01506 | -0.01486 |
| even-negation | 35 | 100252 | -0.01600 | -0.00952 |
| power-sqrt | 51 | 23147 | -0.01730 | -0.02206 |
| power-rsqrt | 53 | 4915 | -0.01732 | -0.01424 |
| identity | 13 | 2878 | -0.01804 | -0.03939 |
| associative | 22 | 24989 | -0.01812 | -0.02420 |
| associative | 23 | 22474 | -0.01898 | -0.01220 |
| power-identity | 49 | 957 | -0.01933 | -0.01251 |
| even-negation | 34 | 13875 | -0.01944 | -0.01143 |
| reverse-associative | 27 | 25188 | -0.02002 | -0.00842 |
| commutative | 12 | 12578 | -0.02177 | -0.01233 |
| recip-sqrt | 60 | 4853 | -0.02239 | -0.01195 |
| power-recip | 52 | 21307 | -0.02257 | -0.01172 |
| fma-fusion | 59 | 60911 | -0.02269 | -0.02331 |
| log2-exp2-cancel | 44 | 257 | -0.02369 | 0.11730 |
| doubling | 20 | 10224 | -0.02599 | -0.02804 |
| halving | 21 | 7384 | -0.02619 | -0.02608 |
| ln-exp-cancel | 42 | 206 | -0.03151 | -0.00000 |
| reverse-angle-addition | 38 | 776 | -0.03347 | 0.00367 |
| canonicalize | 0 | 2896 | -0.03362 | -0.00848 |
| cancellation | 6 | 42 | -0.03511 | 0.00000 |
| exp-ln-cancel | 41 | 454 | -0.03668 | 0.00003 |
| idempotent | 17 | 411 | -0.03833 | -0.04637 |
| sin-angle-addition | 36 | 1086 | -0.04013 | -0.00140 |
| associative | 25 | 1474 | -0.04700 | -0.11134 |
| reverse-associative | 29 | 1555 | -0.05100 | 0.03578 |
| inverse-annihilation | 3 | 756 | -0.05300 | -0.06684 |
| diff-of-squares | 58 | 82 | -0.06349 | 0.00000 |
| canonicalize | 4 | 1441 | -0.06717 | -0.00102 |
| half-angle-product | 39 | 126 | -0.07102 | -0.00000 |
| exp-homomorphism | 45 | 311 | -0.07114 | -0.08982 |
| annihilator | 15 | 322 | -0.07140 | -0.00015 |

Checkpoint written to `pixelflow-pipeline/data/guide_checkpoint_r2g_v1.json`.
