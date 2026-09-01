# Guide cold-start training report (strict-v1 labels)

TRAIN: 660404 samples, 223 families, positive rate 0.5220%.

DEV (held out, never trained on): 155231 samples, 54 families, positive rate 0.5901%.

**Loss weighting**: pos_weight = 190.588 (inverse class frequency (negatives/positives = 656957/3447 measured on this TRAIN split) — the simplest defensible cold-start choice: unweighted BCE lets a trainer collapse to predicting the majority class and still score >99% raw accuracy (flagged by gen_strict_labels's own report), and there is no prior Guide run to tune a fancier (focal-loss-style) weighting against yet).

**Held-out (DEV-family) ranking quality**: AUC-ROC = 0.9898, PR-AUC (average precision) = 0.4544.

**Sanity check** — Spearman correlation between the learned per-rule mean predicted probability and the DEV-measured per-rule strict-positive rate: ρ = 0.7964. A model that learned nothing beyond noise would show ρ near 0; a model that only reproduced each rule's overall base rate (ignoring candidate-local structure) would still show ρ close to 1 here, since both quantities are monotonic in the same underlying per-rule tendency — this check confirms the model tracks the label semantics, not that it beats a per-rule lookup table (a saturation-quality evaluation, out of scope for this report, would be needed for that).

## Training curve

| epoch | lr | train weighted loss | DEV AUC | DEV PR-AUC |
|---:|---:|---:|---:|---:|
| 0 | 0.01000 | 0.807169 | 0.9873 | 0.3870 |
| 1 | 0.00952 | 0.768908 |  |  |
| 2 | 0.00909 | 0.729487 |  |  |
| 3 | 0.00870 | 0.747655 |  |  |
| 4 | 0.00833 | 0.720326 |  |  |
| 5 | 0.00800 | 0.705080 | 0.9892 | 0.3691 |
| 6 | 0.00769 | 0.679273 |  |  |
| 7 | 0.00741 | 0.661310 |  |  |
| 8 | 0.00714 | 0.646522 |  |  |
| 9 | 0.00690 | 0.642105 |  |  |
| 10 | 0.00667 | 0.638281 | 0.9887 | 0.4150 |
| 11 | 0.00645 | 0.618157 |  |  |
| 12 | 0.00625 | 0.621326 |  |  |
| 13 | 0.00606 | 0.621496 |  |  |
| 14 | 0.00588 | 0.610095 |  |  |
| 15 | 0.00571 | 0.586170 | 0.9899 | 0.4304 |
| 16 | 0.00556 | 0.584833 |  |  |
| 17 | 0.00541 | 0.560031 |  |  |
| 18 | 0.00526 | 0.559622 |  |  |
| 19 | 0.00513 | 0.559865 |  |  |
| 20 | 0.00500 | 0.551514 | 0.9898 | 0.4104 |
| 21 | 0.00488 | 0.545076 |  |  |
| 22 | 0.00476 | 0.540550 |  |  |
| 23 | 0.00465 | 0.533268 |  |  |
| 24 | 0.00455 | 0.515950 |  |  |
| 25 | 0.00444 | 0.530378 | 0.9900 | 0.4471 |
| 26 | 0.00435 | 0.523583 |  |  |
| 27 | 0.00426 | 0.516795 |  |  |
| 28 | 0.00417 | 0.511711 |  |  |
| 29 | 0.00408 | 0.516728 | 0.9898 | 0.4544 |

## Calibration (population-quantile buckets, dense toward the top, DEV)

| quantile range | n | predicted range | mean predicted | actual positive rate |
|---|---:|---|---:|---:|
| [0.000, 0.500) | 77616 | [0.000000, 0.000000] | 0.000000 | 0.000000 |
| [0.500, 0.750) | 38807 | [0.000000, 0.000000] | 0.000000 | 0.000026 |
| [0.750, 0.900) | 23285 | [0.000000, 0.000955] | 0.000120 | 0.000258 |
| [0.900, 0.950) | 7761 | [0.000955, 0.283341] | 0.041968 | 0.003865 |
| [0.950, 0.980) | 4657 | [0.284284, 0.952152] | 0.725769 | 0.027056 |
| [0.980, 0.990) | 1553 | [0.952182, 0.984173] | 0.970040 | 0.112685 |
| [0.990, 0.995) | 776 | [0.984181, 0.993249] | 0.989153 | 0.240979 |
| [0.995, 0.998) | 466 | [0.993252, 0.997161] | 0.995210 | 0.409871 |
| [0.998, 0.999) | 155 | [0.997171, 0.998428] | 0.997907 | 0.606452 |
| [0.999, 1.000) | 155 | [0.998445, 0.999808] | 0.999002 | 0.683871 |

## Per-rule: learned priority (DEV mean predicted) vs measured strict-bound rate

| rule | idx | train fired | train rate | DEV fired | DEV measured rate | DEV mean predicted |
|---|---:|---:|---:|---:|---:|---:|
| power-rsqrt | 53 | 707 | 0.18105 | 187 | 0.19251 | 0.56384 |
| power-recip | 52 | 2556 | 0.16080 | 799 | 0.14393 | 0.52895 |
| recip-sqrt | 60 | 741 | 0.16464 | 213 | 0.15962 | 0.50375 |
| power-sqrt | 51 | 3632 | 0.13546 | 994 | 0.12575 | 0.46832 |
| even-negation | 34 | 2440 | 0.05533 | 641 | 0.06240 | 0.43841 |
| even-negation | 35 | 17851 | 0.05557 | 5591 | 0.04686 | 0.40836 |
| power-combine | 47 | 1122 | 0.04278 | 319 | 0.05016 | 0.27909 |
| constant-fold | 8 | 25434 | 0.01860 | 5463 | 0.02105 | 0.23040 |
| odd-negation | 30 | 568 | 0.01408 | 182 | 0.00549 | 0.20149 |
| canonicalize | 0 | 489 | 0.01636 | 175 | 0.01143 | 0.18493 |
| cos-angle-addition | 37 | 340 | 0.00882 | 103 | 0.00000 | 0.12982 |
| log2-exp2-cancel | 44 | 62 | 0.00000 | 14 | 0.00000 | 0.11759 |
| reverse-angle-addition | 38 | 563 | 0.01066 | 134 | 0.00746 | 0.11703 |
| ln-exp-cancel | 42 | 43 | 0.00000 | 6 | 0.00000 | 0.10340 |
| doubling | 20 | 6616 | 0.00499 | 1222 | 0.00655 | 0.06767 |
| exp2-log2-cancel | 43 | 63 | 0.00000 | 14 | 0.00000 | 0.05962 |
| idempotent | 16 | 73 | 0.00000 | 19 | 0.00000 | 0.05137 |
| fma-fusion | 59 | 109823 | 0.00407 | 22242 | 0.00486 | 0.04721 |
| odd-negation | 31 | 309 | 0.00324 | 97 | 0.01031 | 0.04389 |
| idempotent | 17 | 70 | 0.00000 | 24 | 0.00000 | 0.03970 |
| canonicalize | 4 | 179 | 0.00000 | 72 | 0.00000 | 0.03782 |
| factor | 19 | 25516 | 0.00231 | 6538 | 0.00352 | 0.03376 |
| sin-angle-addition | 36 | 381 | 0.00262 | 109 | 0.00000 | 0.03287 |
| associative | 25 | 528 | 0.00379 | 496 | 0.00000 | 0.02943 |
| associative | 24 | 422 | 0.00000 | 45 | 0.00000 | 0.01639 |
| reverse-associative | 28 | 439 | 0.00000 | 47 | 0.00000 | 0.01465 |
| reverse-associative | 27 | 33458 | 0.00078 | 9646 | 0.00187 | 0.01322 |
| exp-ln-cancel | 41 | 43 | 0.00000 | 6 | 0.00000 | 0.01240 |
| exp-homomorphism | 45 | 93 | 0.00000 | 52 | 0.00000 | 0.00797 |
| associative | 23 | 30859 | 0.00055 | 8962 | 0.00045 | 0.00783 |
| associative | 22 | 51592 | 0.00033 | 9634 | 0.00021 | 0.00348 |
| power-identity | 49 | 257 | 0.00000 | 74 | 0.00000 | 0.00307 |
| reverse-associative | 29 | 547 | 0.00000 | 537 | 0.00372 | 0.00175 |
| halving | 21 | 10635 | 0.00009 | 1933 | 0.00000 | 0.00154 |
| inverse-annihilation | 3 | 474 | 0.01477 | 309 | 0.00000 | 0.00116 |
| reverse-associative | 26 | 56485 | 0.00014 | 10496 | 0.00029 | 0.00115 |
| commutative | 11 | 2925 | 0.00000 | 906 | 0.00000 | 0.00052 |
| commutative | 12 | 3166 | 0.00000 | 1258 | 0.00000 | 0.00030 |
| commutative | 10 | 82389 | 0.00000 | 22455 | 0.00000 | 0.00024 |
| commutative | 9 | 131272 | 0.00000 | 28259 | 0.00000 | 0.00017 |
| involution | 1 | 16507 | 0.00000 | 5347 | 0.00000 | 0.00008 |
| distribute | 18 | 22535 | 0.00000 | 5058 | 0.00000 | 0.00002 |
| identity | 13 | 10515 | 0.00000 | 3299 | 0.00000 | 0.00000 |
| annihilator | 15 | 3392 | 0.00000 | 1120 | 0.00000 | 0.00000 |
| half-angle-product | 39 | 221 | 0.00452 | 110 | 0.00000 | 0.00000 |
| cancellation | 2 | 9 | 0.00000 | 9 | 0.00000 | 0.00000 |
| cancellation | 6 | 48 | 0.00000 | 15 | 0.00000 | 0.00000 |
| inverse-annihilation | 7 | 56 | 0.00000 | 0 | 0.00000 | 0.00000 |
| identity | 14 | 1929 | 0.00000 | 0 | 0.00000 | 0.00000 |
| power-zero | 48 | 8 | 0.12500 | 0 | 0.00000 | 0.00000 |
| log2-power | 56 | 11 | 0.00000 | 0 | 0.00000 | 0.00000 |
| diff-of-squares | 58 | 11 | 0.00000 | 0 | 0.00000 | 0.00000 |

Checkpoint written to `pixelflow-pipeline/data/guide_checkpoint_strict_v1.json`.
