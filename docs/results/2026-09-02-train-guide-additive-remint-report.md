# Guide cold-start training report (strict-v1 labels)

> Budget denominator: REGISTERED_PRIMARY_BUDGET_APPLICATIONS = 100 (docs/plans/2026-09-01-phase3-registration.md §4, classical-band primary tier), imported by both gen_strict_labels and saturate_guided_until_applications so budget_fraction means the same thing at mint time and at deploy time. The first mint/train pass of this round (2026-09-01) used a 195-application placeholder (this round's measured median application count, before B was registered) — a train/deploy denominator skew, caught and fixed before this checkpoint (see git history for the placeholder-era numbers).

TRAIN: 22697 samples, 217 families, positive rate 12.4069%.

DEV (held out, never trained on): 6838 samples, 54 families, positive rate 11.8309%.

**Loss weighting**: pos_weight = 7.060 (inverse class frequency (negatives/positives = 19881/2816 measured on this TRAIN split) — the simplest defensible cold-start choice: unweighted BCE lets a trainer collapse to predicting the majority class and still score >99% raw accuracy (flagged by gen_strict_labels's own report), and there is no prior Guide run to tune a fancier (focal-loss-style) weighting against yet).

**Held-out (DEV-family) ranking quality**: AUC-ROC = 0.9451, PR-AUC (average precision) = 0.6837.

**Sanity check** — Spearman correlation between the learned per-rule mean predicted probability and the DEV-measured per-rule strict-positive rate: ρ = 0.9407. A model that learned nothing beyond noise would show ρ near 0; a model that only reproduced each rule's overall base rate (ignoring candidate-local structure) would still show ρ close to 1 here, since both quantities are monotonic in the same underlying per-rule tendency — this check confirms the model tracks the label semantics, not that it beats a per-rule lookup table (a saturation-quality evaluation, out of scope for this report, would be needed for that).

## Training curve

| epoch | lr | train weighted loss | DEV AUC | DEV PR-AUC |
|---:|---:|---:|---:|---:|
| 0 | 0.01000 | 0.663646 | 0.9334 | 0.6150 |
| 1 | 0.00952 | 0.552846 |  |  |
| 2 | 0.00909 | 0.540053 |  |  |
| 3 | 0.00870 | 0.535120 |  |  |
| 4 | 0.00833 | 0.521915 |  |  |
| 5 | 0.00800 | 0.522468 | 0.9398 | 0.6636 |
| 6 | 0.00769 | 0.518014 |  |  |
| 7 | 0.00741 | 0.509385 |  |  |
| 8 | 0.00714 | 0.507348 |  |  |
| 9 | 0.00690 | 0.499487 |  |  |
| 10 | 0.00667 | 0.504132 | 0.9437 | 0.6716 |
| 11 | 0.00645 | 0.495787 |  |  |
| 12 | 0.00625 | 0.493403 |  |  |
| 13 | 0.00606 | 0.495750 |  |  |
| 14 | 0.00588 | 0.490395 |  |  |
| 15 | 0.00571 | 0.491633 | 0.9412 | 0.6648 |
| 16 | 0.00556 | 0.488766 |  |  |
| 17 | 0.00541 | 0.491100 |  |  |
| 18 | 0.00526 | 0.486911 |  |  |
| 19 | 0.00513 | 0.484159 |  |  |
| 20 | 0.00500 | 0.480192 | 0.9427 | 0.6766 |
| 21 | 0.00488 | 0.483189 |  |  |
| 22 | 0.00476 | 0.485076 |  |  |
| 23 | 0.00465 | 0.479628 |  |  |
| 24 | 0.00455 | 0.478696 |  |  |
| 25 | 0.00444 | 0.477174 | 0.9426 | 0.6767 |
| 26 | 0.00435 | 0.480518 |  |  |
| 27 | 0.00426 | 0.476438 |  |  |
| 28 | 0.00417 | 0.474750 |  |  |
| 29 | 0.00408 | 0.473818 | 0.9451 | 0.6837 |

## Calibration (population-quantile buckets, dense toward the top, DEV)

| quantile range | n | predicted range | mean predicted | actual positive rate |
|---|---:|---|---:|---:|
| [0.000, 0.500) | 3419 | [0.000000, 0.008130] | 0.002437 | 0.000000 |
| [0.500, 0.750) | 1710 | [0.008132, 0.482803] | 0.126406 | 0.029240 |
| [0.750, 0.900) | 1025 | [0.483350, 0.868830] | 0.762677 | 0.340488 |
| [0.900, 0.950) | 342 | [0.868859, 0.931560] | 0.892382 | 0.435673 |
| [0.950, 0.980) | 205 | [0.932073, 0.975764] | 0.957903 | 0.634146 |
| [0.980, 0.990) | 69 | [0.975789, 0.978601] | 0.976622 | 0.956522 |
| [0.990, 0.995) | 34 | [0.978855, 0.982511] | 0.980546 | 0.941176 |
| [0.995, 0.998) | 20 | [0.982630, 0.983953] | 0.983306 | 0.950000 |
| [0.998, 0.999) | 7 | [0.985706, 0.989512] | 0.987427 | 1.000000 |
| [0.999, 1.000) | 7 | [0.989948, 0.991409] | 0.990852 | 1.000000 |

## Per-rule: learned priority (DEV mean predicted) vs measured strict-bound rate

| rule | id | train fired | train rate | DEV fired | DEV measured rate | DEV mean predicted |
|---|---:|---:|---:|---:|---:|---:|
| power-rsqrt | e32395e542bd0eeb | 131 | 0.92366 | 36 | 0.97222 | 0.97877 |
| recip-sqrt | 15ac79c3047ad4d1 | 120 | 0.88333 | 35 | 0.85714 | 0.97234 |
| power-recip | 92e3367a6f9e5ccc | 442 | 0.81674 | 136 | 0.80147 | 0.96916 |
| power-sqrt | 2bfc49afae251983 | 542 | 0.75461 | 163 | 0.71779 | 0.94302 |
| even-negation(Cos) | 167b9119efe00893 | 277 | 0.42599 | 79 | 0.48101 | 0.85008 |
| even-negation(Abs) | a1d188080320a7b2 | 1986 | 0.42900 | 599 | 0.38898 | 0.83632 |
| constant-fold | d92019505a8a3247 | 997 | 0.40120 | 289 | 0.36678 | 0.83120 |
| power-combine | 7e1e121b3dacb3ce | 117 | 0.33333 | 36 | 0.38889 | 0.69330 |
| log2-exp2-cancel | 940721a5e42cc290 | 7 | 0.00000 | 2 | 0.00000 | 0.55420 |
| canonicalize(Sub) | a60c541aa296142e | 49 | 0.10204 | 13 | 0.15385 | 0.48279 |
| fma-fusion | 6acd3c44e21839ea | 2016 | 0.14137 | 535 | 0.13271 | 0.46467 |
| ln-exp-cancel | f1ebf9bf7883292e | 7 | 0.00000 | 1 | 0.00000 | 0.41409 |
| exp2-log2-cancel | 9086fe19df53a842 | 7 | 0.00000 | 2 | 0.00000 | 0.29856 |
| factor | 4545eb999aeeeca6 | 677 | 0.05908 | 223 | 0.08520 | 0.26078 |
| doubling | c3e67432f2a1b5e3 | 342 | 0.05556 | 74 | 0.05405 | 0.24565 |
| associative(Max) | ff9d0366e310988d | 30 | 0.06667 | 26 | 0.00000 | 0.23802 |
| odd-negation(Sin) | c6c412afffd3c46b | 86 | 0.04651 | 26 | 0.03846 | 0.22984 |
| idempotent(Min) | bac1aad26a86722d | 13 | 0.00000 | 3 | 0.00000 | 0.22254 |
| canonicalize(Div) | d434c897945f5659 | 20 | 0.00000 | 8 | 0.00000 | 0.21364 |
| idempotent(Max) | ff3aced291136397 | 12 | 0.00000 | 5 | 0.00000 | 0.20711 |
| cos-angle-addition | fd66534f6ee3abc1 | 27 | 0.03704 | 5 | 0.00000 | 0.17999 |
| reverse-angle-addition | 9f5cb6c9195cb5fa | 15 | 0.06667 | 3 | 0.33333 | 0.17806 |
| odd-negation(Tan) | b1ffd2d990da5384 | 47 | 0.02128 | 11 | 0.09091 | 0.17081 |
| associative(Min) | bb313766bc8ec707 | 32 | 0.00000 | 5 | 0.00000 | 0.14914 |
| reverse-associative(Min) | da272d71621a1b82 | 32 | 0.00000 | 5 | 0.00000 | 0.14579 |
| reverse-associative(Mul) | 42de0d719da182d0 | 598 | 0.03010 | 251 | 0.07171 | 0.13382 |
| reverse-associative(Max) | 95add9713b8cd888 | 30 | 0.00000 | 26 | 0.07692 | 0.09169 |
| sin-angle-addition | 567b8b751d23d432 | 15 | 0.00000 | 3 | 0.00000 | 0.08131 |
| associative(Mul) | 68244f671e6f939d | 598 | 0.01672 | 251 | 0.01594 | 0.07817 |
| exp-homomorphism | b1d12260b01d5163 | 12 | 0.00000 | 4 | 0.00000 | 0.06994 |
| power-identity | 479755b99bddb77f | 31 | 0.00000 | 10 | 0.00000 | 0.06211 |
| exp-ln-cancel | d6ecb2b4ffb49d92 | 7 | 0.00000 | 1 | 0.00000 | 0.05855 |
| associative(Add) | 587d0085e2775c26 | 1058 | 0.00851 | 258 | 0.00388 | 0.05457 |
| reverse-associative(Add) | 12f33cd6663ac71f | 1058 | 0.00662 | 258 | 0.01163 | 0.04329 |
| commutative(Min) | 5d63e90e10c3b1a4 | 236 | 0.00000 | 77 | 0.00000 | 0.03787 |
| commutative(Max) | a594cd0e3a796976 | 238 | 0.00000 | 88 | 0.00000 | 0.03218 |
| halving | 5791376f6f4b4304 | 342 | 0.00292 | 74 | 0.00000 | 0.03080 |
| distribute | c55b6b8fee188f1e | 678 | 0.00000 | 223 | 0.00000 | 0.00601 |
| commutative(Mul) | 3d28c10dff31a2de | 2337 | 0.00000 | 809 | 0.00000 | 0.00382 |
| involution(Sub) | 573686fe9a77ffad | 3094 | 0.00000 | 1000 | 0.00000 | 0.00337 |
| commutative(Add) | ce143fa8dbaa336d | 4325 | 0.00000 | 1185 | 0.00000 | 0.00203 |
| inverse-annihilation(Sub) | be54d2411e017373 | 8 | 0.75000 | 0 | 0.00000 | 0.00000 |
| power-zero | a66b8afce49f3b9f | 1 | 1.00000 | 0 | 0.00000 | 0.00000 |

Checkpoint written to `pixelflow-pipeline/data/guide_checkpoint_strict_remint.json`.
