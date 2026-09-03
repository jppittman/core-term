# Bilinear Guide cold-start training report (strict-v1 labels)

Model class: **bilinear (m(x)^T W r + b^T r)**. Registration: `docs/plans/2026-09-02-bilinear-guide-registration.md`.

TRAIN: 22697 samples, 217 families, positive rate 12.4069% (268649 dedup_repeat rows excluded).

DEV (held out, never trained on, never used to select a hyperparameter): 6838 samples, 54 families, positive rate 11.8309% (85751 dedup_repeat rows excluded).

**Loss weighting**: pos_weight = 7.060 (inverse class frequency (negatives/positives = 19881/2816 measured on this TRAIN split) — identical in definition to `train_guide`'s, so the two arms' losses differ in no term).

## Held-out DEV ranking quality, against the zero-predictor floor

| metric | bilinear | constant-predictor floor |
|---|---:|---:|
| AUC-ROC | 0.9558 | 0.5000 |
| PR-AUC (average precision) | 0.7624 | 0.1183 |

Within-expression AUC (the move-ordering question a deployed guide is actually asked), macro-averaged over the 116 expressions whose candidate set holds both classes: **0.9524**.


## Learning-rate selection (TRAIN-internal holdout only)

20% of TRAIN families (43 of them) held out; DEV was not read.

| lr | holdout AUC | holdout PR-AUC |
|---:|---:|---:|
| 0.3000 | 0.9540 | 0.7474 |
| 0.1000 | 0.9741 | 0.8793 |
| 0.0600 | 0.9722 | 0.8753 |
| 0.0300 | 0.9659 | 0.8335 |
| 0.0100 | 0.9516 | 0.7390 |
| 0.0030 | 0.9450 | 0.7021 |
| 0.0010 | 0.9432 | 0.6949 |

Selected lr = 0.1000.


## Training curve (final fit, all TRAIN families)

| epoch | lr | train weighted loss | clip rate | DEV AUC | DEV PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.10000 | 2.010421 | 0.148 | 0.9415 | 0.6859 |
| 1 | 0.09524 | 1.648636 | 0.144 |  |  |
| 2 | 0.09091 | 1.617224 | 0.141 |  |  |
| 3 | 0.08696 | 1.576371 | 0.141 |  |  |
| 4 | 0.08333 | 1.464345 | 0.137 |  |  |
| 5 | 0.08000 | 1.430627 | 0.132 | 0.9512 | 0.7456 |
| 6 | 0.07692 | 1.360565 | 0.126 |  |  |
| 7 | 0.07407 | 1.336429 | 0.125 |  |  |
| 8 | 0.07143 | 1.331786 | 0.121 |  |  |
| 9 | 0.06897 | 1.299681 | 0.117 |  |  |
| 10 | 0.06667 | 1.311504 | 0.112 | 0.9547 | 0.8122 |
| 11 | 0.06452 | 1.294549 | 0.109 |  |  |
| 12 | 0.06250 | 1.287032 | 0.109 |  |  |
| 13 | 0.06061 | 1.253538 | 0.110 |  |  |
| 14 | 0.05882 | 1.305224 | 0.106 |  |  |
| 15 | 0.05714 | 1.256102 | 0.106 | 0.9578 | 0.8009 |
| 16 | 0.05556 | 1.286088 | 0.110 |  |  |
| 17 | 0.05405 | 1.225805 | 0.108 |  |  |
| 18 | 0.05263 | 1.289844 | 0.116 |  |  |
| 19 | 0.05128 | 1.269650 | 0.116 |  |  |
| 20 | 0.05000 | 1.240658 | 0.110 | 0.9484 | 0.7279 |
| 21 | 0.04878 | 1.264933 | 0.117 |  |  |
| 22 | 0.04762 | 1.180907 | 0.105 |  |  |
| 23 | 0.04651 | 1.167140 | 0.106 |  |  |
| 24 | 0.04545 | 1.174314 | 0.105 |  |  |
| 25 | 0.04444 | 1.193440 | 0.108 | 0.9651 | 0.8351 |
| 26 | 0.04348 | 1.146698 | 0.105 |  |  |
| 27 | 0.04255 | 1.104676 | 0.106 |  |  |
| 28 | 0.04167 | 1.194646 | 0.111 |  |  |
| 29 | 0.04082 | 1.346885 | 0.135 | 0.9558 | 0.7624 |

## Rule encoder

The rule embedding is trained through `encode_rule`'s `[LHS | RHS | LHS-RHS | LHS*RHS]` projection (registration §4), over depth-bound poolings of each rule's own templates. Rank of the concatenation matrix: **60 of 62 rules**. A rank equal to the rule count means the projection can realize any per-rule embedding assignment — i.e. this arm is not handicapped against the additive arm's free per-rule scalar.

**Recorded limitation.** 1 rule pair(s) encode identically and therefore score identically in every context, at any weights: [("differentiate", "constant-fold")]. All of them are rules that define no template on either side (["constant-fold", "differentiate"]), so the registered encoding has nothing to encode for them.


Write-side train/deploy skew: max |trainer-in-memory - deployed-from-file| = 0.000e0 over 5000 DEV records (bar 1e-6).


Trained rule-by-context second difference (trig-shaped vs polynomial-shaped neighborhood, two rules): **2.680984**. The additive arm pins this at exactly zero by construction; a value of zero here would mean training collapsed this head into that model class.


## Per-rule: learned priority (DEV mean predicted score) vs measured strict-bound rate

| rule | id | train fired | train rate | DEV fired | DEV measured rate | DEV mean predicted P |
|---|---|---:|---:|---:|---:|---:|
| reverse-associative(Add) | 12f33cd6663ac71f | 1058 | 0.00662 | 258 | 0.01163 | 0.00019 |
| recip-sqrt | 15ac79c3047ad4d1 | 120 | 0.88333 | 35 | 0.85714 | 0.99986 |
| even-negation(Cos) | 167b9119efe00893 | 277 | 0.42599 | 79 | 0.48101 | 0.42071 |
| power-sqrt | 2bfc49afae251983 | 542 | 0.75461 | 163 | 0.71779 | 0.99668 |
| commutative(Mul) | 3d28c10dff31a2de | 2337 | 0.00000 | 809 | 0.00000 | 0.00018 |
| reverse-associative(Mul) | 42de0d719da182d0 | 598 | 0.03010 | 251 | 0.07171 | 0.00121 |
| factor | 4545eb999aeeeca6 | 677 | 0.05908 | 223 | 0.08520 | 0.00520 |
| power-identity | 479755b99bddb77f | 31 | 0.00000 | 10 | 0.00000 | 0.00269 |
| sin-angle-addition | 567b8b751d23d432 | 15 | 0.00000 | 3 | 0.00000 | 0.00049 |
| involution(Sub) | 573686fe9a77ffad | 3094 | 0.00000 | 1000 | 0.00000 | 0.00006 |
| halving | 5791376f6f4b4304 | 342 | 0.00292 | 74 | 0.00000 | 0.00155 |
| associative(Add) | 587d0085e2775c26 | 1058 | 0.00851 | 258 | 0.00388 | 0.00017 |
| commutative(Min) | 5d63e90e10c3b1a4 | 236 | 0.00000 | 77 | 0.00000 | 0.00004 |
| associative(Mul) | 68244f671e6f939d | 598 | 0.01672 | 251 | 0.01594 | 0.00107 |
| fma-fusion | 6acd3c44e21839ea | 2016 | 0.14137 | 535 | 0.13271 | 0.15202 |
| power-combine | 7e1e121b3dacb3ce | 117 | 0.33333 | 36 | 0.38889 | 0.07096 |
| exp2-log2-cancel | 9086fe19df53a842 | 7 | 0.00000 | 2 | 0.00000 | 0.00641 |
| power-recip | 92e3367a6f9e5ccc | 442 | 0.81674 | 136 | 0.80147 | 0.97641 |
| log2-exp2-cancel | 940721a5e42cc290 | 7 | 0.00000 | 2 | 0.00000 | 0.02858 |
| reverse-associative(Max) | 95add9713b8cd888 | 30 | 0.00000 | 26 | 0.07692 | 0.00401 |
| reverse-angle-addition | 9f5cb6c9195cb5fa | 15 | 0.06667 | 3 | 0.33333 | 0.01082 |
| even-negation(Abs) | a1d188080320a7b2 | 1986 | 0.42900 | 599 | 0.38898 | 0.34370 |
| commutative(Max) | a594cd0e3a796976 | 238 | 0.00000 | 88 | 0.00000 | 0.00089 |
| canonicalize(Sub) | a60c541aa296142e | 49 | 0.10204 | 13 | 0.15385 | 0.00155 |
| power-zero | a66b8afce49f3b9f | 1 | 1.00000 | 0 | 0.00000 | 0.00000 |
| exp-homomorphism | b1d12260b01d5163 | 12 | 0.00000 | 4 | 0.00000 | 0.01771 |
| odd-negation(Tan) | b1ffd2d990da5384 | 47 | 0.02128 | 11 | 0.09091 | 0.00206 |
| idempotent(Min) | bac1aad26a86722d | 13 | 0.00000 | 3 | 0.00000 | 0.00000 |
| associative(Min) | bb313766bc8ec707 | 32 | 0.00000 | 5 | 0.00000 | 0.00009 |
| inverse-annihilation(Sub) | be54d2411e017373 | 8 | 0.75000 | 0 | 0.00000 | 0.00000 |
| doubling | c3e67432f2a1b5e3 | 342 | 0.05556 | 74 | 0.05405 | 0.00269 |
| distribute | c55b6b8fee188f1e | 678 | 0.00000 | 223 | 0.00000 | 0.00018 |
| odd-negation(Sin) | c6c412afffd3c46b | 86 | 0.04651 | 26 | 0.03846 | 0.00245 |
| commutative(Add) | ce143fa8dbaa336d | 4325 | 0.00000 | 1185 | 0.00000 | 0.00014 |
| canonicalize(Div) | d434c897945f5659 | 20 | 0.00000 | 8 | 0.00000 | 0.00005 |
| exp-ln-cancel | d6ecb2b4ffb49d92 | 7 | 0.00000 | 1 | 0.00000 | 0.00296 |
| constant-fold | d92019505a8a3247 | 997 | 0.40120 | 289 | 0.36678 | 0.34828 |
| reverse-associative(Min) | da272d71621a1b82 | 32 | 0.00000 | 5 | 0.00000 | 0.00009 |
| power-rsqrt | e32395e542bd0eeb | 131 | 0.92366 | 36 | 0.97222 | 0.99974 |
| ln-exp-cancel | f1ebf9bf7883292e | 7 | 0.00000 | 1 | 0.00000 | 0.00009 |
| cos-angle-addition | fd66534f6ee3abc1 | 27 | 0.03704 | 5 | 0.00000 | 0.00088 |
| idempotent(Max) | ff3aced291136397 | 12 | 0.00000 | 5 | 0.00000 | 0.00000 |
| associative(Max) | ff9d0366e310988d | 30 | 0.06667 | 26 | 0.00000 | 0.00386 |

Checkpoint written to `pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json`.
