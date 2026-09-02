# Counterfactual credit: hindsight bounds vs measured Δ (leave-one-out and confluence-aware)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Sample: 40 `sh` + 60 DEV classical expressions, 3000 applications sampled (3000 at ordinal < B=100, seed 0x5eedc0dec0ffee01). wall_clock_ceiling_hit = false. git cf8814a8b4e83ca117bf69be2be44a3f6ddc449e.

Δ distribution: zero 2525/3000 (84.2%), positive 340/3000 (11.3%), negative 135/3000 (4.5%).

Proxies loaded: r2g = "pixelflow-pipeline/data/guide_checkpoint_r2g_regime_v1.json", strict-v1 = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json", per-rule = "docs/results/2026-09-01-train-guide-report.json". Bootstrap: 1000 paired resamples (seeded).

| proxy | Spearman (pooled) [95% CI] | Δρ vs r2g [95% CI] | Spearman (sh) | Spearman (dev) | Pearson (pooled) | n (excluded) |
|---|---:|---:|---:|---:|---:|---:|
| r2g_linear | 0.198709 [0.159648, 0.234269] | [0.000000, 0.000000] | -0.046326 | 0.268292 | 0.033769 | 3000 (0) |
| strict_v1_linear | 0.246271 [0.201663, 0.286599] | [0.020902, 0.075816] | 0.141893 | 0.282849 | 0.038538 | 3000 (0) |
| per_rule_rate | 0.288063 [0.244601, 0.326235] | [0.059467, 0.116787] | 0.179106 | 0.355071 | 0.050300 | 3000 (0) |
| loose | null [null, null] | [null, null] | null | null | null | 3000 (0) |
| tight | null [null, null] | [null, null] | null | null | null | 3000 (0) |
| strict | 0.509634 [0.457020, 0.553268] | [0.268372, 0.353322] | 0.424064 | 0.499612 | -0.001898 | 3000 (0) |
| strict_by_output_class | 0.010912 [-0.023474, 0.044257] | [-0.233892, -0.137558] | -0.005655 | 0.019600 | -0.031965 | 3000 (0) |

## Confluence-aware credit (multi-mask)

Second mask mode: the seed application AND every later application sharing its `(rule_idx, canonical matched-class content)` are skipped, so an alternative re-derivation cannot silently restore the node leave-one-out removed.

Multi-mask Δ distribution: zero 2415/3000 (80.5%), positive 431/3000 (14.4%), negative 154/3000 (5.1%).

**Confluence blindness of leave-one-out: 90 of the 2525 applications leave-one-out scored Δ = 0 become Δ > 0 under the multi-mask (3.6% of them)**; 20 become Δ < 0. Multi-masks that skipped more than the seed: 425/3000 (mean skips 1.16, max 6).

| proxy | Spearman vs multi-mask Δ (pooled) [95% CI] | Δρ vs r2g [95% CI] | Spearman (sh) | Spearman (dev) | Pearson (pooled) | n (excluded) |
|---|---:|---:|---:|---:|---:|---:|
| r2g_linear | 0.145032 [0.106952, 0.181949] | [0.000000, 0.000000] | -0.110687 | 0.217966 | 0.033862 | 3000 (0) |
| strict_v1_linear | 0.209399 [0.165794, 0.249866] | [0.037848, 0.091505] | 0.092233 | 0.238216 | 0.040559 | 3000 (0) |
| per_rule_rate | 0.201996 [0.155690, 0.242020] | [0.022273, 0.089753] | 0.122724 | 0.256274 | 0.050095 | 3000 (0) |
| loose | null [null, null] | [null, null] | null | null | null | 3000 (0) |
| tight | null [null, null] | [null, null] | null | null | null | 3000 (0) |
| strict | 0.564514 [0.520380, 0.604252] | [0.377990, 0.463847] | 0.573003 | 0.541952 | 0.003107 | 3000 (0) |
| strict_by_output_class | 0.031032 [-0.001347, 0.062993] | [-0.161816, -0.063273] | -0.006431 | 0.048256 | -0.031501 | 3000 (0) |

## Per-rule Δ over the sampled applications

| idx | rule | n (sh) | mean Δ | Δ>0 | Δ<0 | mean f_r2g | mean adv_r2g |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | canonicalize | 44 (28) | 0.008612 | 2 | 4 | -0.045208 | 0.021821 |
| 1 | involution | 135 (0) | -0.149992 | 61 | 16 | 0.006780 | -0.036312 |
| 4 | canonicalize | 20 (0) | 0.019471 | 1 | 2 | -0.100351 | 0.025909 |
| 8 | constant-fold | 153 (22) | 0.008093 | 89 | 6 | -0.057294 | 0.017005 |
| 9 | commutative | 814 (126) | 0.013145 | 19 | 40 | -0.018732 | -0.025101 |
| 10 | commutative | 619 (418) | 0.024698 | 10 | 22 | -0.010233 | -0.002926 |
| 11 | commutative | 44 (0) | -0.000287 | 0 | 4 | 0.007173 | -0.036580 |
| 12 | commutative | 41 (0) | 0.138590 | 1 | 6 | -0.010505 | -0.024763 |
| 16 | idempotent | 2 (0) | -2.916040 | 1 | 1 | -0.017086 | -0.008913 |
| 17 | idempotent | 3 (0) | 0.000909 | 2 | 1 | -0.043598 | 0.024564 |
| 18 | distribute | 58 (35) | 0.244311 | 3 | 5 | 0.037853 | -0.041246 |
| 19 | factor | 21 (0) | -0.000300 | 1 | 3 | -0.017706 | -0.001273 |
| 20 | doubling | 2 (0) | 2.908732 | 1 | 0 | -0.133683 | 0.097135 |
| 21 | halving | 17 (17) | 0.000000 | 0 | 0 | 0.129386 | -0.124850 |
| 22 | associative | 113 (76) | -0.000017 | 0 | 1 | 0.001005 | -0.000655 |
| 23 | associative | 322 (296) | -0.000201 | 0 | 6 | -0.033725 | 0.036124 |
| 24 | associative | 3 (0) | 0.000000 | 0 | 0 | 0.041011 | -0.055172 |
| 25 | associative | 1 (0) | 0.000000 | 0 | 0 | -0.020043 | -0.024833 |
| 26 | reverse-associative | 106 (64) | 0.058455 | 1 | 3 | 0.036219 | -0.027769 |
| 27 | reverse-associative | 138 (108) | -0.000025 | 0 | 1 | -0.025293 | 0.024928 |
| 28 | reverse-associative | 8 (0) | -0.000438 | 0 | 1 | 0.003488 | -0.024705 |
| 29 | reverse-associative | 2 (0) | 0.000000 | 0 | 0 | -0.051457 | 0.036269 |
| 30 | odd-negation | 6 (0) | -0.000934 | 0 | 2 | -0.063681 | 0.043177 |
| 31 | odd-negation | 2 (0) | 0.000000 | 0 | 0 | -0.078938 | 0.020090 |
| 34 | even-negation | 28 (0) | 0.000050 | 11 | 3 | -0.087123 | 0.061208 |
| 35 | even-negation | 152 (0) | 0.126895 | 69 | 6 | -0.074184 | 0.054640 |
| 36 | sin-angle-addition | 2 (2) | -0.006006 | 0 | 1 | -0.131187 | 0.131076 |
| 37 | cos-angle-addition | 3 (2) | -0.004004 | 0 | 1 | -0.044781 | 0.051213 |
| 39 | half-angle-product | 3 (3) | 0.000000 | 0 | 0 | -0.070064 | 0.073959 |
| 41 | exp-ln-cancel | 1 (0) | 0.138033 | 1 | 0 | -0.074168 | 0.041799 |
| 43 | exp2-log2-cancel | 1 (0) | 0.092136 | 1 | 0 | -0.059236 | 0.049706 |
| 47 | power-combine | 4 (0) | 0.000000 | 0 | 0 | -0.062404 | 0.052432 |
| 51 | power-sqrt | 29 (0) | 0.074220 | 29 | 0 | -0.086617 | 0.067879 |
| 52 | power-recip | 29 (0) | 0.067431 | 27 | 0 | -0.079442 | 0.062879 |
| 53 | power-rsqrt | 6 (0) | 0.107204 | 6 | 0 | -0.060851 | 0.045587 |
| 58 | diff-of-squares | 1 (1) | 0.000000 | 0 | 0 | -0.028003 | 0.029128 |
| 59 | fma-fusion | 64 (2) | 0.000029 | 1 | 0 | -0.065492 | 0.057313 |
| 60 | recip-sqrt | 3 (0) | 0.005277 | 3 | 0 | -0.049639 | 0.035908 |
