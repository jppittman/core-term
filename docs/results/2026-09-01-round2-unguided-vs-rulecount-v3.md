## SS3 - the |R| effect, order held fixed

`Delta_U(p) = U(p) - U(OrderMatchedBase(seed, |p|))`, seed = `0x20260901` (DEFAULT_INTERLEAVE_SEED). Classical band (n=188).

**Mode (i)**

| rule set | \|R\| | matched-base ref | U(p)@100 | U(matched)@100 | Delta_U@100 | differing@100 | U(p)@200 | U(matched)@200 | Delta_U@200 | differing@200 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dup:93` | 93 | `base:matched:0x20260901:93` | 43.76 | 60.55 | -16.79 | 125 | 6.69 | 24.82 | -18.13 | 148 |
| `dup:124` | 124 | `base:matched:0x20260901:124` | 41.12 | 43.88 | -2.76 | 161 | 25.44 | 16.89 | 8.55 | 150 |
| `dup:186` | 186 | `base:matched:0x20260901:186` | 15.44 | 13.85 | 1.59 | 158 | 9.47 | 4.70 | 4.76 | 171 |
| `dup:248` | 248 | `base:matched:0x20260901:248` | 38.67 | 4.33 | 34.34 | 181 | 27.02 | 3.36 | 23.66 | 162 |

Spearman rho(Delta_U, |R|): B=100 = 1.000, B=200 = 0.800. Delta_U at max |R|: 34.34% (B=100), 23.66% (B=200).

Delta1(v3) at `dup:93` (95% bootstrap CI half-width of paired median Delta_U): B=100 = 0.05 pts (median 0.00, CI [-0.09, 0.00]), B=200 = 4.82 pts (median -4.21).

**H1(v3) verdict (i):** direction HOLDS (rho >= 0.9), effect @100 HOLDS (|Delta_U(max)| >= Delta1), effect @200 HOLDS.

**Mode (ii)**

| rule set | \|R\| | matched-base ref | U(p)@100 | U(matched)@100 | Delta_U@100 | differing@100 | U(p)@200 | U(matched)@200 | Delta_U@200 | differing@200 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `comp:93` | 93 | `base:matched:0x20260901:93` | 60.55 | 60.55 | 0.00 | 5 | 25.60 | 24.82 | 0.78 | 5 |
| `comp:124` | 124 | `base:matched:0x20260901:124` | 41.69 | 43.88 | -2.19 | 17 | 21.22 | 16.89 | 4.33 | 33 |

Spearman rho(Delta_U, |R|): B=100 = -1.000, B=200 = 1.000. Delta_U at max |R|: -2.19% (B=100), 4.33% (B=200).

Delta1(v3) at `comp:93` (95% bootstrap CI half-width of paired median Delta_U): B=100 = 0.00 pts (median 0.00, CI [0.00, 0.00]), B=200 = 0.00 pts (median 0.00).

**H1(v3) verdict (ii):** direction FAILS (rho >= 0.9), effect @100 FAILS (|Delta_U(max)| >= Delta1), effect @200 HOLDS.

**Mode (iii)**

| rule set | \|R\| | matched-base ref | U(p)@100 | U(matched)@100 | Delta_U@100 | differing@100 | U(p)@200 | U(matched)@200 | Delta_U@200 | differing@200 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `new:95` | 95 | `base:matched:0x20260901:95` | 33.11 | 48.25 | -15.14 | 186 | 52.06 | 27.52 | 24.54 | 184 |

Spearman rho(Delta_U, |R|): B=100 = undefined, B=200 = undefined. Delta_U at max |R|: -15.14% (B=100), 24.54% (B=200).

Delta1(v3) at `new:95` (95% bootstrap CI half-width of paired median Delta_U): B=100 = 11.45 pts (median -0.18, CI [-21.22, 1.67]), B=200 = 16.74 pts (median 6.12).

**H1(v3) verdict (iii):** direction FAILS (rho >= 0.9), effect @100 FAILS (|Delta_U(max)| >= Delta1), effect @200 HOLDS.

## SS4 - the order effect on its own (all rule sets |R|=62)

**classical**

| rule set | U@100 | L@100 | differing-from-base@100 | U@200 | L@200 | differing-from-base@200 |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 96.58 | 48.467 | 0 | 40.49 | 21.922 | 0 |
| `base:shuffled:1` | 43.74 | 13.672 | 175 | 23.57 | 14.957 | 150 |
| `base:shuffled:2` | 46.19 | 18.585 | 186 | 25.70 | 7.934 | 143 |
| `base:shuffled:3` | 26.28 | 13.448 | 186 | 1.49 | 0.499 | 151 |
| `base:static:numeric-first` | 1.12 | 0.599 | 186 | 0.44 | 0.002 | 140 |

**rapid (reported, no claim)**

| rule set | U@100 | L@100 | differing-from-base@100 | U@200 | L@200 | differing-from-base@200 |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:1` | 0.00 | 0.000 | 22 | 0.00 | 0.000 | 7 |
| `base:shuffled:2` | 0.00 | 0.000 | 20 | 0.00 | 0.000 | 5 |
| `base:shuffled:3` | 0.00 | 0.000 | 28 | 0.00 | 0.000 | 6 |
| `base:static:numeric-first` | 0.00 | 0.000 | 23 | 0.00 | 0.000 | 3 |

**blitz (reported, no claim)**

| rule set | U@100 | L@100 | differing-from-base@100 | U@200 | L@200 | differing-from-base@200 |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:1` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:2` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:3` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:static:numeric-first` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |

## Seed sensitivity of an inflated point

Registered seed (`0x20260901`) plus two additional interleave seeds (`1`, `2`), classical band.

**dup:124**

| seed variant | U@100 | U@200 |
|---|---:|---:|
| `dup:124` | 41.12 | 25.44 |
| `dup:124:interleave:1` | 37.80 | 32.37 |
| `dup:124:interleave:2` | 50.33 | 22.62 |

Spread (range) across the 3 seeds: B=100 = 12.53 pts, B=200 = 9.76 pts.

**comp:93**

| seed variant | U@100 | U@200 |
|---|---:|---:|
| `comp:93` | 60.55 | 25.60 |
| `comp:93:interleave:1` | 39.95 | 18.77 |
| `comp:93:interleave:2` | 38.70 | 13.07 |

Spread (range) across the 3 seeds: B=100 = 21.85 pts, B=200 = 12.53 pts.

## Summary

**The |R| effect, order held fixed, is small and — where it is measurable at all — negative,
not positive; the order effect dwarfs it.** Mode (i) (pure duplicates, closure identical to
`base` by construction) is the only mode with enough grid points (4) to read a trend: `Delta_U`
grows monotonically and strongly with `|R|` (Spearman rho = 1.000 at B=100, 0.800 at B=200,
clearing Delta1(v3) by a wide margin at both budgets), but the sign is *increasing regret* —
`Delta_U` rises from -16.8% at `dup:93` to +34.3% at `dup:248` (B=100). Since mode (i) cannot
change what the search can reach (`DuplicateRule` delegates verbatim to its inner rule), this
is not a capacity effect: it is budget dilution — a fixed application-count budget spends a
shrinking fraction of its applications on the productive 62 rules as redundant duplicate slots
are added to every sweep, holding those 62 rules' relative sweep position fixed. Modes (ii)
(2 points: `comp:93`, `comp:124`) and (iii) (1 point: `new:95`) don't have enough grid points to
read a trend at all — mode (ii)'s two-point Spearman rho is definitionally +-1 regardless of
effect size, and its `Delta_U` values are small (-2.2% to +4.3%) and inconsistent in sign
between B=100 and B=200; mode (iii) has one point, so rho is undefined by construction.

**The order effect, measured on the same 62 rules with no rule added or removed, is 2-3x
larger than mode (i)'s |R| effect at its biggest, and runs the opposite direction.** `base`
(production order) sits at U=96.6%/40.5% (B=100/200) — worse than every other order tested.
The three independently-shuffled seeds land at U=26-46% (B=100) / 1.5-25.7% (B=200), a 50-70
point improvement over production order from nothing but a different permutation of the
identical 62 rules. `StaticReorder(NumericFirst)` — the production quick-win candidate, rules
sorted by descending TRAIN strict-positive rate — does far better still: U=1.1%/0.4%, a ~95
and ~40 point drop from `base` at B=100/200 respectively, and better than every random shuffle
tested too. Seed sensitivity on an inflated point (`dup:124`, `comp:93` under the registered
seed plus two more) shows a real but bounded spread — 9.8-21.9 points across three seeds — so
individual `Delta_U` values in SS3 (anchored to one seed) carry that much seed noise, but every
seed tested is far below `base`'s unshuffled value, so the order effect's *existence and
direction* is not seed-fragile even though its exact magnitude at a given point is.

**Reading.** With order held fixed, the |R| effect is not the driver of v2's raw `U(|R|)`
finding — v2 §6b's confound argument holds: order, not rule count, is what moved `U`. Where
`|R|` has a measurable effect at all (mode i only), it is small relative to the order effect
and points the wrong way for a "more rules helps" story — it is a dilution cost, not a capacity
gain. The order effect is the real, large, and reproducible-across-seeds finding here, and
`StaticReorder(NumericFirst)` — the zero-runtime-cost production quick-win the orchestrator
flagged — is the single best order measured, beating even random shuffles.
