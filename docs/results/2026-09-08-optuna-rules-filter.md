# Does the rules × nodes filter's null survive tuning?

**Question.** `docs/results/2026-09-08-rules-filter-bilinear.md` reports the bilinear
rules × nodes filter as NULL on every family against its registered decision rule. That
model was trained at one hand-picked configuration. JP, 2026-09-08:

> I'm a fool. You know when I was running this, none of my models worked at all until I ran
> optuna. We haven't run optuna once since I stopped following this closely.

He is right about the history. The previous sweep harness — `optuna_unified.py` driving
`train_unified --server <unix socket>` — was deleted with the RL loop in `fb8136e3`, and
nothing in this tree has swept since. Every learned result reported after that date (the
linear and bilinear Guides, return-to-go, and this filter) was trained at a single
configuration. **A null from an untuned model cannot separate "this shape does not carry the
signal" from "this learning rate does not."**

So: one question, answered one way or the other. Does the filter's registered null survive a
hyperparameter search?

**Answer.** **It survives, unmoved.** A 250-trial TPE study improves the intrinsic metric by +2.9 %
(mean held-out PR-AUC lift 1.7510 → 1.8021) and finds the registered configuration to rank
**204th of 250** — the median random-ish trial beats it. The extrinsic result does not move at
all: at the registered budget, keep-rate and model, the median `dag_cost` ratio against
`Identity` is **identical to three decimals on every DEV family** (glyph16 1.000, bench 1.000,
bench_wide 1.000, psychedelic 1.042, cellgrid 1.049, shader 1.054), none within the registered
≤ 0.95, and the held-out sign still never turns below 1. **The null is now a tuned null, which
makes it a stronger null.**

Nothing about the registered extrinsic protocol
(`docs/plans/2026-09-08-rules-filter-bilinear-registration.md`) is changed here. It is opened
once, at the end, at the configuration the study picked.

---

## 1. The harness, and why it is a server

A trial is a full family-held-out training run: three held-out folds plus the all-DEV model
over 640,401 labelled applications. Measured on this box (aarch64, release, shared, load
stated beside every timing):

| | wall |
|---|---:|
| cold `train` process, all four models, uncapped | 128 s (`3:22` for the registered run at load 7–21) |
| one trial over the socket, uncapped | 125.1 s (load ≈ 5) |
| one trial over the socket, `--train-cap 100000` | 25.6 s (load ≈ 5) |
| median trial during the study | 11.7 s (load 8–87, gates running beside it) |

The corpus load is ≈ 3 s of the 128 s, so a subprocess per trial would have been *nearly* fine
by the measurement rule — but the training itself is the cost, and the only lever on it is
fidelity. `rules_filter serve` loads the corpus once and trains one configuration per
connection (a JSON line in, the manifest as a JSON line out); the study then runs at reduced
fidelity (`--train-cap 100000`, a deterministic stride over each fold's training set; the
held-out fold is never capped) and **refits the top trials uncapped**, so the winner is chosen
at full fidelity and the reduced-fidelity ranking is never the last word.

Everything the study varies had to be made nameable first. Six of the eleven knobs were
compile-time constants or did not exist; they are now CLI arguments of `train`, defaulting to
the registered values, so **trial 0 is the registered run**. That is not a claim, it is
checked: trial 0 refitted uncapped reproduces the registered intrinsic numbers exactly
(glyph PR-AUC 0.8388 / AUC 0.8794, shader 0.1915 / 0.6628, scene 0.2528 / 0.6253), and
retraining the registered configuration with the refactored binary reproduces all four
checkpoints byte-for-byte (`weights_fnv64` 04e89165a8be108b / 72b1538b0fc38900 /
abe1a700308215c7 / 5144fabe6ca73547) along with every threshold and rule rate. **The untuned
extrinsic column below is therefore the registered run itself, not a re-measurement of it.**

## 2. The objective, registered before the study ran

The intrinsic metric is PR-AUC on TIGHT labels, family-held-out — the metric the registration
already names. It is aggregated over the three folds as **mean PR-AUC *lift***,
`PR-AUC / positive_rate`, not as a plain mean.

The reason is that the folds' base rates differ by 3.5× (glyph 0.406, scene 0.166, shader
0.115), and PR-AUC's floor is the base rate. A plain mean of PR-AUC is those three folds
weighted by how easy they are, and maximising it is mostly maximising glyph — whose untuned
PR-AUC (0.8388) is already twice its floor while shader's (0.1915) is 1.67× its own. Lift is
the same metric made comparable across folds. Both are recorded on every trial; only lift is
optimised.

Not swept: the seed (fixed at 17 — sweeping it measures noise, and its variance is a separate
question this study does not answer), and the keep-threshold rule (a quantile at ρ). The
threshold rule cannot be swept on this objective at all — PR-AUC is threshold-free, so the
intrinsic metric is blind to it. That is a real gap and it is named here rather than papered
over: if the extrinsic null is a thresholding artefact, this study cannot see it.

**Study:** TPE, seed 17, 250 trials in 57 min of trial time (3,425 s), plus 6 uncapped refits, trial 0 enqueued as the registered
configuration, persisted to sqlite. Top 5 refitted uncapped, and trial 0 refitted uncapped
whether or not it ranked.

## 3. What the sweep found

### 3.1 The registered configuration is in the bottom fifth of its own search space

| | value |
|---|---:|
| trials, all complete | 250 |
| trial 0 (registered), reduced fidelity | 1.7943 |
| **rank of trial 0 among the 250** | **204th** |
| median trial | 1.8439 |
| best trial (t203), reduced fidelity | 1.8953 |
| worst trial | 1.1218 |

The median trial beats the registered configuration. That is the finding this study was run
to obtain, and it is the one that generalises past this filter: **the configuration every
learned result in this tree was trained at is not a neutral default — it is a below-median
draw**, and any null reported at it was reported at a handicap.

### 3.2 The winner, chosen at full fidelity

Reduced fidelity ranks, it does not decide: the best capped trial (t203) is only third once
refitted uncapped. Refits, ordered by the full-fidelity objective:

| trial | capped | **full** | full mean PR-AUC |
|---|---:|---:|---:|
| **183** | 1.8945 | **1.8021** | 0.4340 |
| 17 | 1.8917 | 1.7816 | 0.4230 |
| 203 | 1.8953 | 1.7801 | 0.4339 |
| 133 | 1.8921 | 1.7745 | 0.4265 |
| **0 (registered)** | 1.7943 | **1.7510** | 0.4277 |
| 202 | 1.8906 | 1.7426 | 0.4281 |

The winner is **trial 183**: `--epochs 1 --lr 0.0740 --lr-decay 0.7680 --l2 7.87e-5
--max-grad-norm 0.4866 --label tight --pos-weight-power 0.8529 --batch-size 1 --neg-keep 0.8202
--init-scale 1.6275 --relu-warm-bias 0.3626 --seed 17`.

Per fold, held out from its own family:

| fold | metric | trial 0 (registered) | trial 183 (tuned) |
|---|---|---:|---:|
| glyph | PR-AUC (tight) | 0.8388 | 0.8378 |
| | AUC (tight) | 0.8794 | 0.8772 |
| shader | PR-AUC (tight) | 0.1915 | **0.2049** |
| | AUC (tight) | 0.6628 | **0.6752** |
| scene | PR-AUC (tight) | 0.2528 | **0.2595** |
| | AUC (tight) | 0.6253 | **0.6639** |
| | **mean lift** | 1.7510 | **1.8021** |
| | mean PR-AUC | 0.4277 | 0.4340 |

All of the gain is on shader and scene — the two folds whose PR-AUC was near its floor. Glyph,
already at 2.07× its base rate, moves −0.001. The per-rule prior is unchanged by construction
(0.5983 / 0.1429 / 0.1958), so the head's margin over it widens on two folds of three.

### 3.3 What the study actually latched onto

Parameter importances (fANOVA over the 250 trials):

| parameter | importance |
|---|---:|
| `neg_keep` | 0.218 |
| `init_scale` | 0.140 |
| `pos_weight_power` | 0.120 |
| `l2` | 0.117 |
| `max_grad_norm` | 0.100 |
| `relu_warm_bias` | 0.092 |
| `lr` | 0.083 |
| `lr_decay` | 0.078 |
| `epochs` | 0.026 |
| `batch_size` | 0.024 |
| `label` | 0.003 |

The three that matter most are the three that were not adjustable before this run:
**negative subsampling, initialisation scale, and how hard the class imbalance is corrected.**
`neg_keep` alone carries more than learning rate, decay and epochs together — and the top ten
trials all sit at `neg_keep` ≈ 0.75–0.84, i.e. *dropping* a fifth of the negatives helps, which
the previous harness could not have discovered because it had no way to express it. Likewise
every top-ten trial runs `epochs = 1` against the registered 3, and `init_scale` ≈ 1.1–1.9
against the registered 1.0. `label` is inert (0.003): training on strict labels instead of
tight neither helps nor hurts the tight metric, which is consistent with 92.2 % of tight
positives being invisible to strict.

Top ten trials, at reduced fidelity:

| trial | lift | mean PR-AUC | epochs | lr | l2 | pos_w_pow | neg_keep | init_scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 203 | 1.8953 | 0.4465 | 1 | 0.0857 | 7.2e-5 | 0.761 | 0.802 | 1.538 |
| 183 | 1.8945 | 0.4449 | 1 | 0.0740 | 7.9e-5 | 0.853 | 0.820 | 1.628 |
| 133 | 1.8921 | 0.4422 | 1 | 0.0557 | 6.4e-5 | 0.846 | 0.814 | 1.765 |
| 17 | 1.8917 | 0.4379 | 1 | 0.0090 | 7.0e-5 | 0.543 | 0.813 | 1.912 |
| 202 | 1.8906 | 0.4454 | 1 | 0.0825 | 2.0e-5 | 0.800 | 0.840 | 1.261 |
| 211 | 1.8890 | 0.4447 | 1 | 0.1069 | 5.1e-5 | 0.780 | 0.784 | 1.304 |
| 177 | 1.8888 | 0.4464 | 1 | 0.0928 | 3.3e-5 | 0.807 | 0.812 | 1.436 |
| 207 | 1.8886 | 0.4450 | 1 | 0.0616 | 3.6e-5 | 0.841 | 0.823 | 1.181 |
| 222 | 1.8864 | 0.4459 | 1 | 0.0882 | 4.7e-5 | 0.732 | 0.790 | 1.140 |
| 184 | 1.8859 | 0.4453 | 1 | 0.0767 | 1.4e-4 | 0.769 | 0.792 | 1.411 |

Every one of them is `batch_size = 1`, `label = tight`, `epochs = 1`. The study converged on a
region, not a lucky point.

## 4. The extrinsic re-eval, protocol unchanged

The registered decision rule, verbatim from the registration: **median `dag_cost` ratio ≤ 0.95
at B/2 with bytes ≤ 1.00 and the held-out sign agreeing**, ρ = 0.25 primary, family-held-out
model, against `Identity` at the same budget, with `PerRuleRate` and `UniformRandom` at the
bilinear arm's own realized keep-rate as controls.

### 4.1 DEV, at B/2, ρ = 0.25, family-held-out model — the registered cell

Ratios are per kernel against `Identity` at the same budget; median (p10 / p90) over the
family. glyph32 is byte-identical to glyph16 here and is in the JSON, not repeated.

| family | n | Identity | PerRuleRate | Uniform@matched (untuned / tuned) | Bilinear untuned | Bilinear tuned | bytes u / t | keep u / t |
|---|---:|---:|---:|---|---:|---:|---|---|
| glyph16 | 94 | 1.000 | 0.982 | 1.024 / 1.003 | **1.000** (0.851/1.144) | **1.000** (0.743/1.124) | 1.000 / 1.000 | 0.315 / 0.388 |
| bench | 3 | 1.000 | 0.974 | 1.000 / 1.000 | **1.000** (0.622/1.000) | **1.000** (0.663/1.000) | 1.000 / 1.000 | 0.323 / 0.403 |
| bench_wide | 1 | 1.000 | 0.974 | 0.999 / 0.999 | **1.000** (1.000/1.000) | **1.000** (1.000/1.000) | 1.000 / 1.000 | 0.992 / 0.992 |
| shader | 12 | 1.000 | 1.091 | 1.054 / 1.054 | **1.054** (0.959/1.244) | **1.054** (0.959/1.244) | 0.990 / 0.990 | 0.000 / 0.023 |
| psychedelic | 1 | 1.000 | 1.042 | 0.992 / 1.005 | **1.042** (1.042/1.042) | **1.042** (1.042/1.042) | 0.919 / 0.919 | 0.300 / 0.483 |
| cellgrid | 1 | 1.000 | 1.049 | 1.044 / 1.026 | **1.049** (1.049/1.049) | **1.049** (1.049/1.049) | 0.975 / 0.975 | 0.104 / 0.241 |

**Every median is identical to three decimals.** The +2.9 % intrinsic gain moved nothing. What
did change is the *keep-rate*: the tuned head's score distribution is shaped differently, so the
same ρ-quantile threshold keeps 0.315 → 0.388 of glyph cells and 0.104 → 0.241 of cell-grid
cells. It keeps more, it keeps different cells, and the extraction that comes out costs the
same.

The shader row is worth naming on its own: the untuned filter's realized keep-rate on shaders
is **0.000** — a threshold calibrated on a glyph-dominated training set keeps essentially
nothing on a family it never saw, so "the bilinear arm" there was `KeepNothing` wearing a
model's name, and its 1.054 is the cost of firing nothing. Tuning lifts that to 0.023, still
nothing. This is the threshold-transfer failure §2 flagged as unsweepable on a threshold-free
intrinsic metric, now visible in the extrinsic table.

### 4.2 The secondary keep-rate, ρ = 0.5

| family | untuned | tuned |
|---|---:|---:|
| glyph16 | 1.000 | 1.000 |
| bench | 1.000 | 1.000 |
| bench_wide | 1.000 | 1.000 |
| shader | 0.999 | 1.011 |
| psychedelic | 1.029 | **1.000** |
| cellgrid | 1.023 | 1.023 |

Best family median 0.999 untuned, 1.000 tuned. Null at ρ = 0.5 too, both ways.

### 4.3 HELD-OUT, at B/2, ρ = 0.25, all-DEV model

The held-out set was opened once for the registered run and is re-read here at the tuned
configuration; by §B.2 of the benchmark correction it was already promoted to DEV by that first
opening, so these rows are reported as confirmation, not as a fresh held-out test.

| family | n | untuned | tuned | bytes u / t | keep u / t |
|---|---:|---:|---:|---|---|
| bold_glyph16 | 93 | 1.000 | 1.016 | 1.043 / 1.040 | 0.527 / 0.529 |
| bold_bench | 3 | 1.000 | 1.016 | 1.000 / 0.993 | 0.363 / 0.558 |
| bold_bench_wide | 1 | 1.000 | 1.013 | 1.000 / 0.991 | 1.000 / 0.949 |
| chrome | 1 | 1.156 | 1.106 | 0.995 / 1.005 | 0.618 / 0.688 |
| chrome_channel | 1 | 1.178 | 1.115 | 1.000 / 1.006 | 0.641 / 0.574 |

**The sign never turns below 1.** Tuning helps chrome (1.156 → 1.106) and hurts the bold
glyphs (1.000 → 1.016); at ρ = 0.5 it is the other way round (chrome 1.005 → 1.049, glyphs
1.000 → 1.000). Neither direction reaches the rule, and the one arm that dips below 1 anywhere
in the grid — chrome at B/4, ρ = 0.25, tuned 0.959 — is beaten there by `UniformRandom` at its
own matched keep-rate (0.949). A control that wins is not a model that works.

## 5. Verdict

**The registered null survives tuning, and is stronger for it.**

- **Intrinsic: the sweep works.** +2.9 % mean held-out PR-AUC lift, concentrated on the two
  folds that had the most room. The registered configuration ranks 204th of 250 — below the
  median trial. Tuning was not a formality here; it found real signal the hand-picked
  configuration was leaving on the floor.
- **Extrinsic: nothing moves.** Median `dag_cost` ratio identical to three decimals on all six
  DEV families at the registered cell. No family within ≤ 0.95 at either keep-rate. The
  held-out sign never turns below 1. Every place a filtered arm beats `Identity`, a control at
  its own keep-rate beats it too.
- **Therefore the registration's own explanation stands, and is now the only one left.** The
  registered result already showed why (§7 there): on these kernels `Identity` reaches its own
  final `dag_cost` at B/8 on 40 of 94 glyphs, so *any* filter that fires fewer applications
  lands at ratio ≈ 1.00 and *which* cells it fires barely moves the median. A better-ranked
  filter cannot fix a corpus where ranking is not the binding constraint. Before this study one
  could still answer "the model was undertrained"; that answer is now closed.
- **What is not closed:** the keep-threshold rule. PR-AUC is threshold-free, so the intrinsic
  objective is blind to it, and §4.1 shows a threshold that transfers so badly across families
  that the untuned shader arm kept 0.000 of its cells. If the filter shape is ever revisited,
  **the threshold rule is the next thing to make sweepable, and it needs an objective that can
  see it** — a keep-rate-matched precision at the realized operating point, not PR-AUC.

The generalisable finding is the one in §3.1, and it is not about this filter: the
configuration this tree's learned models are trained at is a below-median draw from its own
search space. Every null reported at it was reported at a handicap. That is what §7's rule is
for.

## 6. Reproducing it

```bash
# the samples are deterministic; mint them if they are gone
rules_filter mint --out samples.jsonl

# terminal 1
rules_filter serve --samples samples.jsonl --socket /tmp/rf.sock   # short path: SUN_LEN

# terminal 2 (optuna in a venv, never in the workspace)
pixelflow-pipeline/scripts/optuna_rules_filter.py \
  --socket /tmp/rf.sock --study sqlite:///rf_study.db \
  --trials 250 --timeout 7200 --train-cap 100000 --refit 5 --out sweep_summary.json

# the winner, at full fidelity, then the registered extrinsic protocol
rules_filter train --samples samples.jsonl --out models_tuned <best flags>
rules_filter eval  --models models_tuned --out dev_rows.jsonl            # PIXELFLOW_GUARD_TELEMETRY=1
rules_filter eval  --models models_tuned --out held_rows.jsonl --held-out
rules_filter report --models models_tuned --rows dev_rows.jsonl \
  --held-out-rows held_rows.jsonl --out-prefix docs/results/2026-09-08-optuna-rules-filter
```

The study database, the per-trial records and the row-level eval output are in the session
scratchpad; `2026-09-08-optuna-rules-filter.json` beside this file carries every number quoted
above.

## 7. The rule this establishes

Appended to `docs/results/2026-09-07-claims-ledger.md` §7 as item 11, listing the
learned-model rows it applies to: **no learned model's extrinsic number is quoted without an
Optuna sweep on the intrinsic metric first, with the untuned configuration enqueued as trial 0
so the tuned and untuned numbers land in the same table.** No verdict in the ledger changes;
what changes is what the evidence is taken to support.
