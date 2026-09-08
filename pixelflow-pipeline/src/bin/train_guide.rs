//! Cold-start supervised training for the saturation Guide, on strict labels
//! (`docs/plans/2026-08-31-guide-design-revision.md` §3 option 3 stage 1, §5
//! pre-registered experiment — this binary is the "supervised training
//! pipeline... proceed against strict labels immediately" half of that
//! section, NOT the accept/kill saturation-quality measurement itself).
//!
//! # Scope: cold-start classification, no saturation evaluation
//!
//! This binary trains a small linear classifier — a per-rule bias, a
//! bag-of-neighborhood-ops linear term, and a handful of scalar features —
//! against [`gen_strict_labels`](../gen_strict_labels.rs)'s
//! `strict_labels_train.jsonl` (TRAIN-family candidates only, by
//! construction of that file — see its own module doc's "Fence check"
//! section) to predict [`Label::LoadBearing`](pixelflow_search::egraph::Label)
//! (the strict bound: this application's output node is literally on the
//! extracted derivation path). It reports held-out (DEV-family) ranking
//! quality and writes a checkpoint. It does **not** wire the trained weights
//! into [`pixelflow_search::nnue::guide::Guide`] or run any guided
//! saturation — that is explicitly the next phase (§5's accept/kill
//! measurement), never bundled into this one per the design doc's own
//! phase-gate discipline.
//!
//! # Why a small linear model, not the full `SaturationHead` candidate tower
//!
//! `pixelflow_search::nnue::guide::scoring::SaturationHead`'s candidate-local
//! tower (`forward_candidate`/`compute_candidate_embed`/`score_candidate`,
//! wired 2026-09-01) is `pub(crate)` to `pixelflow-search` — deliberately: the
//! module doc for `nnue::guide` says the trait, `CandidateSummary`, and
//! `Guide`'s constructor are "the entire contract... every mask/rule-
//! projection/candidate-tower weight and the bilinear scorer are private
//! machinery behind it". Reaching into that machinery from
//! `pixelflow-pipeline` to backprop through it would mean widening that
//! crate's public surface before a caller (a Phase-3 wiring task) actually
//! needs to construct one — exactly the anti-row this codebase's cost-model
//! domain doc warns against ("build the enum, if at all, only when a
//! concrete Phase-3 caller needs it"). This binary instead trains directly
//! against [`CandidateFeatures`](pixelflow_search::egraph::candidate::CandidateFeatures)'s
//! own field set as minted into the JSONL (rule identity, one-hop
//! neighborhood ops, budget fraction, matched-class node count) — the same
//! information `CandidateSummary::new` would forward into the tower, just
//! scored by a transparent linear model that a Guide-wiring task can inspect,
//! extend into an MLP, or replace outright once the accept/kill gate (§5)
//! calls for scoring quality beyond what a linear model can express.
//!
//! # Loss weighting
//!
//! The dataset is dominated by the negative (`Wasted`) class — TRAIN's
//! pooled strict-positive rate is ~0.52%
//! (`docs/results/2026-09-01-strict-label-dataset.json`). Plain BCE would let
//! a trainer collapse to predicting `Wasted` everywhere and still score
//! well over 99% raw accuracy — flagged explicitly by `gen_strict_labels`'s
//! own stdout report. This binary applies **inverse-class-frequency weighting**
//! (`pos_weight = negatives / positives`, computed from the TRAIN split
//! actually used, not hand-picked): the standard, simplest-defensible choice
//! for a cold-start baseline that has no prior run to tune against, and the
//! literal quantity `gen_strict_labels`'s report already names as the fix
//! ("needs positive-class loss weighting (e.g. inverse-frequency...)"). The
//! resulting weight (computed at run time, printed, and recorded in the
//! checkpoint) is large (~190x) because the class imbalance is severe;
//! per-step gradient clipping (`--grad-clip`, applied to the loss gradient
//! w.r.t. the logit, not to the loss value) exists purely to keep online SGD
//! numerically stable against that weight's early-training steps — it does
//! not change what the loss weighting means, only how big a single sample's
//! update is allowed to be.
//!
//! # Cold start, TRAIN-only
//!
//! Every weight starts at a fixed value (bias/scalar weights at 0, per-rule
//! and per-op weights at 0) — no warm start from any prior guide/mask-head
//! checkpoint, per design doc §5's "Cold-start" training-protocol bullet
//! (all such prior weights were deleted with the unified-training-flow
//! history anyway, so there is nothing to warm-start from even if this
//! binary wanted to). Training reads only `--train` (default
//! `strict_labels_train.jsonl`, TRAIN-family by the fence check
//! `gen_strict_labels` already enforced when minting it); `--dev` is read
//! only for held-out evaluation after training completes and never touches
//! a gradient.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin train_guide -- \
//!     --train pixelflow-pipeline/data/strict_labels_train.jsonl \
//!     --dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --out-checkpoint pixelflow-pipeline/data/guide_checkpoint_strict_v1.json \
//!     --report-json docs/results/2026-09-01-train-guide-report.json \
//!     --report-md docs/results/2026-09-01-train-guide-report.md
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use clap::Parser;
use serde::Serialize;

use pixelflow_pipeline::schema::{fnv1a64_hex, unix_now_s};
use pixelflow_pipeline::training::guide_linear::{
    GuideCheckpoint, Model, Record, Sample, auc_roc, auc_roc_within_groups, average_precision,
    op_index_table, rule_index_table, to_sample,
};
use pixelflow_search::egraph::{RuleId, RuleSet};

#[derive(Parser)]
#[command(name = "train_guide")]
#[command(about = "Cold-start supervised training of the saturation Guide on strict labels")]
struct Args {
    /// TRAIN-family strict-label JSONL (`gen_strict_labels`'s `--out-train`).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_train.jsonl"
    )]
    train: String,

    /// DEV-family strict-label JSONL (`gen_strict_labels`'s `--out-dev`) —
    /// held out, evaluated only, never trained on.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_dev.jsonl"
    )]
    dev: String,

    /// Training epochs (full passes over the shuffled TRAIN set).
    #[arg(long, default_value_t = 30)]
    epochs: usize,

    /// Initial learning rate (per-sample SGD step), inverse-time decayed by
    /// `--lr-decay` per epoch.
    #[arg(long, default_value_t = 0.01)]
    lr: f32,

    /// Inverse-time learning-rate decay: `lr_t = lr0 / (1 + lr_decay * t)`.
    #[arg(long, default_value_t = 0.05)]
    lr_decay: f32,

    /// L2 weight decay applied to every weight (not the bias) each step.
    #[arg(long, default_value_t = 1e-4)]
    l2: f32,

    /// Seed for the cold-start weight init (all-zero, so this only seeds the
    /// per-epoch shuffle) and the shuffle itself.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Clip bound on the loss gradient w.r.t. the logit — see module doc
    /// "Loss weighting" for why this is needed at this pos_weight scale.
    #[arg(long, default_value_t = 20.0)]
    grad_clip: f32,

    /// Evaluate held-out DEV AUC every this many epochs (and at the last
    /// epoch), to produce the training curve's held-out arm without paying
    /// for a DEV forward pass every epoch.
    #[arg(long, default_value_t = 5)]
    eval_every: usize,

    /// Where to write the trained checkpoint (weights + mint-style
    /// provenance).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    out_checkpoint: String,

    /// Optional JSON report (metrics, per-rule table, calibration, training
    /// curve).
    #[arg(long)]
    report_json: Option<String>,

    /// Optional Markdown report (human-readable mirror of `--report-json`).
    #[arg(long)]
    report_md: Option<String>,
}

// ── Input schema + feature encoding: shared with the skew test ──────────────
//
// `Record`/`Sample`/`op_index_table`/`to_sample` moved to
// `pixelflow_pipeline::training::guide_linear` (2026-09-02) so this trainer
// and `skew_test_linear_guide` (which checks this binary's forward pass
// against `nnue::guide::linear::LinearCandidateGuide`'s deployed one) share
// exactly one definition of "what a JSONL row means as a feature vector" —
// see that module's doc for why a second copy here would undermine the very
// skew test it exists to support.

/// Everything one JSONL split contributes: encoded samples plus the
/// aggregates the report needs (rule name table, family count, measured
/// per-rule positive rate — the "measured strict-bound table" the learned
/// priorities get checked against).
struct LoadedSplit {
    samples: Vec<Sample>,
    /// Parallel to `samples`: which expression each one came from — the
    /// decision set a live guide actually ranks within (see
    /// `auc_within_groups`).
    groups: Vec<String>,
    families: HashSet<(u32, u64)>,
    /// rule slot (see [`Sample::rule_slot`]) -> (fired, positive), measured
    /// directly from this split.
    per_rule: BTreeMap<usize, (usize, usize)>,
    positives: usize,
    /// Rows dropped because `dedup_repeat` was set — reported, never silent.
    repeats_excluded: usize,
}

fn load_jsonl(
    path: &Path,
    op_index: &HashMap<String, usize>,
    rule_index: &HashMap<String, usize>,
) -> LoadedSplit {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("train_guide: cannot open {}: {e}", path.display()));
    let reader = BufReader::new(file);

    let mut samples = Vec::new();
    let mut groups = Vec::new();
    let mut families = HashSet::new();
    let mut per_rule: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    let mut positives = 0usize;
    let mut repeats_excluded = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            panic!(
                "train_guide: I/O error reading {} at line {}: {e}",
                path.display(),
                line_no + 1
            )
        });
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line).unwrap_or_else(|e| {
            panic!(
                "train_guide: malformed JSONL record in {} at line {}: {e}",
                path.display(),
                line_no + 1
            )
        });
        families.insert((record.family_band, record.family_seed));
        // Train and evaluate on exactly the candidate population a deployed
        // guide is asked to rank. `GuidedSaturation` removes an already-seen
        // `CandidateKey` BEFORE scoring, so a `dedup_repeat` row can never
        // enter a live decision set; keeping them would let idempotent
        // re-fires (the large majority of every episode's log, and almost
        // entirely negative) set both the learned per-rule priorities and
        // the control's base rates.
        if record.dedup_repeat {
            repeats_excluded += 1;
            continue;
        }
        let sample = to_sample(&record, op_index, rule_index);
        let agg = per_rule.entry(sample.rule_slot).or_insert((0, 0));
        agg.0 += 1;
        if record.label_positive {
            agg.1 += 1;
            positives += 1;
        }
        groups.push(record.expr_name.clone());
        samples.push(sample);
    }

    eprintln!(
        "train_guide: {} — {} first-seen candidates kept, {repeats_excluded} dedup_repeat \
         rows excluded (a deployed guide never scores them)",
        path.display(),
        samples.len()
    );

    assert!(
        !samples.is_empty(),
        "train_guide: {} contained zero first-seen candidate records ({repeats_excluded} \
         dedup_repeat rows were excluded) — refusing to train/evaluate on an empty split",
        path.display()
    );

    LoadedSplit {
        samples,
        groups,
        families,
        per_rule,
        positives,
        repeats_excluded,
    }
}

// `Model` (per-rule bias + bag-of-ops linear term + scalar features) also
// moved to `pixelflow_pipeline::training::guide_linear` — see this file's
// "Input schema + feature encoding" note above.

/// Numerically stable logistic sigmoid.
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Weighted BCE-with-logits, PyTorch's stable form:
/// `(1-y)*z + (1 + (pos_weight-1)*y) * (log(1+exp(-|z|)) + max(-z, 0))`.
/// Used only for the reported loss value; the gradient used for the SGD
/// step is computed directly from `sigmoid`, not by differentiating this
/// expression symbolically at each step.
fn weighted_bce_loss(z: f32, y: f32, pos_weight: f32) -> f32 {
    let log_weight = 1.0 + (pos_weight - 1.0) * y;
    (1.0 - y) * z + log_weight * ((-z.abs()).exp().ln_1p() + (-z).max(0.0))
}

/// `dLoss/dz` for weighted BCE-with-logits: `pos_weight*(p-1)` for a positive
/// label, `p` for a negative one (derived from `d/dz[-log(sigmoid(z))] =
/// sigmoid(z)-1` and `d/dz[-log(1-sigmoid(z))] = sigmoid(z)`).
fn weighted_bce_grad(p: f32, y: f32, pos_weight: f32) -> f32 {
    if y > 0.5 { pos_weight * (p - 1.0) } else { p }
}

// ── Deterministic shuffle (same LCG constant `SaturationHead::randomize` uses) ──

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// Fisher-Yates shuffle of `0..n`, seeded deterministically so a training
/// run's epoch order is reproducible from `--seed` alone.
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        let r = (lcg_next(&mut state) >> 33) as usize % (i + 1);
        idx.swap(i, r);
    }
    idx
}

// Ranking metrics (`auc_roc`/`average_precision`) moved to
// `pixelflow_pipeline::training::guide_linear` (2026-09-02) so
// `eval_control_guides` (DEV AUC/PR-AUC for the `PerRuleRateGuide` control
// arm, design doc §5 task 2) scores its table by the exact same formula this
// binary reports for the linear model — a side-by-side comparison is only
// honest if both sides are measured the same way.

/// Spearman rank correlation (average-rank tie handling) — shared pattern
/// with `tightened_labeler_rank.rs`'s `spearman`/`pearson` (kept as a
/// separate small copy rather than a shared crate helper: no `pixelflow-
/// pipeline` module holds general-purpose stats helpers today, and lifting
/// one for two four-line functions is exactly the "abstraction before
/// subtraction has failed" this codebase's style guide warns against).
fn spearman(xs: &[f64], ys: &[f64]) -> Option<f64> {
    assert_eq!(xs.len(), ys.len(), "spearman: mismatched lengths");
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("spearman: NaN input"));
        let mut ranks = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
            for k in idx.iter().take(j + 1).skip(i) {
                ranks[*k] = avg_rank;
            }
            i = j + 1;
        }
        ranks
    };
    pearson(&rank(xs), &rank(ys))
}

fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return None;
    }
    Some(cov / (vx.sqrt() * vy.sqrt()))
}

/// One rank-based calibration bucket: a slice of samples between two
/// [`CALIBRATION_QUANTILES`] cut points, ranked by predicted probability.
#[derive(Serialize)]
struct CalibrationBucket {
    /// The `[lo, hi)` population-quantile range this bucket covers, e.g.
    /// `(0.99, 0.995)` for "the 99th-99.5th percentile by predicted score".
    quantile_lo: f64,
    quantile_hi: f64,
    count: usize,
    mean_predicted: f64,
    actual_positive_rate: f64,
    min_predicted: f64,
    max_predicted: f64,
}

/// Population-quantile cut points for the calibration table, deliberately
/// dense toward the top: this dataset's strict-positive rate is ~0.5-0.6%
/// (design doc §2.1, `docs/results/2026-09-01-strict-label-dataset.json`),
/// so a well-trained classifier pushes the overwhelming majority of
/// predicted probabilities toward exactly zero. Equal-*count* deciles (the
/// first version of this function) still put ~90% of DEV in one bucket whose
/// predicted range rounds to `0.00000` at 5 decimal places — technically
/// rank-based, but uninformative for exactly the skew it was trying to
/// handle. These cut points instead concentrate resolution in the top 10%,
/// where essentially all of the model's discriminative signal (and all of
/// the true positives worth calibrating) actually lives.
const CALIBRATION_QUANTILES: &[f64] = &[
    0.0, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0,
];

fn calibration_tail_buckets(scores: &[f32], labels: &[f32]) -> Vec<CalibrationBucket> {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or_else(|| panic!("calibration_tail_buckets: NaN score"))
    });
    let mut out = Vec::with_capacity(CALIBRATION_QUANTILES.len() - 1);
    for w in CALIBRATION_QUANTILES.windows(2) {
        let (q_lo, q_hi) = (w[0], w[1]);
        let lo = (q_lo * n as f64).round() as usize;
        let hi = if q_hi >= 1.0 {
            n
        } else {
            (q_hi * n as f64).round() as usize
        };
        if lo >= hi {
            continue;
        }
        let slice = &idx[lo..hi];
        let count = slice.len();
        let sum_pred: f64 = slice.iter().map(|&i| scores[i] as f64).sum();
        let sum_label: f64 = slice.iter().map(|&i| labels[i] as f64).sum();
        let min_predicted = scores[slice[0]] as f64;
        let max_predicted = scores[slice[count - 1]] as f64;
        out.push(CalibrationBucket {
            quantile_lo: q_lo,
            quantile_hi: q_hi,
            count,
            mean_predicted: sum_pred / count as f64,
            actual_positive_rate: sum_label / count as f64,
            min_predicted,
            max_predicted,
        });
    }
    out
}

// ── Report ───────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EpochPoint {
    epoch: usize,
    lr: f32,
    train_mean_weighted_loss: f64,
    dev_auc: Option<f64>,
    dev_pr_auc: Option<f64>,
}

#[derive(Serialize)]
struct RuleRow {
    /// The rule's stable identity ([`RuleId::get`]) — never a position.
    rule_id: u64,
    rule_name: String,
    train_fired: usize,
    train_positive: usize,
    train_positive_rate: f64,
    dev_fired: usize,
    dev_positive: usize,
    dev_positive_rate: f64,
    dev_mean_predicted: f64,
}

#[derive(Serialize)]
struct Report {
    label_source: String,
    /// The DEV file these metrics were computed on, and its FNV-1a 64
    /// content hash. `eval_control_guides` subtracts this report's linear
    /// AUCs from control AUCs it computes itself; without a fingerprint to
    /// check, pointing it at a regenerated or filtered DEV file produces a
    /// plausible gap between two different populations and calls it a
    /// same-split comparison.
    dev_path: String,
    dev_fnv64: String,
    /// Provenance note for the 2026-09-02 budget-denominator correction —
    /// see `write_md_report`'s leading banner for the human-readable form.
    denominator_note: String,
    pos_weight: f32,
    pos_weight_rationale: String,
    train_samples: usize,
    train_families: usize,
    train_positive_rate: f64,
    dev_samples: usize,
    dev_families: usize,
    dev_positive_rate: f64,
    dev_auc: f64,
    dev_pr_auc: f64,
    /// AUC computed WITHIN each expression's candidate set and macro-averaged
    /// — the move-ordering question a deployed guide is actually asked. See
    /// `guide_linear::auc_roc_within_groups` for why the pooled `dev_auc`
    /// above can credit the model for between-expression variation two of
    /// its features cannot act on at deploy time.
    dev_auc_within_expression: Option<f64>,
    dev_within_expression_groups: usize,
    /// `dedup_repeat` rows excluded from each split (see
    /// `guide_linear::Record::dedup_repeat`).
    train_dedup_repeats_excluded: usize,
    dev_dedup_repeats_excluded: usize,
    training_curve: Vec<EpochPoint>,
    calibration: Vec<CalibrationBucket>,
    per_rule: Vec<RuleRow>,
    rule_priority_spearman_learned_vs_dev_measured: Option<f64>,
    checkpoint_path: String,
}

fn write_json_report(path: &str, report: &Report) {
    let json = serde_json::to_string_pretty(report)
        .unwrap_or_else(|e| panic!("train_guide: report serialization failed: {e}"));
    std::fs::write(path, json).unwrap_or_else(|e| panic!("train_guide: cannot write {path}: {e}"));
    eprintln!("train_guide: wrote {path}");
}

fn write_md_report(path: &str, report: &Report) {
    let mut md = String::new();
    md.push_str("# Guide cold-start training report (strict-v1 labels)\n\n");
    md.push_str(&format!("> {}\n\n", report.denominator_note));
    md.push_str(&format!(
        "TRAIN: {} samples, {} families, positive rate {:.4}%.\n\n",
        report.train_samples,
        report.train_families,
        report.train_positive_rate * 100.0
    ));
    md.push_str(&format!(
        "DEV (held out, never trained on): {} samples, {} families, positive rate {:.4}%.\n\n",
        report.dev_samples,
        report.dev_families,
        report.dev_positive_rate * 100.0
    ));
    md.push_str(&format!(
        "**Loss weighting**: pos_weight = {:.3} ({}).\n\n",
        report.pos_weight, report.pos_weight_rationale
    ));
    md.push_str(&format!(
        "**Held-out (DEV-family) ranking quality**: AUC-ROC = {:.4}, PR-AUC (average \
         precision) = {:.4}.\n\n",
        report.dev_auc, report.dev_pr_auc
    ));
    if let Some(rho) = report.rule_priority_spearman_learned_vs_dev_measured {
        md.push_str(&format!(
            "**Sanity check** — Spearman correlation between the learned per-rule mean \
             predicted probability and the DEV-measured per-rule strict-positive rate: \
             ρ = {rho:.4}. A model that learned nothing beyond noise would show ρ near 0; a \
             model that only reproduced each rule's overall base rate (ignoring \
             candidate-local structure) would still show ρ close to 1 here, since both \
             quantities are monotonic in the same underlying per-rule tendency — this check \
             confirms the model tracks the label semantics, not that it beats a per-rule \
             lookup table (a saturation-quality evaluation, out of scope for this report, \
             would be needed for that).\n\n"
        ));
    }

    md.push_str("## Training curve\n\n");
    md.push_str("| epoch | lr | train weighted loss | DEV AUC | DEV PR-AUC |\n");
    md.push_str("|---:|---:|---:|---:|---:|\n");
    for p in &report.training_curve {
        md.push_str(&format!(
            "| {} | {:.5} | {:.6} | {} | {} |\n",
            p.epoch,
            p.lr,
            p.train_mean_weighted_loss,
            p.dev_auc.map(|v| format!("{v:.4}")).unwrap_or_default(),
            p.dev_pr_auc.map(|v| format!("{v:.4}")).unwrap_or_default(),
        ));
    }

    md.push_str("\n## Calibration (population-quantile buckets, dense toward the top, DEV)\n\n");
    md.push_str(
        "| quantile range | n | predicted range | mean predicted | actual positive rate |\n",
    );
    md.push_str("|---|---:|---|---:|---:|\n");
    for b in &report.calibration {
        md.push_str(&format!(
            "| [{:.3}, {:.3}) | {} | [{:.6}, {:.6}] | {:.6} | {:.6} |\n",
            b.quantile_lo,
            b.quantile_hi,
            b.count,
            b.min_predicted,
            b.max_predicted,
            b.mean_predicted,
            b.actual_positive_rate
        ));
    }

    md.push_str(
        "\n## Per-rule: learned priority (DEV mean predicted) vs measured strict-bound rate\n\n",
    );
    md.push_str(
        "| rule | id | train fired | train rate | DEV fired | DEV measured rate | DEV mean predicted |\n",
    );
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for r in &report.per_rule {
        md.push_str(&format!(
            "| {} | {:016x} | {} | {:.5} | {} | {:.5} | {:.5} |\n",
            r.rule_name,
            r.rule_id,
            r.train_fired,
            r.train_positive_rate,
            r.dev_fired,
            r.dev_positive_rate,
            r.dev_mean_predicted
        ));
    }

    md.push_str(&format!(
        "\nCheckpoint written to `{}`.\n",
        report.checkpoint_path
    ));

    std::fs::write(path, md).unwrap_or_else(|e| panic!("train_guide: cannot write {path}: {e}"));
    eprintln!("train_guide: wrote {path}");
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let (op_names, op_index) = op_index_table();
    let num_ops = op_names.len();

    // The rule table is sized and named from the LIVE registered rule set,
    // not from whichever rules happened to appear in these two JSONL splits:
    // a rule that never fired in TRAIN or DEV still needs a (zero) weight,
    // or the checkpoint is short and the first live candidate for that rule
    // makes `LinearCandidateGuide::logit` panic. Slots are an internal
    // layout; the checkpoint records the label per slot and the vocabulary's
    // fingerprint, which is what makes the weights readable elsewhere.
    let rules = RuleSet::production();
    let (rule_names, rule_index) = rule_index_table(&rules);
    let num_rules = rule_names.len();

    eprintln!("train_guide: loading TRAIN from {}", args.train);
    let train = load_jsonl(&PathBuf::from(&args.train), &op_index, &rule_index);
    eprintln!("train_guide: loading DEV from {}", args.dev);
    let dev = load_jsonl(&PathBuf::from(&args.dev), &op_index, &rule_index);

    // TRAIN and DEV must be disjoint at family granularity, or "held out"
    // is a claim about data that contributed gradients.
    let overlap: Vec<(u32, u64)> = train
        .families
        .intersection(&dev.families)
        .copied()
        .collect();
    assert!(
        overlap.is_empty(),
        "train_guide: TRAIN ({}) and DEV ({}) share {} generator families (e.g. {:?}) — the \
         reported DEV metrics would not be held out; regenerate the splits against the \
         corpus split manifest",
        args.train,
        args.dev,
        overlap.len(),
        &overlap[..overlap.len().min(3)]
    );

    let train_positive_rate = train.positives as f64 / train.samples.len() as f64;
    let neg = train.samples.len() - train.positives;
    assert!(
        train.positives > 0 && neg > 0,
        "train_guide: TRAIN has {} positive and {neg} negative samples — binary \
         supervision needs both. With no positives `pos_weight` would be invented from a \
         `max(1)` denominator and the run would train an all-negative model; with no \
         negatives it would be zero and every positive gradient would vanish. Either way \
         the checkpoint would be written as if it had been trained.",
        train.positives
    );
    let pos_weight = neg as f32 / train.positives as f32;
    let pos_weight_rationale = format!(
        "inverse class frequency (negatives/positives = {neg}/{} measured on this TRAIN split) \
         — the simplest defensible cold-start choice: unweighted BCE lets a trainer collapse to \
         predicting the majority class and still score >99% raw accuracy (flagged by \
         gen_strict_labels's own report), and there is no prior Guide run to tune a fancier \
         (focal-loss-style) weighting against yet",
        train.positives
    );

    eprintln!(
        "train_guide: TRAIN {} samples / {} families, positive rate {:.4}%, pos_weight = {:.3}",
        train.samples.len(),
        train.families.len(),
        train_positive_rate * 100.0,
        pos_weight
    );
    eprintln!(
        "train_guide: DEV {} samples / {} families, positive rate {:.4}%",
        dev.samples.len(),
        dev.families.len(),
        dev.positives as f64 / dev.samples.len() as f64 * 100.0
    );

    let mut model = Model::zero(num_rules, num_ops);
    let mut training_curve = Vec::new();

    for epoch in 0..args.epochs {
        let lr = args.lr / (1.0 + args.lr_decay * epoch as f32);
        let order = shuffled_indices(train.samples.len(), args.seed.wrapping_add(epoch as u64));

        let mut loss_sum = 0.0f64;
        for &i in &order {
            let s = &train.samples[i];
            let z = model.logit(s);
            let p = sigmoid(z);
            loss_sum += weighted_bce_loss(z, s.label, pos_weight) as f64;
            let grad_z =
                weighted_bce_grad(p, s.label, pos_weight).clamp(-args.grad_clip, args.grad_clip);
            model.sgd_step(s, grad_z, lr, args.l2);
        }
        let mean_loss = loss_sum / train.samples.len() as f64;

        let is_last = epoch + 1 == args.epochs;
        let (dev_auc, dev_pr_auc) =
            if args.eval_every > 0 && (epoch % args.eval_every == 0 || is_last) {
                let scores: Vec<f32> = dev
                    .samples
                    .iter()
                    .map(|s| sigmoid(model.logit(s)))
                    .collect();
                let labels: Vec<f32> = dev.samples.iter().map(|s| s.label).collect();
                (
                    auc_roc(&scores, &labels),
                    average_precision(&scores, &labels),
                )
            } else {
                (None, None)
            };

        eprintln!(
            "train_guide: epoch {epoch:>3} lr={lr:.5} train_weighted_loss={mean_loss:.6}{}",
            match (dev_auc, dev_pr_auc) {
                (Some(a), Some(p)) => format!("  dev_auc={a:.4} dev_pr_auc={p:.4}"),
                _ => String::new(),
            }
        );

        training_curve.push(EpochPoint {
            epoch,
            lr,
            train_mean_weighted_loss: mean_loss,
            dev_auc,
            dev_pr_auc,
        });
    }

    // Final held-out evaluation (guaranteed present regardless of `--eval-every`).
    let dev_scores: Vec<f32> = dev
        .samples
        .iter()
        .map(|s| sigmoid(model.logit(s)))
        .collect();
    let dev_labels: Vec<f32> = dev.samples.iter().map(|s| s.label).collect();
    let dev_auc = auc_roc(&dev_scores, &dev_labels).unwrap_or_else(|| {
        panic!("train_guide: DEV AUC undefined — one class is entirely absent from DEV")
    });
    let dev_pr_auc = average_precision(&dev_scores, &dev_labels).unwrap_or_else(|| {
        panic!("train_guide: DEV PR-AUC undefined — DEV has zero positive labels")
    });

    let dev_auc_grouped = auc_roc_within_groups(&dev_scores, &dev_labels, &dev.groups);
    match dev_auc_grouped {
        Some((a, n)) => println!(
            "=== held out DEV, WITHIN-expression AUC (the move-ordering question): {a:.4} \
             over {n} expressions with both classes; pooled DEV AUC above also contains \
             between-expression variation the deployed guide cannot act on ==="
        ),
        None => println!(
            "=== held out DEV, WITHIN-expression AUC: undefined — no single expression's \
             candidate set contains both classes ==="
        ),
    }

    let dev_fnv64 = {
        let bytes = std::fs::read(&args.dev).unwrap_or_else(|e| {
            panic!(
                "train_guide: cannot re-read {} to fingerprint it: {e}",
                args.dev
            )
        });
        fnv1a64_hex(&bytes)
    };

    let calibration = calibration_tail_buckets(&dev_scores, &dev_labels);

    // Per-rule table: learned DEV mean predicted probability vs measured
    // strict-bound rate (TRAIN and DEV, both reported).
    let mut dev_mean_pred_by_rule: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for (s, &p) in dev.samples.iter().zip(dev_scores.iter()) {
        let e = dev_mean_pred_by_rule.entry(s.rule_slot).or_insert((0.0, 0));
        e.0 += p as f64;
        e.1 += 1;
    }

    let mut all_rule_idxs: Vec<usize> = train
        .per_rule
        .keys()
        .chain(dev.per_rule.keys())
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    all_rule_idxs.sort_unstable();

    let mut per_rule_rows = Vec::new();
    let mut learned_ranks = Vec::new();
    let mut measured_ranks = Vec::new();
    for idx in all_rule_idxs {
        let (train_fired, train_positive) = train.per_rule.get(&idx).copied().unwrap_or((0, 0));
        let (dev_fired, dev_positive) = dev.per_rule.get(&idx).copied().unwrap_or((0, 0));
        let (pred_sum, pred_n) = dev_mean_pred_by_rule.get(&idx).copied().unwrap_or((0.0, 0));
        let dev_mean_predicted = if pred_n > 0 {
            pred_sum / pred_n as f64
        } else {
            0.0
        };
        let dev_positive_rate = if dev_fired > 0 {
            dev_positive as f64 / dev_fired as f64
        } else {
            0.0
        };
        if dev_fired >= 20 {
            // Only rank rules with enough DEV support to measure a
            // meaningful positive rate — a rule that fired twice on DEV
            // contributes noise, not signal, to the sanity-check
            // correlation.
            learned_ranks.push(dev_mean_predicted);
            measured_ranks.push(dev_positive_rate);
        }
        per_rule_rows.push(RuleRow {
            rule_id: RuleId::from_label(&rule_names[idx]).get(),
            rule_name: rule_names[idx].clone(),
            train_fired,
            train_positive,
            train_positive_rate: if train_fired > 0 {
                train_positive as f64 / train_fired as f64
            } else {
                0.0
            },
            dev_fired,
            dev_positive,
            dev_positive_rate,
            dev_mean_predicted,
        });
    }
    per_rule_rows.sort_by(|a, b| {
        b.dev_mean_predicted
            .partial_cmp(&a.dev_mean_predicted)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let rule_priority_spearman = spearman(&learned_ranks, &measured_ranks);

    let mut checkpoint = GuideCheckpoint {
        schema_identity: String::new(),
        label_source: "strict-v1".to_string(),
        trainer: "train_guide".to_string(),
        written_at_unix_s: unix_now_s(),
        seed: args.seed,
        epochs: args.epochs,
        lr_initial: args.lr,
        lr_decay: args.lr_decay,
        l2: args.l2,
        grad_clip: args.grad_clip,
        pos_weight,
        num_rules,
        num_ops,
        rule_names: rule_names.clone(),
        rule_fingerprint: format!("{}", rules.fingerprint()),
        op_names: op_names.clone(),
        bias: model.bias,
        w_rule: model.w_rule.clone(),
        w_op: model.w_op.clone(),
        w_budget: model.w_budget,
        w_match_class: model.w_match_class,
        w_neighborhood: model.w_neighborhood,
        w_expr_size: model.w_expr_size,
        train_samples: train.samples.len(),
        train_families: train.families.len(),
        train_positive_rate,
        dev_samples: dev.samples.len(),
        dev_families: dev.families.len(),
        dev_auc,
        dev_pr_auc,
        weights_fnv64: String::new(),
    };
    checkpoint
        .write(&PathBuf::from(&args.out_checkpoint))
        .unwrap_or_else(|e| panic!("train_guide: cannot write {}: {e}", args.out_checkpoint));
    eprintln!("train_guide: wrote checkpoint {}", args.out_checkpoint);

    println!(
        "=== train_guide: TRAIN {} samples / {} families (positive rate {:.4}%), pos_weight \
         {:.3} ===",
        train.samples.len(),
        train.families.len(),
        train_positive_rate * 100.0,
        pos_weight
    );
    println!(
        "=== held out DEV {} samples / {} families: AUC-ROC {:.4}, PR-AUC {:.4} ===",
        dev.samples.len(),
        dev.families.len(),
        dev_auc,
        dev_pr_auc
    );
    if let Some(rho) = rule_priority_spearman {
        println!(
            "=== per-rule learned-priority vs DEV-measured-rate Spearman rho = {rho:.4} ({} rules with >=20 DEV firings) ===",
            learned_ranks.len()
        );
    }

    let report = Report {
        label_source: "strict-v1".to_string(),
        dev_path: args.dev.clone(),
        dev_fnv64: dev_fnv64.clone(),
        denominator_note: format!(
            "Budget denominator: REGISTERED_PRIMARY_BUDGET_APPLICATIONS = {} \
             (docs/plans/2026-09-01-phase3-registration.md §4, classical-band primary tier), \
             imported by both gen_strict_labels and saturate_guided_until_applications so \
             budget_fraction means the same thing at mint time and at deploy time. The first \
             mint/train pass of this round (2026-09-01) used a 195-application placeholder \
             (this round's measured median application count, before B was registered) — a \
             train/deploy denominator skew, caught and fixed before this checkpoint (see git \
             history for the placeholder-era numbers).",
            pixelflow_search::egraph::REGISTERED_PRIMARY_BUDGET_APPLICATIONS
        ),
        pos_weight,
        pos_weight_rationale,
        train_samples: train.samples.len(),
        train_families: train.families.len(),
        train_positive_rate,
        dev_samples: dev.samples.len(),
        dev_families: dev.families.len(),
        dev_positive_rate: dev.positives as f64 / dev.samples.len() as f64,
        dev_auc,
        dev_pr_auc,
        dev_auc_within_expression: dev_auc_grouped.map(|(a, _)| a),
        dev_within_expression_groups: dev_auc_grouped.map_or(0, |(_, n)| n),
        train_dedup_repeats_excluded: train.repeats_excluded,
        dev_dedup_repeats_excluded: dev.repeats_excluded,
        training_curve,
        calibration,
        per_rule: per_rule_rows,
        rule_priority_spearman_learned_vs_dev_measured: rule_priority_spearman,
        checkpoint_path: args.out_checkpoint.clone(),
    };

    if let Some(path) = &args.report_json {
        write_json_report(path, &report);
    }
    if let Some(path) = &args.report_md {
        write_md_report(path, &report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── sigmoid / loss / gradient ────────────────────────────────────────

    #[test]
    fn sigmoid_matches_naive_formula_away_from_the_stable_split() {
        for &z in &[-3.0f32, -0.5, 0.0, 0.5, 3.0] {
            let naive = 1.0 / (1.0 + (-z).exp());
            assert!((sigmoid(z) - naive).abs() < 1e-6, "z={z}");
        }
    }

    #[test]
    fn sigmoid_is_finite_for_large_magnitude_inputs() {
        assert!(sigmoid(100.0).is_finite());
        assert!(sigmoid(-100.0).is_finite());
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn weighted_bce_loss_matches_unweighted_bce_when_pos_weight_is_one() {
        let z = 0.7f32;
        let p = sigmoid(z);
        for &y in &[0.0f32, 1.0] {
            let weighted = weighted_bce_loss(z, y, 1.0);
            let naive = -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            assert!(
                (weighted - naive).abs() < 1e-4,
                "y={y} weighted={weighted} naive={naive}"
            );
        }
    }

    #[test]
    fn weighted_bce_grad_matches_a_positive_label_scaled_by_pos_weight() {
        let p = 0.2f32;
        assert!((weighted_bce_grad(p, 1.0, 10.0) - 10.0 * (p - 1.0)).abs() < 1e-6);
        assert!((weighted_bce_grad(p, 0.0, 10.0) - p).abs() < 1e-6);
    }

    // ── shuffle ──────────────────────────────────────────────────────────

    #[test]
    fn shuffled_indices_is_a_permutation() {
        let idx = shuffled_indices(500, 7);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..500).collect::<Vec<_>>());
    }

    #[test]
    fn shuffled_indices_is_deterministic_given_the_same_seed() {
        assert_eq!(shuffled_indices(200, 99), shuffled_indices(200, 99));
    }

    #[test]
    fn shuffled_indices_differs_across_seeds() {
        assert_ne!(shuffled_indices(200, 1), shuffled_indices(200, 2));
    }

    // ── ranking metrics ──────────────────────────────────────────────────

    #[test]
    fn auc_roc_is_one_for_perfectly_separated_scores() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!((auc_roc(&scores, &labels).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn auc_roc_is_zero_for_perfectly_inverted_scores() {
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!(auc_roc(&scores, &labels).unwrap().abs() < 1e-9);
    }

    #[test]
    fn auc_roc_is_one_half_for_random_scores_indistinguishable_by_class() {
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![0.0, 1.0, 0.0, 1.0];
        assert!((auc_roc(&scores, &labels).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn auc_roc_is_none_when_a_class_is_absent() {
        let scores = vec![0.1, 0.2, 0.3];
        let labels = vec![0.0, 0.0, 0.0];
        assert!(auc_roc(&scores, &labels).is_none());
    }

    #[test]
    fn average_precision_is_one_for_perfectly_separated_scores() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!((average_precision(&scores, &labels).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn average_precision_is_none_with_no_positives() {
        let scores = vec![0.1, 0.2, 0.3];
        let labels = vec![0.0, 0.0, 0.0];
        assert!(average_precision(&scores, &labels).is_none());
    }

    #[test]
    fn spearman_is_one_for_identical_rankings() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        assert!((spearman(&xs, &ys).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spearman_is_negative_one_for_inverted_rankings() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![40.0, 30.0, 20.0, 10.0];
        assert!((spearman(&xs, &ys).unwrap() + 1.0).abs() < 1e-9);
    }

    // ── calibration ──────────────────────────────────────────────────────

    #[test]
    fn calibration_tail_buckets_covers_the_whole_population_and_orders_by_score() {
        let scores: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let labels: Vec<f32> = (0..1000)
            .map(|i| if i >= 500 { 1.0 } else { 0.0 })
            .collect();
        let buckets = calibration_tail_buckets(&scores, &labels);
        assert_eq!(buckets.len(), CALIBRATION_QUANTILES.len() - 1);
        let total: usize = buckets.iter().map(|b| b.count).sum();
        assert_eq!(
            total,
            scores.len(),
            "buckets must partition the whole population"
        );
        // Bottom bucket (lowest predicted scores) is all-negative, top
        // bucket (highest predicted scores) is all-positive for this
        // perfectly-ordered synthetic input.
        assert!((buckets.first().unwrap().actual_positive_rate - 0.0).abs() < 1e-9);
        assert!((buckets.last().unwrap().actual_positive_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn calibration_tail_buckets_is_monotonic_by_quantile_range() {
        let scores: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let labels: Vec<f32> = vec![0.0; 1000];
        let buckets = calibration_tail_buckets(&scores, &labels);
        for w in buckets.windows(2) {
            assert!(w[0].quantile_hi <= w[1].quantile_lo + 1e-9);
        }
    }

    // ── model / op index table ───────────────────────────────────────────

    #[test]
    fn op_index_table_has_one_entry_per_opkind_and_no_duplicate_names() {
        let (names, index) = op_index_table();
        assert_eq!(names.len(), index.len(), "duplicate OpKind Debug names");
        assert!(!names.is_empty());
        for (i, n) in names.iter().enumerate() {
            assert_eq!(index[n], i);
        }
    }

    #[test]
    fn model_zero_start_scores_every_sample_at_the_bias_only() {
        let model = Model::zero(5, 3);
        let s = Sample {
            rule_slot: 2,
            op_feats: vec![(0, 1.5), (2, 0.5)],
            budget_fraction: 0.3,
            log_match_class: 1.0,
            log_neighborhood: 1.0,
            log_expr_size: 1.0,
            label: 1.0,
        };
        assert_eq!(
            model.logit(&s),
            0.0,
            "a zero-initialized model must score zero everywhere"
        );
    }

    #[test]
    fn sgd_step_moves_a_positive_samples_logit_upward() {
        let mut model = Model::zero(3, 2);
        let s = Sample {
            rule_slot: 1,
            op_feats: vec![(0, 1.0)],
            budget_fraction: 0.5,
            log_match_class: 0.5,
            log_neighborhood: 0.5,
            log_expr_size: 0.5,
            label: 1.0,
        };
        let before = model.logit(&s);
        let z = model.logit(&s);
        let p = sigmoid(z);
        let grad = weighted_bce_grad(p, s.label, 5.0);
        model.sgd_step(&s, grad, 0.1, 0.0);
        let after = model.logit(&s);
        assert!(
            after > before,
            "a positive-label SGD step should raise this sample's own logit: before={before} after={after}"
        );
    }

    #[test]
    fn sgd_step_moves_a_negative_samples_logit_downward() {
        let mut model = Model::zero(3, 2);
        let s = Sample {
            rule_slot: 1,
            op_feats: vec![(0, 1.0)],
            budget_fraction: 0.5,
            log_match_class: 0.5,
            log_neighborhood: 0.5,
            log_expr_size: 0.5,
            label: 0.0,
        };
        let before = model.logit(&s);
        let z = model.logit(&s);
        let p = sigmoid(z);
        let grad = weighted_bce_grad(p, s.label, 5.0);
        model.sgd_step(&s, grad, 0.1, 0.0);
        let after = model.logit(&s);
        assert!(
            after < before,
            "a negative-label SGD step should lower this sample's own logit: before={before} after={after}"
        );
    }

    // ── end-to-end: a tiny synthetic dataset should be learnable ─────────

    #[test]
    fn a_tiny_linearly_separable_dataset_is_learned_to_near_perfect_dev_auc() {
        // Rule 0 is always LoadBearing, rule 1 never is — a linear model
        // with a per-rule bias term should separate this perfectly.
        let mut samples = Vec::new();
        for i in 0..200 {
            let rule_idx = i % 2;
            samples.push(Sample {
                rule_slot: rule_idx,
                op_feats: vec![],
                budget_fraction: 0.1,
                log_match_class: 1.0,
                log_neighborhood: 1.0,
                log_expr_size: 1.0,
                label: if rule_idx == 0 { 1.0 } else { 0.0 },
            });
        }

        let mut model = Model::zero(2, 0);
        let pos_weight = 1.0; // balanced by construction
        for epoch in 0..50 {
            let order = shuffled_indices(samples.len(), 1000 + epoch);
            for &i in &order {
                let s = &samples[i];
                let z = model.logit(s);
                let p = sigmoid(z);
                let grad = weighted_bce_grad(p, s.label, pos_weight);
                model.sgd_step(s, grad, 0.2, 0.0);
            }
        }

        let scores: Vec<f32> = samples.iter().map(|s| sigmoid(model.logit(s))).collect();
        let labels: Vec<f32> = samples.iter().map(|s| s.label).collect();
        let auc = auc_roc(&scores, &labels).unwrap();
        assert!(
            auc > 0.99,
            "expected near-perfect AUC on a linearly separable toy set, got {auc}"
        );
    }
}
