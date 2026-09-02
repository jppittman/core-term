//! Linear return-to-go regressor for the saturation Guide
//! (`docs/plans/2026-09-01-guide-return-to-go.md` §3/§8 — the "linear model
//! first" half of the redirect: hindsight return-to-go as the training
//! target, in place of the strict/tight/loose hand-drawn bounds).
//!
//! # Scope: a sibling of `train_guide`, not a rewrite of it
//!
//! §8 names two acceptable shapes for this task — extend `train_guide`
//! with an `--objective` switch, or a sibling binary. This is the sibling:
//! `train_guide`'s strict-bit classifier (weighted BCE, inverse-class-
//! frequency loss weighting, a calibration table keyed to a ~0.5% positive
//! rate) and this regressor (MSE or pairwise-rank on a continuous
//! log-regret target, no class imbalance to correct for) are different
//! enough training loops that folding them into one `main` would mean a
//! `match` on `--objective` forking almost every step of it — exactly the
//! "namespace stacked into one name" smell this codebase's style guide
//! flags. What they *do* share — the feature encoding
//! (`Record`/`Sample`/`to_sample`) and the linear model's forward pass/SGD
//! step (`Model`) — already lives in
//! [`pixelflow_pipeline::training::guide_linear`] and is imported here
//! unchanged, not restated.
//!
//! # Two objectives, one shared linear form
//!
//! `--loss mse` (default, §3.2 primary): online SGD regressing
//! `f(x) = Model::logit(x)` against the requested label column
//! (`--target centered|raw`, `--label-b 100|200`) by squared error,
//! `dL/dz = 2(f - y)`, clipped by `--grad-clip` — `Model::sgd_step` is
//! loss-agnostic (it only consumes an already-computed `dL/dz`), so this
//! reuses it exactly as `train_guide` does for its own (very different)
//! loss.
//!
//! `--loss pairwise-rank` (§3.4 ablation): [`sample_rank_pairs`] draws,
//! per epoch, cross-trajectory pairs `(a, b)` of the same expression at the
//! same budget-fraction decile, signed by which trajectory's return was
//! better (§1.4 — the only shape a ranking loss has signal under a
//! return-to-go label, since every application within one trajectory
//! carries the *same* label). Standard pairwise logistic loss on
//! `sign·(f(x_a) - f(x_b))`, updated with two sequential `sgd_step` calls
//! (one per side of the pair) using the same shared `Model`.
//!
//! # What this binary does NOT do
//!
//! It does not mint the R2G trajectory dataset (`gen_r2g_trajectories`, a
//! separate binary/task) and does not run the counterfactual replay (§4) or
//! the at-budget evaluation ladder (§7) — this binary trains a checkpoint
//! and reports held-out regression/ranking quality on whatever
//! `--train`/`--dev` JSONL it is given, exactly as far as `train_guide`
//! goes for the strict-bit head. If the minter has not landed in this tree
//! yet, point `--train`/`--dev` at a hand-built fixture in
//! [`pixelflow_pipeline::training::r2g::R2gRecord`]'s schema — every field
//! that schema requires, this binary requires too, so a real mint's output
//! reads in unchanged.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin train_guide_r2g -- \
//!     --train pixelflow-pipeline/data/r2g_train.jsonl \
//!     --dev pixelflow-pipeline/data/r2g_dev.jsonl \
//!     --target centered --label-b 100 \
//!     --out-checkpoint pixelflow-pipeline/data/guide_checkpoint_r2g_v1.json \
//!     --report-json docs/results/2026-09-01-train-guide-r2g-report.json \
//!     --report-md docs/results/2026-09-01-train-guide-r2g-report.md
//! ```

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader};

use clap::Parser;
use serde::Serialize;

use pixelflow_pipeline::schema::unix_now_s;
use pixelflow_pipeline::training::guide_linear::{
    Model, Sample, op_index_table, rule_index_table, to_sample,
};
use pixelflow_pipeline::training::r2g::{
    LabelBudget, R2gCheckpoint, R2gRecord, RegressionTarget, sample_rank_pairs, target_value,
};
use pixelflow_search::egraph::{RuleId, RuleSet};

#[derive(Parser)]
#[command(name = "train_guide_r2g")]
#[command(about = "Linear return-to-go regressor for the saturation Guide")]
struct Args {
    /// TRAIN-family R2G JSONL (`R2gRecord` schema, §8).
    #[arg(long, default_value = "pixelflow-pipeline/data/r2g_train.jsonl")]
    train: String,

    /// DEV-family R2G JSONL — held out, evaluated only, never trained on.
    #[arg(long, default_value = "pixelflow-pipeline/data/r2g_dev.jsonl")]
    dev: String,

    /// Which label column trains the model: the expression-centered
    /// log-regret (§3.2 primary) or the raw (uncentered) log-regret (§3.4
    /// ablation).
    #[arg(long, default_value = "centered")]
    target: String,

    /// Which registered budget tier's label to train on: 100 (primary) or
    /// 200 (secondary).
    #[arg(long, default_value_t = 100)]
    label_b: u32,

    /// `mse` (§3.2 primary: regress the label directly) or `pairwise-rank`
    /// (§3.4 ablation: cross-trajectory ranking loss).
    #[arg(long, default_value = "mse")]
    loss: String,

    /// Pairs sampled per record per epoch under `--loss pairwise-rank`
    /// (§3.4: "8 pairs per record").
    #[arg(long, default_value_t = 8)]
    pairs_per_record: usize,

    /// Training epochs (full passes over the shuffled TRAIN set).
    #[arg(long, default_value_t = 30)]
    epochs: usize,

    /// Initial learning rate, inverse-time decayed by `--lr-decay` per
    /// epoch.
    #[arg(long, default_value_t = 0.01)]
    lr: f32,

    /// Inverse-time learning-rate decay: `lr_t = lr0 / (1 + lr_decay * t)`.
    #[arg(long, default_value_t = 0.05)]
    lr_decay: f32,

    /// L2 weight decay applied to every weight (not the bias) each step.
    #[arg(long, default_value_t = 1e-4)]
    l2: f32,

    /// Seed for the shuffle (and, under `--loss pairwise-rank`, the
    /// per-epoch pair sampling).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Clip bound on the loss gradient w.r.t. the logit. A return-to-go
    /// label is a log-regret (§1.2) — small in magnitude compared to
    /// `train_guide`'s ~190x class-imbalance pos_weight — so this defaults
    /// far lower than that binary's `--grad-clip`.
    #[arg(long, default_value_t = 10.0)]
    grad_clip: f32,

    /// Evaluate held-out DEV metrics every this many epochs (and at the
    /// last epoch).
    #[arg(long, default_value_t = 5)]
    eval_every: usize,

    /// Where to write the trained checkpoint.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_r2g_v1.json"
    )]
    out_checkpoint: String,

    /// Lower bound (inclusive) on the source expression's arena node count.
    /// `0` = unbounded. Round 3's registered training regime
    /// (docs/plans/2026-09-01-guide-return-to-go.md §2b.3) is node band
    /// 101-1000, i.e. `--min-expr-nodes 101 --max-expr-nodes 1000`: the
    /// cells where guided orderings measurably disagree.
    #[arg(long, default_value_t = 0)]
    min_expr_nodes: usize,

    /// Upper bound (inclusive) on the source expression's arena node count.
    /// `0` = unbounded. See `--min-expr-nodes`.
    #[arg(long, default_value_t = 0)]
    max_expr_nodes: usize,

    /// Drop records whose `application_ordinal` is `>= --label-b`, per the
    /// registration's §1.3: "an application with `t > B` carries no label
    /// for that `B`" — its firing cannot have influenced the return read at
    /// `B`, so training on it attaches the label to a decision that came
    /// after the measurement. The minter attaches the trajectory's return to
    /// EVERY application it records, so this filter (not the mint) is what
    /// enforces §1.3; it is off by default so a pre-round-3 run reproduces
    /// bit-for-bit, and the count it drops is reported, never silent.
    #[arg(long, default_value_t = false)]
    enforce_label_ordinal: bool,

    /// Optional JSON report.
    #[arg(long)]
    report_json: Option<String>,

    /// Optional Markdown report.
    #[arg(long)]
    report_md: Option<String>,
}

// ── Loading ──────────────────────────────────────────────────────────────

/// The record-level selection that defines a training regime
/// (docs/plans/2026-09-01-guide-return-to-go.md §2b.3). One struct rather
/// than four loose arguments so a caller cannot pass the bounds in the wrong
/// order, and so the report can print the regime that actually ran instead
/// of the flags someone believes they passed.
#[derive(Clone, Copy, Debug)]
struct Regime {
    min_expr_nodes: usize,
    max_expr_nodes: usize,
    /// `Some(b)` drops records with `application_ordinal >= b` (§1.3).
    max_label_ordinal: Option<u64>,
}

impl Regime {
    fn admits(&self, r: &R2gRecord) -> bool {
        let n = r.base.expr_node_count;
        if self.min_expr_nodes > 0 && n < self.min_expr_nodes {
            return false;
        }
        if self.max_expr_nodes > 0 && n > self.max_expr_nodes {
            return false;
        }
        !matches!(self.max_label_ordinal, Some(b) if r.application_ordinal >= b)
    }

    fn describe(&self) -> String {
        let lo = if self.min_expr_nodes == 0 {
            "0".to_string()
        } else {
            self.min_expr_nodes.to_string()
        };
        let hi = if self.max_expr_nodes == 0 {
            "inf".to_string()
        } else {
            self.max_expr_nodes.to_string()
        };
        let ord = match self.max_label_ordinal {
            Some(b) => format!("application_ordinal < {b}"),
            None => "application_ordinal unrestricted".to_string(),
        };
        format!("expr_node_count in [{lo}, {hi}], {ord}")
    }
}

struct LoadedSplit {
    records: Vec<R2gRecord>,
    samples: Vec<Sample>,
    /// `target_value(record, label_b, target)` per record, `None` where the
    /// mint recorded no label for this `(label_b, target)` at this record
    /// (§1.3: `t > B` or the expression's `c*_e = 0`).
    targets: Vec<Option<f32>>,
    /// Distinct `Policy` strings observed (§2's ordering-diversity table),
    /// sorted — the "policies used" provenance line item.
    policies: Vec<String>,
    /// FNV-1a 64 content hash of the JSONL file's own bytes — this
    /// codebase's established content-identity mechanism (`schema.rs`),
    /// standing in for the "corpus MD5" provenance line item: it is what
    /// this trainer can compute without shelling out to an external tool,
    /// and it has the same property an MD5 would (any byte change anywhere
    /// in the input changes it).
    source_fnv64: String,
    /// Records read from the file before the regime filter.
    read_total: usize,
    /// Records the regime filter rejected — reported, never silent.
    dropped_by_regime: usize,
}

/// The tables and label selection one `load_jsonl` call needs, grouped
/// rather than passed as six loose arguments (this codebase's style guide:
/// "Functions < 4 arguments").
struct SplitLoad<'a> {
    op_index: &'a HashMap<String, usize>,
    rule_index: &'a HashMap<String, usize>,
    label_b: LabelBudget,
    target: RegressionTarget,
    regime: Regime,
}

fn load_jsonl(path: &str, load: &SplitLoad<'_>) -> LoadedSplit {
    let SplitLoad {
        op_index,
        rule_index,
        label_b,
        target,
        regime,
    } = *load;
    // Streamed, not slurped: round 3's TRAIN mint is ~10 GB of JSONL and the
    // regime filter keeps a small fraction of it, so reading the whole file
    // into a `Vec<u8>` first would cost gigabytes of resident memory to
    // discard most of it. The file's content hash is still over its whole
    // bytes — accumulated chunk by chunk, the same FNV-1a the one-shot
    // helper computes (pinned by `streamed_fnv_matches_the_one_shot_helper`).
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("train_guide_r2g: cannot read {path}: {e}"));
    let mut hasher = StreamingFnv1a64::new();
    let reader = BufReader::new(HashingRead {
        inner: file,
        hasher: &mut hasher,
    });
    let mut records = Vec::new();
    let mut samples = Vec::new();
    let mut targets = Vec::new();
    let mut policies: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut read_total = 0usize;
    let mut dropped_by_regime = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            panic!(
                "train_guide_r2g: I/O error reading {path} at line {}: {e}",
                line_no + 1
            )
        });
        if line.trim().is_empty() {
            continue;
        }
        let record: R2gRecord = serde_json::from_str(&line).unwrap_or_else(|e| {
            panic!(
                "train_guide_r2g: malformed JSONL record in {path} at line {}: {e}",
                line_no + 1
            )
        });
        read_total += 1;
        if !regime.admits(&record) {
            dropped_by_regime += 1;
            continue;
        }
        policies.insert(record.policy.clone());
        samples.push(to_sample(&record.base, op_index, rule_index));
        targets.push(target_value(&record, label_b, target));
        records.push(record);
    }

    assert!(
        !records.is_empty(),
        "train_guide_r2g: {path} contained zero usable records after the regime filter \
         ({}) — {read_total} records read, all {dropped_by_regime} rejected. Refusing to \
         train/evaluate on an empty split",
        regime.describe()
    );
    let labelled = targets.iter().filter(|t| t.is_some()).count();
    assert!(
        labelled > 0,
        "train_guide_r2g: {path} has {} records but none carry a label for the requested \
         (target={target:?}, label_b={}) — every record's t > B or its expression's c*_e = 0",
        records.len(),
        label_b.as_u32()
    );

    LoadedSplit {
        records,
        samples,
        targets,
        policies: policies.into_iter().collect(),
        source_fnv64: hasher.hex(),
        read_total,
        dropped_by_regime,
    }
}

// ── Streaming content hash ──────────────────────────────────────────────

/// FNV-1a 64, fed incrementally. `schema::fnv1a64_hex` is the one-shot form
/// over a slice already in memory; this is the same automaton over a stream,
/// so a 10 GB corpus can be identified without being resident. The two agree
/// by construction and are pinned to agree by a test.
struct StreamingFnv1a64(u64);

impl StreamingFnv1a64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// A `Read` that hashes everything it hands on — so the file is traversed
/// exactly once for both parsing and identity.
struct HashingRead<'a, R> {
    inner: R,
    hasher: &'a mut StreamingFnv1a64,
}

impl<R: std::io::Read> std::io::Read for HashingRead<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.write(&buf[..n]);
        Ok(n)
    }
}

// ── Small stats helpers ─────────────────────────────────────────────────
//
// Kept as local copies rather than lifted into a shared crate module — the
// same call `train_guide`'s own module doc makes for its `spearman`/
// `pearson`: no `pixelflow-pipeline` module holds general-purpose stats
// helpers today, and lifting one for a handful of four-line functions is
// the "abstraction before subtraction has failed" smell this codebase's
// style guide warns against.

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
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

fn spearman(xs: &[f64], ys: &[f64]) -> Option<f64> {
    assert_eq!(xs.len(), ys.len(), "spearman: mismatched lengths");
    if xs.len() < 2 {
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

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        let r = (lcg_next(&mut state) >> 33) as usize % (i + 1);
        idx.swap(i, r);
    }
    idx
}

// ── Report ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EpochPoint {
    epoch: usize,
    lr: f32,
    train_mean_loss: f64,
    dev_mse: Option<f64>,
    dev_spearman: Option<f64>,
}

#[derive(Serialize)]
struct RuleRow {
    /// The rule's stable identity ([`RuleId::get`]) — never a position.
    rule_id: u64,
    rule_name: String,
    dev_fired: usize,
    dev_mean_predicted_return: f64,
    dev_mean_actual_return: f64,
}

#[derive(Serialize)]
struct Report {
    objective: String,
    regression_target: String,
    label_b: u32,
    /// The registered training regime this run's records were selected by
    /// (`Regime::describe`) — printed so a report can never be read as
    /// covering a population it never saw.
    regime: String,
    train_read_total: usize,
    train_dropped_by_regime: usize,
    dev_read_total: usize,
    dev_dropped_by_regime: usize,
    /// MSE of the constant predictor `f ≡ 0` on the same held-out labelled
    /// DEV samples — the floor `dev_mse` must beat to have learned anything.
    /// For a centered target this is the labels' own second moment, so a
    /// model that ties it has learned exactly nothing.
    dev_mse_zero_predictor: f64,
    train_records: usize,
    train_labelled_samples: usize,
    dev_records: usize,
    dev_labelled_samples: usize,
    train_final_loss: f64,
    dev_mse: f64,
    dev_spearman: Option<f64>,
    training_curve: Vec<EpochPoint>,
    per_rule: Vec<RuleRow>,
    checkpoint_path: String,
}

fn write_json_report(path: &str, report: &Report) {
    let json = serde_json::to_string_pretty(report)
        .unwrap_or_else(|e| panic!("train_guide_r2g: report serialization failed: {e}"));
    std::fs::write(path, json)
        .unwrap_or_else(|e| panic!("train_guide_r2g: cannot write {path}: {e}"));
    eprintln!("train_guide_r2g: wrote {path}");
}

fn write_md_report(path: &str, report: &Report) {
    let mut md = String::new();
    md.push_str("# Guide return-to-go training report\n\n");
    md.push_str(&format!(
        "Objective: **{}**. Regression target: **{}** at B = **{}**.\n\n",
        report.objective, report.regression_target, report.label_b
    ));
    md.push_str(&format!(
        "Training regime (docs/plans/2026-09-01-guide-return-to-go.md §2b.3): **{}**.\n\n",
        report.regime
    ));
    md.push_str(&format!(
        "TRAIN: {} records read, {} dropped by the regime, {} kept, {} labelled samples.\n\n",
        report.train_read_total,
        report.train_dropped_by_regime,
        report.train_records,
        report.train_labelled_samples
    ));
    md.push_str(&format!(
        "DEV (held out, never trained on): {} records read, {} dropped by the regime, {} kept, \
         {} labelled samples.\n\n",
        report.dev_read_total,
        report.dev_dropped_by_regime,
        report.dev_records,
        report.dev_labelled_samples
    ));
    md.push_str(&format!(
        "**TRAIN final-epoch mean loss**: {:.6}.\n\n",
        report.train_final_loss
    ));
    md.push_str(&format!(
        "**Zero-predictor floor** (constant `f ≡ 0` on the same DEV samples): MSE = {:.6}. \
         A model at or above this floor has learned nothing.\n\n",
        report.dev_mse_zero_predictor
    ));
    md.push_str(&format!(
        "**Held-out (DEV) regression quality**: MSE = {:.6}, Spearman(predicted return, \
         realized return) = {}.\n\n",
        report.dev_mse,
        report
            .dev_spearman
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "undefined (< 2 labelled DEV samples)".to_string())
    ));

    md.push_str("## Training curve\n\n");
    md.push_str("| epoch | lr | train mean loss | DEV MSE | DEV Spearman |\n");
    md.push_str("|---:|---:|---:|---:|---:|\n");
    for p in &report.training_curve {
        md.push_str(&format!(
            "| {} | {:.5} | {:.6} | {} | {} |\n",
            p.epoch,
            p.lr,
            p.train_mean_loss,
            p.dev_mse.map(|v| format!("{v:.6}")).unwrap_or_default(),
            p.dev_spearman
                .map(|v| format!("{v:.4}"))
                .unwrap_or_default(),
        ));
    }

    md.push_str("\n## Per-rule: DEV mean predicted return vs mean realized return\n\n");
    md.push_str("| rule | id | DEV fired | mean predicted return | mean realized return |\n");
    md.push_str("|---|---:|---:|---:|---:|\n");
    for r in &report.per_rule {
        md.push_str(&format!(
            "| {} | {:016x} | {} | {:.5} | {:.5} |\n",
            r.rule_name,
            r.rule_id,
            r.dev_fired,
            r.dev_mean_predicted_return,
            r.dev_mean_actual_return
        ));
    }

    md.push_str(&format!(
        "\nCheckpoint written to `{}`.\n",
        report.checkpoint_path
    ));

    std::fs::write(path, md)
        .unwrap_or_else(|e| panic!("train_guide_r2g: cannot write {path}: {e}"));
    eprintln!("train_guide_r2g: wrote {path}");
}

// ── main ─────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let label_b: LabelBudget = args
        .label_b
        .to_string()
        .parse()
        .unwrap_or_else(|e| panic!("train_guide_r2g: --label-b: {e}"));
    let target: RegressionTarget = args
        .target
        .parse()
        .unwrap_or_else(|e| panic!("train_guide_r2g: --target: {e}"));
    let pairwise = match args.loss.as_str() {
        "mse" => false,
        "pairwise-rank" => true,
        other => {
            panic!("train_guide_r2g: --loss must be \"mse\" or \"pairwise-rank\", got {other:?}")
        }
    };
    let objective = if pairwise {
        "return-rank"
    } else {
        "return-mse"
    };

    let (op_names, op_index) = op_index_table();
    let num_ops = op_names.len();

    // The rule table is sized and named from the LIVE registered rule set,
    // never from whichever rules these two splits happened to name — see
    // `train_guide`'s own note. Slots are an internal layout; the checkpoint
    // records the label per slot and the vocabulary's fingerprint.
    let rules = RuleSet::production();
    let (rule_names, rule_index) = rule_index_table(&rules);
    let num_rules = rule_names.len();

    let regime = Regime {
        min_expr_nodes: args.min_expr_nodes,
        max_expr_nodes: args.max_expr_nodes,
        max_label_ordinal: args
            .enforce_label_ordinal
            .then(|| u64::from(label_b.as_u32())),
    };
    eprintln!("train_guide_r2g: training regime — {}", regime.describe());
    eprintln!("train_guide_r2g: loading TRAIN from {}", args.train);
    let split_load = SplitLoad {
        op_index: &op_index,
        rule_index: &rule_index,
        label_b,
        target,
        regime,
    };
    let train = load_jsonl(&args.train, &split_load);
    eprintln!(
        "train_guide_r2g: TRAIN {} records read, {} dropped by the regime, {} kept",
        train.read_total,
        train.dropped_by_regime,
        train.records.len()
    );
    eprintln!("train_guide_r2g: loading DEV from {}", args.dev);
    let dev = load_jsonl(&args.dev, &split_load);
    eprintln!(
        "train_guide_r2g: DEV {} records read, {} dropped by the regime, {} kept",
        dev.read_total,
        dev.dropped_by_regime,
        dev.records.len()
    );

    let train_labelled: Vec<usize> = (0..train.records.len())
        .filter(|&i| train.targets[i].is_some())
        .collect();

    eprintln!(
        "train_guide_r2g: TRAIN {} records ({} labelled for target={target:?} label_b={}), \
         objective={objective}",
        train.records.len(),
        train_labelled.len(),
        label_b.as_u32()
    );
    eprintln!(
        "train_guide_r2g: DEV {} records ({} labelled)",
        dev.records.len(),
        dev.targets.iter().filter(|t| t.is_some()).count()
    );

    let mut model = Model::zero(num_rules, num_ops);
    let mut training_curve = Vec::new();
    let mut train_final_loss = 0.0f64;

    for epoch in 0..args.epochs {
        let lr = args.lr / (1.0 + args.lr_decay * epoch as f32);
        let epoch_seed = args.seed.wrapping_add(epoch as u64);

        let mean_loss = if pairwise {
            let pairs =
                sample_rank_pairs(&train.records, label_b, args.pairs_per_record, epoch_seed);
            assert!(
                !pairs.is_empty(),
                "train_guide_r2g: --loss pairwise-rank produced zero training pairs — every \
                 TRAIN expression has at most one trajectory, or every pair tied \
                 (spread_e(B) = 0 everywhere); §2's dataset gate would fire on a real mint \
                 before reaching this point"
            );
            let order = shuffled_indices(pairs.len(), epoch_seed);
            let mut loss_sum = 0.0f64;
            for &pi in &order {
                let pair = &pairs[pi];
                let za = model.logit(&train.samples[pair.a]);
                let zb = model.logit(&train.samples[pair.b]);
                let p = sigmoid(pair.sign * (za - zb));
                loss_sum += -(p.max(1e-12) as f64).ln();
                let grad_shared = -pair.sign * (1.0 - p);
                let grad_a = grad_shared.clamp(-args.grad_clip, args.grad_clip);
                let grad_b = (-grad_shared).clamp(-args.grad_clip, args.grad_clip);
                model.sgd_step(&train.samples[pair.a], grad_a, lr, args.l2);
                model.sgd_step(&train.samples[pair.b], grad_b, lr, args.l2);
            }
            loss_sum / order.len() as f64
        } else {
            let order = shuffled_indices(train_labelled.len(), epoch_seed);
            let mut loss_sum = 0.0f64;
            for &oi in &order {
                let i = train_labelled[oi];
                let s = &train.samples[i];
                let y = train.targets[i].expect("filtered to labelled indices");
                let z = model.logit(s);
                let diff = z - y;
                loss_sum += (diff * diff) as f64;
                let grad_z = (2.0 * diff).clamp(-args.grad_clip, args.grad_clip);
                model.sgd_step(s, grad_z, lr, args.l2);
            }
            loss_sum / order.len() as f64
        };
        train_final_loss = mean_loss;

        let is_last = epoch + 1 == args.epochs;
        let (dev_mse, dev_spearman) =
            if args.eval_every > 0 && (epoch % args.eval_every == 0 || is_last) {
                eval_dev(&model, &dev)
            } else {
                (None, None)
            };

        eprintln!(
            "train_guide_r2g: epoch {epoch:>3} lr={lr:.5} train_mean_loss={mean_loss:.6}{}",
            match (dev_mse, dev_spearman) {
                (Some(m), Some(s)) => format!("  dev_mse={m:.6} dev_spearman={s:.4}"),
                _ => String::new(),
            }
        );

        training_curve.push(EpochPoint {
            epoch,
            lr,
            train_mean_loss: mean_loss,
            dev_mse,
            dev_spearman,
        });
    }

    let (dev_mse, dev_spearman) = eval_dev(&model, &dev);
    let dev_mse = dev_mse.unwrap_or_else(|| {
        panic!("train_guide_r2g: DEV MSE undefined — DEV has zero labelled samples")
    });

    // Per-rule table: DEV mean predicted vs mean realized return.
    let mut agg: BTreeMap<usize, (f64, f64, usize)> = BTreeMap::new();
    for i in 0..dev.records.len() {
        let Some(y) = dev.targets[i] else { continue };
        let rule_slot = dev.samples[i].rule_slot;
        let z = model.logit(&dev.samples[i]);
        let e = agg.entry(rule_slot).or_insert((0.0, 0.0, 0));
        e.0 += z as f64;
        e.1 += y as f64;
        e.2 += 1;
    }
    let mut per_rule_rows: Vec<RuleRow> = agg
        .into_iter()
        .map(|(rule_slot, (pred_sum, actual_sum, n))| RuleRow {
            rule_id: RuleId::from_label(&rule_names[rule_slot]).get(),
            rule_name: rule_names[rule_slot].clone(),
            dev_fired: n,
            dev_mean_predicted_return: pred_sum / n as f64,
            dev_mean_actual_return: actual_sum / n as f64,
        })
        .collect();
    per_rule_rows.sort_by(|a, b| {
        b.dev_mean_predicted_return
            .partial_cmp(&a.dev_mean_predicted_return)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut checkpoint = R2gCheckpoint {
        schema_identity: String::new(),
        label_family: "return-to-go".to_string(),
        regression_target: args.target.clone(),
        label_b: label_b.as_u32(),
        objective: objective.to_string(),
        trainer: "train_guide_r2g".to_string(),
        written_at_unix_s: unix_now_s(),
        seed: args.seed,
        epochs: args.epochs,
        lr_initial: args.lr,
        lr_decay: args.lr_decay,
        l2: args.l2,
        grad_clip: args.grad_clip,
        pairs_per_record: args.pairs_per_record,
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
        train_records: train.records.len(),
        train_labelled_samples: train_labelled.len(),
        train_policies: train.policies.clone(),
        train_source_fnv64: train.source_fnv64.clone(),
        dev_records: dev.records.len(),
        dev_labelled_samples: dev.targets.iter().filter(|t| t.is_some()).count(),
        dev_policies: dev.policies.clone(),
        dev_source_fnv64: dev.source_fnv64.clone(),
        dev_mse,
        dev_spearman,
        weights_fnv64: String::new(),
    };
    checkpoint
        .write(std::path::Path::new(&args.out_checkpoint))
        .unwrap_or_else(|e| panic!("train_guide_r2g: cannot write {}: {e}", args.out_checkpoint));
    eprintln!("train_guide_r2g: wrote checkpoint {}", args.out_checkpoint);

    println!(
        "=== train_guide_r2g: objective={objective} target={} label_b={} ===",
        args.target,
        label_b.as_u32()
    );
    println!(
        "=== TRAIN {} records ({} labelled), final mean loss {:.6} ===",
        train.records.len(),
        train_labelled.len(),
        train_final_loss
    );
    println!(
        "=== held out DEV {} records ({} labelled): MSE {:.6}, Spearman {} ===",
        dev.records.len(),
        dev.targets.iter().filter(|t| t.is_some()).count(),
        dev_mse,
        dev_spearman
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "undefined".to_string())
    );

    let dev_mse_zero_predictor = {
        let ys: Vec<f64> = dev
            .targets
            .iter()
            .filter_map(|t| t.map(f64::from))
            .collect();
        assert!(
            !ys.is_empty(),
            "train_guide_r2g: DEV has zero labelled samples — the zero-predictor floor is \
             undefined and the reported dev_mse would be meaningless"
        );
        ys.iter().map(|y| y * y).sum::<f64>() / ys.len() as f64
    };
    let report = Report {
        objective: objective.to_string(),
        regression_target: args.target.clone(),
        label_b: label_b.as_u32(),
        regime: regime.describe(),
        train_read_total: train.read_total,
        train_dropped_by_regime: train.dropped_by_regime,
        dev_read_total: dev.read_total,
        dev_dropped_by_regime: dev.dropped_by_regime,
        dev_mse_zero_predictor,
        train_records: train.records.len(),
        train_labelled_samples: train_labelled.len(),
        dev_records: dev.records.len(),
        dev_labelled_samples: dev.targets.iter().filter(|t| t.is_some()).count(),
        train_final_loss,
        dev_mse,
        dev_spearman,
        training_curve,
        per_rule: per_rule_rows,
        checkpoint_path: args.out_checkpoint.clone(),
    };

    if let Some(path) = &args.report_json {
        write_json_report(path, &report);
    }
    if let Some(path) = &args.report_md {
        write_md_report(path, &report);
    }
}

/// DEV MSE + Spearman(predicted, realized), pooled over every labelled DEV
/// record — `None`/`None` if DEV has zero labelled samples (MSE), or fewer
/// than 2 (Spearman, undefined below that).
fn eval_dev(model: &Model, dev: &LoadedSplit) -> (Option<f64>, Option<f64>) {
    let mut predicted = Vec::new();
    let mut actual = Vec::new();
    for i in 0..dev.records.len() {
        if let Some(y) = dev.targets[i] {
            predicted.push(model.logit(&dev.samples[i]) as f64);
            actual.push(y as f64);
        }
    }
    if predicted.is_empty() {
        return (None, None);
    }
    let mse = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a) * (p - a))
        .sum::<f64>()
        / predicted.len() as f64;
    (Some(mse), spearman(&predicted, &actual))
}

#[cfg(test)]
mod tests {
    /// The identity regime: every record admitted. Round 3's filters are
    /// exercised by their own tests below; every pre-existing test asserts
    /// the unfiltered behavior it always asserted.
    const UNFILTERED: super::Regime = super::Regime {
        min_expr_nodes: 0,
        max_expr_nodes: 0,
        max_label_ordinal: None,
    };

    use super::*;
    use std::fs::File;
    use std::io::Write as _;

    /// The streaming hash and the one-shot `schema::fnv1a64_hex` are the
    /// same automaton — pinned here because the trainer stopped slurping
    /// its (10 GB) input and the provenance hash must not have silently
    /// changed meaning when it did.
    #[test]
    fn streamed_fnv_matches_the_one_shot_helper() {
        for bytes in [
            b"".as_slice(),
            b"a".as_slice(),
            b"{\"expr_name\":\"e0\"}\n{\"expr_name\":\"e1\"}\n".as_slice(),
        ] {
            let mut h = StreamingFnv1a64::new();
            // Fed in awkward chunks: a streaming hash that only agrees when
            // handed the whole slice at once is not a streaming hash.
            for chunk in bytes.chunks(3) {
                h.write(chunk);
            }
            assert_eq!(h.hex(), pixelflow_pipeline::schema::fnv1a64_hex(bytes));
        }
    }

    /// The regime is a filter on records, and it is the ONLY thing that
    /// enforces §1.3 — the minter attaches a trajectory's return to every
    /// application it records, including ones that fired after `B`.
    #[test]
    fn regime_rejects_out_of_band_sizes_and_post_budget_ordinals() {
        let regime = Regime {
            min_expr_nodes: 101,
            max_expr_nodes: 1000,
            max_label_ordinal: Some(100),
        };
        let json = fixture_row("e0", 0, "unguided", 0, 0.5, 0.25, 0.1);
        let mut r: R2gRecord = serde_json::from_value(json).unwrap();
        r.base.expr_node_count = 150;
        r.application_ordinal = 40;
        assert!(regime.admits(&r), "in band, before B");

        r.base.expr_node_count = 100;
        assert!(!regime.admits(&r), "below the band");
        r.base.expr_node_count = 1001;
        assert!(!regime.admits(&r), "above the band");

        r.base.expr_node_count = 150;
        r.application_ordinal = 100;
        assert!(!regime.admits(&r), "t >= B carries no label for B (§1.3)");

        assert!(
            UNFILTERED.admits(&r),
            "the identity regime admits what the round-3 regime rejects"
        );
    }

    fn write_fixture(dir: &std::path::Path, name: &str, rows: &[serde_json::Value]) -> String {
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        for row in rows {
            writeln!(f, "{}", serde_json::to_string(row).unwrap()).unwrap();
        }
        path.to_str().unwrap().to_string()
    }

    #[allow(clippy::too_many_arguments)]
    /// The canonical label of the `i`-th production rule — the vocabulary a
    /// fixture row has to name for `to_sample` to resolve it.
    fn fixture_rule_label(i: usize) -> String {
        RuleSet::production()
            .label_of(i)
            .expect("fixture rule slot within the production set")
    }

    #[allow(clippy::too_many_arguments)]
    fn fixture_row(
        expr_name: &str,
        trajectory_id: u32,
        policy: &str,
        rule_slot: usize,
        budget_fraction: f32,
        return_b100: f32,
        centered_b100: f32,
    ) -> serde_json::Value {
        // Fixtures name REAL rules: `to_sample` resolves a row's label
        // against the live vocabulary and refuses one it does not have, so a
        // made-up `rule{n}` would fail at load rather than at the assertion
        // the test is actually about.
        let label = fixture_rule_label(rule_slot);
        serde_json::json!({
            "expr_name": expr_name,
            "family_band": 0,
            "family_seed": 0,
            "expr_node_count": 10,
            "rule_id": pixelflow_search::egraph::RuleId::from_label(&label).get(),
            "rule_name": label,
            "budget_fraction": budget_fraction,
            "match_class_node_count": 3,
            "neighborhood_op_count": 1,
            "neighborhood_op_hist": {"Add": 1},
            "dedup_repeat": false,
            "label_positive": rule_slot == 0,
            "trajectory_id": trajectory_id,
            "policy": policy,
            "round_ordinal": 0,
            "application_ordinal": 1,
            "changed": true,
            "cost_b100": 10,
            "cost_b200": 10,
            "expr_best_cost": 5,
            "return_b100": return_b100,
            "return_b200": return_b100,
            "centered_b100": centered_b100,
            "centered_b200": centered_b100,
        })
    }

    // ── sigmoid / pearson / spearman (small helper sanity) ──────────────

    #[test]
    fn sigmoid_is_finite_and_bounded() {
        assert!(sigmoid(50.0) > 0.999);
        assert!(sigmoid(-50.0) < 0.001);
        assert!(sigmoid(0.0) - 0.5 < 1e-6);
    }

    #[test]
    fn spearman_is_one_for_identical_rankings() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        assert!((spearman(&xs, &ys).unwrap() - 1.0).abs() < 1e-9);
    }

    // ── shuffle ──────────────────────────────────────────────────────────

    #[test]
    fn shuffled_indices_is_a_permutation_and_deterministic() {
        let a = shuffled_indices(100, 7);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        assert_eq!(a, shuffled_indices(100, 7));
    }

    // ── load_jsonl / target selection ───────────────────────────────────

    #[test]
    fn load_jsonl_reads_records_computes_targets_and_hashes_the_source() {
        let dir = std::env::temp_dir().join(format!("train_guide_r2g_load_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rows = vec![
            fixture_row("e0", 0, "unguided", 0, 0.1, 0.2, 0.15),
            fixture_row("e0", 1, "random:1", 1, 0.2, 0.4, 0.35),
        ];
        let path = write_fixture(&dir, "fixture.jsonl", &rows);

        let (_, op_index) = op_index_table();
        let (_, rule_index) = rule_index_table(&RuleSet::production());
        let split = load_jsonl(
            &path,
            &SplitLoad {
                op_index: &op_index,
                rule_index: &rule_index,
                label_b: LabelBudget::B100,
                target: RegressionTarget::Centered,
                regime: UNFILTERED,
            },
        );

        assert_eq!(split.records.len(), 2);
        assert_eq!(split.samples.len(), 2);
        assert!((split.targets[0].unwrap() - 0.15).abs() < 1e-6);
        assert!((split.targets[1].unwrap() - 0.35).abs() < 1e-6);
        assert_eq!(
            split.policies,
            vec!["random:1".to_string(), "unguided".to_string()]
        );
        assert!(!split.source_fnv64.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "zero usable records")]
    fn load_jsonl_refuses_an_empty_file() {
        let dir =
            std::env::temp_dir().join(format!("train_guide_r2g_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_fixture(&dir, "empty.jsonl", &[]);
        let (_, op_index) = op_index_table();
        let (_, rule_index) = rule_index_table(&RuleSet::production());
        let _ = load_jsonl(
            &path,
            &SplitLoad {
                op_index: &op_index,
                rule_index: &rule_index,
                label_b: LabelBudget::B100,
                target: RegressionTarget::Centered,
                regime: UNFILTERED,
            },
        );
    }

    // ── end-to-end: a tiny linearly-regressable dataset (MSE) ───────────

    #[test]
    fn a_tiny_synthetic_dataset_trains_a_model_that_beats_a_zero_baseline_on_dev_mse() {
        let dir = std::env::temp_dir().join(format!("train_guide_r2g_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Rule 0's centered_b100 is consistently -0.5, rule 1's is
        // consistently +0.5 — a linear model with a per-rule bias should
        // learn this trivially.
        let mut rows = Vec::new();
        for i in 0..100u32 {
            let (rule_idx, y) = if i % 2 == 0 { (0, -0.5) } else { (1, 0.5) };
            rows.push(fixture_row(
                &format!("e{i}"),
                i,
                "unguided",
                rule_idx,
                0.3,
                y,
                y,
            ));
        }
        let train_path = write_fixture(&dir, "train.jsonl", &rows);
        let dev_path = write_fixture(&dir, "dev.jsonl", &rows);

        let (_, op_index) = op_index_table();
        let (_, rule_index) = rule_index_table(&RuleSet::production());
        let train = load_jsonl(
            &train_path,
            &SplitLoad {
                op_index: &op_index,
                rule_index: &rule_index,
                label_b: LabelBudget::B100,
                target: RegressionTarget::Centered,
                regime: UNFILTERED,
            },
        );
        let dev = load_jsonl(
            &dev_path,
            &SplitLoad {
                op_index: &op_index,
                rule_index: &rule_index,
                label_b: LabelBudget::B100,
                target: RegressionTarget::Centered,
                regime: UNFILTERED,
            },
        );

        let labelled: Vec<usize> = (0..train.records.len()).collect();
        let mut model = Model::zero(2, op_index.len());
        for epoch in 0..150 {
            let lr = 0.02 / (1.0 + 0.05 * epoch as f32);
            let order = shuffled_indices(labelled.len(), 1000 + epoch);
            for &oi in &order {
                let i = labelled[oi];
                let s = &train.samples[i];
                let y = train.targets[i].unwrap();
                let z = model.logit(s);
                let grad_z = (2.0 * (z - y)).clamp(-5.0, 5.0);
                model.sgd_step(s, grad_z, lr, 1e-3);
            }
        }

        let (dev_mse, dev_spearman) = eval_dev(&model, &dev);
        let dev_mse = dev_mse.unwrap();
        // A zero-weight baseline would score MSE = mean(y^2) = 0.25 here;
        // a model that learned the per-rule split should do far better.
        assert!(
            dev_mse < 0.05,
            "expected DEV MSE well under 0.25, got {dev_mse}"
        );
        assert!(dev_spearman.unwrap() > 0.9, "got {dev_spearman:?}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
