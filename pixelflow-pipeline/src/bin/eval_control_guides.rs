//! Zero-candidate-local-information control arm for the cold-start linear
//! Guide (`docs/plans/2026-08-31-guide-design-revision.md` §5, task 2 of the
//! 2026-09-02 cold-start-training round): "implement the CONTROL arm:
//! `PerRuleRateGuide`... Report DEV AUC/PR-AUC for the table AND the linear
//! model side by side — if they are close, say plainly that the model
//! learned little beyond per-rule base rates."
//!
//! # Why this comparison is the one that answers the question
//!
//! The Phase 3 registration doc's pre-flight (§8) already found that
//! rule-granularity oracle filtering barely moves the classical-band curve
//! (94.9% -> 85.4% median regret at B=100). That measured a perfect oracle
//! at rule granularity. This binary asks the analogous question of a
//! *trained* model: does [`LinearCandidateGuide`]'s ranking quality (DEV
//! AUC/PR-AUC, read from `train_guide`'s own report — bit-exact with this
//! deployed type per the mandatory skew test,
//! `bin/skew_test_linear_guide.rs`) meaningfully exceed
//! [`PerRuleRateGuide`]'s — a lookup table with *no* candidate-local
//! information at all, scored on this exact same DEV split by this exact
//! same metric ([`auc_roc`]/[`average_precision`], the identical functions
//! `train_guide` used). A small gap says the linear model mostly learned a
//! per-rule base rate; a real gap is evidence the candidate-local features
//! (design doc §4: neighborhood ops, budget state, matched-class size)
//! earned their place.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin eval_control_guides -- \
//!     --dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --train-guide-report docs/results/2026-09-01-train-guide-report.json \
//!     --report-json docs/results/2026-09-01-control-guide-comparison.json \
//!     --report-md docs/results/2026-09-01-control-guide-comparison.md
//! ```

use std::io::{BufRead, BufReader};

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_search::egraph::RuleId;
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

use pixelflow_pipeline::schema::fnv1a64_hex;
use pixelflow_pipeline::training::guide_linear::{
    Record, auc_roc, average_precision, per_rule_rate_guide_from_report,
};

#[derive(Parser)]
#[command(name = "eval_control_guides")]
#[command(
    about = "Score PerRuleRateGuide's DEV AUC/PR-AUC and compare against train_guide's \
             reported linear-model numbers on the same split"
)]
struct Args {
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_dev.jsonl"
    )]
    dev: String,

    /// `train_guide`'s `--report-json` output — supplies both the per-rule
    /// TRAIN rates [`PerRuleRateGuide`] is built from and the linear
    /// model's own DEV AUC/PR-AUC to compare against.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-train-guide-report.json"
    )]
    train_guide_report: String,

    #[arg(long)]
    report_json: Option<String>,

    #[arg(long)]
    report_md: Option<String>,
}

#[derive(Deserialize)]
struct TrainGuideReport {
    /// The DEV file `train_guide` measured `dev_auc`/`dev_pr_auc` on, and
    /// its content hash. Both are required: subtracting a report's
    /// historical AUCs from AUCs computed here is only a same-split
    /// comparison if it IS the same split, and a regenerated or filtered DEV
    /// file produces a plausible, meaningless gap otherwise.
    dev_path: String,
    dev_fnv64: String,
    dev_samples: usize,
    dev_auc: f64,
    dev_pr_auc: f64,
}

#[derive(Serialize)]
struct ComparisonReport {
    dev_path: String,
    dev_samples_checked: usize,
    linear_dev_auc: f64,
    linear_dev_pr_auc: f64,
    per_rule_rate_dev_auc: f64,
    per_rule_rate_dev_pr_auc: f64,
    auc_gap: f64,
    pr_auc_gap: f64,
    verdict: String,
}

/// Below this AUC-gap, the design doc's own framing calls the two guides
/// "close" — a threshold stated explicitly rather than left to eyeballing
/// the numbers. Not load-bearing for any accept/kill gate (§5/registration
/// doc's gates are about saturation-quality regret, not this ranking
/// metric); this only decides which sentence the report prints.
const CLOSE_AUC_GAP: f64 = 0.02;

fn main() {
    let args = Args::parse();

    let report_json = std::fs::read_to_string(&args.train_guide_report).unwrap_or_else(|e| {
        panic!(
            "eval_control_guides: cannot read {}: {e}",
            args.train_guide_report
        )
    });
    let train_report: TrainGuideReport = serde_json::from_str(&report_json).unwrap_or_else(|e| {
        panic!(
            "eval_control_guides: cannot parse {}: {e}",
            args.train_guide_report
        )
    });

    let dev_bytes = std::fs::read(&args.dev)
        .unwrap_or_else(|e| panic!("eval_control_guides: cannot read {}: {e}", args.dev));
    let dev_fnv64 = fnv1a64_hex(&dev_bytes);
    assert_eq!(
        dev_fnv64, train_report.dev_fnv64,
        "eval_control_guides: --dev {} (fnv {dev_fnv64}) is not the file {} was measured \
         on ({}, fnv {}) — this comparison subtracts that report's linear AUCs from \
         control AUCs computed here, which is only meaningful on one split. Re-run \
         train_guide against this DEV file, or point --dev at the one the report names.",
        args.dev, args.train_guide_report, train_report.dev_path, train_report.dev_fnv64
    );

    let per_rule_guide =
        per_rule_rate_guide_from_report(std::path::Path::new(&args.train_guide_report))
            .unwrap_or_else(|e| panic!("eval_control_guides: {e}"));

    let file = std::fs::File::open(&args.dev)
        .unwrap_or_else(|e| panic!("eval_control_guides: cannot open {}: {e}", args.dev));
    let reader = BufReader::new(file);

    let mut scores = Vec::new();
    let mut labels = Vec::new();
    let mut repeats_excluded = 0usize;
    for line in reader.lines() {
        let line = line.unwrap_or_else(|e| panic!("eval_control_guides: I/O error: {e}"));
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("eval_control_guides: malformed JSONL row: {e}"));

        // Same population `train_guide` trains and evaluates on: the
        // first-seen candidates a deployed `GuidedSaturation` loop actually
        // scores. Including `dedup_repeat` re-fires here while the linear
        // model's reported AUCs exclude them would compare two guides on two
        // different datasets.
        if record.dedup_repeat {
            repeats_excluded += 1;
            continue;
        }

        // PerRuleRateGuide reads only the rule identity (see that type's
        // own doc) — every other field is a don't-care filler.
        let summary = CandidateSummary {
            rule_embed: [0.0; pixelflow_search::nnue::factored::EMBED_DIM],
            neighborhood_ops: Vec::new(),
            budget_fraction: 0.0,
            rule: RuleId::from_label(&record.rule_name),
            match_class_node_count: 0,
            expr_node_count: 0,
        };
        let score = per_rule_guide.score_candidates(std::slice::from_ref(&summary))[0];
        scores.push(score);
        labels.push(if record.label_positive { 1.0 } else { 0.0 });
    }

    assert!(
        !scores.is_empty(),
        "eval_control_guides: {} contained zero first-seen candidate records \
         ({repeats_excluded} dedup_repeat rows were excluded)",
        args.dev
    );
    assert_eq!(
        scores.len(),
        train_report.dev_samples,
        "eval_control_guides: this run scored {} DEV samples but {} reports {} — the two \
         guides' metrics are not over the same records",
        scores.len(),
        args.train_guide_report,
        train_report.dev_samples
    );
    eprintln!(
        "eval_control_guides: {} first-seen candidates scored, {repeats_excluded} \
         dedup_repeat rows excluded",
        scores.len()
    );

    let per_rule_auc = auc_roc(&scores, &labels)
        .unwrap_or_else(|| panic!("eval_control_guides: DEV AUC undefined for PerRuleRateGuide"));
    let per_rule_pr_auc = average_precision(&scores, &labels).unwrap_or_else(|| {
        panic!("eval_control_guides: DEV PR-AUC undefined for PerRuleRateGuide")
    });

    let auc_gap = train_report.dev_auc - per_rule_auc;
    let pr_auc_gap = train_report.dev_pr_auc - per_rule_pr_auc;

    // Three outcomes, not two: the gap is signed, and `auc_gap.abs()`
    // reported a linear model that LOST to the control as evidence that
    // candidate-local features carry signal.
    let verdict = if auc_gap.abs() < CLOSE_AUC_GAP {
        format!(
            "CLOSE (AUC gap {auc_gap:+.4}, |gap| < {CLOSE_AUC_GAP:.2}): the linear model's DEV \
             AUC is barely above a per-rule lookup table's — most of its ranking quality is \
             explained by which rule fired, not by this candidate's own \
             neighborhood/budget/matched-class features. PR-AUC gap is {pr_auc_gap:+.4}."
        )
    } else if auc_gap > 0.0 {
        format!(
            "REAL GAP (AUC gap {auc_gap:+.4} >= {CLOSE_AUC_GAP:.2}): the linear model \
             meaningfully outranks a per-rule lookup table — candidate-local features are \
             carrying signal beyond each rule's base rate. PR-AUC gap is {pr_auc_gap:+.4}."
        )
    } else {
        format!(
            "CONTROL WINS (AUC gap {auc_gap:+.4} <= -{CLOSE_AUC_GAP:.2}): the per-rule lookup \
             table meaningfully OUTRANKS the linear model on held-out DEV — the \
             candidate-local features did not merely fail to help, they cost ranking \
             quality. This is a training/feature failure to investigate, not evidence for \
             candidate-local features. PR-AUC gap is {pr_auc_gap:+.4}."
        )
    };

    println!("=== eval_control_guides ===");
    println!(
        "linear model   (train_guide report): DEV AUC {:.4}, PR-AUC {:.4} ({} samples)",
        train_report.dev_auc, train_report.dev_pr_auc, train_report.dev_samples
    );
    println!(
        "PerRuleRateGuide (this run):          DEV AUC {per_rule_auc:.4}, PR-AUC \
         {per_rule_pr_auc:.4} ({} samples)",
        scores.len()
    );
    println!("{verdict}");

    let report = ComparisonReport {
        dev_path: args.dev.clone(),
        dev_samples_checked: scores.len(),
        linear_dev_auc: train_report.dev_auc,
        linear_dev_pr_auc: train_report.dev_pr_auc,
        per_rule_rate_dev_auc: per_rule_auc,
        per_rule_rate_dev_pr_auc: per_rule_pr_auc,
        auc_gap,
        pr_auc_gap,
        verdict: verdict.clone(),
    };

    if let Some(path) = &args.report_json {
        let json = serde_json::to_string_pretty(&report)
            .unwrap_or_else(|e| panic!("eval_control_guides: report serialization failed: {e}"));
        std::fs::write(path, json)
            .unwrap_or_else(|e| panic!("eval_control_guides: cannot write {path}: {e}"));
        println!("wrote {path}");
    }
    if let Some(path) = &args.report_md {
        let md = format!(
            "# Control-arm comparison: linear Guide vs per-rule rate lookup\n\n\
             DEV split: `{}` ({} samples).\n\n\
             | guide | DEV AUC-ROC | DEV PR-AUC |\n\
             |---|---:|---:|\n\
             | linear model (candidate-local features) | {:.4} | {:.4} |\n\
             | PerRuleRateGuide (rule_idx only, control) | {:.4} | {:.4} |\n\n\
             **Gap**: AUC {:+.4}, PR-AUC {:+.4}.\n\n\
             {}\n",
            args.dev,
            scores.len(),
            train_report.dev_auc,
            train_report.dev_pr_auc,
            per_rule_auc,
            per_rule_pr_auc,
            auc_gap,
            pr_auc_gap,
            verdict
        );
        std::fs::write(path, md)
            .unwrap_or_else(|e| panic!("eval_control_guides: cannot write {path}: {e}"));
        println!("wrote {path}");
    }
}
