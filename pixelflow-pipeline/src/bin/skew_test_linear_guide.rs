//! Mandatory train/deploy skew test for the cold-start linear saturation
//! Guide (`docs/plans/2026-08-31-guide-design-revision.md` §5, task 2 of the
//! 2026-09-02 cold-start-training round: "for >=1000 DEV records, the
//! trainer's forward and the deployed impl's score_candidates agree to
//! <=1e-6").
//!
//! # What this checks, and why it is two independent code paths
//!
//! `train_guide` trains `pixelflow_pipeline::training::guide_linear::Model`
//! directly against [`Record`]/[`Sample`] (JSONL fields in, a logit out).
//! `pixelflow_search::nnue::guide::linear::LinearCandidateGuide` is a
//! *separate* implementation, in a different crate, that loads the exact
//! same JSON checkpoint and computes a score from a [`CandidateSummary`] —
//! the type the real guided-saturation loop
//! (`saturate_guided_until_applications`) actually calls
//! `SaturationGuide::score_candidates` with. This binary is the only place
//! that runs *both* paths on the *same* input and checks they agree: it is
//! the difference between "we trained a model" and "the model we trained is
//! the model that gets deployed" — exactly the train/deploy skew class of
//! bug that invalidated Round 1 of the extraction-head program before any
//! evaluation ran (per this round's own task brief).
//!
//! `rule_embed` is filled with an arbitrary constant when building each
//! [`CandidateSummary`] here: [`LinearCandidateGuide`] never reads it (see
//! that type's own doc), so its value cannot affect the comparison — using
//! a fixed non-zero constant rather than all-zero is a small extra check
//! that the deployed impl really does ignore it, not silently drop the
//! rule_embed lane for some other reason.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin skew_test_linear_guide -- \
//!     --dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --checkpoint pixelflow-pipeline/data/guide_checkpoint_strict_v1.json \
//!     --report-json docs/results/2026-09-01-skew-test-linear-guide.json
//! ```

use std::io::{BufRead, BufReader};

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_ir::OpKind;
use pixelflow_search::nnue::factored::EMBED_DIM;
use pixelflow_search::nnue::guide::linear::LinearCandidateGuide;
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

use pixelflow_pipeline::training::guide_linear::{Model, Record, op_index_table, to_sample};

#[derive(Parser)]
#[command(name = "skew_test_linear_guide")]
#[command(
    about = "Verify train_guide's forward pass and LinearCandidateGuide's score_candidates \
             agree bit-exactly on the same checkpoint and DEV records"
)]
struct Args {
    /// DEV-family strict-label JSONL to replay.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_dev.jsonl"
    )]
    dev: String,

    /// Checkpoint written by `train_guide`.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    checkpoint: String,

    /// Number of leading DEV records to check. Design doc mandates >= 1000;
    /// refused below that (a skew test that ran on too few records to
    /// matter is not the mandatory test).
    #[arg(long, default_value_t = 5000)]
    n: usize,

    /// Bit-exactness tolerance on `|trainer_logit - deployed_logit|`.
    #[arg(long, default_value_t = 1e-6)]
    tol: f32,

    /// Optional JSON report (max/mean diff, pass/fail, sample count).
    #[arg(long)]
    report_json: Option<String>,
}

/// Checkpoint fields `Model` needs, read independently of
/// [`LinearCandidateGuide::load`] — see module doc for why this and the
/// deployed loader are deliberately two separate parses of the same file.
#[derive(Deserialize)]
struct CheckpointWeights {
    bias: f32,
    w_rule: Vec<f32>,
    w_op: Vec<f32>,
    w_budget: f32,
    w_match_class: f32,
    w_neighborhood: f32,
    w_expr_size: f32,
    op_names: Vec<String>,
}

/// Rebuild a `Vec<OpKind>` whose per-op occurrence counts match a JSONL
/// row's `neighborhood_op_hist` exactly — the input `op_features` (inside
/// `LinearCandidateGuide`) re-groups and counts, so any ordering here is
/// fine as long as the counts round-trip.
fn ops_from_histogram(hist: &std::collections::BTreeMap<String, usize>) -> Vec<OpKind> {
    let by_name: std::collections::HashMap<String, OpKind> =
        OpKind::all().map(|op| (format!("{op:?}"), op)).collect();
    let mut ops = Vec::new();
    for (name, &count) in hist {
        let op = *by_name.get(name).unwrap_or_else(|| {
            panic!(
                "skew_test_linear_guide: neighborhood_op_hist names op {name:?}, not in \
                 OpKind::all() — dataset/binary OpKind table mismatch"
            )
        });
        ops.extend(std::iter::repeat_n(op, count));
    }
    ops
}

#[derive(Serialize)]
struct SkewReport {
    checkpoint: String,
    dev_path: String,
    records_checked: usize,
    tol: f32,
    max_abs_diff: f64,
    mean_abs_diff: f64,
    exceeding_tol: usize,
    passed: bool,
}

fn main() {
    let args = Args::parse();
    assert!(
        args.n >= 1000,
        "skew_test_linear_guide: --n must be >= 1000 (design doc §5 task 2's mandatory bar); \
         got {}",
        args.n
    );

    let weights_json = std::fs::read_to_string(&args.checkpoint).unwrap_or_else(|e| {
        panic!(
            "skew_test_linear_guide: cannot read checkpoint {}: {e}",
            args.checkpoint
        )
    });
    let weights: CheckpointWeights = serde_json::from_str(&weights_json).unwrap_or_else(|e| {
        panic!(
            "skew_test_linear_guide: cannot parse checkpoint {}: {e}",
            args.checkpoint
        )
    });

    let (op_names, op_index) = op_index_table();
    assert_eq!(
        op_names.len(),
        weights.op_names.len(),
        "skew_test_linear_guide: this binary's OpKind table has {} entries, checkpoint {} \
         has {} — built against a different pixelflow-ir revision than the checkpoint was \
         trained with; retrain or rebuild before running the skew test",
        op_names.len(),
        args.checkpoint,
        weights.op_names.len()
    );
    for (i, (built, stored)) in op_names.iter().zip(weights.op_names.iter()).enumerate() {
        assert_eq!(
            built, stored,
            "skew_test_linear_guide: op index {i} disagrees between this binary's \
             OpKind::all() order ({built:?}) and the checkpoint's op_names ({stored:?})"
        );
    }

    let model = Model {
        bias: weights.bias,
        w_rule: weights.w_rule,
        w_op: weights.w_op,
        w_budget: weights.w_budget,
        w_match_class: weights.w_match_class,
        w_neighborhood: weights.w_neighborhood,
        w_expr_size: weights.w_expr_size,
    };

    let rules = pixelflow_search::egraph::all_rules();
    let rule_names: Vec<&str> = rules.iter().map(|r| r.name()).collect();
    let guide = LinearCandidateGuide::load(std::path::Path::new(&args.checkpoint), &rule_names)
        .unwrap_or_else(|e| panic!("skew_test_linear_guide: {e}"));

    let file = std::fs::File::open(&args.dev)
        .unwrap_or_else(|e| panic!("skew_test_linear_guide: cannot open {}: {e}", args.dev));
    let reader = BufReader::new(file);

    let mut max_abs_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut exceeding_tol = 0usize;
    let mut checked = 0usize;

    for line in reader.lines().take(args.n) {
        let line = line.unwrap_or_else(|e| panic!("skew_test_linear_guide: I/O error: {e}"));
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("skew_test_linear_guide: malformed JSONL row: {e}"));

        let sample = to_sample(&record, &op_index);
        let trainer_logit = model.logit(&sample);

        let summary = CandidateSummary {
            rule_embed: [0.13; EMBED_DIM],
            neighborhood_ops: ops_from_histogram(&record.neighborhood_op_hist),
            budget_fraction: record.budget_fraction,
            rule_idx: record.rule_idx,
            match_class_node_count: record.match_class_node_count,
            expr_node_count: record.expr_node_count,
        };
        let deployed_logit = guide.score_candidates(std::slice::from_ref(&summary))[0];

        let diff = (trainer_logit - deployed_logit).abs() as f64;
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
        if diff as f32 > args.tol {
            exceeding_tol += 1;
        }
        checked += 1;
    }

    assert!(
        checked >= 1000,
        "skew_test_linear_guide: only {checked} usable DEV records in {} — need >= 1000 for \
         the mandatory skew test to mean anything",
        args.dev
    );

    let mean_abs_diff = sum_abs_diff / checked as f64;
    let passed = exceeding_tol == 0;

    println!(
        "=== skew_test_linear_guide: {checked} DEV records, tol={:.2e} ===",
        args.tol
    );
    println!("max |trainer - deployed| = {max_abs_diff:.3e}");
    println!("mean |trainer - deployed| = {mean_abs_diff:.3e}");
    println!("records exceeding tol = {exceeding_tol}/{checked}");
    println!("RESULT: {}", if passed { "PASS" } else { "FAIL" });

    if let Some(path) = &args.report_json {
        let report = SkewReport {
            checkpoint: args.checkpoint.clone(),
            dev_path: args.dev.clone(),
            records_checked: checked,
            tol: args.tol,
            max_abs_diff,
            mean_abs_diff,
            exceeding_tol,
            passed,
        };
        let json = serde_json::to_string_pretty(&report)
            .unwrap_or_else(|e| panic!("skew_test_linear_guide: report serialization failed: {e}"));
        std::fs::write(path, json)
            .unwrap_or_else(|e| panic!("skew_test_linear_guide: cannot write {path}: {e}"));
        println!("wrote {path}");
    }

    assert!(
        passed,
        "skew_test_linear_guide: FAILED — {exceeding_tol}/{checked} DEV records exceeded \
         tol={:.2e} (max diff {max_abs_diff:.3e}) — LinearCandidateGuide does not deploy the \
         same model train_guide trained; do not wire it into a guided saturation loop until \
         this passes",
        args.tol
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_from_histogram_round_trips_per_op_counts() {
        let mut hist = std::collections::BTreeMap::new();
        hist.insert("Add".to_string(), 3usize);
        hist.insert("Mul".to_string(), 1usize);
        let ops = ops_from_histogram(&hist);
        assert_eq!(ops.iter().filter(|&&o| o == OpKind::Add).count(), 3);
        assert_eq!(ops.iter().filter(|&&o| o == OpKind::Mul).count(), 1);
        assert_eq!(ops.len(), 4);
    }

    #[test]
    #[should_panic(expected = "not in OpKind::all()")]
    fn ops_from_histogram_panics_loudly_on_an_unknown_op_name() {
        let mut hist = std::collections::BTreeMap::new();
        hist.insert("NotARealOp".to_string(), 1usize);
        ops_from_histogram(&hist);
    }
}
