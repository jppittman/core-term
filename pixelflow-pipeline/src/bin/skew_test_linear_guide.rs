//! Mandatory train/deploy skew test for the linear saturation Guide heads:
//! the strict-bit classifier (`docs/plans/2026-08-31-guide-design-revision.md`
//! §5, task 2 of the 2026-09-02 cold-start-training round: "for >=1000 DEV
//! records, the trainer's forward and the deployed impl's score_candidates
//! agree to <=1e-6") and, as of `--model return`, the return-to-go
//! regressor (`docs/plans/2026-09-01-guide-return-to-go.md` §3.3: "for
//! >=1000 DEV records, `|f_trainer + score_deployed| <= 1e-6`").
//!
//! # What this checks, and why it is two independent code paths
//!
//! `train_guide`/`train_guide_r2g` train
//! `pixelflow_pipeline::training::guide_linear::Model` directly against
//! [`Record`]/[`Sample`] (JSONL fields in, a logit out — the formula is
//! objective-agnostic; only the label it was fit to differs). Deploy-side,
//! `pixelflow_search::nnue::guide::linear`'s [`LinearCandidateGuide`] /
//! [`LinearReturnGuide`] are *separate* implementations, in a different
//! crate, that load the exact same JSON checkpoint and compute a score from
//! a [`CandidateSummary`] — the type the real guided-saturation loop
//! (`saturate_guided_until_applications`) actually calls
//! `SaturationGuide::score_candidates` with. This binary is the only place
//! that runs *both* paths on the *same* input and checks they agree: it is
//! the difference between "we trained a model" and "the model we trained is
//! the model that gets deployed" — exactly the train/deploy skew class of
//! bug that invalidated Round 1 of the extraction-head program before any
//! evaluation ran (per this round's own task brief).
//!
//! `--model strict` (default) replays `strict_labels_dev.jsonl` against
//! [`LinearCandidateGuide`], comparing `Model::logit` to
//! `score_candidates` directly (both already in "bigger is better"
//! orientation, so no sign to account for). `--model return` replays an
//! `R2gRecord` JSONL (`docs/plans/2026-09-01-guide-return-to-go.md` §8's
//! schema) against [`LinearReturnGuide`], whose `score_candidates` is
//! `-predicted_return` (that type's own doc, "The sign, spelled out") — so
//! the bit-exactness check there is `|trainer_logit + score_deployed| <=
//! tol`, the `+` accounting for the deliberate sign flip rather than
//! hiding it.
//!
//! `rule_embed` is filled with an arbitrary constant when building each
//! [`CandidateSummary`] here: neither deployed Guide ever reads it (see
//! their own docs), so its value cannot affect the comparison — using a
//! fixed non-zero constant rather than all-zero is a small extra check that
//! the deployed impl really does ignore it, not silently drop the
//! rule_embed lane for some other reason.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin skew_test_linear_guide -- \
//!     --model strict \
//!     --dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --checkpoint pixelflow-pipeline/data/guide_checkpoint_strict_v1.json \
//!     --report-json docs/results/2026-09-01-skew-test-linear-guide.json
//!
//! cargo run --release -p pixelflow-pipeline --features training --bin skew_test_linear_guide -- \
//!     --model return \
//!     --dev pixelflow-pipeline/data/r2g_dev.jsonl \
//!     --checkpoint pixelflow-pipeline/data/guide_checkpoint_r2g_v1.json \
//!     --target centered --label-b 100 \
//!     --report-json docs/results/2026-09-01-skew-test-r2g.json
//! ```

use std::io::{BufRead, BufReader};

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{RuleId, RuleSet};
use pixelflow_search::nnue::factored::EMBED_DIM;
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

use pixelflow_pipeline::training::guide_linear::{
    Model, Record, load_linear_guide, op_index_table, to_sample,
};
use pixelflow_pipeline::training::r2g::{
    LabelBudget, R2gRecord, RegressionTarget, load_return_guide, target_value,
};

#[derive(Parser)]
#[command(name = "skew_test_linear_guide")]
#[command(
    about = "Verify a trainer's forward pass and its deployed Guide's score_candidates agree \
             bit-exactly on the same checkpoint and DEV records"
)]
struct Args {
    /// Which head to check: `strict` (train_guide / LinearCandidateGuide,
    /// default) or `return` (train_guide_r2g / LinearReturnGuide).
    #[arg(long, default_value = "strict")]
    model: String,

    /// DEV-family JSONL to replay — `strict_labels_dev.jsonl`'s schema
    /// under `--model strict`, an `R2gRecord` JSONL under `--model return`.
    /// No hardcoded default (the two modes read different schemas from
    /// different default files); required.
    #[arg(long)]
    dev: String,

    /// Checkpoint written by `train_guide` (`--model strict`) or
    /// `train_guide_r2g` (`--model return`).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    checkpoint: String,

    /// `--model return` only: which label column the checkpoint was
    /// trained on (must match how it was trained; not re-derived from the
    /// checkpoint since a mismatch there is exactly the kind of skew this
    /// test exists to catch).
    #[arg(long, default_value = "centered")]
    target: String,

    /// `--model return` only: which registered tier's label to check
    /// against.
    #[arg(long, default_value_t = 100)]
    label_b: u32,

    /// Number of leading DEV records to check. Design doc mandates >= 1000;
    /// refused below that (a skew test that ran on too few records to
    /// matter is not the mandatory test).
    #[arg(long, default_value_t = 5000)]
    n: usize,

    /// Bit-exactness tolerance on `|trainer_logit - deployed_logit|`
    /// (`--model strict`) or `|trainer_logit + score_deployed|`
    /// (`--model return`).
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
    /// `w_rule` slot -> canonical rule label. Read from the CHECKPOINT, not
    /// from the live rule set: this side of the skew test is a deliberately
    /// independent parse, and a checkpoint whose slot layout disagrees with
    /// the live vocabulary must surface here as a scoring difference rather
    /// than be silently re-indexed into agreement.
    rule_names: Vec<String>,
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
    model: String,
    checkpoint: String,
    dev_path: String,
    records_checked: usize,
    tol: f32,
    max_abs_diff: f64,
    mean_abs_diff: f64,
    exceeding_tol: usize,
    passed: bool,
}

/// Parse the checkpoint's op-name table and build the trainer-side `Model`
/// from it. Shared by both `--model` modes: the weight layout is identical
/// (`pixelflow_search::nnue::guide::linear::LinearWeights`'s formula, one
/// definition on the deploy side, one on the trainer side — see module
/// doc), only the label a checkpoint was fit to differs.
fn load_trainer_model(checkpoint_path: &str) -> (Model, std::collections::HashMap<String, usize>) {
    let weights_json = std::fs::read_to_string(checkpoint_path).unwrap_or_else(|e| {
        panic!("skew_test_linear_guide: cannot read checkpoint {checkpoint_path}: {e}")
    });
    let weights: CheckpointWeights = serde_json::from_str(&weights_json).unwrap_or_else(|e| {
        panic!("skew_test_linear_guide: cannot parse checkpoint {checkpoint_path}: {e}")
    });

    let (op_names, _) = op_index_table();
    assert_eq!(
        op_names.len(),
        weights.op_names.len(),
        "skew_test_linear_guide: this binary's OpKind table has {} entries, checkpoint \
         {checkpoint_path} has {} — built against a different pixelflow-ir revision than the \
         checkpoint was trained with; retrain or rebuild before running the skew test",
        op_names.len(),
        weights.op_names.len()
    );
    for (i, (built, stored)) in op_names.iter().zip(weights.op_names.iter()).enumerate() {
        assert_eq!(
            built, stored,
            "skew_test_linear_guide: op index {i} disagrees between this binary's \
             OpKind::all() order ({built:?}) and the checkpoint's op_names ({stored:?})"
        );
    }

    let rule_index: std::collections::HashMap<String, usize> = weights
        .rule_names
        .iter()
        .enumerate()
        .map(|(i, l)| (l.clone(), i))
        .collect();
    assert_eq!(
        rule_index.len(),
        weights.rule_names.len(),
        "skew_test_linear_guide: checkpoint {checkpoint_path} repeats a rule label in \
         rule_names — the w_rule slot a label names would be ambiguous"
    );

    (
        Model {
            bias: weights.bias,
            w_rule: weights.w_rule,
            w_op: weights.w_op,
            w_budget: weights.w_budget,
            w_match_class: weights.w_match_class,
            w_neighborhood: weights.w_neighborhood,
            w_expr_size: weights.w_expr_size,
        },
        rule_index,
    )
}

fn open_dev_lines(dev_path: &str, n: usize) -> impl Iterator<Item = String> {
    let file = std::fs::File::open(dev_path)
        .unwrap_or_else(|e| panic!("skew_test_linear_guide: cannot open {dev_path}: {e}"));
    BufReader::new(file)
        .lines()
        .take(n)
        .map(|l| l.unwrap_or_else(|e| panic!("skew_test_linear_guide: I/O error: {e}")))
        .filter(|l| !l.trim().is_empty())
}

/// Everything one `--model` mode's pass over DEV accumulated, plus the
/// identifying strings the report/panic messages need — grouped into one
/// struct (rather than nine loose parameters) per this codebase's style
/// guide ("Functions < 4 arguments").
struct SkewRun<'a> {
    model_name: &'a str,
    checkpoint: &'a str,
    dev: &'a str,
    tol: f32,
    checked: usize,
    max_abs_diff: f64,
    sum_abs_diff: f64,
    exceeding_tol: usize,
    report_json: &'a Option<String>,
}

fn report_and_check(run: SkewRun<'_>) {
    let SkewRun {
        model_name,
        checkpoint,
        dev,
        tol,
        checked,
        max_abs_diff,
        sum_abs_diff,
        exceeding_tol,
        report_json,
    } = run;

    assert!(
        checked >= 1000,
        "skew_test_linear_guide: only {checked} usable DEV records in {dev} — need >= 1000 for \
         the mandatory skew test to mean anything"
    );

    let mean_abs_diff = sum_abs_diff / checked as f64;
    let passed = exceeding_tol == 0;

    println!(
        "=== skew_test_linear_guide --model {model_name}: {checked} DEV records, tol={tol:.2e} ==="
    );
    println!("max |trainer - deployed| = {max_abs_diff:.3e}");
    println!("mean |trainer - deployed| = {mean_abs_diff:.3e}");
    println!("records exceeding tol = {exceeding_tol}/{checked}");
    println!("RESULT: {}", if passed { "PASS" } else { "FAIL" });

    if let Some(path) = report_json {
        let report = SkewReport {
            model: model_name.to_string(),
            checkpoint: checkpoint.to_string(),
            dev_path: dev.to_string(),
            records_checked: checked,
            tol,
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
        "skew_test_linear_guide --model {model_name}: FAILED — {exceeding_tol}/{checked} DEV \
         records exceeded tol={tol:.2e} (max diff {max_abs_diff:.3e}) — the deployed Guide does \
         not score the same model the trainer trained; do not wire it into a guided saturation \
         loop until this passes"
    );
}

/// `--model strict`: `train_guide` vs `LinearCandidateGuide`. Both already
/// share "bigger is better" orientation, so the comparison is a direct
/// difference — see module doc.
fn run_strict(args: &Args) {
    let (model, rule_index) = load_trainer_model(&args.checkpoint);
    let guide = load_linear_guide(
        std::path::Path::new(&args.checkpoint),
        &RuleSet::production(),
    )
    .unwrap_or_else(|e| panic!("skew_test_linear_guide: {e}"));
    let (_, op_index) = op_index_table();

    let mut max_abs_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut exceeding_tol = 0usize;
    let mut checked = 0usize;

    for line in open_dev_lines(&args.dev, args.n) {
        let record: Record = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("skew_test_linear_guide: malformed JSONL row: {e}"));

        let sample = to_sample(&record, &op_index, &rule_index);
        let trainer_logit = model.logit(&sample);

        let summary = CandidateSummary {
            rule_embed: [0.13; EMBED_DIM],
            neighborhood_ops: ops_from_histogram(&record.neighborhood_op_hist),
            budget_fraction: record.budget_fraction,
            rule: RuleId::from_label(&record.rule_name),
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

    report_and_check(SkewRun {
        model_name: "strict",
        checkpoint: &args.checkpoint,
        dev: &args.dev,
        tol: args.tol,
        checked,
        max_abs_diff,
        sum_abs_diff,
        exceeding_tol,
        report_json: &args.report_json,
    });
}

/// `--model return`: `train_guide_r2g` vs `LinearReturnGuide`. The deployed
/// score is `-predicted_return` (that type's own doc), so the bit-exactness
/// check is `|trainer_logit + score_deployed| <= tol` — the plan's own
/// framing (§3.3) — rather than a bare difference, so the deliberate sign
/// flip is part of what is being checked, not papered over.
fn run_return(args: &Args) {
    let (model, rule_index) = load_trainer_model(&args.checkpoint);
    let guide = load_return_guide(
        std::path::Path::new(&args.checkpoint),
        &RuleSet::production(),
    )
    .unwrap_or_else(|e| panic!("skew_test_linear_guide: {e}"));
    let (_, op_index) = op_index_table();

    let label_b: LabelBudget = args
        .label_b
        .to_string()
        .parse()
        .unwrap_or_else(|e| panic!("skew_test_linear_guide: --label-b: {e}"));
    let target: RegressionTarget = args
        .target
        .parse()
        .unwrap_or_else(|e| panic!("skew_test_linear_guide: --target: {e}"));

    let mut max_abs_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut exceeding_tol = 0usize;
    let mut checked = 0usize;
    let mut skipped_unlabelled = 0usize;

    for line in open_dev_lines(&args.dev, args.n) {
        let record: R2gRecord = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("skew_test_linear_guide: malformed JSONL row: {e}"));

        // A record with no label for this (target, label_b) carries no
        // trained-against value to compare against — skip it rather than
        // comparing against a fabricated one, and do not count it toward
        // `--n`'s budget of *checked* records.
        if target_value(&record, label_b, target).is_none() {
            skipped_unlabelled += 1;
            continue;
        }

        let sample = to_sample(&record.base, &op_index, &rule_index);
        let trainer_logit = model.logit(&sample);

        let summary = CandidateSummary {
            rule_embed: [0.13; EMBED_DIM],
            neighborhood_ops: ops_from_histogram(&record.base.neighborhood_op_hist),
            budget_fraction: record.base.budget_fraction,
            rule: RuleId::from_label(&record.base.rule_name),
            match_class_node_count: record.base.match_class_node_count,
            expr_node_count: record.base.expr_node_count,
        };
        let deployed_score = guide.score_candidates(std::slice::from_ref(&summary))[0];

        // §3.3: |f_trainer + score_deployed| <= tol (score_deployed =
        // -predicted_return, so this is trainer_logit - predicted_return
        // under the deliberate sign flip).
        let diff = (trainer_logit + deployed_score).abs() as f64;
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
        if diff as f32 > args.tol {
            exceeding_tol += 1;
        }
        checked += 1;
    }

    if skipped_unlabelled > 0 {
        eprintln!(
            "skew_test_linear_guide --model return: skipped {skipped_unlabelled} DEV records \
             with no label for target={target:?} label_b={} (t > B or c*_e = 0)",
            label_b.as_u32()
        );
    }

    report_and_check(SkewRun {
        model_name: "return",
        checkpoint: &args.checkpoint,
        dev: &args.dev,
        tol: args.tol,
        checked,
        max_abs_diff,
        sum_abs_diff,
        exceeding_tol,
        report_json: &args.report_json,
    });
}

fn main() {
    let args = Args::parse();
    assert!(
        args.n >= 1000,
        "skew_test_linear_guide: --n must be >= 1000 (the mandatory skew test's bar); got {}",
        args.n
    );

    match args.model.as_str() {
        "strict" => run_strict(&args),
        "return" => run_return(&args),
        other => panic!(
            "skew_test_linear_guide: --model must be \"strict\" or \"return\", got {other:?}"
        ),
    }
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
