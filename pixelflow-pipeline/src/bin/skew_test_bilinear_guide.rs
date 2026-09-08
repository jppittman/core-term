//! Mandatory train/deploy skew test for the **bilinear** saturation Guide
//! (`docs/plans/2026-09-02-bilinear-guide-registration.md` §9: "for >= 1000
//! DEV records, the trainer's forward and the deployed impl's
//! `score_candidates` agree to <= 1e-6", and "no guided evaluation run may be
//! started before that test passes").
//!
//! # What the risk actually is here, and why this test is not circular
//!
//! For the additive head, `train_guide` and `LinearCandidateGuide` are two
//! independent transcriptions of one formula, and `skew_test_linear_guide`
//! checks the transcriptions agree. That is not the shape of the risk for
//! this head, and pretending otherwise would produce a test that looks
//! stronger and checks less. `BilinearTrainer` and `BilinearCandidateGuide`
//! both route through `SaturationHead::score_candidate` **on purpose** — a
//! hand-written second copy of a four-layer forward pass with a bilinear top
//! would be a new bug surface, not a control.
//!
//! What *can* differ between the trained model and the deployed one, and is
//! what this binary checks:
//!
//! - **The checkpoint boundary.** 16,000-odd floats are flattened into one
//!   vector by a visitor and read back by the same visitor into a *freshly
//!   zeroed* head. A tensor missing from that order, a transposed row-major
//!   assumption, or a float that does not survive JSON would all deploy
//!   silently as a different model.
//! - **The derived rule embeddings.** The checkpoint deliberately does not
//!   store them: the trainer computes `rule_proj(concat(templates))` and so
//!   does `BilinearCandidateGuide::new`, from templates rebuilt on the
//!   deploy side. If those two derivations ever disagree — a different
//!   traversal order, a different depth binding — every score is wrong by a
//!   rule-dependent amount, and nothing else would notice.
//! - **The op embeddings.** They are frozen, carried in the checkpoint, and
//!   reloaded into an `OpEmbeddings` by `OpKind::all()` order. An off-by-one
//!   there shifts every op's embedding by one and still produces plausible
//!   scores.
//!
//! Each of those is a whole-model corruption that a loss curve cannot see
//! and a saturation run would attribute to the model class.
//!
//! # The one direction this binary structurally cannot check, and who does
//!
//! Loading a checkpoint into a `BilinearTrainer` and into a
//! `BilinearCandidateGuide` compares the two *reading* paths. It cannot see
//! a bug in the *writing* path — a tensor missing from the flatten order is
//! missing from both reads identically. That direction is checked where the
//! only in-memory trained head exists: `train_guide_bilinear` compares its
//! own post-training `BilinearTrainer::score` against the guide it builds
//! from the file it just wrote, over the same >= 1000 DEV records and the
//! same tolerance, and hard-fails before writing a report. This binary is
//! the committed artifact for the reading half; that assertion is the
//! writing half, and neither substitutes for the other.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin skew_test_bilinear_guide -- \
//!     --dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --checkpoint pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json \
//!     --report-json docs/results/2026-09-02-skew-test-bilinear-guide.json
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};

use clap::Parser;
use serde::Serialize;

use pixelflow_pipeline::training::guide_bilinear::{BilinearGuideCheckpoint, BilinearSample};
use pixelflow_pipeline::training::guide_linear::Record;
use pixelflow_search::egraph::RuleSet;
use pixelflow_search::nnue::guide::SaturationGuide;
use pixelflow_search::nnue::guide::bilinear::{BilinearCandidateGuide, BilinearTrainer};

#[derive(Parser)]
#[command(name = "skew_test_bilinear_guide")]
#[command(about = "Mandatory bit-exact train/deploy skew test for the bilinear Guide")]
struct Args {
    /// DEV-family strict-label JSONL to replay.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_dev.jsonl"
    )]
    dev: String,

    /// The trained checkpoint under test.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json"
    )]
    checkpoint: String,

    /// How many DEV records to check. The registration's bar is >= 1000 and
    /// this binary refuses less.
    #[arg(long, default_value_t = 5000)]
    n: usize,

    /// Agreement tolerance.
    #[arg(long, default_value_t = 1e-6)]
    tol: f32,

    /// Optional JSON artifact, committed alongside the results.
    #[arg(long)]
    report_json: Option<String>,
}

#[derive(Serialize)]
struct SkewReport {
    model: String,
    checkpoint: String,
    checkpoint_weights_fnv64: String,
    dev_path: String,
    records_checked: usize,
    tol: f32,
    max_abs_diff: f64,
    mean_abs_diff: f64,
    exceeding_tol: usize,
    /// How many of the checked records scored to a value that is not
    /// bit-identical between the two sides. Reported next to the tolerance
    /// because this head's two sides *should* be bit-identical, not merely
    /// close: anything above zero means a real difference the tolerance
    /// happened to absorb.
    not_bit_identical: usize,
    /// Rules whose deployed embedding differs from the trainer's, by name.
    /// Must be empty.
    rule_embed_mismatches: Vec<String>,
    passed: bool,
}

fn main() {
    let args = Args::parse();
    assert!(
        args.n >= 1000,
        "skew_test_bilinear_guide: --n must be >= 1000 (the registration's bar); got {}",
        args.n
    );

    let rules = RuleSet::production();
    let text = std::fs::read_to_string(&args.checkpoint).unwrap_or_else(|e| {
        panic!(
            "skew_test_bilinear_guide: cannot read {}: {e}",
            args.checkpoint
        )
    });
    let checkpoint: BilinearGuideCheckpoint = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "skew_test_bilinear_guide: malformed checkpoint {}: {e}",
            args.checkpoint
        )
    });
    let weights = checkpoint.to_weights(&rules);

    // The trainer's own forward pass over the trained weights, against the
    // guide a saturation loop would build from the same file. Two different
    // objects, each deriving its own rule embeddings from templates it
    // rebuilds itself.
    let trainer = BilinearTrainer::from_weights(&weights, &rules)
        .unwrap_or_else(|e| panic!("skew_test_bilinear_guide: {e}"));
    let deployed = BilinearCandidateGuide::new(&weights, &rules)
        .unwrap_or_else(|e| panic!("skew_test_bilinear_guide: {e}"));

    let mut rule_embed_mismatches = Vec::new();
    for i in 0..rules.len() {
        let id = rules.id_of(i).expect("index within the rule set");
        if trainer.rule_embed(id) != deployed.rule_embed(id) {
            rule_embed_mismatches.push(rules.label_of(i).unwrap_or_else(|| format!("{id}")));
        }
    }

    let file = File::open(&args.dev)
        .unwrap_or_else(|e| panic!("skew_test_bilinear_guide: cannot open {}: {e}", args.dev));
    let mut max_abs_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut exceeding_tol = 0usize;
    let mut not_bit_identical = 0usize;
    let mut checked = 0usize;

    for line in BufReader::new(file).lines() {
        if checked >= args.n {
            break;
        }
        let line = line.unwrap_or_else(|e| panic!("skew_test_bilinear_guide: read error: {e}"));
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("skew_test_bilinear_guide: malformed JSONL row: {e}"));
        let sample = BilinearSample::from_record(&record);

        // `rule_embed` comes from the *trainer* side, exactly as the guided
        // loop fills it from `SaturationGuide::rule_embed` once per rule per
        // episode. A deployed guide that derived a different embedding would
        // then score a candidate built from the other one — which is the
        // skew this field exists to expose.
        let summary = sample.to_summary(trainer.rule_embed(sample.rule));
        let trainer_score = trainer.score(&summary);
        let deployed_score = deployed.score_candidates(std::slice::from_ref(&summary))[0];

        if trainer_score.to_bits() != deployed_score.to_bits() {
            not_bit_identical += 1;
        }
        let diff = f64::from((trainer_score - deployed_score).abs());
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
        if diff as f32 > args.tol {
            exceeding_tol += 1;
        }
        checked += 1;
    }

    assert!(
        checked >= 1000,
        "skew_test_bilinear_guide: only {checked} DEV records were available, the bar is \
         >= 1000 — the test cannot pass on a short file"
    );

    let passed = exceeding_tol == 0 && rule_embed_mismatches.is_empty();
    let report = SkewReport {
        model: "bilinear".into(),
        checkpoint: args.checkpoint.clone(),
        checkpoint_weights_fnv64: checkpoint.weights_fnv64.clone(),
        dev_path: args.dev.clone(),
        records_checked: checked,
        tol: args.tol,
        max_abs_diff,
        mean_abs_diff: sum_abs_diff / checked as f64,
        exceeding_tol,
        not_bit_identical,
        rule_embed_mismatches: rule_embed_mismatches.clone(),
        passed,
    };

    println!(
        "skew_test_bilinear_guide: {checked} records, max |diff| = {max_abs_diff:.3e}, \
         {exceeding_tol} over tol {}, {not_bit_identical} not bit-identical, \
         {} rule-embedding mismatches",
        args.tol,
        rule_embed_mismatches.len()
    );

    if let Some(path) = &args.report_json {
        let json = serde_json::to_string_pretty(&report)
            .unwrap_or_else(|e| panic!("skew_test_bilinear_guide: serialization failed: {e}"));
        std::fs::write(path, json)
            .unwrap_or_else(|e| panic!("skew_test_bilinear_guide: cannot write {path}: {e}"));
        println!("skew_test_bilinear_guide: wrote {path}");
    }

    assert!(
        rule_embed_mismatches.is_empty(),
        "skew_test_bilinear_guide: {} rule(s) derive a different embedding on the deployed \
         side than on the trainer side: {rule_embed_mismatches:?} — every candidate for \
         those rules would be scored against a vector the model was never fit to",
        rule_embed_mismatches.len()
    );
    assert!(
        passed,
        "skew_test_bilinear_guide: {exceeding_tol} of {checked} records exceed the {} \
         tolerance (max |diff| {max_abs_diff:.3e}) — the deployed guide is not the model \
         that was trained; no guided evaluation may run against this checkpoint",
        args.tol
    );
    println!("skew_test_bilinear_guide: PASSED");
}
