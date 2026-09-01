//! Round 2b: the contrastive objective (docs/plans/2026-08-05-egraph-nnue-research-workflow.md
//! §4.2 item 2). Trains the extraction head with a combined loss —
//! regression MSE plus a `lambda`-weighted pairwise ranking hinge over
//! within-`VariantSet` candidate pairs (J6) — at a small grid of `lambda`,
//! each COLD-STARTED from the same seed so lambda is the only thing that
//! differs between runs, and reports the diagnostic Round 0/1 established
//! this dataset exists to answer: does the pairwise term move WITHIN-SET
//! ranking accuracy on held-out DEV-family variant sets, where corpus-level
//! regression is already saturated (DEV Spearman rho ~0.98) and has nothing
//! left to show?
//!
//! # What this measures and how
//!
//! One `BenchSession` (one clock) mints every label in this run:
//! - TRAIN-tier regression samples (`corpus_train.bin`, capped at
//!   `--train-max`) — the existing regression objective.
//! - TRAIN-family `VariantSet`s (`training::variant_set::mint_variant_set_with_arenas`)
//!   — the pairwise objective's training data. Every candidate is
//!   fenced against DEV/FINAL and quarantined, per the mint contract.
//! - DEV-family `VariantSet`s (`training::variant_set::mint_variant_set_dev_eval_with_arenas`)
//!   — held out of every training step, used ONLY for the within-set
//!   pairwise-accuracy metric this round exists to report.
//!
//! Pairs (both TRAIN and DEV) whose measured `|delta|` (percent of the
//! smaller member) is below the run's own noise floor — measured from
//! repeated re-measurement of a fixed probe kernel, same method as
//! `bin/mint_variant_sets` — are excluded from both the training loss and
//! the accuracy metric: a pair the harness cannot order is not training
//! signal and not evaluable signal either (measurement-discipline
//! requirement, docs/plans/2026-08-05-egraph-nnue-research-workflow.md §4).
//!
//! Every sample (regression and pairwise, TRAIN and DEV) is walked into an
//! [`EdgeTrace`](pixelflow_search::nnue::factored::EdgeTrace) exactly once,
//! against a throwaway reference `ExprNnue::new_random(seed)` used only to
//! run the walk — the trace itself is embedding-independent (structure +
//! PE rotation only) and is reused across every lambda in the sweep, so the
//! O(edges) walk is not repeated `k` times.
//!
//! What IS repeated per lambda, per forward pass, is `trace.realize(&model
//! .embeddings)`: `unified_backward`'s `backward_value`/`backward_pairwise`
//! both differentiate into `grads.d_embeddings` through the recorded edge
//! stream (P1(a) — embeddings are a real, trained parameter now, not a
//! frozen reference table), so each lambda's model — cold-started from the
//! SAME seed but trained under a different loss — moves its OWN embeddings
//! independently from step one, and every forward/backward for that model
//! must realize features from that model's current table, not a shared
//! frozen one. This is the corollary the module doc for `unified_backward`
//! flags: a contrastive run trains embeddings too, which past-2b framing
//! (written when neither loss had an accumulator-backward path) did not
//! anticipate — see that module's history note. DEV evaluation realizes
//! from each lambda's own FINAL trained embeddings for the same reason: a
//! DEV accumulator built against the discarded reference table would score
//! the wrong function.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin train_contrastive -- \
//!   --lambdas 0.0,0.25,0.5,1.0,2.0 --epochs 40
//! ```

use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_search::egraph::CostModel;
use pixelflow_search::nnue::ExprNnue;
use pixelflow_search::nnue::factored::{EdgeAccumulator, EdgeTrace};

use pixelflow_pipeline::extraction_head_weights_path;
use pixelflow_pipeline::jit_bench::{
    BenchMode, BenchPosition, BenchResult, BenchSession, CostLabel,
};
use pixelflow_pipeline::journal::{
    ArtifactId, ConfigHash, DIFF_HASH_EXCLUDES, JournalEntry, SourceVersion, fnv1a64_hex,
};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::mint::{
    MintMetadata, NormalizationStats, bench_mode_slug, unix_now_s, weights_identity,
};
use pixelflow_pipeline::training::quarantine::Quarantine;
use pixelflow_pipeline::training::split::{DevSide, Fence, FinalSide};
use pixelflow_pipeline::training::stats::{noise_floor_pct, noise_probe_arena, spearman_rho};
use pixelflow_pipeline::training::unified_backward::{
    PairwiseSide, SgdConfig, UnifiedGradients, ValueObjective, apply_unified_sgd,
    backward_pairwise, backward_value, forward_cached,
};
use pixelflow_pipeline::training::variant_set::{
    MintTally, VariantMember, mint_variant_set_dev_eval_with_arenas, mint_variant_set_with_arenas,
};

/// E-graph saturation budget — matches `bin/mint_variant_sets` /
/// `bench_extraction_3way` / `prod_kernel_jit`.
const SATURATE_LIMIT: usize = 40;

/// Regression-sample benchmark exclusion rate above which this run refuses
/// to trust its own labels (mirrors `bootstrap_extraction_head`'s
/// `MAX_EXCLUSION_RATE`).
const MAX_EXCLUSION_RATE: f64 = 0.10;

#[derive(Parser, Debug)]
#[command(name = "train_contrastive")]
#[command(about = "Round 2b: sweep the contrastive (regression + lambda*pairwise-hinge) objective")]
struct Args {
    /// Comma-separated lambda grid. `0.0` is the regression-only sanity
    /// baseline — it must be present for the ablation comparison to mean
    /// anything, so it is always trained even if omitted.
    #[arg(long, default_value = "0.0,0.25,0.5,1.0,2.0")]
    lambdas: String,

    /// Training epochs (each lambda config trains for this many, from a
    /// fresh cold start).
    #[arg(long, default_value_t = 40)]
    epochs: usize,

    #[arg(long, default_value_t = 0.001)]
    lr: f32,
    #[arg(long, default_value_t = 0.9)]
    momentum: f32,
    #[arg(long, default_value_t = 1e-5)]
    weight_decay: f32,
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,
    #[arg(long, default_value_t = 1.0)]
    value_coeff: f32,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    /// Cold-start seed. IDENTICAL for every lambda in the sweep — lambda
    /// must be the only thing that differs between runs, or a difference in
    /// held-out metrics could be initialization noise instead of the
    /// objective.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Directory holding the tiered corpus (`gen_bench_corpus` output).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: PathBuf,

    /// Trained extraction-head weights — used ONLY as the NNUE arm of
    /// `enumerate_candidates`'s diversity enumeration when minting
    /// VariantSets, never as a warm start for the models this binary trains
    /// (those are always cold-started per lambda, see `--seed`).
    #[arg(long, default_value_os_t = extraction_head_weights_path())]
    enumeration_weights: PathBuf,

    /// Max TRAIN-corpus regression samples (post-shuffle prefix of
    /// `corpus_train.bin`).
    #[arg(long, default_value_t = 2000)]
    train_max: usize,

    /// Max DEV-corpus regression samples for the DEV MAE/Spearman metric.
    #[arg(long, default_value_t = 400)]
    dev_max: usize,

    /// TRAIN-family VariantSets to mint for the pairwise training term.
    #[arg(long, default_value_t = 150)]
    train_variant_sets: usize,

    /// DEV-family VariantSets to mint for the held-out pairwise-accuracy
    /// metric (task requirement: "~50 sets suffices").
    #[arg(long, default_value_t = 50)]
    dev_variant_sets: usize,

    /// Candidates requested per VariantSet base (both TRAIN and DEV).
    #[arg(long, default_value_t = 8)]
    variant_k: usize,

    /// Order-shuffle seeds (TRAIN regression corpus, TRAIN variant bases,
    /// DEV variant bases) — distinct from `--seed` (model init) so the data
    /// composition is independently reproducible from the model's cold
    /// start.
    #[arg(long, default_value_t = 20260827)]
    order_seed: u64,

    /// Re-measurements of the fixed noise-probe kernel used to establish
    /// this run's pair-orderability floor.
    #[arg(long, default_value_t = 25)]
    noise_probe_samples: usize,

    /// Quarantine sidecar JSONL (overwritten each run).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/train_contrastive_quarantine.jsonl"
    )]
    quarantine_log: PathBuf,

    /// Persist the trained model for this one lambda (must be a value present
    /// in `--lambdas`) as a checkpoint, with a `MintMetadata` sidecar, so a
    /// downstream consumer (`bench_extraction_3way`) can evaluate it. The
    /// sweep is a diagnostic by default — no lambda's model survives past
    /// this process — so a checkpoint is opt-in and the lambda it names must
    /// be chosen before looking at the sweep's own results.
    #[arg(long)]
    checkpoint_lambda: Option<f32>,

    /// Where to write the `--checkpoint-lambda` checkpoint. Defaults to the
    /// path `bench_extraction_3way` reads unconditionally.
    #[arg(long, default_value_os_t = extraction_head_weights_path())]
    checkpoint_out: PathBuf,
}

// ── Seeded shuffle (xorshift64*, same family as bin/mint_variant_sets) ─────

struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdead_beef } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            s.swap(i, j);
        }
    }
}

// ── Regression samples ──────────────────────────────────────────────────────

struct RegressionSample {
    /// The typed record of the feature walk, NOT a baked accumulator: every
    /// forward pass realizes this against the training model's OWN live
    /// embeddings ([`EdgeTrace::realize`]), because embeddings now move
    /// under `backward_value`/`backward_pairwise` (P1(a)) and each lambda's
    /// model trains its own copy independently. See the module doc.
    trace: EdgeTrace,
    target_log_ns: f32,
}

/// Benchmark every entry of `entries` under `session`, quarantine-check and
/// mint a label for each, and walk it into an [`EdgeTrace`] (embeddings
/// only run the walk once — the trace itself doesn't depend on their
/// values, see [`EdgeAccumulator::from_arena_dag_traced`]). Returns samples
/// plus (attempted, bench_failed, quarantine_failed) tallies.
///
/// `drift_factors` collects each minted label's [`CostLabel::drift`] — pass
/// the run's shared TRAIN-label accumulator here, or a throwaway `Vec` for
/// DEV entries (the checkpoint sidecar's normalization summary describes the
/// TRAINING population only, matching `bootstrap_extraction_head`'s "Training
/// labels only" contract: DEV is measured on a different part of the drift
/// curve and its factors don't describe what the weights were fit on).
#[allow(clippy::too_many_arguments)]
fn build_regression_samples(
    session: &mut BenchSession,
    quarantine: &mut Quarantine,
    entries: &[(String, ExprArena, ExprId)],
    walk_embeddings: &pixelflow_search::nnue::factored::OpEmbeddings,
    name_prefix: &str,
    position: &mut usize,
    drift_factors: &mut Vec<f64>,
) -> (Vec<RegressionSample>, usize, usize, usize) {
    let mut samples = Vec::with_capacity(entries.len());
    let mut attempted = 0usize;
    let mut bench_failed = 0usize;
    let mut quarantine_failed = 0usize;

    for (name, arena, root) in entries {
        attempted += 1;
        let qname = format!("{name_prefix}_{name}");
        if !quarantine.check(&qname, arena, *root) {
            quarantine_failed += 1;
            continue;
        }
        let bench: BenchResult = match session.benchmark_arena(arena, *root, BenchMode::Latency) {
            Ok(b) => b,
            Err(_e) => {
                bench_failed += 1;
                continue;
            }
        };
        let label = CostLabel::mint(&bench, BenchPosition(*position), "train_contrastive");
        *position += 1;
        drift_factors.push(label.drift().get());
        let (_, trace) = EdgeAccumulator::from_arena_dag_traced(arena, *root, walk_embeddings);
        samples.push(RegressionSample {
            trace,
            target_log_ns: label.target_log_ns(),
        });
    }
    (samples, attempted, bench_failed, quarantine_failed)
}

// ── Pairwise samples ────────────────────────────────────────────────────────

struct PairSample {
    /// Each side's typed feature-walk record — realized against the
    /// training model's live embeddings on every forward pass, same as
    /// [`RegressionSample::trace`].
    trace_cheaper: EdgeTrace,
    trace_pricier: EdgeTrace,
}

/// Every orderable pair (measured delta at/above `floor_pct`) inside one
/// minted `(members, arenas)` VariantSet, as `PairSample`s (each side
/// walked into an [`EdgeTrace`] once, against `walk_embeddings` — see
/// [`build_regression_samples`]). Also returns (total pairs in this set,
/// orderable pairs in this set) for the dataset-quality report.
fn set_pairs(
    members: &[VariantMember],
    arenas: &[(ExprArena, ExprId)],
    floor_pct: f64,
    walk_embeddings: &pixelflow_search::nnue::factored::OpEmbeddings,
) -> (Vec<PairSample>, usize, usize) {
    let mut pairs = Vec::new();
    let mut total = 0usize;
    let mut orderable = 0usize;
    for i in 0..members.len() {
        for j in (i + 1)..members.len() {
            total += 1;
            let (a, b) = (&members[i], &members[j]);
            let lo = a.session_ns.min(b.session_ns);
            if lo <= 0.0 {
                continue;
            }
            let delta_pct = (a.session_ns - b.session_ns).abs() / lo * 100.0;
            if delta_pct < floor_pct {
                continue;
            }
            orderable += 1;
            let (cheaper_idx, pricier_idx) = if a.session_ns <= b.session_ns {
                (i, j)
            } else {
                (j, i)
            };
            let (ca, ra) = &arenas[cheaper_idx];
            let (cp, rp) = &arenas[pricier_idx];
            let (_, trace_cheaper) =
                EdgeAccumulator::from_arena_dag_traced(ca, *ra, walk_embeddings);
            let (_, trace_pricier) =
                EdgeAccumulator::from_arena_dag_traced(cp, *rp, walk_embeddings);
            pairs.push(PairSample {
                trace_cheaper,
                trace_pricier,
            });
        }
    }
    (pairs, total, orderable)
}

// ── Training item (one combined SGD stream per lambda config) ──────────────

#[derive(Clone, Copy)]
enum TrainItem {
    Regression(usize),
    Pair(usize),
}

/// Cold-start-train ONE model at `lambda`, evaluate it, return its metrics
/// alongside the trained model itself (the checkpoint the caller may persist
/// via `--checkpoint-lambda`).
#[allow(clippy::too_many_arguments)]
fn train_one_lambda(
    lambda: f32,
    args: &Args,
    train_samples: &[RegressionSample],
    train_pairs: &[PairSample],
    dev_samples: &[RegressionSample],
    dev_pairs: &[PairSample],
) -> (LambdaResult, ExprNnue) {
    let mut model = ExprNnue::new_random(args.seed);
    let mut momentum_buf = UnifiedGradients::zero();

    let mut items: Vec<TrainItem> = (0..train_samples.len())
        .map(TrainItem::Regression)
        .collect();
    if lambda != 0.0 {
        items.extend((0..train_pairs.len()).map(TrainItem::Pair));
    }

    let mut rng_state = args.seed ^ 0x5151_5151_5151_5151;
    let mut rng_next = || -> u64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng_state >> 33
    };

    let mut final_epoch_reg_rmse = 0.0f64;
    let mut final_epoch_pair_hinge = 0.0f64;

    for epoch in 0..args.epochs {
        for i in (1..items.len()).rev() {
            let j = rng_next() as usize % (i + 1);
            items.swap(i, j);
        }

        let mut epoch_reg_sq_err = 0.0f64;
        let mut epoch_reg_n = 0usize;
        let mut epoch_pair_hinge_sum = 0.0f64;
        let mut epoch_pair_n = 0usize;

        for chunk in items.chunks(args.batch_size) {
            let mut grads = UnifiedGradients::zero();
            for &item in chunk {
                match item {
                    TrainItem::Regression(idx) => {
                        let s = &train_samples[idx];
                        // Realized from the LIVE table on every step — this
                        // model's embeddings move under its own gradient
                        // from epoch 0, so a cached accumulator would go
                        // stale the moment SGD touches `model.embeddings`.
                        let acc = s.trace.realize(&model.embeddings);
                        let cache = forward_cached(&model, &acc);
                        backward_value(
                            &model,
                            &cache,
                            &s.trace,
                            ValueObjective {
                                target_log_ns: s.target_log_ns,
                                coeff: args.value_coeff,
                            },
                            &mut grads,
                        );
                        let err = cache.value_pred - s.target_log_ns;
                        epoch_reg_sq_err += f64::from(err * err);
                        epoch_reg_n += 1;
                    }
                    TrainItem::Pair(idx) => {
                        let p = &train_pairs[idx];
                        let acc_cheaper = p.trace_cheaper.realize(&model.embeddings);
                        let acc_pricier = p.trace_pricier.realize(&model.embeddings);
                        let cache_cheaper = forward_cached(&model, &acc_cheaper);
                        let cache_pricier = forward_cached(&model, &acc_pricier);
                        let violation = backward_pairwise(
                            &model,
                            PairwiseSide {
                                cache: &cache_cheaper,
                                trace: &p.trace_cheaper,
                            },
                            PairwiseSide {
                                cache: &cache_pricier,
                                trace: &p.trace_pricier,
                            },
                            lambda,
                            &mut grads,
                        );
                        epoch_pair_hinge_sum += f64::from(violation);
                        epoch_pair_n += 1;
                    }
                }
            }
            grads.scale(1.0 / chunk.len().max(1) as f32);
            apply_unified_sgd(
                &mut model,
                &grads,
                &mut momentum_buf,
                SgdConfig {
                    lr: args.lr,
                    momentum: args.momentum,
                    weight_decay: args.weight_decay,
                    grad_clip: args.grad_clip,
                },
            );
        }

        if epoch == args.epochs - 1 {
            final_epoch_reg_rmse = (epoch_reg_sq_err / epoch_reg_n.max(1) as f64).sqrt();
            final_epoch_pair_hinge = epoch_pair_hinge_sum / epoch_pair_n.max(1) as f64;
        }
        if epoch == 0 || epoch == args.epochs - 1 || (epoch + 1) % 10 == 0 {
            eprintln!(
                "  [lambda={lambda:.2}] epoch {:>3}/{}: train_reg_RMSE={:.4} train_pair_mean_hinge={:.5} (n_reg={epoch_reg_n}, n_pair={epoch_pair_n})",
                epoch + 1,
                args.epochs,
                (epoch_reg_sq_err / epoch_reg_n.max(1) as f64).sqrt(),
                epoch_pair_hinge_sum / epoch_pair_n.max(1) as f64,
            );
        }
    }

    // ── DEV regression metrics ──────────────────────────────────────────
    // Realized from THIS model's final trained embeddings, not the
    // discarded walk-time reference table — DEV must score the function
    // this lambda actually learned, and since P1(a) that function includes
    // the embedding table.
    let predicted: Vec<f32> = dev_samples
        .iter()
        .map(|s| model.predict_log_cost_with_features(&s.trace.realize(&model.embeddings)))
        .collect();
    let measured: Vec<f32> = dev_samples.iter().map(|s| s.target_log_ns).collect();
    let dev_mae = predicted
        .iter()
        .zip(&measured)
        .map(|(p, m)| f64::from((p - m).abs()))
        .sum::<f64>()
        / predicted.len().max(1) as f64;
    let dev_spearman = spearman_rho(&predicted, &measured);

    // ── DEV pairwise accuracy (the diagnostic this round exists for) ────
    let mut correct = 0usize;
    for p in dev_pairs {
        let pred_cheaper =
            model.predict_log_cost_with_features(&p.trace_cheaper.realize(&model.embeddings));
        let pred_pricier =
            model.predict_log_cost_with_features(&p.trace_pricier.realize(&model.embeddings));
        if pred_cheaper < pred_pricier {
            correct += 1;
        }
    }
    let dev_pairwise_accuracy = if dev_pairs.is_empty() {
        None
    } else {
        Some(correct as f64 / dev_pairs.len() as f64)
    };

    let result = LambdaResult {
        lambda,
        train_final_reg_rmse: final_epoch_reg_rmse,
        train_final_pair_mean_hinge: final_epoch_pair_hinge,
        dev_samples: dev_samples.len(),
        dev_mae_log_ns: dev_mae,
        dev_spearman_rho: dev_spearman,
        dev_pairs_evaluated: dev_pairs.len(),
        dev_pairwise_correct: correct,
        dev_pairwise_accuracy,
    };
    (result, model)
}

#[derive(Serialize, Clone, Debug)]
struct LambdaResult {
    lambda: f32,
    train_final_reg_rmse: f64,
    train_final_pair_mean_hinge: f64,
    dev_samples: usize,
    dev_mae_log_ns: f64,
    dev_spearman_rho: Option<f64>,
    dev_pairs_evaluated: usize,
    dev_pairwise_correct: usize,
    dev_pairwise_accuracy: Option<f64>,
}

#[derive(Serialize)]
struct RunMetrics {
    train_regression_samples: usize,
    dev_regression_samples: usize,
    train_variant_sets_minted: usize,
    dev_variant_sets_minted: usize,
    train_pairs_total: usize,
    train_pairs_orderable: usize,
    dev_pairs_total: usize,
    dev_pairs_orderable: usize,
    noise_floor_pct: f64,
    results: Vec<LambdaResult>,
}

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();

    let mut lambdas: Vec<f32> = args
        .lambdas
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .unwrap_or_else(|e| panic!("--lambdas: `{s}` is not a valid f32: {e}"))
        })
        .collect();
    assert!(
        !lambdas.is_empty(),
        "--lambdas must name at least one value"
    );
    if !lambdas.contains(&0.0) {
        eprintln!(
            "--lambdas did not include 0.0 (the regression-only baseline) — adding it, \
                    since the ablation comparison is meaningless without it"
        );
        lambdas.insert(0, 0.0);
    }
    eprintln!(
        "train_contrastive: lambda grid = {lambdas:?}, epochs={}, seed={}",
        args.epochs, args.seed
    );

    // ── Fences (DEV/FINAL) ──────────────────────────────────────────────
    let dev_fence = Fence::<DevSide>::build(&args.corpus_dir);
    let final_fence = Fence::<FinalSide>::build(&args.corpus_dir);
    eprintln!(
        "Holdout fence: {} DEV keys ({} entries), {} FINAL keys ({} entries)",
        dev_fence.len(),
        dev_fence.entries(),
        final_fence.len(),
        final_fence.entries()
    );

    // ── Enumeration-arm NNUE (diversity source for VariantSet minting only) ──
    let weights_bytes = std::fs::read(&args.enumeration_weights).unwrap_or_else(|e| {
        panic!(
            "failed to read extraction-head weights {}: {e}\nTrain them first with \
             bootstrap_extraction_head",
            args.enumeration_weights.display()
        )
    });
    let enumeration_nnue = ExprNnue::from_bytes(&weights_bytes).unwrap_or_else(|e| {
        panic!(
            "ExprNnue::from_bytes rejected {}: {e}",
            args.enumeration_weights.display()
        )
    });
    let mint_meta = MintMetadata::read_for(&args.enumeration_weights).unwrap_or_else(|e| {
        panic!(
            "failed to read mint sidecar for {}: {e}",
            args.enumeration_weights.display()
        )
    });
    mint_meta.require_mode(BenchMode::Latency);
    mint_meta.require_weights(&weights_bytes);
    let static_costs = CostModel::latency_prior();

    // Walk-time reference model: used ONLY to run each sample's feature walk
    // once and record its `EdgeTrace` (structural — PE rotation + op kinds —
    // and independent of embedding VALUES, so any consistent table works
    // here). Every lambda's model is cold-started from the SAME seed as this
    // one, but — since embeddings now train under `backward_value`/
    // `backward_pairwise` (P1(a)) — each one moves its OWN copy from step
    // one; every actual forward/backward realizes the trace against that
    // model's live `model.embeddings`, never against this reference's.
    let ref_model = ExprNnue::new_random(args.seed);

    let mut session = BenchSession::new();
    let mut quarantine = Quarantine::new(
        args.quarantine_log
            .to_str()
            .expect("quarantine log path must be utf8"),
    );
    let mut position = 0usize;

    // ── Noise floor ──────────────────────────────────────────────────────
    let (probe_arena, probe_root) = noise_probe_arena();
    let mut probe_ns = Vec::with_capacity(args.noise_probe_samples);
    for _ in 0..args.noise_probe_samples {
        match session.benchmark_arena(&probe_arena, probe_root, BenchMode::Latency) {
            Ok(bench) => {
                let label = CostLabel::mint(&bench, BenchPosition(position), "noise_probe");
                position += 1;
                probe_ns.push(libm::expf(label.target_log_ns()) as f64);
            }
            Err(e) => eprintln!("noise probe measurement failed: {e}"),
        }
    }
    let floor_pct = noise_floor_pct(&probe_ns).unwrap_or_else(|| {
        panic!(
            "could not establish a noise floor from {} probe samples — need at least 2",
            probe_ns.len()
        )
    });
    eprintln!(
        "Noise floor: {floor_pct:.3}% (from {} probe re-measurements)",
        probe_ns.len()
    );

    // ── TRAIN regression samples ─────────────────────────────────────────
    let train_path = args.corpus_dir.join("corpus_train.bin");
    let mut train_entries = read_corpus(&train_path)
        .unwrap_or_else(|e| panic!("failed to read TRAIN corpus {}: {e}", train_path.display()));
    assert!(!train_entries.is_empty(), "TRAIN corpus is empty");
    SeededRng::new(args.order_seed).shuffle(&mut train_entries);
    let reg_budget = args.train_max.min(train_entries.len());
    // TRAIN-label drift factors — the population `--checkpoint-lambda`'s
    // sidecar summarizes (see `build_regression_samples` doc).
    let mut train_drift_factors: Vec<f64> = Vec::new();
    let (train_samples, reg_attempted, reg_bench_failed, reg_quarantine_failed) =
        build_regression_samples(
            &mut session,
            &mut quarantine,
            &train_entries[..reg_budget],
            &ref_model.embeddings,
            "reg_train",
            &mut position,
            &mut train_drift_factors,
        );
    let reg_excluded = reg_bench_failed + reg_quarantine_failed;
    assert!(
        (reg_excluded as f64) / (reg_attempted.max(1) as f64) <= MAX_EXCLUSION_RATE,
        "TRAIN regression exclusion rate {:.1}% ({reg_excluded}/{reg_attempted}) exceeds {:.0}% \
         — labels are not trustworthy enough to train on",
        100.0 * reg_excluded as f64 / reg_attempted.max(1) as f64,
        100.0 * MAX_EXCLUSION_RATE
    );
    eprintln!(
        "TRAIN regression samples: {} minted ({} attempted, {} bench-failed, {} quarantine-failed)",
        train_samples.len(),
        reg_attempted,
        reg_bench_failed,
        reg_quarantine_failed
    );

    // ── DEV regression samples ───────────────────────────────────────────
    let dev_path = args.corpus_dir.join("corpus_dev.bin");
    let mut dev_entries = read_corpus(&dev_path)
        .unwrap_or_else(|e| panic!("failed to read DEV corpus {}: {e}", dev_path.display()));
    assert!(!dev_entries.is_empty(), "DEV corpus is empty");
    SeededRng::new(args.order_seed ^ 0xD1D1).shuffle(&mut dev_entries);
    let dev_reg_budget = args.dev_max.min(dev_entries.len());
    // DEV labels are eval-only — their drift factors describe the evaluation
    // population, not the training targets, so they get a throwaway sink
    // rather than folding into `train_drift_factors`.
    let mut dev_drift_factors: Vec<f64> = Vec::new();
    let (dev_samples, dev_attempted, dev_bench_failed, dev_quarantine_failed) =
        build_regression_samples(
            &mut session,
            &mut quarantine,
            &dev_entries[..dev_reg_budget],
            &ref_model.embeddings,
            "reg_dev",
            &mut position,
            &mut dev_drift_factors,
        );
    eprintln!(
        "DEV regression samples: {} minted ({} attempted, {} bench-failed, {} quarantine-failed)",
        dev_samples.len(),
        dev_attempted,
        dev_bench_failed,
        dev_quarantine_failed
    );
    assert!(
        dev_samples.len() >= 2,
        "DEV regression evaluation needs at least 2 samples, got {}",
        dev_samples.len()
    );

    // ── TRAIN VariantSets → pairwise training data ──────────────────────
    let mut train_bases = train_entries.clone();
    SeededRng::new(args.order_seed ^ 0x7A21).shuffle(&mut train_bases);
    let mut train_pairs: Vec<PairSample> = Vec::new();
    let mut train_sets_minted = 0usize;
    let mut train_pairs_total = 0usize;
    let mut train_pairs_orderable = 0usize;
    let mut train_variant_tally = MintTally::default();
    for (name, arena, root) in train_bases.iter().take(args.train_variant_sets * 2) {
        if train_sets_minted >= args.train_variant_sets {
            break;
        }
        match mint_variant_set_with_arenas(
            &mut session,
            &mut quarantine,
            Some(&dev_fence),
            &final_fence,
            &enumeration_nnue,
            &static_costs,
            name,
            arena,
            *root,
            args.variant_k,
            SATURATE_LIMIT,
            &mut position,
            &mut train_variant_tally,
        ) {
            Ok((set, arenas)) => {
                train_sets_minted += 1;
                let (mut pairs, total, orderable) =
                    set_pairs(&set.members, &arenas, floor_pct, &ref_model.embeddings);
                train_pairs_total += total;
                train_pairs_orderable += orderable;
                train_pairs.append(&mut pairs);
            }
            Err(_reason) => {}
        }
    }
    eprintln!(
        "TRAIN VariantSets: {} minted ({} candidates minted), {} orderable pairs / {} total ({:.1}%)",
        train_sets_minted,
        train_variant_tally.candidates_minted,
        train_pairs_orderable,
        train_pairs_total,
        100.0 * train_pairs_orderable as f64 / train_pairs_total.max(1) as f64
    );

    // ── DEV VariantSets → held-out pairwise-accuracy evaluation ─────────
    let mut dev_bases = dev_entries.clone();
    SeededRng::new(args.order_seed ^ 0xDE00).shuffle(&mut dev_bases);
    let mut dev_pairs: Vec<PairSample> = Vec::new();
    let mut dev_sets_minted = 0usize;
    let mut dev_pairs_total = 0usize;
    let mut dev_pairs_orderable = 0usize;
    let mut dev_variant_tally = MintTally::default();
    for (name, arena, root) in dev_bases.iter().take(args.dev_variant_sets * 2) {
        if dev_sets_minted >= args.dev_variant_sets {
            break;
        }
        match mint_variant_set_dev_eval_with_arenas(
            &mut session,
            &mut quarantine,
            &final_fence,
            &enumeration_nnue,
            &static_costs,
            name,
            arena,
            *root,
            args.variant_k,
            SATURATE_LIMIT,
            &mut position,
            &mut dev_variant_tally,
        ) {
            Ok((set, arenas)) => {
                dev_sets_minted += 1;
                let (mut pairs, total, orderable) =
                    set_pairs(&set.members, &arenas, floor_pct, &ref_model.embeddings);
                dev_pairs_total += total;
                dev_pairs_orderable += orderable;
                dev_pairs.append(&mut pairs);
            }
            Err(_reason) => {}
        }
    }
    eprintln!(
        "DEV VariantSets: {} minted ({} candidates minted), {} orderable pairs / {} total ({:.1}%)",
        dev_sets_minted,
        dev_variant_tally.candidates_minted,
        dev_pairs_orderable,
        dev_pairs_total,
        100.0 * dev_pairs_orderable as f64 / dev_pairs_total.max(1) as f64
    );
    assert!(
        !dev_pairs.is_empty(),
        "DEV pairwise-accuracy evaluation has zero orderable pairs — cannot report the metric \
         this round exists for"
    );

    quarantine.finish();

    // ── Sweep ─────────────────────────────────────────────────────────────
    eprintln!("\n=== Training sweep: {} lambda configs ===", lambdas.len());
    let mut results = Vec::with_capacity(lambdas.len());
    // Every lambda's trained model, kept only long enough for an optional
    // `--checkpoint-lambda` save below — the sweep is a diagnostic, so
    // nothing here is a warm start for anything.
    let mut trained_models: Vec<(f32, ExprNnue)> = Vec::with_capacity(lambdas.len());
    for &lambda in &lambdas {
        eprintln!("\n--- lambda = {lambda} ---");
        let (r, model) = train_one_lambda(
            lambda,
            &args,
            &train_samples,
            &train_pairs,
            &dev_samples,
            &dev_pairs,
        );
        eprintln!(
            "  DEV: MAE={:.4} spearman={} pairwise_acc={} ({}/{})",
            r.dev_mae_log_ns,
            r.dev_spearman_rho
                .map_or("undefined".to_string(), |v| format!("{v:.4}")),
            r.dev_pairwise_accuracy
                .map_or("undefined".to_string(), |v| format!("{:.4}", v)),
            r.dev_pairwise_correct,
            r.dev_pairs_evaluated,
        );
        results.push(r);
        trained_models.push((lambda, model));
    }

    // ── Ablation report (task requirement 3: say so plainly either way) ──
    let baseline = results
        .iter()
        .find(|r| r.lambda == 0.0)
        .expect("lambda=0.0 baseline is always present");
    let baseline_acc = baseline
        .dev_pairwise_accuracy
        .expect("dev_pairs is non-empty, checked above");

    println!("\n=== Round 2b contrastive objective: results ===");
    println!(
        "{:<8} {:>12} {:>10} {:>16} {:>10}",
        "lambda", "dev_MAE", "dev_rho", "pairwise_acc", "vs_base"
    );
    for r in &results {
        let acc = r.dev_pairwise_accuracy.unwrap_or(f64::NAN);
        let delta = acc - baseline_acc;
        println!(
            "{:<8.2} {:>12.4} {:>10} {:>16.4} {:>+10.4}",
            r.lambda,
            r.dev_mae_log_ns,
            r.dev_spearman_rho
                .map_or("n/a".to_string(), |v| format!("{v:.4}")),
            acc,
            delta,
        );
    }

    let best = results.iter().filter(|r| r.lambda != 0.0).max_by(|a, b| {
        a.dev_pairwise_accuracy
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&b.dev_pairwise_accuracy.unwrap_or(f64::NEG_INFINITY))
            .expect("finite")
    });
    match best {
        Some(b) if b.dev_pairwise_accuracy.unwrap_or(0.0) > baseline_acc => {
            println!(
                "\nVERDICT: lambda={:.2} beats the regression-only baseline on held-out \
                 within-set pairwise accuracy: {:.4} vs {:.4} baseline ({:+.4}).",
                b.lambda,
                b.dev_pairwise_accuracy.unwrap(),
                baseline_acc,
                b.dev_pairwise_accuracy.unwrap() - baseline_acc,
            );
        }
        Some(b) => {
            println!(
                "\nVERDICT: NO lambda beat the regression-only baseline's within-set pairwise \
                 accuracy on held-out DEV VariantSets. Best swept lambda={:.2} scored {:.4} vs \
                 {:.4} baseline ({:+.4}). The pairwise term did not move the diagnostic this \
                 round exists to test — this is the round result, reported plainly, not \
                 papered over. (Single seed per lambda: this comparison has no variance \
                 estimate: a wider seed/lambda sweep would be needed before treating a small \
                 positive OR negative delta as more than noise.)",
                b.lambda,
                b.dev_pairwise_accuracy.unwrap(),
                baseline_acc,
                b.dev_pairwise_accuracy.unwrap() - baseline_acc,
            );
        }
        None => println!("\nVERDICT: no non-zero lambda was swept — nothing to ablate against."),
    }

    // ── Checkpoint (opt-in) ─────────────────────────────────────────────
    //
    // Written LAST among the sweep's own outputs, mirroring
    // `bootstrap_extraction_head`'s Phase-4 ordering rationale: this run's
    // sweep report above already stands on its own, so a save failure here
    // must not be allowed to look like a failed measurement run — but a save
    // that DOES succeed must name a model this run actually evaluated, and
    // that evaluation is now complete.
    if let Some(target_lambda) = args.checkpoint_lambda {
        let (_, model) = trained_models
            .iter()
            .find(|(l, _)| (*l - target_lambda).abs() < 1e-6)
            .unwrap_or_else(|| {
                panic!(
                    "--checkpoint-lambda {target_lambda} was not one of the swept --lambdas {lambdas:?} \
                     — nothing was trained at that value to checkpoint"
                )
            });
        model.save(&args.checkpoint_out).unwrap_or_else(|e| {
            panic!(
                "failed to save checkpoint (lambda={target_lambda}) to {}: {e}",
                args.checkpoint_out.display()
            )
        });
        let saved_bytes = std::fs::read(&args.checkpoint_out).unwrap_or_else(|e| {
            panic!(
                "failed to read back the just-saved checkpoint at {}: {e} — cannot record its \
                 identity in the mint sidecar",
                args.checkpoint_out.display()
            )
        });
        let checkpoint_identity = weights_identity(&saved_bytes);
        let metadata = MintMetadata {
            schema_identity: MintMetadata::current_schema_identity(),
            bench_mode: bench_mode_slug(BenchMode::Latency).to_string(),
            trainer: format!("train_contrastive(lambda={target_lambda})"),
            samples: train_samples.len(),
            order_shuffle_seed: args.order_seed,
            sentinel_calibration_ns: session.calibration_ns(),
            normalization: NormalizationStats::summarize(&train_drift_factors),
            weights_fnv64: checkpoint_identity.clone(),
            written_at_unix_s: unix_now_s(),
        };
        metadata
            .write_for(&args.checkpoint_out)
            .unwrap_or_else(|e| {
                panic!(
                    "failed to write the mint-metadata sidecar for {}: {e}. The checkpoint on disk \
                 now has no recorded objective — delete it and rerun",
                    args.checkpoint_out.display()
                )
            });
        println!(
            "\nCheckpoint: saved lambda={target_lambda} model to {} ({} bytes, weights_fnv64={checkpoint_identity})",
            args.checkpoint_out.display(),
            saved_bytes.len(),
        );
    }

    // ── Journal ──────────────────────────────────────────────────────────
    let metrics = RunMetrics {
        train_regression_samples: train_samples.len(),
        dev_regression_samples: dev_samples.len(),
        train_variant_sets_minted: train_sets_minted,
        dev_variant_sets_minted: dev_sets_minted,
        train_pairs_total,
        train_pairs_orderable,
        dev_pairs_total,
        dev_pairs_orderable,
        noise_floor_pct: floor_pct,
        results: results.clone(),
    };

    let source = SourceVersion::current(&DIFF_HASH_EXCLUDES);
    let train_bytes = std::fs::read(&train_path).unwrap_or_default();
    let corpus_hash = fnv1a64_hex(&train_bytes);
    let protocol = format!(
        "lambdas={:?};epochs={};seed={};order_seed={};train_max={};dev_max={};train_variant_sets={};dev_variant_sets={};k={}",
        lambdas,
        args.epochs,
        args.seed,
        args.order_seed,
        args.train_max,
        args.dev_max,
        args.train_variant_sets,
        args.dev_variant_sets,
        args.variant_k,
    );
    let config = ConfigHash::compose(&source, &corpus_hash, &mint_meta.weights_fnv64, &protocol);
    let weights_artifact = ArtifactId::with_identity(
        args.enumeration_weights.display().to_string(),
        mint_meta.weights_fnv64.clone(),
    );
    let journal_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/results/journal.jsonl"
    ));
    JournalEntry::new("train_contrastive", config, weights_artifact, metrics).append(&journal_path);

    eprintln!("\nWall time: {:.1}s", start.elapsed().as_secs_f64());
}
