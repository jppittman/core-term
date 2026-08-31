//! Phase 2 gate of docs/plans/2026-07-07-guided-saturation-redesign.md,
//! rebuilt on the hardened bench harness per
//! docs/plans/2026-08-05-egraph-nnue-research-workflow.md (0.3 items 2/7,
//! 0.3b) and the 2026-08-05 integrity audit (H1, H2, M3, L2).
//!
//! On the SAME saturated e-graph, compares three extraction policies with
//! real JIT wall-clock:
//!   (a) NNUE      — `IncrementalExtractor` guided by the trained Judge
//!   (b) STATIC    — `extract` + `CostModel::latency_prior()`
//!   (c) NO-SWAP   — the original expression form, un-extracted
//! plus a fourth A/A arm (STATIC measured twice, `static_aa`) whose delta is
//! the run's noise floor: any verdict inside that floor is INCONCLUSIVE.
//!
//! Correctness gate (H1/H2, plan 0.4 as REFRAMED 2026-08-06): the contract
//! line is ALGEBRAIC validity — pixelflow is a math library over the reals,
//! not an IEEE library, so two algebraically-equivalent forms disagreeing at
//! or near singularities is CONTRACT, not a failure. The reference is the
//! *scalar oracle* ([`pixelflow_ir::DifferentialCheck`]), independent of the
//! JIT backend, evaluated on the harness's 64 seeded input tuples (log-uniform
//! magnitudes 1e-4..1e4, both signs, ±0.0 — recovered from the harness itself
//! via identity kernels so check inputs and timed inputs are identical) plus
//! the original 8-point GRID.
//!
//! Acceptance is decided by the **compositional per-point error bound**: each
//! node carries an absolute radius, seeded at the estimate instructions
//! (`Recip`/`Rsqrt`) and widened by each op's own rounding, and a lane is
//! accepted iff it lands inside `value ± radius`. The predecessor folded a
//! max-over-ops tolerance (`Tolerance::loosest`) into one whole-expression
//! allowance, which has no compositional justification: `recip`'s 12-bit
//! estimate error, amplified through `sin(K·Y²)` at a large argument, exceeds
//! every per-op row while each step stays inside contract. That fold called
//! 14.4% of a real corpus "compiler bugs" (smoke B1, Codex R1). The magnitude
//! threshold and the cancellation-ratio heuristic it used for conditioning are
//! deleted too, not kept as second opinions — the ratio rule condemned
//! `x · 1e-4` on every point, which is exactly where a broken extraction hid
//! as metadata (Codex 3741889906).
//!
//! Points where the language promises no single value are skipped, `Select`-
//! aware (a divergent op in an unchosen branch does not condemn the point,
//! Codex R3), and a policy whose points are ALL skipped is excluded rather
//! than passing unchecked. The gate has two tiers:
//!   (a) SAME-FORM hard gate — JIT(form) vs oracle(form) on the SAME arena,
//!       under that point's composed bound. Divergence is a miscompile: the
//!       kernel (for no-swap) or the policy arm (for extracted forms) is
//!       EXCLUDED LOUDLY and counted, and the run fails at the end if the
//!       same-form miscompile rate exceeds [`MAX_SAME_FORM_MISCOMPILE_RATE`].
//!       Mask-valued roots gate on bit patterns with threshold proximity: an
//!       opposite mask is contract only when the deciding comparison's
//!       operands sit inside their own composed error, otherwise it is a
//!       miscompile (Codex R2).
//!   (b) CROSS-FORM conditioned gate — oracle(extracted) vs oracle(original),
//!       compared only where BOTH points report `is_well_conditioned()`, and
//!       accepted within each form's composed bound plus the documented
//!       [`CROSS_FORM_REL`] rewrite-rounding band (the composed bound is a
//!       SAME-form quantity — zero for exact inputs — so it cannot by itself
//!       judge two forms that execute different op sequences).
//!       Disagreement there is an extraction bug — but ONE synthetic
//!       expression must not kill a multi-hour run that already gated other
//!       kernels (smoke B4), so it EXCLUDES AND COUNTS the arm, printing both
//!       forms, and only the RATE exceeding
//!       [`MAX_CROSS_FORM_DIVERGENCE_RATE`] fails the run. Disagreement at
//!       ill-conditioned points is recorded as metadata and never excludes.
//! Every timed repeat's `BenchResult::outputs` (recorded from the exact
//! `ExecutableCode` the harness timed) is re-verified against the SAME
//! `PointCheck` that gated it in phase A, so the checked artifact IS the timed
//! artifact at every measurement.
//!
//! Selection discipline (plan 0.2/4.3, Codex R5): the default run consumes the
//! **DEV** tier of the corpus written by `gen_bench_corpus` — the same loader
//! and the same tier files the trainer holds out. The five named production
//! kernels are FINAL-tier and are NOT in the selection set; the FINAL tier is
//! measured only under `--final-eval`, which prints a publication banner and
//! refuses to emit a selection verdict at all.
//!
//! Objective consistency (Codex R6): the extraction head is trained on
//! `BenchMode::Latency` targets, so this gate measures LATENCY, asserts at
//! startup that the weights' mint sidecar
//! ([`pixelflow_pipeline::training::mint::MintMetadata`]) records that mode,
//! and writes the mode into every record.
//!
//! Interleaved protocol (M3): all saturation/extraction/compilation happens in
//! phase A; phase B then runs `TIMED_REPEATS` passes where each kernel's four
//! arms are measured in a seeded-random order, through one `BenchSession`
//! (QoS pinning, DVFS burn-in, drift sentinel, overhead subtraction). Ratios
//! use per-policy medians of raw ns across repeats — raw rather than
//! overhead-adjusted because adjusted ns is legitimately ~0 for folded forms,
//! and the shared call overhead only compresses ratios toward 1 (conservative
//! for any claimed win); both are persisted per measurement.
//!
//! Output (0.3b): a human-readable report on stdout, a machine-readable JSONL
//! (one record per (kernel, policy, repeat) + prepared/failure records + a
//! run summary with config hash and sentinel calibration) under
//! `pixelflow-pipeline/data/`, and one summary line appended to
//! `docs/results/journal.jsonl`.
//!
//! Run:
//! ```bash
//! # selection run (DEV tier)
//! cargo run --release -p pixelflow-pipeline --features training --bin bench_extraction_3way
//! # publication run (FINAL tier, no selection verdict)
//! cargo run --release -p pixelflow-pipeline --features training --bin bench_extraction_3way -- \
//!     --final-eval
//! ```

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use serde::Serialize;

use pixelflow_codegen::emit::compile_arena_dag;
use pixelflow_codegen::emit::executable::ExecutableCode;
use pixelflow_ir::{
    BindingTable, DifferentialCheck, ExprArena, ExprId, MaskComparison, MaskVerdict, PointCheck,
    PointVerdict, compare_mask_root,
};
use pixelflow_search::egraph::{
    CostModel, EGraph, IncrementalExtractor, all_rules, choices_to_arena, extract,
};
use pixelflow_search::nnue::ExprNnue;

use pixelflow_pipeline::extraction_head_weights_path;
use pixelflow_pipeline::jit_bench::{BenchMode, BenchSession, INPUT_TUPLES, benchmark_jit_arena};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::factored::arena_to_kernel_code;
use pixelflow_pipeline::training::mint::{MintMetadata, bench_mode_slug};
use pixelflow_pipeline::training::quarantine::screen_for_oracle;
use pixelflow_pipeline::training::split::Tier;

/// E-graph saturation budget — matches `prod_kernel_jit.rs`.
const SATURATE_LIMIT: usize = 40;

/// Alternatives considered per e-class during NNUE refinement — matches
/// `prod_kernel_jit.rs` and `pixelflow-compiler/src/optimize.rs`.
const EXTRACT_TOP_K: usize = 8;

/// Timed passes per (kernel, arm) in phase B (audit M3: >= 3, per-policy
/// medians). Arm order is re-randomized per (kernel, repeat).
const TIMED_REPEATS: usize = 5;

/// Seed for the per-repeat arm-order shuffle. Fixed so the interleaving is
/// reproducible run to run.
const ORDER_SEED: u64 = 0x3A11_5EED_2026_0805;

/// Dependency structure of the timed loop.
///
/// **Latency, to match the objective the extraction head was trained on**
/// (Codex R6). The trainer mints every target with `BenchMode::Latency`; a
/// gate measuring throughput would accept or reject the model on a metric it
/// never saw, because the two modes rank expression shapes differently (a form
/// with more independent work wins on throughput and can lose on a serialized
/// chain). [`MintMetadata::require_mode`] asserts the weights agree with this
/// constant at startup, so the two can never drift apart silently.
const BENCH_MODE: BenchMode = BenchMode::Latency;

/// Maximum tolerated SAME-FORM miscompile rate over every (kernel, policy)
/// gate. Same-form means both tiers evaluate the IDENTICAL arena under the
/// point's own composed bound, so a disagreement is a JIT/lowering bug or an
/// oracle regression — never a rewrite judgment call. One is already a bug
/// report; this bar exists so a broken backend cannot be laundered into
/// "excluded and counted" (plan 0.4).
const MAX_SAME_FORM_MISCOMPILE_RATE: f64 = 0.01;

/// Maximum tolerated CROSS-FORM divergence rate over every (kernel, policy)
/// gate.
///
/// Cross-form divergence between two algebraically-valid forms is CONTRACT at
/// or near singularities (plan 0.4), and the well-conditioned filter is a
/// bound, not a proof of benignity — so a single occurrence excludes the arm
/// and is counted, never aborts the process (smoke B4: one synthetic
/// expression killed a multi-hour run whose named kernels had already passed,
/// and every rewrite in its diff was valid — `mul_add` one-vs-two roundings,
/// the sin angle-addition identity, `x·recip(y) → x/y`). A RATE this high,
/// though, means the extractor is systematically changing values, which the
/// e-graph must not do; that is a real bug and fails the run.
///
/// **Read the rate knowing what the instrument can see.** The 2026-08-05 dev
/// run tripped this at 19.44% (7 of 36 gates) with zero same-form miscompiles,
/// and all seven were inspected from
/// `data/bench_extraction_3way_1786294245_*.jsonl`: every candidate/original
/// pair was algebraically equal by inspection (double negation, `cos(-t) =
/// cos(t)`, the cos angle-addition identity, `pow(x,-1) → recip`, `pow(x,-0.5)
/// → rsqrt`, reassociation), and none changed the function. What differed was
/// the *accuracy of the route*, which [`PointCheck`]'s composed bound does not
/// model: it seeds width at the estimate instructions and reports `± 0e0` for a
/// transcendental expansion. Two forms reaching the same value through
/// different transcendental routes therefore read "well-conditioned" while
/// legitimately landing far apart — at `|t| = 17.38` the two routes to `cos t`
/// gave `1.03866e-1` (8e-8 from libm) and `4.80013e-2`, and the form that
/// replaced `pow(x,-0.5)` with the *estimate* `rsqrt` was the one CLOSER to the
/// exact answer (15.0265 vs 15.0284; the `pow` form gave 14.8661). So a rate
/// above this bar is evidence about the bound model first and the extractor
/// second. Recalibrating means giving the cross-form comparison an accuracy
/// model for transcendental expansions (or classifying divergences by cause and
/// thresholding the residual class) — not moving this number on its own.
const MAX_CROSS_FORM_DIVERGENCE_RATE: f64 = 0.10;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "bench_extraction_3way")]
#[command(about = "Phase 2 gate: NNUE vs static-prior vs no-swap extraction, on real JIT ns")]
struct Args {
    /// Directory holding the tiered corpus written by `gen_bench_corpus`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// PUBLICATION run: measure the FINAL (held-out) tier instead of DEV.
    ///
    /// A `--final-eval` run prints a banner naming itself as such and refuses
    /// to compute a selection verdict: repeatedly tuning against FINAL is
    /// exactly what reserving it prevents (plan 0.2/4.3, Codex R5).
    #[arg(long, default_value_t = false)]
    final_eval: bool,

    /// Maximum kernels to time from the tier (0 = the whole tier). More than
    /// this many are subsampled deterministically from `--sample-seed`, and
    /// both values enter the config hash.
    ///
    /// The default is a POWER decision, not a convenience: round 0 measured a
    /// geomean run-to-run spread of 0.8815/0.9285/1.0389 across 40-kernel runs
    /// while the within-run A/A floor was 0.11–0.44% — a ~40x gap driven by
    /// the kernel SET changing between runs. The per-kernel log-ratio
    /// dispersion behind that spread is ~0.35, so the bootstrap CI half-width
    /// on the geomean shrinks as ~0.35/√n: at n=40 it is ±11% (the ±5% verdict
    /// gate sits INSIDE it, unreachable), at n=400 it is ±3.4% and the gate is
    /// resolvable. Use `0` (the whole tier) for publication runs.
    #[arg(long, default_value_t = 400)]
    max_kernels: usize,

    /// Seed for the tier subsample. Fixed by default so the SAME eval set is
    /// drawn run after run (given an unchanged tier file — the corpus content
    /// hash in the journal pins that): run-to-run comparisons are then over a
    /// constant kernel set, not a redraw.
    #[arg(long, default_value_t = 0x5A11_2026_0805_u64)]
    sample_seed: u64,
}

/// The original 8-point correctness grid, kept alongside the 64 harness
/// tuples (plan 0.3 item 2: "keep the original 8 points too").
const GRID: [(f32, f32, f32, f32); 8] = [
    (0.0, 0.0, 0.0, 0.0),
    (0.3, 0.2, 0.1, -0.1),
    (-0.5, 0.4, 0.2, 0.3),
    (0.8, -0.6, -0.3, 0.5),
    (1.0, 1.0, 0.5, -0.5),
    (-1.2, 0.1, 0.7, 0.2),
    (0.15, -0.95, -0.4, 0.6),
    (0.6, 0.6, -0.2, -0.3),
];

// ---------------------------------------------------------------------------
// Seeded RNG for the arm-order shuffle (M3). xorshift64* — the same family
// the harness's input buffer uses; `rand`/`StdRng` is not a dependency of
// this crate and the shuffle only needs determinism, not quality.
// ---------------------------------------------------------------------------

struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        assert_ne!(seed, 0, "xorshift seed must be nonzero");
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Fisher-Yates shuffle.
    fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            s.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Config hash: source revision + corpus identity + weights identity +
// environment + this binary's protocol params, composed into one journal
// line's provenance. The composition itself (`SourceVersion`,
// `environment_fingerprint`, `ConfigHash::compose`, the FNV-1a primitive) now
// lives in `pixelflow_pipeline::journal` — `bootstrap_extraction_head` needs
// the exact same shape, and hand-rolling it independently in each binary was
// the structurally-divergent-journal-line risk J15 exists to close
// (docs/plans/2026-08-17-cost-model-domain.md).
// ---------------------------------------------------------------------------

use pixelflow_pipeline::journal::{
    ArtifactId, ConfigHash, DIFF_HASH_EXCLUDES, JournalEntry, SourceVersion, fnv1a64_hex,
};

// ---------------------------------------------------------------------------
// The four measurement arms. STATIC appears twice: the `static_aa` arm is the
// A/A repeat whose delta from `static` is the run noise floor (plan 0.3b).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    NoSwap,
    Static,
    StaticAa,
    Nnue,
}

const ARMS: [Arm; 4] = [Arm::NoSwap, Arm::Static, Arm::StaticAa, Arm::Nnue];

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::NoSwap => "noswap",
            Arm::Static => "static",
            Arm::StaticAa => "static_aa",
            Arm::Nnue => "nnue",
        }
    }

    fn idx(self) -> usize {
        match self {
            Arm::NoSwap => 0,
            Arm::Static => 1,
            Arm::StaticAa => 2,
            Arm::Nnue => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// The five named production kernels. They are FINAL-tier per
// `corpus_split.toml` and enter this bench ONLY through `corpus_final.bin`
// under `--final-eval` (Codex R5) — the builders below exist so the tests can
// exercise the gate on known shapes without a corpus on disk.
// ---------------------------------------------------------------------------

// The builders below are the only users of `OpKind` outside the tests.
#[cfg(test)]
use pixelflow_ir::OpKind;

/// Names `gen_bench_corpus` writes the production kernels under, so a corpus
/// row can be labeled `named` vs `synthetic` in the report. The last twelve
/// are the withheld ShaderToy benchmark set
/// (`pixelflow_pipeline::shader_bench::SHADERTOY_KERNEL_NAMES`) — listed
/// literally here rather than via the const import, matching how this array
/// already duplicates (rather than imports) the original five names.
const NAMED_KERNELS: [&str; 17] = [
    "swirl",
    "circle_sdf",
    "poly",
    "redundant",
    "normalize",
    "cosine_palette",
    "smooth_min_scene",
    "mandelbrot_distance",
    "star_sdf",
    "gyroid_slice",
    "plasma",
    "domain_warp_fbm",
    "kaleidoscope_fold",
    "metaballs",
    "julia_set",
    "smoothstep_vignette",
    "torus_slice",
];

/// Report family of one corpus entry.
fn family_of(name: &str) -> &'static str {
    if NAMED_KERNELS.contains(&name) {
        "named"
    } else {
        "synthetic"
    }
}

/// sin(sqrt(x*x + y*y) * freq) * amp + bias — the swirl shader core.
#[cfg(test)]
fn swirl() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let kf = a.push_const(3.0);
    let sf = a.push_binary(OpKind::Mul, s, kf);
    let sn = a.push_unary(OpKind::Sin, sf);
    let ka = a.push_const(0.5);
    let prod = a.push_binary(OpKind::Mul, sn, ka);
    let kb = a.push_const(0.5);
    let out = a.push_binary(OpKind::Add, prod, kb);
    (a, out)
}

/// Circle SDF: sqrt((x-cx)^2 + (y-cy)^2) - r.
#[cfg(test)]
fn circle_sdf() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let cx = a.push_const(0.3);
    let cy = a.push_const(-0.2);
    let dx = a.push_binary(OpKind::Sub, x, cx);
    let dy = a.push_binary(OpKind::Sub, y, cy);
    let dx2 = a.push_binary(OpKind::Mul, dx, dx);
    let dy2 = a.push_binary(OpKind::Mul, dy, dy);
    let sum = a.push_binary(OpKind::Add, dx2, dy2);
    let dist = a.push_unary(OpKind::Sqrt, sum);
    let r = a.push_const(0.5);
    let out = a.push_binary(OpKind::Sub, dist, r);
    (a, out)
}

/// FMA-bait polynomial: a*x*x + b*x + c (Horner-able, fusion-able).
#[cfg(test)]
fn poly() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let ka = a.push_const(2.0);
    let kb = a.push_const(-3.0);
    let kc = a.push_const(1.0);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let ax2 = a.push_binary(OpKind::Mul, ka, xx);
    let bx = a.push_binary(OpKind::Mul, kb, x);
    let s1 = a.push_binary(OpKind::Add, ax2, bx);
    let out = a.push_binary(OpKind::Add, s1, kc);
    (a, out)
}

/// Redundancy bait: (x+y)*(x+y) + 2*(x+y) — CSE + distribution territory.
#[cfg(test)]
fn redundant() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let s = a.push_binary(OpKind::Add, x, y);
    let s2 = a.push_binary(OpKind::Mul, s, s);
    let two = a.push_const(2.0);
    let ts = a.push_binary(OpKind::Mul, two, s);
    let out = a.push_binary(OpKind::Add, s2, ts);
    (a, out)
}

/// Division/sqrt bait: x / sqrt(x*x + y*y) (normalize — rsqrt rewrites).
#[cfg(test)]
fn normalize() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let out = a.push_binary(OpKind::Div, x, s);
    (a, out)
}

#[cfg(test)]
fn named_kernels() -> Vec<(&'static str, ExprArena, ExprId)> {
    let (a, ra) = swirl();
    let (b, rb) = circle_sdf();
    let (c, rc) = poly();
    let (d, rd) = redundant();
    let (e, re) = normalize();
    vec![
        ("swirl", a, ra),
        ("circle_sdf", b, rb),
        ("poly", c, rc),
        ("redundant", d, rd),
        ("normalize", e, re),
    ]
}

// ---------------------------------------------------------------------------
// JIT execution helper: `run_scalar` (broadcast to all lanes, read lane 0)
// lives in `pixelflow_pipeline::oracle_compare` — the one copy every
// JIT/oracle cross-check harness imports
// (docs/plans/2026-08-17-cost-model-domain.md, J14).
// ---------------------------------------------------------------------------
use pixelflow_pipeline::oracle_compare::run_scalar;

// ---------------------------------------------------------------------------
// Oracle-referenced correctness gate (H1/H2)
// ---------------------------------------------------------------------------

/// Recover the harness's 64 seeded input tuples by benchmarking the four
/// identity kernels `|x,y,z,w| -> v`: `BenchResult::outputs` records each
/// coordinate at every tuple, so the check grid is read *from the harness
/// itself* rather than duplicated — check inputs and timed inputs cannot
/// drift apart.
fn recover_harness_tuples() -> Vec<[f32; 4]> {
    let mut per_var: Vec<Vec<f32>> = Vec::with_capacity(4);
    for v in 0..4u8 {
        let mut arena = ExprArena::new();
        let root = arena.push_var(v);
        let bench = benchmark_jit_arena(&arena, root).unwrap_or_else(|e| {
            panic!("recover_harness_tuples: identity kernel Var({v}) failed to benchmark: {e}")
        });
        assert_eq!(
            bench.outputs.len(),
            INPUT_TUPLES,
            "harness recorded {} outputs, expected INPUT_TUPLES={INPUT_TUPLES}",
            bench.outputs.len()
        );
        let coords = bench
            .outputs
            .iter()
            .map(|lanes| {
                for lane in lanes {
                    assert_eq!(
                        lane.to_bits(),
                        lanes[0].to_bits(),
                        "identity kernel Var({v}) returned non-uniform lanes — splat is broken"
                    );
                }
                lanes[0]
            })
            .collect();
        per_var.push(coords);
    }
    let tuples: Vec<[f32; 4]> = (0..INPUT_TUPLES)
        .map(|i| [per_var[0][i], per_var[1][i], per_var[2][i], per_var[3][i]])
        .collect();
    assert_eq!(
        tuples[0],
        [0.5, 0.7, 1.3, -0.2],
        "input tuple 0 must be the documented legacy test point"
    );
    tuples
}

/// The full check grid: the 64 harness tuples (also used per-repeat) plus the
/// original 8 GRID points (phase-A only).
struct CheckGrid {
    points: Vec<[f32; 4]>,
    tuples: Vec<[f32; 4]>,
}

impl CheckGrid {
    fn build() -> Self {
        let tuples = recover_harness_tuples();
        let mut points = tuples.clone();
        points.extend(GRID.iter().map(|&(x, y, z, w)| [x, y, z, w]));
        Self { points, tuples }
    }
}

/// Why one policy could not be timed.
///
/// The variants are not decoration: the run-level alarms score them
/// differently. A same-form miscompile accuses the JIT; a cross-form
/// divergence accuses the extractor; a compile failure or an unquarantinable
/// shape accuses neither and only censors the comparison set.
#[derive(Debug)]
enum GateFailure {
    /// The oracle cannot evaluate this shape (params, buffers, reductions), so
    /// no independent cross-check exists. Excluded, counted, no alarm.
    OracleUnsupported(String),
    /// Every grid point was skipped or unbounded, so the same-form gate never
    /// accepted OR rejected this form anywhere: passing it would mean only
    /// "was never tested" (Codex 3744586366). Excluded, counted, no alarm — but
    /// counted SEPARATELY from an unsupported shape, because this one says how
    /// much of the run's correctness claim is vacuous.
    NoBoundedPoints(String),
    /// The JIT refused the expression.
    CompileFailed(String),
    /// JIT(form) disagreed with oracle(form) on the SAME arena beyond that
    /// point's composed bound — a miscompile ([`MAX_SAME_FORM_MISCOMPILE_RATE`]).
    SameForm(String),
    /// oracle(extracted) disagreed with oracle(original) at a point BOTH forms
    /// determine well ([`MAX_CROSS_FORM_DIVERGENCE_RATE`]).
    CrossForm(String),
}

impl GateFailure {
    /// Stable slug for the JSONL `kind` field.
    fn kind(&self) -> &'static str {
        match self {
            GateFailure::OracleUnsupported(_) => "oracle_unsupported",
            GateFailure::NoBoundedPoints(_) => "no_bounded_points",
            GateFailure::CompileFailed(_) => "compile_failed",
            GateFailure::SameForm(_) => "same_form_miscompile",
            GateFailure::CrossForm(_) => "cross_form_divergence",
        }
    }

    fn detail(&self) -> &str {
        match self {
            GateFailure::OracleUnsupported(d)
            | GateFailure::NoBoundedPoints(d)
            | GateFailure::CompileFailed(d)
            | GateFailure::SameForm(d)
            | GateFailure::CrossForm(d) => d,
        }
    }
}

/// Whether two algebraically-equal forms agree at one point.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FormAgreement {
    /// The two oracle values are bit-identical, both NaN, or within both
    /// composed bounds plus the [`CROSS_FORM_REL`] rewrite band.
    Agree,
    /// They are not: the forms compute different numbers (or opposite masks
    /// with both comparisons pinned).
    Disagree,
    /// A bound is not finite (the interval crossed a pole or a domain edge) or
    /// the deciding mask comparison is unpinned, so the comparison decides
    /// nothing. Metadata.
    Indeterminate,
}

/// Rewrite-rounding band for the CROSS-form comparison, relative to the larger
/// of the two forms' values.
///
/// The composed bound cannot carry this comparison by itself, and the reason is
/// worth stating precisely: `PointCheck::error_bound` answers *"how far from
/// this value may a conforming backend running THIS form land"*, which is
/// **zero** whenever the inputs are exact — both tiers execute the same op
/// sequence and round identically. Two algebraically-equal forms execute
/// DIFFERENT op sequences, so they legitimately land on different f32 values
/// with the same exact inputs (one rounding for `mul_add` versus two, a
/// reassociated sum, `x·recip(y)` versus `x/y`). Comparing them at bound-sum
/// slack alone would demand bit-equality of every rewrite — the e-graph's
/// entire purpose — so a band for rewrite rounding is unavoidable.
///
/// `1e-3` is the library's own conditioning scale (`WELL_CONDITIONED_REL`): a
/// point that pins its answer to a part in a thousand, whose two forms differ
/// by more than that, has had its VALUE changed, not merely its rounding. This
/// is a single documented number, not a per-op table folded with
/// `Tolerance::loosest` — the fold this gate deleted was wrong because it
/// claimed compositional authority it did not have, not because a band exists.
///
/// Where it can be wrong: relative to the OUTPUT under-counts a rewrite's
/// rounding when large intermediates cancel (the classical bound is relative to
/// the sum of intermediate magnitudes), so a reassociation across a
/// cancellation can be flagged. That is why a flag EXCLUDES AND COUNTS one arm
/// with both forms printed for a human, and only [`MAX_CROSS_FORM_DIVERGENCE_RATE`]
/// fails a run.
const CROSS_FORM_REL: f32 = 1e-3;

/// Compare two forms' oracle values at one point: each form's own composed
/// bound (implementation error — estimates, gathers, blends) plus the
/// [`CROSS_FORM_REL`] rewrite-rounding band.
///
/// # Panics
///
/// Panics when the two ORACLE evaluations produce lanes that are not both
/// valid mask patterns at a mask-valued root. That accuses the oracle, not the
/// rewrite, and no rate threshold should absorb it.
fn forms_agree(label: &str, point: &[f32; 4], cand: PointCheck, orig: PointCheck) -> FormAgreement {
    if cand.root_is_mask_valued() != orig.root_is_mask_valued() {
        // A rewrite that changes the output DOMAIN (mask vs number) is not a
        // near-singularity artifact; it is a different function.
        return FormAgreement::Disagree;
    }
    if cand.root_is_mask_valued() {
        return match compare_mask_root(cand.value(), orig.value()) {
            MaskComparison::Agree => FormAgreement::Agree,
            MaskComparison::Disagree => {
                if cand.mask_is_indeterminate() || orig.mask_is_indeterminate() {
                    FormAgreement::Indeterminate
                } else {
                    FormAgreement::Disagree
                }
            }
            MaskComparison::InvalidPattern {
                got_is_mask,
                want_is_mask,
            } => panic!(
                "{label}: the ORACLE produced a non-mask lane at a mask-valued root at \
                 ({}, {}, {}, {}): candidate {:#010x} (is_mask={got_is_mask}), original \
                 {:#010x} (is_mask={want_is_mask}) — eval bug, hard failure",
                point[0],
                point[1],
                point[2],
                point[3],
                cand.value().to_bits(),
                orig.value().to_bits()
            ),
        };
    }
    let (a, b) = (cand.value(), orig.value());
    if a.to_bits() == b.to_bits() {
        return FormAgreement::Agree;
    }
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            FormAgreement::Agree
        } else {
            FormAgreement::Disagree
        };
    }
    if !a.is_finite() || !b.is_finite() {
        // Not bit-identical and at least one side is infinite: no band and no
        // bound bridges that, so the forms disagree. (Whether it MATTERS is the
        // caller's conditioning question — at an ill-conditioned point this is
        // recorded as metadata, which is exactly the overflow-recovery case
        // plan 0.4 calls contract.)
        return FormAgreement::Disagree;
    }
    let slack = cand.error_bound() + orig.error_bound() + CROSS_FORM_REL * a.abs().max(b.abs());
    if !slack.is_finite() {
        return FormAgreement::Indeterminate;
    }
    if (a - b).abs() <= slack {
        FormAgreement::Agree
    } else {
        FormAgreement::Disagree
    }
}

/// The first cross-form divergence at a well-conditioned point, held until the
/// grid is exhausted so the report can state the FULL point tallies rather
/// than the ones accumulated up to the failure.
struct CrossFormDivergence {
    point: [f32; 4],
    cand: PointCheck,
    orig: PointCheck,
}

/// Correctness data for one policy: the [`PointCheck`] that gated each harness
/// input tuple (`None` = nothing was promised there, so per-repeat
/// verification skips it) plus the point tallies the records and diagnostics
/// report.
///
/// Storing the `PointCheck` itself rather than a value and a tolerance is what
/// makes phase B re-verification use the EXACT acceptance rule phase A used —
/// there is no second, looser copy of the criterion anywhere.
struct PolicyCheck {
    expected: Vec<Option<PointCheck>>,
    /// Whether the candidate's root is mask-valued: its lanes are gated by
    /// [`PointCheck::classify_mask_root`] (bit patterns plus threshold
    /// proximity), never by a numeric bound whose NaN-vs-NaN arm would wave a
    /// broken `0x7fc00000` through where the hardware owed `0xFFFFFFFF`.
    mask_root: bool,
    /// Mask roots only: same-form points where JIT and oracle produced
    /// opposite VALID masks with the deciding comparison's operands inside
    /// their own composed error. Contract (Codex R2), metadata.
    mask_near_threshold: usize,
    /// Same-form points where JIT-vs-oracle was actually compared.
    checked: usize,
    /// Points skipped as platform-divergent for the candidate.
    skipped: usize,
    /// Points whose composed bound was not finite: not a pass, not a failure.
    unbounded: usize,
    /// Points where BOTH forms were well-conditioned and were compared.
    cross_compared: usize,
    /// Points ill-conditioned for either form (cross-form only).
    cross_ill_points: usize,
    /// Ill-conditioned cross-form points that also disagreed — contract at or
    /// near a singularity, metadata only.
    cross_ill_disagreements: usize,
    /// Well-conditioned cross-form points that disagreed. Nonzero means the
    /// arm was excluded as [`GateFailure::CrossForm`].
    cross_wc_disagreements: usize,
}

impl PolicyCheck {
    /// The point-population line every diagnostic prints: a gate that says
    /// "extraction bug" without saying how much of the grid it actually
    /// compared cannot be investigated.
    fn tally_line(&self) -> String {
        format!(
            "points: {} same-form checked, {} cross-form well-conditioned, {} ill-conditioned, \
             {} platform-divergent (skipped), {} unbounded, {} mask near-threshold",
            self.checked,
            self.cross_compared,
            self.cross_ill_points,
            self.skipped,
            self.unbounded,
            self.mask_near_threshold
        )
    }
}

/// Compile `candidate`, then gate it (H1/H2, plan 0.4 two-tier):
///   (a) SAME-FORM hard gate — JIT(candidate) vs oracle(candidate) on the
///       candidate's own arena, each point judged against the bound THAT POINT
///       composed. Divergence is a miscompile → `Err(GateFailure::SameForm)`;
///       the caller excludes the arm loudly and counts it.
///   (b) CROSS-FORM conditioned gate — oracle(candidate) vs oracle(original),
///       compared only where both points are well-conditioned, accepted within
///       the sum of their bounds. Disagreement → `Err(GateFailure::CrossForm)`
///       with BOTH forms printed. At ill-conditioned points disagreement is
///       algebraic-rewrite contract, counted as metadata only.
///
/// The original form is gated against ITS own oracle by its own `noswap`
/// invocation of this function, so every form in the run is same-form checked
/// exactly once.
///
/// On success returns the gated [`PolicyCheck`] together with the exact
/// [`ExecutableCode`] that passed: phase B times THAT object
/// (`BenchSession::benchmark_compiled`), so no compilation happens after phase
/// A and the timed artifact is bit-identical to the checked artifact.
fn check_policy(
    label: &str,
    original: (&ExprArena, ExprId),
    candidate: (&ExprArena, ExprId),
    grid: &CheckGrid,
) -> Result<(ExecutableCode, PolicyCheck), GateFailure> {
    let (orig_arena, orig_root) = original;
    let (cand_arena, cand_root) = candidate;

    // `DifferentialCheck::at` panics (by design) on shapes the oracle cannot
    // evaluate, so screen first and report them as an exclusion.
    for (which, (arena, root)) in [("candidate", candidate), ("original", original)] {
        if let Err(detail) = screen_for_oracle(arena, root) {
            return Err(GateFailure::OracleUnsupported(format!(
                "{label}: the {which} form is not oracle-checkable: {detail}"
            )));
        }
    }

    let compiled = compile_arena_dag(cand_arena, cand_root)
        .map_err(|e| GateFailure::CompileFailed(format!("{label}: JIT compile failed: {e}")))?;

    let bindings = BindingTable::empty();
    // Prepared ONCE per form: `new` clones and rebuilds the arena, so building
    // it inside the point loop would pay that 72 times.
    let cand_check = DifferentialCheck::new(cand_arena, cand_root);
    let orig_check = DifferentialCheck::new(orig_arena, orig_root);
    let mask_root = cand_check.root_is_mask_valued();

    let mut check = PolicyCheck {
        expected: vec![None; grid.tuples.len()],
        mask_root,
        mask_near_threshold: 0,
        checked: 0,
        skipped: 0,
        unbounded: 0,
        cross_compared: 0,
        cross_ill_points: 0,
        cross_ill_disagreements: 0,
        cross_wc_disagreements: 0,
    };
    let mut divergence: Option<CrossFormDivergence> = None;

    for (pi, &p) in grid.points.iter().enumerate() {
        let pc = cand_check.at(&p, &bindings);
        if pc.is_platform_divergent() {
            // Contract, not a defect: no backend's answer is "the" answer here.
            check.skipped += 1;
            continue;
        }

        // ── Tier (a): same-form hard gate ───────────────────────────────────
        let got = run_scalar(&compiled.code, p[0], p[1], p[2], p[3]);
        if mask_root {
            match pc.classify_mask_root(got) {
                MaskVerdict::Agree => {}
                MaskVerdict::NearThreshold => {
                    // Either verdict is legal here, so the point binds neither
                    // tier and is withheld from per-repeat verification.
                    check.mask_near_threshold += 1;
                    continue;
                }
                MaskVerdict::Miscompile => {
                    return Err(GateFailure::SameForm(format!(
                        "{label}: mask-valued root produced the OPPOSITE mask at \
                         ({}, {}, {}, {}), with the deciding comparison well clear of its \
                         threshold: got {:#010x}, want {:#010x} — a wrong answer, not \
                         near-threshold contract (Codex R2)",
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        got.to_bits(),
                        pc.value().to_bits()
                    )));
                }
                MaskVerdict::InvalidPattern {
                    got_is_mask,
                    want_is_mask,
                } => {
                    return Err(GateFailure::SameForm(format!(
                        "{label}: mask-valued root produced an INVALID lane pattern at \
                         ({}, {}, {}, {}): got {:#010x} (is_mask={got_is_mask}), want \
                         {:#010x} (is_mask={want_is_mask}) — a broken mask corrupts every \
                         bitwise consumer; miscompile, no proximity and no bound excuses it",
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        got.to_bits(),
                        pc.value().to_bits()
                    )));
                }
            }
        } else {
            match pc.verdict(got) {
                PointVerdict::Accept => {}
                PointVerdict::Unbounded => {
                    check.unbounded += 1;
                    continue;
                }
                PointVerdict::Reject => {
                    return Err(GateFailure::SameForm(format!(
                        "{label}: JIT disagrees with the scalar oracle on its OWN arena at \
                         ({}, {}, {}, {}): got {got:e}, want {:e}, composed bound {:e} \
                         (relative {:e}) — miscompile/lowering bug, not a rewrite issue \
                         (audit H1/H2)",
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        pc.value(),
                        pc.error_bound(),
                        pc.relative_error_bound()
                    )));
                }
            }
        }
        check.checked += 1;
        // grid.points lists the harness tuples first, in order; only points
        // that just passed the same-form gate are re-verified per repeat.
        if pi < check.expected.len() {
            check.expected[pi] = Some(pc);
        }

        // ── Tier (b): cross-form conditioned gate ───────────────────────────
        let po = orig_check.at(&p, &bindings);
        if po.is_platform_divergent() {
            continue;
        }
        if !pc.is_well_conditioned() || !po.is_well_conditioned() {
            check.cross_ill_points += 1;
            if forms_agree(label, &p, pc, po) == FormAgreement::Disagree {
                // Algebraically-valid rewrites legitimately change values at or
                // near singularities (plan 0.4) — metadata, never an exclusion.
                check.cross_ill_disagreements += 1;
            }
            continue;
        }
        check.cross_compared += 1;
        if forms_agree(label, &p, pc, po) == FormAgreement::Disagree {
            check.cross_wc_disagreements += 1;
            if divergence.is_none() {
                divergence = Some(CrossFormDivergence {
                    point: p,
                    cand: pc,
                    orig: po,
                });
            }
        }
    }

    if check.checked == 0 {
        return Err(GateFailure::NoBoundedPoints(format!(
            "{label}: all {} check points were skipped (platform-divergent or unbounded) — the \
             same-form gate never ran, so no timing from this arm is trustworthy. {}",
            grid.points.len(),
            check.tally_line()
        )));
    }

    if let Some(d) = divergence {
        return Err(GateFailure::CrossForm(format!(
            "{label}: the extracted form disagrees with the ORIGINAL in the scalar oracle at a \
             WELL-CONDITIONED point ({}, {}, {}, {}) — extraction/e-graph bug.\n  \
             candidate {:e} ± {:e} (relative {:e})\n  original  {:e} ± {:e} (relative {:e})\n  \
             {} of them disagreed\n  {}\n  ORIGINAL kernel:  {}\n  CANDIDATE kernel: {}",
            d.point[0],
            d.point[1],
            d.point[2],
            d.point[3],
            d.cand.value(),
            d.cand.error_bound(),
            d.cand.relative_error_bound(),
            d.orig.value(),
            d.orig.error_bound(),
            d.orig.relative_error_bound(),
            check.cross_wc_disagreements,
            check.tally_line(),
            arena_to_kernel_code(orig_arena, orig_root),
            arena_to_kernel_code(cand_arena, cand_root),
        )));
    }

    Ok((compiled.code, check))
}

/// Re-verify one timed repeat: the harness records lane-0 outputs of the exact
/// `ExecutableCode` it timed at every input tuple, so this comparison pins "the
/// artifact that was checked is the artifact that was measured" per measurement
/// (H2's check-vs-time gap). It replays the SAME [`PointCheck`] that gated the
/// tuple in phase A, so there is no second acceptance rule to drift. A mismatch
/// after phase A passed means nondeterministic codegen or harness corruption —
/// panic, never skip.
fn verify_repeat_outputs(label: &str, repeat: usize, check: &PolicyCheck, outputs: &[[f32; 4]]) {
    assert_eq!(
        outputs.len(),
        check.expected.len(),
        "{label} repeat {repeat}: harness recorded {} outputs for {} tuples",
        outputs.len(),
        check.expected.len()
    );
    for (i, (lanes, want)) in outputs.iter().zip(&check.expected).enumerate() {
        let Some(pc) = *want else { continue };
        let got = lanes[0];
        if check.mask_root {
            // Phase A saw Agree at every tuple that kept an expected value, so
            // ANY other verdict here means the code object's output changed
            // between check and timing.
            let verdict = pc.classify_mask_root(got);
            assert_eq!(
                verdict,
                MaskVerdict::Agree,
                "{label} repeat {repeat}: the timed code object's mask lane changed from the \
                 phase-A verdict at input tuple {i}: got {:#010x}, want {:#010x} ({verdict:?}) — \
                 nondeterministic codegen or harness corruption",
                got.to_bits(),
                pc.value().to_bits()
            );
            continue;
        }
        let verdict = pc.verdict(got);
        assert_eq!(
            verdict,
            PointVerdict::Accept,
            "{label} repeat {repeat}: the timed code object disagrees with the oracle value that \
             gated it at input tuple {i}: got {got:e}, want {:e} ± {:e} ({verdict:?}) — \
             nondeterministic codegen or harness corruption",
            pc.value(),
            pc.error_bound()
        );
    }
}

// ---------------------------------------------------------------------------
// Phase A: saturate, extract, compile, gate — everything CPU-heavy happens
// here, before any timing (M3).
// ---------------------------------------------------------------------------

/// One gated, ready-to-time policy. `code` is the EXACT executable that
/// passed the phase-A oracle gate; phase B times it directly
/// (`BenchSession::benchmark_compiled`) — recompiling there would interleave
/// compilation, allocation, and icache pollution with timing AND break the
/// "time the same code object that passed the check" invariant.
struct PreparedPolicy {
    arena: ExprArena,
    root: ExprId,
    code: ExecutableCode,
    check: PolicyCheck,
}

/// One extraction policy's phase-A outcome. `extract_us` is always recorded —
/// extraction itself precedes (and is independent of) the compile/gate.
struct PolicyPrep {
    extract_us: f64,
    ready: Option<PreparedPolicy>,
    fail: Option<GateFailure>,
}

struct PreparedKernel {
    name: String,
    family: &'static str,
    n_orig: usize,
    noswap: PreparedPolicy,
    static_prep: PolicyPrep,
    nnue_prep: PolicyPrep,
}

impl PreparedKernel {
    /// The policy an arm times, plus its extraction overhead (µs). `None` if
    /// the policy failed its phase-A gate (already recorded loudly).
    fn arm_policy(&self, arm: Arm) -> Option<(&PreparedPolicy, f64)> {
        match arm {
            Arm::NoSwap => Some((&self.noswap, 0.0)),
            Arm::Static | Arm::StaticAa => self
                .static_prep
                .ready
                .as_ref()
                .map(|p| (p, self.static_prep.extract_us)),
            Arm::Nnue => self
                .nnue_prep
                .ready
                .as_ref()
                .map(|p| (p, self.nnue_prep.extract_us)),
        }
    }
}

struct KernelInput {
    name: String,
    family: &'static str,
    arena: ExprArena,
    root: ExprId,
}

struct PrepareCtx<'a> {
    nnue: &'a ExprNnue,
    grid: &'a CheckGrid,
}

/// Gate outcomes over every (kernel, policy) invocation of [`check_policy`].
///
/// The two rates it feeds are the reason a single bad expression no longer
/// kills a run: an exclusion is recorded here and the process continues, and
/// only the RATE decides whether the run is trustworthy (smoke B4).
#[derive(Clone, Copy, Debug, Default)]
struct GateTally {
    gates: usize,
    /// Gates run on an EXTRACTED policy (static / nnue) — the only attempts
    /// that can produce a [`GateFailure::CrossForm`]. The no-swap gate
    /// compares a form with itself, so counting it in the cross-form
    /// denominator systematically diluted the rate: 12% of extractions
    /// changing values reported as 8% under the healthy three-gate shape and
    /// never tripped the 10% alarm (Codex 3759014680).
    extracted_gates: usize,
    oracle_unsupported: usize,
    no_bounded_points: usize,
    compile_failed: usize,
    same_form: usize,
    cross_form: usize,
}

impl GateTally {
    /// Record the no-swap gate's outcome. `None` is a pass. A no-swap form is
    /// compared with itself, so it can never yield `CrossForm` — it belongs in
    /// the all-gate denominator only.
    fn record_noswap(&mut self, outcome: Option<&GateFailure>) {
        assert!(
            !matches!(outcome, Some(GateFailure::CrossForm(_))),
            "a no-swap gate produced CrossForm — it compares a form with itself, so this is a \
             harness bug, not a divergence"
        );
        self.gates += 1;
        self.count(outcome);
    }

    /// Record an extracted-policy gate's outcome. `None` is a pass.
    fn record_extracted(&mut self, outcome: Option<&GateFailure>) {
        self.gates += 1;
        self.extracted_gates += 1;
        self.count(outcome);
    }

    fn count(&mut self, outcome: Option<&GateFailure>) {
        match outcome {
            None => {}
            Some(GateFailure::OracleUnsupported(_)) => self.oracle_unsupported += 1,
            Some(GateFailure::NoBoundedPoints(_)) => self.no_bounded_points += 1,
            Some(GateFailure::CompileFailed(_)) => self.compile_failed += 1,
            Some(GateFailure::SameForm(_)) => self.same_form += 1,
            Some(GateFailure::CrossForm(_)) => self.cross_form += 1,
        }
    }

    /// # Panics
    ///
    /// Panics if no gate ran at all — a rate over an empty denominator is not
    /// a number, and printing one would be the silent failure this project
    /// forbids.
    fn rate(&self, count: usize) -> f64 {
        assert!(
            self.gates > 0,
            "gate rates requested before any policy was gated"
        );
        count as f64 / self.gates as f64
    }

    fn same_form_rate(&self) -> f64 {
        self.rate(self.same_form)
    }

    /// Cross-form divergence rate over EXTRACTED-policy attempts only
    /// (Codex 3759014680) — the attempts that could have diverged.
    ///
    /// # Panics
    ///
    /// Panics if divergences were recorded with no extracted gate — that is a
    /// tally-keeping bug, not a rate.
    fn cross_form_rate(&self) -> f64 {
        if self.extracted_gates == 0 {
            assert_eq!(
                self.cross_form, 0,
                "cross-form divergences recorded without any extracted-policy gate"
            );
            return 0.0;
        }
        self.cross_form as f64 / self.extracted_gates as f64
    }

    fn describe(&self) -> String {
        format!(
            "{} gates ({} extracted): {} same-form miscompiles ({:.2}% of all), {} cross-form \
             divergences ({:.2}% of extracted), {} compile failures, {} oracle-unsupported, \
             {} bounded nothing",
            self.gates,
            self.extracted_gates,
            self.same_form,
            self.same_form_rate() * 100.0,
            self.cross_form,
            self.cross_form_rate() * 100.0,
            self.compile_failed,
            self.oracle_unsupported,
            self.no_bounded_points
        )
    }

    /// The alarms this tally trips, as ready-to-print sentences. Empty means
    /// the run's exclusions stayed inside their documented thresholds.
    fn alarms(&self) -> Vec<String> {
        let mut alarms = Vec::new();
        if self.same_form_rate() > MAX_SAME_FORM_MISCOMPILE_RATE {
            alarms.push(format!(
                "SAME-FORM miscompile rate {:.2}% exceeds {:.0}%: the JIT and the scalar oracle \
                 disagree on the SAME arena beyond the composed error bound in {} of {} gates — \
                 a miscompile/lowering bug or an oracle regression, never a rewrite judgment call",
                self.same_form_rate() * 100.0,
                MAX_SAME_FORM_MISCOMPILE_RATE * 100.0,
                self.same_form,
                self.gates
            ));
        }
        if self.cross_form_rate() > MAX_CROSS_FORM_DIVERGENCE_RATE {
            alarms.push(format!(
                "CROSS-FORM divergence rate {:.2}% exceeds {:.0}%: extracted forms disagreed with \
                 their originals at WELL-CONDITIONED points in {} of {} extracted-policy gates \
                 (no-swap gates cannot diverge and are excluded from this denominator, Codex \
                 3759014680). One such case is contract at a singularity; this many means the \
                 extractor is systematically changing values",
                self.cross_form_rate() * 100.0,
                MAX_CROSS_FORM_DIVERGENCE_RATE * 100.0,
                self.cross_form,
                self.extracted_gates
            ));
        }
        alarms
    }
}

fn prepare_kernel(
    input: KernelInput,
    ctx: &PrepareCtx<'_>,
    tally: &mut GateTally,
    out: &mut Jsonl,
) -> Option<PreparedKernel> {
    let KernelInput {
        name,
        family,
        arena,
        root,
    } = input;
    let n_orig = arena.node_count_subtree(root);

    // NO-SWAP: the unmodified input, gated against the independent scalar
    // oracle. A same-form failure here is a miscompile/lowering bug in the JIT
    // backend (plan 0.4 tier a): exclude the whole kernel LOUDLY and count it —
    // without a trustworthy baseline no arm can be timed. Excluding rather than
    // aborting is deliberate: the remaining kernels are still measurable, and
    // the rate alarm at the end is what judges the run.
    let gated = check_policy(
        &format!("{name}/noswap"),
        (&arena, root),
        (&arena, root),
        ctx.grid,
    );
    tally.record_noswap(gated.as_ref().err());
    let (noswap_code, noswap_check) = match gated {
        Ok(gated) => gated,
        Err(failure) => {
            eprintln!(
                "[KERNEL EXCLUDED] {name}: the *original*, unmodified expression failed its \
                 phase-A gate ({}): {}",
                failure.kind(),
                failure.detail()
            );
            out.write(&KernelExcludedRecord {
                record: "kernel_excluded",
                kernel: &name,
                family,
                kind: failure.kind(),
                detail: failure.detail(),
            });
            return None;
        }
    };
    out.write(&PolicyPreparedRecord::new(
        &name,
        family,
        "noswap",
        0.0,
        &noswap_check,
    ));

    // Shared saturated e-graph for both learned and static extraction. This
    // is the CPU-heavy step (M3) — it runs here, in phase A, never between
    // timed measurements.
    let mut eg = EGraph::with_rules(all_rules());
    let root_class = eg.add_arena(&arena, root);
    eg.saturate_with_limit(SATURATE_LIMIT);

    // (a) NNUE extraction.
    let t_nnue = Instant::now();
    let extractor = IncrementalExtractor::new(ctx.nnue, EXTRACT_TOP_K);
    let (_cost, nnue_choices) = extractor.extract_choices_only(&eg, root_class);
    let (nnue_arena, nnue_root) = choices_to_arena(&nnue_choices);
    let extract_us_nnue = t_nnue.elapsed().as_secs_f64() * 1e6;

    // (b) STATIC extraction — CostModel::latency_prior() via `extract`, which
    // seals its DP choices into an `Extraction` and calls `choices_to_arena`
    // itself (one definition, imported, not restated). `extract_dag`'s extra
    // DAG-sharing metadata (`.shared`/`.schedule`, for codegen let-binding)
    // is unused here — this call site only ever wanted the arena.
    let t_static = Instant::now();
    let static_costs = CostModel::latency_prior();
    let (static_arena, static_root, _static_cost) = extract(&eg, root_class, &static_costs);
    let extract_us_static = t_static.elapsed().as_secs_f64() * 1e6;

    let mut gate = |policy: &'static str,
                    cand_arena: ExprArena,
                    cand_root: ExprId,
                    extract_us: f64,
                    tally: &mut GateTally|
     -> PolicyPrep {
        let gated = check_policy(
            &format!("{name}/{policy}"),
            (&arena, root),
            (&cand_arena, cand_root),
            ctx.grid,
        );
        tally.record_extracted(gated.as_ref().err());
        match gated {
            Ok((code, check)) => {
                out.write(&PolicyPreparedRecord::new(
                    &name, family, policy, extract_us, &check,
                ));
                PolicyPrep {
                    extract_us,
                    ready: Some(PreparedPolicy {
                        arena: cand_arena,
                        root: cand_root,
                        code,
                        check,
                    }),
                    fail: None,
                }
            }
            Err(failure) => {
                // Loud, with both forms already inside the detail for a
                // cross-form divergence — a gate that says "extraction bug"
                // without showing either form cannot be investigated.
                eprintln!("[ARM EXCLUDED: {}] {}", failure.kind(), failure.detail());
                out.write(&PolicyFailureRecord {
                    record: "policy_failure",
                    kernel: &name,
                    family,
                    policy,
                    extract_us,
                    kind: failure.kind(),
                    detail: failure.detail(),
                });
                PolicyPrep {
                    extract_us,
                    ready: None,
                    fail: Some(failure),
                }
            }
        }
    };

    let static_prep = gate(
        "static",
        static_arena,
        static_root,
        extract_us_static,
        tally,
    );
    let nnue_prep = gate("nnue", nnue_arena, nnue_root, extract_us_nnue, tally);

    Some(PreparedKernel {
        name,
        family,
        n_orig,
        noswap: PreparedPolicy {
            arena,
            root,
            code: noswap_code,
            check: noswap_check,
        },
        static_prep,
        nnue_prep,
    })
}

// ---------------------------------------------------------------------------
// Machine-readable output (0.3b): JSONL records
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PolicyPreparedRecord<'a> {
    record: &'static str,
    kernel: &'a str,
    family: &'static str,
    policy: &'static str,
    extract_us: f64,
    /// Same-form points where JIT-vs-oracle was actually compared.
    checked_points: usize,
    /// Points skipped as platform-divergent (`Select`-aware).
    skipped_points: usize,
    /// Points whose composed bound was not finite: not a pass, not a failure.
    unbounded_points: usize,
    /// Whether the policy's root is mask-valued (bit-pattern gate).
    mask_root: bool,
    /// Mask-valued roots only: same-form near-threshold valid-mask
    /// disagreements (contract metadata, plan 0.4 / Codex R2).
    mask_near_threshold_points: usize,
    /// Points where BOTH forms were well-conditioned and were compared.
    cross_compared_points: usize,
    /// Cross-form points filtered as ill-conditioned (plan 0.4 metadata).
    cross_ill_points: usize,
    /// Ill-conditioned cross-form disagreements — contract, not failure.
    cross_ill_disagreements: usize,
}

impl<'a> PolicyPreparedRecord<'a> {
    fn new(
        kernel: &'a str,
        family: &'static str,
        policy: &'static str,
        extract_us: f64,
        check: &PolicyCheck,
    ) -> Self {
        Self {
            record: "policy_prepared",
            kernel,
            family,
            policy,
            extract_us,
            checked_points: check.checked,
            skipped_points: check.skipped,
            unbounded_points: check.unbounded,
            mask_root: check.mask_root,
            mask_near_threshold_points: check.mask_near_threshold,
            cross_compared_points: check.cross_compared,
            cross_ill_points: check.cross_ill_points,
            cross_ill_disagreements: check.cross_ill_disagreements,
        }
    }
}

/// A kernel whose *original* form failed its phase-A gate: excluded from all
/// timing, loudly (plan 0.4). `kind` is the [`GateFailure`] slug, so the
/// analysis can separate miscompiles from unquarantinable shapes.
#[derive(Serialize)]
struct KernelExcludedRecord<'a> {
    record: &'static str,
    kernel: &'a str,
    family: &'static str,
    kind: &'static str,
    detail: &'a str,
}

#[derive(Serialize)]
struct PolicyFailureRecord<'a> {
    record: &'static str,
    kernel: &'a str,
    family: &'static str,
    policy: &'static str,
    extract_us: f64,
    /// [`GateFailure::kind`] slug: `same_form_miscompile` accuses the JIT,
    /// `cross_form_divergence` accuses the extractor, and the other two accuse
    /// neither.
    kind: &'static str,
    detail: &'a str,
}

#[derive(Serialize)]
struct MeasurementRecord<'a> {
    record: &'static str,
    kernel: &'a str,
    family: &'static str,
    policy: &'static str,
    repeat: usize,
    ns: f64,
    adjusted_ns: f64,
    call_overhead_ns: f64,
    iqr_ns: f64,
    mode: &'static str,
    repeat_batches: usize,
    extract_us: f64,
    correctness: &'static str,
    /// Most recent sentinel reading when this measurement was taken (audit
    /// H4). The run's opening calibration lives in the summary record.
    local_sentinel_ns: f64,
    /// `calibration_ns / local_sentinel_ns` — the running clock-speed
    /// correction: multiply `ns`/`adjusted_ns` by it to re-express the
    /// measurement in the run's opening clock.
    sentinel_normalization: f64,
}

#[derive(Serialize)]
struct SentinelSampleRecord {
    expr_index: usize,
    ns: f64,
}

#[derive(Serialize)]
struct RunSummaryRecord<'a> {
    record: &'static str,
    ts_unix: u64,
    config_hash: &'a str,
    /// `git rev-parse HEAD` (suffixed `-dirty`), or `"unversioned"` when git
    /// could not answer — see [`SourceVersion`]. Part of the config-hash input
    /// too, so a measurement-logic change cannot reuse a hash.
    source_rev: &'a str,
    /// Hash of the uncommitted diff when `source_rev` is `-dirty` (Codex
    /// 3741889899): the suffix alone collapses every uncommitted variant onto
    /// one identity, so two different working trees at the same HEAD produced
    /// indistinguishable journal lines.
    source_diff_hash: Option<&'a str>,
    weights_path: String,
    weights_bytes: usize,
    /// The mode the extraction head's targets were minted in, read from the
    /// weights' mint sidecar and asserted equal to `mode` (Codex R6).
    weights_mint_mode: String,
    /// Which corpus tier was measured (`dev` for a selection run, `final` only
    /// under `--final-eval`).
    tier: &'static str,
    /// Whether this is a publication run against FINAL — such a run emits no
    /// selection verdict at all (plan 0.2/4.3, Codex R5).
    final_eval: bool,
    corpus_path: String,
    corpus_entries: usize,
    /// Content hash of the tier file this run measured (Codex 3744586352).
    /// Without it, regenerating a tier from a different seed, target, or
    /// manifest produces a run that is indistinguishable in `config_hash` from
    /// the previous one, and the journal claims two different experiments were
    /// the same setup.
    corpus_hash: &'a str,
    max_kernels: usize,
    sample_seed: String,
    saturate_limit: usize,
    extract_top_k: usize,
    timed_repeats: usize,
    order_seed: String,
    mode: &'static str,
    check_points: usize,
    geomean_nnue_static: Option<f64>,
    /// Bootstrap 95% CI on `geomean_nnue_static` over kernels — the interval
    /// the verdict is decided against (kernel-set variance dominated round 0's
    /// run-to-run spread by ~40x over the A/A floor).
    geomean_nnue_static_ci_lo: Option<f64>,
    geomean_nnue_static_ci_hi: Option<f64>,
    bootstrap_iters: usize,
    bootstrap_seed: String,
    /// Kernels contributing a paired nnue/static ratio.
    nnue_static_pairs: usize,
    /// Paired per-kernel ratio distribution — the geomean has repeatedly been
    /// dominated by single kernels, so the distribution rides beside it.
    ratio_median: Option<f64>,
    ratio_q1: Option<f64>,
    ratio_q3: Option<f64>,
    ratio_wins: usize,
    ratio_losses: usize,
    /// Leave-one-out sensitivity: the range of geomeans dropping each kernel
    /// in turn, and the kernel whose removal shifts it most.
    loo_geomean_min: Option<f64>,
    loo_geomean_max: Option<f64>,
    loo_most_influential: Option<String>,
    loo_max_shift_pct: Option<f64>,
    geomean_static_noswap: Option<f64>,
    aa_geomean: Option<f64>,
    noise_floor_pct: Option<f64>,
    sentinel_calibration_ns: f64,
    sentinel_samples: Vec<SentinelSampleRecord>,
    call_overhead_ns: f64,
    kernels_excluded: usize,
    named_static_fail: usize,
    named_nnue_fail: usize,
    syn_static_fail: usize,
    syn_nnue_fail: usize,
    /// Per-policy failure/exclusion rate over every kernel entering phase A.
    failure_rate_noswap: f64,
    failure_rate_static: f64,
    failure_rate_nnue: f64,
    /// Whether any policy's failure rate exceeded
    /// [`MAX_POLICY_FAILURE_RATE`], censoring the verdict (Codex P1).
    verdict_censored: bool,
    /// Total ill-conditioned cross-form disagreements across all prepared
    /// policies (plan 0.4 metadata — contract, never an exclusion).
    cross_ill_disagreements_total: usize,
    /// Gate outcomes over every (kernel, policy) gate, and the two rates whose
    /// documented thresholds fail the run (smoke B4).
    gates: usize,
    same_form_miscompiles: usize,
    same_form_miscompile_rate: f64,
    cross_form_divergences: usize,
    cross_form_divergence_rate: f64,
    gate_compile_failures: usize,
    gate_oracle_unsupported: usize,
    /// Gates excluded because not one grid point produced a bounded verdict —
    /// the same-form gate never decided anything there.
    gate_no_bounded_points: usize,
    gate_alarms: Vec<String>,
    /// Whether [`GateTally::alarms`] tripped. The verdict is derived from this
    /// FIRST (Codex 3744586360), so a consumer never has to parse the verdict
    /// string to learn that the correctness gate failed.
    correctness_gate_failed: bool,
    /// How much of the grid the gated policies actually decided: the
    /// denominator behind `same_form_miscompiles`. Zero miscompiles over a grid
    /// that bounded nothing would be a vacuous headline (Codex 3744586366).
    bound_coverage_mean_pct: Option<f64>,
    bound_coverage_gates: usize,
    mean_extract_us_static: Option<f64>,
    mean_extract_us_nnue: Option<f64>,
    verdict: &'a str,
}

#[derive(Serialize)]
/// This binary's own metrics for one line of `docs/results/journal.jsonl` —
/// wrapped by [`pixelflow_pipeline::journal::JournalEntry`] for the
/// provenance envelope (`record`, `ts_unix`, `config: ConfigHash`, `weights:
/// ArtifactId`) every journal-writing binary now shares
/// (docs/plans/2026-08-17-cost-model-domain.md, J15) — `config` already
/// carries the corpus identity and source revision this struct used to
/// duplicate as flat fields.
struct BenchMetrics<'a> {
    tier: &'static str,
    corpus_path: String,
    corpus_entries: usize,
    final_eval: bool,
    mode: &'static str,
    weights_mint_mode: String,
    geomean_nnue_static: Option<f64>,
    /// Bootstrap 95% CI over kernels — the verdict's instrument. A journal
    /// line whose geomean sits inside another line's CI is a rerun, not a
    /// change.
    geomean_nnue_static_ci_lo: Option<f64>,
    geomean_nnue_static_ci_hi: Option<f64>,
    nnue_static_pairs: usize,
    ratio_median: Option<f64>,
    ratio_q1: Option<f64>,
    ratio_q3: Option<f64>,
    ratio_wins: usize,
    ratio_losses: usize,
    loo_geomean_min: Option<f64>,
    loo_geomean_max: Option<f64>,
    loo_most_influential: Option<String>,
    geomean_static_noswap: Option<f64>,
    noise_floor_pct: Option<f64>,
    kernels_excluded: usize,
    named_static_fail: usize,
    named_nnue_fail: usize,
    syn_static_fail: usize,
    syn_nnue_fail: usize,
    failure_rate_noswap: f64,
    failure_rate_static: f64,
    failure_rate_nnue: f64,
    same_form_miscompile_rate: f64,
    cross_form_divergence_rate: f64,
    /// The correctness gate's own answer, recorded beside the verdict so a
    /// journal reader never has to infer it from the two rates (Codex
    /// 3744586360).
    correctness_gate_failed: bool,
    gate_alarms: Vec<String>,
    /// Mean fraction of grid points the gates actually decided — what the
    /// same-form miscompile rate is a rate *of* (Codex 3744586366).
    bound_coverage_mean_pct: Option<f64>,
    verdict_censored: bool,
    verdict: &'a str,
    data_jsonl: String,
}

/// Line-per-record JSON writer that fails loudly on any serialization or IO
/// error, flushing per record so a mid-run sentinel panic still leaves every
/// completed measurement on disk.
struct Jsonl {
    path: PathBuf,
    file: fs::File,
}

impl Jsonl {
    fn create(path: PathBuf) -> Self {
        let file = fs::File::create(&path)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
        Self { path, file }
    }

    fn write<T: Serialize>(&mut self, record: &T) {
        let line = serde_json::to_string(record).unwrap_or_else(|e| {
            panic!(
                "failed to serialize record for {}: {e}",
                self.path.display()
            )
        });
        writeln!(self.file, "{line}")
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", self.path.display()));
    }
}

/// serde_json refuses non-finite floats; a NaN geomean (no valid pairs) is
/// represented as `null` in the JSONL, never dropped silently.
fn finite_or_none(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

// ---------------------------------------------------------------------------
// Aggregation + reporting helpers
// ---------------------------------------------------------------------------

fn median(mut values: Vec<f64>, context: &str) -> f64 {
    assert!(!values.is_empty(), "median({context}): no samples");
    values.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("median({context}): non-comparable samples {a} vs {b}"))
    });
    values[values.len() / 2]
}

/// Geometric mean; panics on any non-positive value (audit L2 — one zero used
/// to make the whole geomean 0/−inf with no error). Callers assert positivity
/// with kernel/policy context before values get here; this is the backstop.
fn geomean(values: &[f64], context: &str) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let sum_ln: f64 = values
        .iter()
        .map(|&v| {
            assert!(
                v > 0.0,
                "geomean({context}): non-positive value {v} would poison ln (audit L2)"
            );
            v.ln()
        })
        .sum();
    (sum_ln / values.len() as f64).exp()
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(v) => format!("{v:.3}"),
        None => "FAIL".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Verdict statistics: bootstrap CI over kernels, paired ratio distribution,
// leave-one-out sensitivity.
//
// Round 0's estimator problem in one line: the geomean's run-to-run spread
// (0.8815 / 0.9285 / 1.0389) was ~40x the within-run A/A noise floor
// (0.11–0.44%), because the dominant variance source is WHICH kernels are in
// the set, not timing noise. So the interval the verdict needs is over
// KERNELS — a percentile bootstrap resampling the paired per-kernel ratios —
// and the verdict compares that interval against the ±5% gate instead of a
// point estimate that has repeatedly flipped sign on a set change (a 12-kernel
// run's 0.9128 became 1.1031 by dropping ONE kernel).
// ---------------------------------------------------------------------------

/// Bootstrap resamples for the geomean CI. 10k keeps the percentile estimate's
/// own Monte-Carlo error well under 0.1% at negligible cost (the resample is
/// over at most a few thousand precomputed log-ratios).
const BOOTSTRAP_ITERS: usize = 10_000;

/// Seed for the bootstrap resampling — fixed so the CI is reproducible run to
/// run, and recorded in the run summary.
const BOOTSTRAP_SEED: u64 = 0xB007_57A9_2026_0810;

/// Percentile-bootstrap 95% CI on the geomean of `ratios`, resampling over
/// kernels with replacement. `None` below 2 ratios — an interval over one
/// kernel is not an interval.
///
/// # Panics
///
/// Panics on non-positive ratios (audit L2 — same contract as [`geomean`]).
fn bootstrap_geomean_ci(ratios: &[f64], iters: usize, seed: u64) -> Option<(f64, f64)> {
    if ratios.len() < 2 {
        return None;
    }
    let logs: Vec<f64> = ratios
        .iter()
        .map(|&r| {
            assert!(
                r > 0.0,
                "bootstrap_geomean_ci: non-positive ratio {r} would poison ln (audit L2)"
            );
            r.ln()
        })
        .collect();
    let n = logs.len();
    let mut rng = SeededRng::new(seed);
    let mut means: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += logs[(rng.next_u64() % n as u64) as usize];
        }
        means.push((sum / n as f64).exp());
    }
    means.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("bootstrap_geomean_ci: non-comparable means {a} vs {b}"))
    });
    let pick = |p: f64| means[(p * (iters - 1) as f64).round() as usize];
    Some((pick(0.025), pick(0.975)))
}

/// The paired per-kernel ratio distribution the geomean summarizes — printed
/// beside it because the geomean has repeatedly been dominated by single
/// kernels.
struct RatioDistribution {
    median: f64,
    q1: f64,
    q3: f64,
    /// Ratios < 1.0 (NNUE strictly faster).
    wins: usize,
    /// Ratios > 1.0 (NNUE strictly slower).
    losses: usize,
    ties: usize,
}

impl RatioDistribution {
    fn of(ratios: &[f64]) -> Option<Self> {
        if ratios.is_empty() {
            return None;
        }
        let mut sorted = ratios.to_vec();
        sorted.sort_unstable_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("RatioDistribution: non-comparable ratios {a} vs {b}"))
        });
        let n = sorted.len();
        Some(Self {
            median: sorted[n / 2],
            q1: sorted[n / 4],
            q3: sorted[3 * n / 4],
            wins: ratios.iter().filter(|&&r| r < 1.0).count(),
            losses: ratios.iter().filter(|&&r| r > 1.0).count(),
            ties: ratios.iter().filter(|&&r| r == 1.0).count(),
        })
    }
}

/// Leave-one-out sensitivity of the geomean: the range of geomeans obtained by
/// dropping each kernel in turn, plus the single most influential kernel.
struct LooSensitivity {
    min: f64,
    max: f64,
    /// Kernel whose removal shifts the geomean the most.
    most_influential: String,
    /// That shift, in percent of the full geomean.
    shift_pct: f64,
}

impl LooSensitivity {
    fn of(pairs: &[(String, f64)]) -> Option<Self> {
        if pairs.len() < 2 {
            return None;
        }
        let logs: Vec<f64> = pairs.iter().map(|(_, r)| r.ln()).collect();
        let total: f64 = logs.iter().sum();
        let n = pairs.len();
        let full = (total / n as f64).exp();
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut most_influential = String::new();
        let mut best_shift = -1.0_f64;
        for (i, (name, _)) in pairs.iter().enumerate() {
            let loo = ((total - logs[i]) / (n - 1) as f64).exp();
            min = min.min(loo);
            max = max.max(loo);
            let shift = ((loo - full) / full).abs() * 100.0;
            if shift > best_shift {
                best_shift = shift;
                most_influential = name.clone();
            }
        }
        Some(Self {
            min,
            max,
            most_influential,
            shift_pct: best_shift,
        })
    }
}

/// Per-kernel medians across the timed repeats.
struct KernelSummary {
    name: String,
    family: &'static str,
    n_orig: usize,
    ns_noswap: f64,
    ns_static: Option<f64>,
    ns_static_aa: Option<f64>,
    ns_nnue: Option<f64>,
    extract_us_static: f64,
    extract_us_nnue: f64,
    static_fail: bool,
    nnue_fail: bool,
}

fn summarize(kernel: &PreparedKernel, samples: &[Vec<f64>; 4]) -> KernelSummary {
    let med = |arm: Arm| -> Option<f64> {
        let v = &samples[arm.idx()];
        if v.is_empty() {
            return None;
        }
        assert_eq!(
            v.len(),
            TIMED_REPEATS,
            "kernel {} policy {}: expected {TIMED_REPEATS} timed repeats, got {}",
            kernel.name,
            arm.name(),
            v.len()
        );
        let m = median(v.clone(), &format!("{}/{}", kernel.name, arm.name()));
        assert!(
            m > 0.0,
            "kernel {} policy {}: median opening-clock ns {m} must be positive before entering \
             any geomean (audit L2)",
            kernel.name,
            arm.name()
        );
        Some(m)
    };
    let ns_noswap = med(Arm::NoSwap).unwrap_or_else(|| {
        panic!(
            "kernel {}: no-swap arm has no samples — no-swap cannot fail preparation, so this \
             is a bench bug",
            kernel.name
        )
    });
    KernelSummary {
        name: kernel.name.clone(),
        family: kernel.family,
        n_orig: kernel.n_orig,
        ns_noswap,
        ns_static: med(Arm::Static),
        ns_static_aa: med(Arm::StaticAa),
        ns_nnue: med(Arm::Nnue),
        extract_us_static: kernel.static_prep.extract_us,
        extract_us_nnue: kernel.nnue_prep.extract_us,
        static_fail: kernel.static_prep.fail.is_some(),
        nnue_fail: kernel.nnue_prep.fail.is_some(),
    }
}

/// Maximum tolerated per-policy failure/exclusion rate before the verdict is
/// censored (Codex P1): a policy that fails correctness or compilation on
/// more than this fraction of kernels has selected its own comparison set —
/// a geomean over the surviving easy subset can still look like a PASS, so
/// no directional claim is allowed at all.
const MAX_POLICY_FAILURE_RATE: f64 = 0.10;

/// Per-policy failure/exclusion rates over ALL kernels entering phase A.
/// Kernels excluded by the noswap same-form gate count against every policy:
/// none of their arms produced a timeable, gated artifact.
#[derive(Clone, Copy, Debug)]
struct PolicyFailureRates {
    noswap: f64,
    static_: f64,
    nnue: f64,
}

impl PolicyFailureRates {
    fn compute(
        total_kernels: usize,
        kernels_excluded: usize,
        static_fail: usize,
        nnue_fail: usize,
    ) -> Self {
        assert!(total_kernels > 0, "no kernels entered phase A");
        let n = total_kernels as f64;
        Self {
            noswap: kernels_excluded as f64 / n,
            static_: (kernels_excluded + static_fail) as f64 / n,
            nnue: (kernels_excluded + nnue_fail) as f64 / n,
        }
    }

    /// Policies whose failure rate exceeds [`MAX_POLICY_FAILURE_RATE`].
    fn censoring(&self) -> Vec<(&'static str, f64)> {
        [
            ("noswap", self.noswap),
            ("static", self.static_),
            ("nnue", self.nnue),
        ]
        .into_iter()
        .filter(|&(_, rate)| rate > MAX_POLICY_FAILURE_RATE)
        .collect()
    }

    fn describe(&self) -> String {
        format!(
            "noswap {:.1}%, static {:.1}%, nnue {:.1}%",
            self.noswap * 100.0,
            self.static_ * 100.0,
            self.nnue * 100.0
        )
    }
}

/// Everything the run's one verdict is formed from.
///
/// A struct rather than five arguments because the ORDER these are consulted in
/// is the whole point (Codex 3744586360): correctness first, then censoring,
/// then margin. Passing them separately invited a call site that had not yet
/// computed the correctness alarms.
struct VerdictInputs<'a> {
    /// A FINAL-tier run reports numbers and emits no selection verdict.
    final_eval: bool,
    /// Correctness-gate alarms ([`GateTally::alarms`]) — non-empty means the
    /// run's own correctness gate failed, and no timing claim may be published.
    gate_alarms: &'a [String],
    geomean_nnue_static: f64,
    /// Bootstrap 95% CI on the geomean over kernels
    /// ([`bootstrap_geomean_ci`]), ratio space. `None` below 2 pairs. The
    /// verdict compares THIS against the ±5% gate, never the point estimate:
    /// round 0's point estimate flipped sign on a kernel-set change while the
    /// A/A floor claimed 0.11% precision.
    ci: Option<(f64, f64)>,
    /// How many kernels contributed an nnue/static ratio.
    nnue_static_pairs: usize,
    noise_floor_pct: f64,
    rates: &'a PolicyFailureRates,
}

/// The run's verdict — the single place it is formed, so a correctness alarm
/// cannot be computed after the fact.
///
/// Precedence, and the reason for it:
/// 1. **Correctness alarms.** The run measured a compiler that disagrees with
///    its own oracle (or an extractor that changes values) at a rate the plan
///    calls failing. Timing over that is not a result, so neither a PASS nor a
///    publication claim may be printed or journaled — the previous ordering
///    computed the alarms only *after* writing the journal line, so a run whose
///    correctness gate failed could publish a PASS (Codex 3744586360).
/// 2. **Censoring** and the no-pairs guard, for publication and selection runs
///    alike (Codex 3758853801): a geomean over a policy-selected survivor
///    subset — or over nothing — grounds neither a verdict nor a publication
///    claim, so `--final-eval` must not reach its report past these.
/// 3. **Publication runs** report without selecting (plan 0.2/4.3, Codex R5).
/// 4. **The margin** — as the bootstrap CI over kernels against the ±5% gate,
///    never a point estimate. The point estimate's run-to-run spread was ~40x
///    the A/A floor (kernel-set variance), so a point comparison against the
///    gate was a coin whose bias nobody had measured. A directional claim now
///    requires the WHOLE interval on the claim's side of its gate line; an
///    interval straddling a gate line is UNDERPOWERED, and the fix it names is
///    more kernels, not another rerun.
fn verdict_text(inputs: &VerdictInputs<'_>) -> String {
    let VerdictInputs {
        final_eval,
        gate_alarms,
        geomean_nnue_static,
        ci,
        nnue_static_pairs,
        noise_floor_pct,
        rates,
    } = inputs;
    let (final_eval, geomean_nnue_static, ci, nnue_static_pairs, noise_floor_pct) = (
        *final_eval,
        *geomean_nnue_static,
        *ci,
        *nnue_static_pairs,
        *noise_floor_pct,
    );

    if !gate_alarms.is_empty() {
        return format!(
            "INVALID-CORRECTNESS-GATE: {} correctness alarm(s) tripped, so this run measured a \
             pipeline that failed its own gate. No timing claim — PASS, FAIL, or publication — is \
             derivable from it; the numbers below describe an untrustworthy run and must not be \
             published or used to select weights, thresholds, or extraction policy (Codex \
             3744586360). Alarms:\n  {}",
            gate_alarms.len(),
            gate_alarms.join("\n  ")
        );
    }
    // Censoring gate next (Codex P1 + 3758853801): with a policy failing >10%
    // of kernels the geomean is computed over a policy-selected survivor
    // subset, and no margin — however large — supports a directional claim.
    // This guard sits BEFORE the publication return on purpose: a FINAL-tier
    // geomean over a survivor subset is exactly as unpublishable as it is
    // unselectable, and the earlier ordering let `--final-eval` say "may be
    // published" over numbers the censoring rule rejects.
    let censoring = rates.censoring();
    if !censoring.is_empty() {
        let offenders = censoring
            .iter()
            .map(|(name, rate)| format!("{name} {:.1}%", rate * 100.0))
            .collect::<Vec<_>>()
            .join(", ");
        return format!(
            "INCONCLUSIVE-CENSORED: policy failure rate over {:.0}% of kernels ({offenders}; \
             all rates: {}) — the surviving kernel set is selected on policy success, so the \
             geomean cannot ground any verdict or publication claim (Codex P1, 3758853801).",
            MAX_POLICY_FAILURE_RATE * 100.0,
            rates.describe()
        );
    }
    if geomean_nnue_static.is_nan() {
        return "INCONCLUSIVE: no valid nnue/static pairs (all failed correctness/compile) — \
                nothing to select on, nothing to publish."
            .to_string();
    }
    let ci_text = match ci {
        Some((lo, hi)) => format!("95% CI [{lo:.4}, {hi:.4}] over kernels"),
        None => "no CI (fewer than 2 pairs)".to_string(),
    };
    if final_eval {
        return format!(
            "PUBLICATION RUN (FINAL tier) — NO SELECTION VERDICT. Measured nnue/static geomean \
             {geomean_nnue_static:.4} ({ci_text}) over {nnue_static_pairs} pairs, noise floor \
             ±{noise_floor_pct:.2}%. These numbers may be published; they may NOT be used to \
             choose weights, thresholds, or extraction policy. Selection runs use the DEV tier."
        );
    }
    let Some((ci_lo, ci_hi)) = ci else {
        return format!(
            "INCONCLUSIVE: only {nnue_static_pairs} nnue/static pair(s) — no confidence interval \
             is computable, and a verdict from a point estimate is what this protocol exists to \
             refuse. Raise --max-kernels."
        );
    };
    // Positive pct = NNUE faster than static. The interval maps ratio-space
    // [lo, hi] onto pct-space [pct_lo, pct_hi] with the endpoints swapping.
    let pct = (geomean_nnue_static - 1.0) * -100.0;
    let pct_lo = (ci_hi - 1.0) * -100.0;
    let pct_hi = (ci_lo - 1.0) * -100.0;
    let interval = format!(
        "geomean {pct:+.2}%, 95% CI [{pct_lo:+.2}%, {pct_hi:+.2}%] over {nnue_static_pairs} \
         kernels, A/A noise floor ±{noise_floor_pct:.2}%"
    );
    if pct_lo > 5.0 {
        return format!(
            "NNUE > static latency prior: the ENTIRE 95% CI clears the +5% gate ({interval}) — \
             Phase 2 gate PASSES, NNUE extraction earns its keep (survived failure rates: {}).",
            rates.describe()
        );
    }
    if pct_hi < -5.0 {
        return format!(
            "NNUE < static latency prior: the ENTIRE 95% CI is below the -5% line ({interval}) — \
             Phase 2 gate FAILS per the plan: keep the static prior as default, NNUE opt-in only."
        );
    }
    if pct_lo > -5.0 && pct_hi < 5.0 {
        return format!(
            "NNUE approx-ties static latency prior: the ENTIRE 95% CI sits inside the ±5% \
             decision band ({interval}) — Phase 2 gate FAILS per the plan (no meaningful win): \
             keep the static prior as default, NNUE opt-in only, iterate (plan 4.3)."
        );
    }
    format!(
        "INCONCLUSIVE-UNDERPOWERED: the 95% CI straddles a ±5% gate line ({interval}) — the \
         estimator cannot resolve a verdict at this eval-set size. Re-run with a larger \
         --max-kernels (CI width shrinks as 1/sqrt(n); 0 = the whole tier); do NOT re-roll the \
         same size until a verdict appears."
    )
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/// The corpus file a tier lives in — the same layout and the same loader the
/// trainer uses, so "held out" means the same thing on both sides.
fn tier_corpus_path(corpus_dir: &str, tier: Tier) -> PathBuf {
    PathBuf::from(format!("{corpus_dir}/corpus_{}.bin", tier.name()))
}

/// Deterministically choose at most `max_kernels` entries of a tier, always
/// keeping the ones `is_pinned` marks (Codex P1, comment 3744800529).
///
/// Selection is a seeded shuffle of the *unpinned* INDICES followed by a sort,
/// so the chosen subset is reproducible from `(sample_seed, max_kernels, tier
/// file)` and is measured in corpus order. `max_kernels == 0` means the whole
/// tier.
///
/// The pinning is not a convenience. A uniform draw over FINAL is
/// overwhelmingly a draw over its synthetic bulk: a healthy build has roughly
/// 9,480 synthetic entries beside the named production kernels (the five
/// original plus the withheld ShaderToy set, `NAMED_KERNELS`), so the
/// default `--final-eval --max-kernels 40` has about a 2% chance of including
/// even one of them. The named kernels ARE the real-world half of what a
/// publication run claims to have measured, so a run that quietly dropped
/// all of them would journal a synthetic-only result under a publication
/// verdict.
/// Pinned entries count against `max_kernels`; if they alone exceed it, every
/// pinned entry is still kept and no unpinned one is.
fn subsample<T>(
    entries: Vec<T>,
    max_kernels: usize,
    sample_seed: u64,
    is_pinned: impl Fn(&T) -> bool,
) -> Vec<T> {
    if max_kernels == 0 || entries.len() <= max_kernels {
        return entries;
    }
    let mut keep: Vec<bool> = entries.iter().map(&is_pinned).collect();
    let pinned = keep.iter().filter(|k| **k).count();
    let budget = max_kernels.saturating_sub(pinned);

    let mut idx: Vec<usize> = (0..entries.len()).filter(|i| !keep[*i]).collect();
    SeededRng::new(sample_seed).shuffle(&mut idx);
    idx.truncate(budget);
    for i in &idx {
        keep[*i] = true;
    }
    entries
        .into_iter()
        .zip(keep)
        .filter_map(|(e, k)| k.then_some(e))
        .collect()
}

/// The tier file's content hash — this run's corpus identity.
///
/// The executable, the flags, and the source revision do not pin WHICH
/// expressions were measured: `gen_bench_corpus` rewrites a tier in place, and
/// a tier regenerated from a different seed, target, or manifest is a different
/// experiment (Codex 3744586352). FNV rather than a cryptographic hash for the
/// same reason [`pixelflow_pipeline::journal::SourceVersion::diff_hash`] uses
/// it — the job is distinguishing setups in a local research journal, not
/// resisting an adversary. A thin, semantically-named wrapper around
/// [`fnv1a64_hex`] (`pixelflow_pipeline::journal`'s shared primitive) rather
/// than a second hash implementation.
fn corpus_identity(bytes: &[u8]) -> String {
    fnv1a64_hex(bytes)
}

// `refuse_unsmudged_lfs_pointer`, `SourceVersion`, `environment_fingerprint`,
// and the FNV-1a primitive all moved to `pixelflow_pipeline::journal`, which
// now performs the whole append (serialize, mkdir, LFS guard, append-or-panic)
// AND composes the `ConfigHash` below — the one writer/composer every
// journal-producing binary shares (docs/plans/2026-08-17-cost-model-domain.md,
// J15).

/// This binary's own protocol parameters, as the free-text `protocol` field
/// [`ConfigHash::compose`] folds in beside the source revision, corpus
/// identity, weights identity, and environment it composes generically. Only
/// what's specific to THIS measurement protocol lives here now — the shared
/// parts moved to `pixelflow_pipeline::journal`.
fn config_params(tier: Tier, args: &Args) -> String {
    format!(
        "saturate={SATURATE_LIMIT};top_k={EXTRACT_TOP_K};tier={};max_kernels={};\
         sample_seed={:#x};repeats={TIMED_REPEATS};order_seed={ORDER_SEED:#x};\
         boot={BOOTSTRAP_ITERS}:{BOOTSTRAP_SEED:#x};mode={};tuples={INPUT_TUPLES};grid={}",
        tier.name(),
        args.max_kernels,
        args.sample_seed,
        bench_mode_slug(BENCH_MODE),
        GRID.len(),
    )
}

/// Which tier a run measures. FINAL is reachable ONLY through `--final-eval`
/// (plan 0.2/4.3, Codex R5) — a selection run must never touch it, and this is
/// the single place that decides.
fn selection_tier(final_eval: bool) -> Tier {
    if final_eval { Tier::Final } else { Tier::Dev }
}

fn main() {
    let args = Args::parse();
    let tier = selection_tier(args.final_eval);

    eprintln!("=== Extraction 3-way bench (hardened): NNUE vs STATIC vs NO-SWAP + A/A ===");
    if args.final_eval {
        // A publication run is not a selection run, and the difference has to
        // be impossible to miss in a scrollback (plan 0.2/4.3, Codex R5).
        eprintln!(
            "\n\
             ############################################################################\n\
             ##  PUBLICATION RUN — FINAL (held-out) TIER                               ##\n\
             ##  This run reports numbers. It does NOT emit a selection verdict, and   ##\n\
             ##  nothing measured here may be used to choose weights, thresholds, or   ##\n\
             ##  extraction policy. Selection runs use the DEV tier (no --final-eval). ##\n\
             ##  Every FINAL evaluation spends part of the tier's one-shot value.      ##\n\
             ############################################################################\n"
        );
    }

    let weights_path = extraction_head_weights_path();
    if !weights_path.exists() {
        eprintln!(
            "ERROR: extraction-head weights not found at {}.",
            weights_path.display()
        );
        eprintln!("Train them first:");
        eprintln!(
            "  cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus"
        );
        eprintln!(
            "  cargo run --release -p pixelflow-pipeline --features training --bin bootstrap_extraction_head"
        );
        std::process::exit(1);
    }
    eprintln!("Loading Judge weights from {}", weights_path.display());
    let weights_bytes = fs::read(&weights_path).unwrap_or_else(|e| {
        panic!(
            "failed to read Judge weights at {}: {e}",
            weights_path.display()
        )
    });
    let nnue = ExprNnue::from_bytes(&weights_bytes).unwrap_or_else(|e| {
        panic!(
            "ExprNnue::from_bytes rejected {}: {e}. \
             Retrain (bootstrap_extraction_head) must produce valid weights before this bench \
             can run.",
            weights_path.display()
        )
    });
    eprintln!("Judge weights loaded OK ({} bytes).", weights_bytes.len());

    // Objective consistency (Codex R6). The head is fit on targets measured in
    // ONE dependency mode; scoring it in the other mode asks it to predict a
    // ranking it never saw. The sidecar records the minting mode and
    // `require_mode` panics on a mismatch — a missing or unreadable sidecar is
    // equally fatal, because "unknown objective" is not a safe default.
    let mint = MintMetadata::read_for(&weights_path).unwrap_or_else(|e| {
        panic!(
            "cannot verify what objective {} was trained against: {e}\n\
             This gate measures BenchMode::{}; running it against weights whose minting mode is \
             unknown would score the model on an objective it may never have seen. Retrain with \
             bootstrap_extraction_head (which writes the sidecar) before gating.",
            weights_path.display(),
            bench_mode_slug(BENCH_MODE)
        )
    });
    mint.require_mode(BENCH_MODE);
    // The sidecar must describe THESE bytes (Codex 3758853787): weights and
    // sidecar are separate files, and an aborted training run can replace one
    // without the other — a mode check alone would score stale metadata as if
    // it described the replacement weights.
    mint.require_weights(&weights_bytes);
    eprintln!(
        "Weights minted by {} on {} samples in BenchMode::{} — matches this gate's mode and \
         weights identity {}.",
        mint.trainer,
        mint.samples,
        bench_mode_slug(BENCH_MODE),
        mint.weights_fnv64
    );

    // Config hash: source revision + corpus identity + weights identity +
    // environment + this binary's protocol params, composed by
    // `pixelflow_pipeline::journal::ConfigHash` so every journal line is
    // attributable to one exact setup the same way `bootstrap_extraction_head`
    // now composes its own. Without the revision (Codex P2) two commits that
    // change the measurement logic itself — the grid, the gate, the timing
    // loop — hash identically while measuring different things; without the
    // diff hash (Codex 3741889899) every uncommitted variant of one commit
    // does.
    let source = SourceVersion::current(&DIFF_HASH_EXCLUDES);
    eprintln!("Source revision: {}", source.rev);
    if let Some(h) = &source.diff_hash {
        eprintln!("Uncommitted diff hash: {h}");
    }
    // The corpus is part of the setup, so its identity is part of the hash
    // (Codex 3744586352). `gen_bench_corpus` rewrites a tier in place from a
    // different seed, target, or manifest; without this, the next run collides
    // in `config_hash` with the previous one and the journal claims two
    // different experiments were the same. Hashed immediately before the
    // parse below, so the identity names the bytes this run measures.
    let corpus_path = tier_corpus_path(&args.corpus_dir, tier);
    let corpus_bytes = fs::read(&corpus_path).unwrap_or_else(|e| {
        panic!(
            "failed to read the {} tier at {}: {e}\n\
             Build the tiered corpus first:\n  \
             cargo run --release -p pixelflow-pipeline --features training \
             --bin gen_bench_corpus",
            tier.name(),
            corpus_path.display()
        )
    });
    let corpus_hash = corpus_identity(&corpus_bytes);
    eprintln!(
        "Corpus identity: {} ({} bytes, content hash {corpus_hash})",
        corpus_path.display(),
        corpus_bytes.len()
    );
    let protocol = config_params(tier, &args);
    // `mint.weights_fnv64` is already this exact FNV-1a hash of
    // `weights_bytes` (Codex 3758853787's `weights_identity`, asserted equal
    // to these bytes above by `require_weights`) — reusing it here means the
    // config hash names the weights the sidecar itself vouches for, not a
    // second independently-computed hash of the same bytes.
    let config = ConfigHash::compose(&source, &corpus_hash, &mint.weights_fnv64, &protocol);
    let config_hash = config.value.clone();
    eprintln!("Config hash: {config_hash}");

    let ts_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_secs();
    let data_dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data"));
    fs::create_dir_all(&data_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", data_dir.display()));
    let data_path = data_dir.join(format!(
        "bench_extraction_3way_{ts_unix}_{config_hash}.jsonl"
    ));
    let mut out = Jsonl::create(data_path.clone());
    eprintln!("Machine-readable output: {}", data_path.display());

    let grid = CheckGrid::build();
    eprintln!(
        "Correctness grid: {} points ({} harness tuples + {} legacy GRID), oracle-referenced, \
         compositional per-point bounds.\n",
        grid.points.len(),
        grid.tuples.len(),
        GRID.len()
    );

    // ---------------------------------------------------------------
    // Corpus assembly — the tier file, never an ad-hoc generator stream
    // (Codex R5). The named production kernels are FINAL-tier members and
    // reach this bench only through corpus_final.bin under --final-eval.
    // ---------------------------------------------------------------
    let entries = read_corpus(&corpus_path).unwrap_or_else(|e| {
        panic!(
            "failed to parse the {} tier at {} (content hash {corpus_hash}): {e}\n\
             Build the tiered corpus first:\n  \
             cargo run --release -p pixelflow-pipeline --features training \
             --bin gen_bench_corpus",
            tier.name(),
            corpus_path.display()
        )
    });
    assert!(
        !entries.is_empty(),
        "the {} tier at {} is empty — gen_bench_corpus refuses to write an empty tier, so this \
         file is stale or truncated",
        tier.name(),
        corpus_path.display()
    );
    let corpus_entries = entries.len();
    let selected = subsample(
        entries,
        args.max_kernels,
        args.sample_seed,
        |(name, _, _)| family_of(name) == "named",
    );
    let selected_named = selected
        .iter()
        .filter(|(name, _, _)| family_of(name) == "named")
        .count();
    eprintln!(
        "Corpus: {} of {} {}-tier expressions from {} (subsample seed {:#x}; {} named entries \
         retained unconditionally — Codex P1)",
        selected.len(),
        corpus_entries,
        tier.name(),
        corpus_path.display(),
        args.sample_seed,
        selected_named,
    );

    let inputs: Vec<KernelInput> = selected
        .into_iter()
        .map(|(name, arena, root)| {
            // The corpus was quarantined at generation time; a shape the oracle
            // cannot evaluate here means the file is not what it claims to be,
            // which is a hard error rather than a skipped check.
            screen_for_oracle(&arena, root).unwrap_or_else(|e| {
                panic!(
                    "corpus entry '{name}' from {} is not oracle-checkable: {e} — the tier file \
                     disagrees with the quarantine that supposedly produced it",
                    corpus_path.display()
                )
            });
            let family = family_of(&name);
            KernelInput {
                name,
                family,
                arena,
                root,
            }
        })
        .collect();

    // ---------------------------------------------------------------
    // Phase A: saturate + extract + compile + oracle gate (M3: all the
    // CPU-heavy work, strictly before any timing)
    // ---------------------------------------------------------------
    eprintln!(
        "\n--- phase A: saturate + extract + oracle gate ({} kernels) ---",
        inputs.len()
    );
    let ctx = PrepareCtx {
        nnue: &nnue,
        grid: &grid,
    };
    let total_kernels = inputs.len();
    let mut gate_tally = GateTally::default();
    let prepared: Vec<PreparedKernel> = inputs
        .into_iter()
        .filter_map(|input| {
            eprintln!("  preparing {}", input.name);
            prepare_kernel(input, &ctx, &mut gate_tally, &mut out)
        })
        .collect();
    let kernels_excluded = total_kernels - prepared.len();
    if kernels_excluded > 0 {
        eprintln!(
            "  {kernels_excluded}/{total_kernels} kernels EXCLUDED at the phase-A gate \
             (see kernel_excluded records)"
        );
    }
    assert!(
        !prepared.is_empty(),
        "every one of the {total_kernels} kernels was excluded at the phase-A gate — nothing to \
         measure. {}",
        gate_tally.describe()
    );

    // ---------------------------------------------------------------
    // Phase B: interleaved timing (M3) — one BenchSession (QoS pinning,
    // burn-in, sentinel, overhead), arm order re-randomized per (kernel,
    // repeat), TIMED_REPEATS passes, per-policy medians.
    // ---------------------------------------------------------------
    eprintln!(
        "\n--- phase B: interleaved timing ({TIMED_REPEATS} repeats x {} kernels x up to 4 arms, \
         seeded-random order, mode={}) ---",
        prepared.len(),
        bench_mode_slug(BENCH_MODE)
    );
    let mut session = BenchSession::new();
    let mut rng = SeededRng::new(ORDER_SEED);
    let mut samples: Vec<[Vec<f64>; 4]> = prepared
        .iter()
        .map(|_| std::array::from_fn(|_| Vec::with_capacity(TIMED_REPEATS)))
        .collect();

    for repeat in 0..TIMED_REPEATS {
        for (ki, kernel) in prepared.iter().enumerate() {
            let mut arms = ARMS;
            rng.shuffle(&mut arms);
            for arm in arms {
                // Policies that failed phase A were recorded loudly there;
                // they simply have no arm to time.
                let Some((policy, extract_us)) = kernel.arm_policy(arm) else {
                    continue;
                };
                let label = format!("{}/{}", kernel.name, arm.name());
                // Time the EXACT code object that passed the phase-A gate —
                // no compilation happens in phase B (Codex P2: recompiling
                // per repeat interleaved compile/alloc/icache pollution with
                // timing and severed check-vs-time identity).
                let bench = session
                    .benchmark_compiled(&policy.code, &policy.arena, policy.root, BENCH_MODE)
                    .unwrap_or_else(|e| {
                        panic!(
                            "{label} repeat {repeat}: benchmark failed after phase-A \
                             compile+check succeeded: {e}"
                        )
                    });
                verify_repeat_outputs(&label, repeat, &policy.check, &bench.outputs);
                let sentinel = bench
                    .sentinel
                    .expect("session-minted BenchResult always carries a SentinelContext");
                let normalization = sentinel.normalization().get();
                out.write(&MeasurementRecord {
                    record: "measurement",
                    kernel: &kernel.name,
                    family: kernel.family,
                    policy: arm.name(),
                    repeat,
                    ns: bench.ns,
                    adjusted_ns: bench.adjusted_ns,
                    call_overhead_ns: bench.call_overhead_ns,
                    iqr_ns: bench.iqr_ns,
                    mode: bench_mode_slug(BENCH_MODE),
                    repeat_batches: bench.repeat_batches,
                    extract_us,
                    correctness: "ok",
                    local_sentinel_ns: sentinel.local_sentinel_ns,
                    sentinel_normalization: normalization,
                });
                // Aggregate opening-clock values (Codex 3758853795): the raw
                // reading stays in the record beside its correction factor,
                // but every median, policy ratio, A/A noise floor, and verdict
                // downstream sees the drift-corrected value. The seeded arm
                // shuffle only randomizes clock drift across arms; it cannot
                // cancel it in a finite run.
                samples[ki][arm.idx()].push(bench.ns * normalization);
            }
        }
        eprintln!("  repeat {} / {TIMED_REPEATS} done", repeat + 1);
    }

    // ---------------------------------------------------------------
    // Aggregate: per-kernel medians, ratios, noise floor
    // ---------------------------------------------------------------
    let summaries: Vec<KernelSummary> = prepared
        .iter()
        .zip(&samples)
        .map(|(k, s)| summarize(k, s))
        .collect();

    let mut nnue_static_pairs: Vec<(String, f64)> = Vec::new();
    let mut all_static_noswap: Vec<f64> = Vec::new();
    let mut aa_ratios: Vec<f64> = Vec::new();
    for s in &summaries {
        if let (Some(nnue), Some(stat)) = (s.ns_nnue, s.ns_static) {
            nnue_static_pairs.push((s.name.clone(), nnue / stat));
        }
        if let Some(stat) = s.ns_static {
            all_static_noswap.push(stat / s.ns_noswap);
        }
        if let (Some(stat), Some(aa)) = (s.ns_static, s.ns_static_aa) {
            aa_ratios.push(aa / stat);
        }
    }

    // ---------------------------------------------------------------
    // Human-readable report
    // ---------------------------------------------------------------
    println!(
        "\n=== RESULTS ({} tier, median of {TIMED_REPEATS} interleaved repeats, raw ns, \
         mode={}) ===\n",
        tier.name(),
        bench_mode_slug(BENCH_MODE)
    );
    println!(
        "{:<24} {:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>8} {:>8}",
        "kernel",
        "nodes",
        "ns_nnue",
        "ns_static",
        "ns_noswap",
        "nnue/stc",
        "stc/nosw",
        "ex_nnue",
        "ex_stc"
    );
    let rule_width = 24 + 6 + 10 * 3 + 9 * 2 + 8 * 2 + 16;
    println!("{}", "-".repeat(rule_width));

    let named: Vec<&KernelSummary> = summaries.iter().filter(|s| s.family == "named").collect();
    let synth: Vec<&KernelSummary> = summaries
        .iter()
        .filter(|s| s.family == "synthetic")
        .collect();

    for s in &named {
        let ratio_ns = match (s.ns_nnue, s.ns_static) {
            (Some(n), Some(st)) => Some(n / st),
            _ => None,
        };
        let ratio_sn = s.ns_static.map(|st| st / s.ns_noswap);
        println!(
            "{:<24} {:>6} {:>10} {:>10} {:>10.3} {:>9} {:>9} {:>8.2} {:>8.2}",
            s.name,
            s.n_orig,
            fmt_opt(s.ns_nnue),
            fmt_opt(s.ns_static),
            s.ns_noswap,
            ratio_ns
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into()),
            ratio_sn
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into()),
            s.extract_us_nnue,
            s.extract_us_static,
        );
    }

    println!("{}", "-".repeat(rule_width));

    // Synthetic aggregate row (geomean of per-kernel medians per policy).
    let syn_total = synth.len();
    let syn_nnue_ns: Vec<f64> = synth.iter().filter_map(|s| s.ns_nnue).collect();
    let syn_static_ns: Vec<f64> = synth.iter().filter_map(|s| s.ns_static).collect();
    let syn_noswap_ns: Vec<f64> = synth.iter().map(|s| s.ns_noswap).collect();
    let syn_extract_nnue: Vec<f64> = synth.iter().map(|s| s.extract_us_nnue).collect();
    let syn_extract_static: Vec<f64> = synth.iter().map(|s| s.extract_us_static).collect();
    let syn_nnue_fail = synth.iter().filter(|s| s.nnue_fail).count();
    let syn_static_fail = synth.iter().filter(|s| s.static_fail).count();

    if syn_total > 0 {
        let g_syn_nnue = geomean(&syn_nnue_ns, "synthetic nnue ns");
        let g_syn_static = geomean(&syn_static_ns, "synthetic static ns");
        let g_syn_noswap = geomean(&syn_noswap_ns, "synthetic noswap ns");
        println!(
            "{:<24} {:>6} {:>10} {:>10} {:>10.3} {:>9} {:>9} {:>8.2} {:>8.2}",
            format!("synthetic(n={syn_total})"),
            syn_noswap_ns.len(),
            format!("{g_syn_nnue:.3}"),
            format!("{g_syn_static:.3}"),
            g_syn_noswap,
            format!("{:.3}", g_syn_nnue / g_syn_static),
            format!("{:.3}", g_syn_static / g_syn_noswap),
            geomean(&syn_extract_nnue, "synthetic nnue extract_us"),
            geomean(&syn_extract_static, "synthetic static extract_us"),
        );
        println!(
            "  (synthetic failures: nnue={syn_nnue_fail}/{syn_total}, \
             static={syn_static_fail}/{syn_total})"
        );
        println!("{}", "-".repeat(rule_width));
    }

    let all_nnue_static: Vec<f64> = nnue_static_pairs.iter().map(|(_, r)| *r).collect();
    let geomean_nnue_static = geomean(&all_nnue_static, "nnue/static ratios");
    let geomean_static_noswap = geomean(&all_static_noswap, "static/noswap ratios");
    println!(
        "GEOMEAN (n={} nnue/static pairs, n={} static/noswap pairs): nnue/static={:.4}  static/noswap={:.4}",
        all_nnue_static.len(),
        all_static_noswap.len(),
        geomean_nnue_static,
        geomean_static_noswap,
    );

    // The verdict's instrument (plan 4.3 rework): a CI over KERNELS — the
    // variance source that actually dominated round 0 — plus the paired ratio
    // distribution and leave-one-out sensitivity, because the geomean has
    // repeatedly been dominated by single kernels.
    let ratio_ci = bootstrap_geomean_ci(&all_nnue_static, BOOTSTRAP_ITERS, BOOTSTRAP_SEED);
    let ratio_dist = RatioDistribution::of(&all_nnue_static);
    let loo = LooSensitivity::of(&nnue_static_pairs);
    match &ratio_dist {
        Some(d) => println!(
            "Paired nnue/static ratios (n={}): median {:.4}, IQR [{:.4}, {:.4}], \
             wins {} (<1.0) / losses {} (>1.0) / ties {}",
            all_nnue_static.len(),
            d.median,
            d.q1,
            d.q3,
            d.wins,
            d.losses,
            d.ties
        ),
        None => println!("Paired nnue/static ratios: none (no kernel produced both arms)"),
    }
    match ratio_ci {
        Some((lo, hi)) => println!(
            "Bootstrap 95% CI on the geomean ({BOOTSTRAP_ITERS} resamples over kernels, seed \
             {BOOTSTRAP_SEED:#x}): [{lo:.4}, {hi:.4}]"
        ),
        None => println!(
            "Bootstrap 95% CI on the geomean: unavailable ({} pair(s) < 2)",
            all_nnue_static.len()
        ),
    }
    match &loo {
        Some(l) => println!(
            "Leave-one-out geomean range: [{:.4}, {:.4}]; most influential kernel '{}' shifts \
             the geomean by {:.2}%",
            l.min, l.max, l.most_influential, l.shift_pct
        ),
        None => println!("Leave-one-out sensitivity: unavailable (fewer than 2 pairs)"),
    }

    let named_nnue_fail = named.iter().filter(|s| s.nnue_fail).count();
    let named_static_fail = named.iter().filter(|s| s.static_fail).count();
    if !named.is_empty() {
        println!(
            "\nNamed-kernel failures: nnue={named_nnue_fail}/{}, static={named_static_fail}/{}",
            named.len(),
            named.len(),
        );
    }
    println!("Kernels excluded at the phase-A gate: {kernels_excluded}/{total_kernels}");

    // Plan 0.4 metadata: ill-conditioned cross-form disagreements are contract
    // (algebraic rewrites at/near singularities), reported but never excluded.
    let cross_ill_disagreements_total: usize = prepared
        .iter()
        .flat_map(|k| {
            [
                Some(&k.noswap),
                k.static_prep.ready.as_ref(),
                k.nnue_prep.ready.as_ref(),
            ]
        })
        .flatten()
        .map(|p| p.check.cross_ill_disagreements)
        .sum();
    println!(
        "Cross-form ill-conditioned disagreements (contract, metadata only): \
         {cross_ill_disagreements_total}"
    );
    println!("Gate outcomes: {}", gate_tally.describe());

    // How much of the grid the same-form gate actually DECIDED, per gated
    // policy (Codex 3744586366). "Zero same-form miscompiles" is a claim about
    // this denominator: a form whose points are all divergent or all unbounded
    // was never accepted or rejected anywhere, and a headline computed over
    // such forms is vacuous. Printed on its own line so the question is
    // answerable from the scrollback.
    let bounded_fractions: Vec<f64> = prepared
        .iter()
        .flat_map(|k| {
            [
                Some(&k.noswap),
                k.static_prep.ready.as_ref(),
                k.nnue_prep.ready.as_ref(),
            ]
        })
        .flatten()
        .map(|p| p.check.checked as f64 / grid.points.len() as f64)
        .collect();
    let bound_coverage_mean_pct = (!bounded_fractions.is_empty())
        .then(|| 100.0 * bounded_fractions.iter().sum::<f64>() / bounded_fractions.len() as f64);
    let worst_bound_coverage_pct = bounded_fractions
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b * 100.0));
    match bound_coverage_mean_pct {
        Some(pct) => println!(
            "Bound coverage: {pct:.1}% of the {} grid points decided on average over {} gated \
             policies (worst {worst_bound_coverage_pct:.1}%); {} gates bounded nothing at all and \
             were excluded — this is the denominator the same-form miscompile rate is a rate of",
            grid.points.len(),
            bounded_fractions.len(),
            gate_tally.no_bounded_points,
        ),
        None => println!(
            "Bound coverage: no gated policy survived, so nothing was verified at all ({} gates \
             bounded nothing)",
            gate_tally.no_bounded_points
        ),
    }

    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let extract_us_nnue_all: Vec<f64> = summaries.iter().map(|s| s.extract_us_nnue).collect();
    let extract_us_static_all: Vec<f64> = summaries.iter().map(|s| s.extract_us_static).collect();
    let mean_extract_us_nnue = mean(&extract_us_nnue_all);
    let mean_extract_us_static = mean(&extract_us_static_all);
    println!(
        "Mean extraction overhead: nnue={mean_extract_us_nnue:.2}us static={mean_extract_us_static:.2}us (n={})",
        extract_us_nnue_all.len(),
    );

    // Noise floor (plan 0.3b): the same static policy measured as two
    // interleaved arms. |geomean − 1| alone is a single draw of the null
    // statistic — roughly half of all runs understate the true null
    // dispersion, letting a pure-noise margin print as "above the noise
    // floor" — so the floor is the max of that draw and a MAD-derived
    // standard error of the A/A log-ratio distribution: sigma ≈ 1.4826·MAD
    // under a normal null, and the geomean of n ratios concentrates as
    // sigma/√n.
    let aa_geomean = geomean(&aa_ratios, "static A/A ratios");
    let (noise_floor_pct, aa_dispersion_floor_pct) = if aa_ratios.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        let logs: Vec<f64> = aa_ratios.iter().map(|r| r.ln()).collect();
        let center = median(logs.clone(), "static A/A log-ratios");
        let deviations: Vec<f64> = logs.iter().map(|l| (l - center).abs()).collect();
        let mad = median(deviations, "static A/A log-ratio deviations");
        let dispersion_floor = 1.4826 * mad / (aa_ratios.len() as f64).sqrt();
        (
            (aa_geomean - 1.0).abs().max(dispersion_floor) * 100.0,
            dispersion_floor * 100.0,
        )
    };
    let worst_aa_pct = aa_ratios
        .iter()
        .map(|r| (r - 1.0).abs() * 100.0)
        .fold(f64::NAN, f64::max);
    println!(
        "\nNoise floor (static A/A, n={} kernels): geomean ratio {:.4} (|delta| {:.2}%), \
         MAD-derived dispersion ±{:.2}% -> floor ±{:.2}%; worst per-kernel |delta| {:.2}%",
        aa_ratios.len(),
        aa_geomean,
        (aa_geomean - 1.0).abs() * 100.0,
        aa_dispersion_floor_pct,
        noise_floor_pct,
        worst_aa_pct,
    );
    println!(
        "Sentinel: calibration {:.3}ns, {} samples, call overhead ({}) {:.3}ns",
        session.calibration_ns(),
        session.sentinel_samples().len(),
        bench_mode_slug(BENCH_MODE),
        session.call_overhead_ns(BENCH_MODE),
    );

    // Per-policy failure rates over every kernel that entered phase A (Codex
    // P1): the geomean above is computed only over kernels where BOTH compared
    // arms produced a gated artifact, so a policy that fails often has quietly
    // chosen its own comparison set. The rates decide whether any directional
    // claim is allowed at all, and they are printed with the verdict either
    // way — a PASS must state what it survived.
    let failure_rates = PolicyFailureRates::compute(
        total_kernels,
        kernels_excluded,
        named_static_fail + syn_static_fail,
        named_nnue_fail + syn_nnue_fail,
    );
    println!(
        "\nPer-policy failure rates over all {total_kernels} kernels entering phase A \
         (exclusions count against every policy): {}",
        failure_rates.describe()
    );

    // The correctness alarms are computed BEFORE the verdict, not after the
    // journal write (Codex 3744586360): they are an input to the verdict, and a
    // run whose correctness gate failed must never journal a PASS. The abort
    // still happens last, after every record is on disk.
    let gate_alarms = gate_tally.alarms();
    if !gate_alarms.is_empty() {
        println!(
            "\nCORRECTNESS GATE FAILED — {} alarm(s); the verdict below is forced invalid:\n  {}",
            gate_alarms.len(),
            gate_alarms.join("\n  ")
        );
    }
    let verdict = verdict_text(&VerdictInputs {
        final_eval: args.final_eval,
        gate_alarms: &gate_alarms,
        geomean_nnue_static,
        ci: ratio_ci,
        nnue_static_pairs: all_nnue_static.len(),
        noise_floor_pct,
        rates: &failure_rates,
    });
    println!(
        "\n=== {} ===",
        if !gate_alarms.is_empty() {
            "INVALID RUN"
        } else if args.final_eval {
            "PUBLICATION REPORT"
        } else {
            "GATE VERDICT"
        }
    );
    println!("{verdict}");

    // ---------------------------------------------------------------
    // Run summary record + research journal (0.3b)
    // ---------------------------------------------------------------
    let sentinel_samples: Vec<SentinelSampleRecord> = session
        .sentinel_samples()
        .iter()
        .map(|s| SentinelSampleRecord {
            expr_index: s.expr_index,
            ns: s.ns,
        })
        .collect();
    out.write(&RunSummaryRecord {
        record: "run_summary",
        ts_unix,
        config_hash: &config_hash,
        source_rev: &source.rev,
        source_diff_hash: source.diff_hash.as_deref(),
        weights_path: weights_path.display().to_string(),
        weights_bytes: weights_bytes.len(),
        weights_mint_mode: mint.bench_mode.clone(),
        tier: tier.name(),
        final_eval: args.final_eval,
        corpus_path: corpus_path.display().to_string(),
        corpus_entries,
        corpus_hash: &corpus_hash,
        max_kernels: args.max_kernels,
        sample_seed: format!("{:#x}", args.sample_seed),
        saturate_limit: SATURATE_LIMIT,
        extract_top_k: EXTRACT_TOP_K,
        timed_repeats: TIMED_REPEATS,
        order_seed: format!("{ORDER_SEED:#x}"),
        mode: bench_mode_slug(BENCH_MODE),
        check_points: grid.points.len(),
        geomean_nnue_static: finite_or_none(geomean_nnue_static),
        geomean_nnue_static_ci_lo: ratio_ci.map(|(lo, _)| lo),
        geomean_nnue_static_ci_hi: ratio_ci.map(|(_, hi)| hi),
        bootstrap_iters: BOOTSTRAP_ITERS,
        bootstrap_seed: format!("{BOOTSTRAP_SEED:#x}"),
        nnue_static_pairs: all_nnue_static.len(),
        ratio_median: ratio_dist.as_ref().map(|d| d.median),
        ratio_q1: ratio_dist.as_ref().map(|d| d.q1),
        ratio_q3: ratio_dist.as_ref().map(|d| d.q3),
        ratio_wins: ratio_dist.as_ref().map_or(0, |d| d.wins),
        ratio_losses: ratio_dist.as_ref().map_or(0, |d| d.losses),
        loo_geomean_min: loo.as_ref().map(|l| l.min),
        loo_geomean_max: loo.as_ref().map(|l| l.max),
        loo_most_influential: loo.as_ref().map(|l| l.most_influential.clone()),
        loo_max_shift_pct: loo.as_ref().map(|l| l.shift_pct),
        geomean_static_noswap: finite_or_none(geomean_static_noswap),
        aa_geomean: finite_or_none(aa_geomean),
        noise_floor_pct: finite_or_none(noise_floor_pct),
        sentinel_calibration_ns: session.calibration_ns(),
        sentinel_samples,
        call_overhead_ns: session.call_overhead_ns(BENCH_MODE),
        kernels_excluded,
        named_static_fail,
        named_nnue_fail,
        syn_static_fail,
        syn_nnue_fail,
        failure_rate_noswap: failure_rates.noswap,
        failure_rate_static: failure_rates.static_,
        failure_rate_nnue: failure_rates.nnue,
        verdict_censored: !failure_rates.censoring().is_empty(),
        cross_ill_disagreements_total,
        gates: gate_tally.gates,
        same_form_miscompiles: gate_tally.same_form,
        same_form_miscompile_rate: gate_tally.same_form_rate(),
        cross_form_divergences: gate_tally.cross_form,
        cross_form_divergence_rate: gate_tally.cross_form_rate(),
        gate_compile_failures: gate_tally.compile_failed,
        gate_oracle_unsupported: gate_tally.oracle_unsupported,
        gate_no_bounded_points: gate_tally.no_bounded_points,
        gate_alarms: gate_alarms.clone(),
        correctness_gate_failed: !gate_alarms.is_empty(),
        bound_coverage_mean_pct,
        bound_coverage_gates: bounded_fractions.len(),
        mean_extract_us_static: finite_or_none(mean_extract_us_static),
        mean_extract_us_nnue: finite_or_none(mean_extract_us_nnue),
        verdict: &verdict,
    });

    let journal_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/results/journal.jsonl"
    ));
    let weights_artifact = ArtifactId::with_identity(
        weights_path.display().to_string(),
        mint.weights_fnv64.clone(),
    );
    let journal_entry = JournalEntry::new(
        "bench_extraction_3way",
        config,
        weights_artifact,
        BenchMetrics {
            tier: tier.name(),
            corpus_path: corpus_path.display().to_string(),
            corpus_entries,
            final_eval: args.final_eval,
            mode: bench_mode_slug(BENCH_MODE),
            weights_mint_mode: mint.bench_mode.clone(),
            geomean_nnue_static: finite_or_none(geomean_nnue_static),
            geomean_nnue_static_ci_lo: ratio_ci.map(|(lo, _)| lo),
            geomean_nnue_static_ci_hi: ratio_ci.map(|(_, hi)| hi),
            nnue_static_pairs: all_nnue_static.len(),
            ratio_median: ratio_dist.as_ref().map(|d| d.median),
            ratio_q1: ratio_dist.as_ref().map(|d| d.q1),
            ratio_q3: ratio_dist.as_ref().map(|d| d.q3),
            ratio_wins: ratio_dist.as_ref().map_or(0, |d| d.wins),
            ratio_losses: ratio_dist.as_ref().map_or(0, |d| d.losses),
            loo_geomean_min: loo.as_ref().map(|l| l.min),
            loo_geomean_max: loo.as_ref().map(|l| l.max),
            loo_most_influential: loo.as_ref().map(|l| l.most_influential.clone()),
            geomean_static_noswap: finite_or_none(geomean_static_noswap),
            noise_floor_pct: finite_or_none(noise_floor_pct),
            kernels_excluded,
            named_static_fail,
            named_nnue_fail,
            syn_static_fail,
            syn_nnue_fail,
            failure_rate_noswap: failure_rates.noswap,
            failure_rate_static: failure_rates.static_,
            failure_rate_nnue: failure_rates.nnue,
            same_form_miscompile_rate: gate_tally.same_form_rate(),
            cross_form_divergence_rate: gate_tally.cross_form_rate(),
            correctness_gate_failed: !gate_alarms.is_empty(),
            gate_alarms: gate_alarms.clone(),
            bound_coverage_mean_pct,
            verdict_censored: !failure_rates.censoring().is_empty(),
            verdict: &verdict,
            data_jsonl: data_path.display().to_string(),
        },
    );
    pixelflow_pipeline::journal::append_record(&journal_path, &journal_entry);

    println!("\nMachine-readable results: {}", data_path.display());

    // The gate alarms fire LAST, after every record is on disk: they are a
    // verdict on the run's exclusions, and the exclusions they cite must be
    // readable in the JSONL that this panic names.
    assert!(
        gate_alarms.is_empty(),
        "{} gate alarm(s) — the exclusions are individually recorded in {}:\n  {}",
        gate_alarms.len(),
        data_path.display(),
        gate_alarms.join("\n  ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named_kernel_list() -> Vec<(&'static str, ExprArena, ExprId)> {
        named_kernels()
    }

    #[test]
    fn recovered_tuples_match_harness_contract() {
        let tuples = recover_harness_tuples();
        assert_eq!(tuples.len(), INPUT_TUPLES);
        // Tuple 0 is asserted inside recover_harness_tuples; tuple 1 carries
        // the documented ±0.0 / ±1.0 special values.
        assert_eq!(tuples[1], [0.0, -0.0, 1.0, -1.0]);
        assert!(
            tuples[1][1].is_sign_negative(),
            "-0.0 must survive recovery"
        );
        // The buffer must span the documented magnitude sweep and both signs.
        let all: Vec<f32> = tuples.iter().flatten().copied().collect();
        assert!(all.iter().any(|v| v.abs() >= 1e4));
        assert!(all.iter().any(|v| *v != 0.0 && v.abs() <= 1e-4));
        assert!(all.iter().any(|v| *v > 0.0));
        assert!(all.iter().any(|v| *v < 0.0));
    }

    // ── Tier discipline (Codex R5) ──────────────────────────────────────────

    #[test]
    fn a_selection_run_measures_dev_and_never_final() {
        assert_eq!(selection_tier(false), Tier::Dev);
        assert_eq!(selection_tier(true), Tier::Final);
        let default_args = Args::parse_from(["bench_extraction_3way"]);
        assert!(
            !default_args.final_eval,
            "the default run must not touch the FINAL tier"
        );
        assert_eq!(selection_tier(default_args.final_eval), Tier::Dev);
    }

    #[test]
    fn tier_paths_are_the_files_gen_bench_corpus_writes() {
        assert!(
            tier_corpus_path("d", Tier::Dev).ends_with("corpus_dev.bin"),
            "the selection tier must be the trainer's held-out DEV file"
        );
        assert!(tier_corpus_path("d", Tier::Final).ends_with("corpus_final.bin"));
    }

    #[test]
    fn subsample_is_deterministic_bounded_and_ordered() {
        let entries: Vec<usize> = (0..100).collect();
        let none = |_: &usize| false;
        let a = subsample(entries.clone(), 10, 0x1234, none);
        let b = subsample(entries.clone(), 10, 0x1234, none);
        assert_eq!(a, b, "same seed must select the same kernels");
        assert_eq!(a.len(), 10);
        assert!(a.windows(2).all(|w| w[0] < w[1]), "corpus order is kept");
        let c = subsample(entries.clone(), 10, 0x5678, none);
        assert_ne!(a, c, "a different seed must select a different subset");
        // 0 and "fewer than the cap" both mean "the whole tier".
        assert_eq!(subsample(entries.clone(), 0, 1, none).len(), 100);
        assert_eq!(subsample(entries, 500, 1, none).len(), 100);
    }

    #[test]
    fn subsample_never_drops_a_pinned_entry() {
        // FINAL's real shape: a handful of named kernels lost in a synthetic
        // bulk. Uniformly drawing 40 of these would take the five named ones
        // about 2% of the time, so a publication run would almost always
        // report synthetic-only numbers under a named-inclusive claim.
        let entries: Vec<usize> = (0..9_485).collect();
        let pinned = [0usize, 2_000, 4_000, 6_000, 9_484];
        let is_pinned = |e: &usize| pinned.contains(e);

        let got = subsample(entries.clone(), 40, 0x1234, is_pinned);
        assert_eq!(got.len(), 40, "the cap still binds");
        for p in pinned {
            assert!(got.contains(&p), "pinned entry {p} was dropped");
        }
        assert!(got.windows(2).all(|w| w[0] < w[1]), "corpus order is kept");

        // Pinned entries spend the budget rather than adding to it.
        let uniform = subsample(entries.clone(), 40, 0x1234, |_: &usize| false);
        assert_eq!(uniform.len(), got.len());

        // A cap below the pinned count keeps every pinned entry and nothing
        // else — never a run that silently measured fewer than it was told.
        let tight = subsample(entries, 3, 0x1234, is_pinned);
        assert_eq!(tight, pinned.to_vec());
    }

    #[test]
    fn subsample_retains_every_pinned_entry_at_every_seed() {
        // Codex P1 (comment 3744800529): ~9,480 synthetic + 5 named at
        // --max-kernels 40 gave a ~2% chance of drawing even ONE named kernel.
        // The named entries are the real-world portion of the claim: they must
        // survive every subsample, at every seed.
        let named: Vec<i64> = (0..5).map(|i| -(i + 1)).collect(); // negatives = named
        let mut entries: Vec<i64> = (0..9_480).collect();
        entries.extend(&named);
        let is_named = |e: &i64| *e < 0;
        for seed in [0x1_u64, 0x1234, 0x5A11_2026_0805, u64::MAX] {
            let picked = subsample(entries.clone(), 40, seed, is_named);
            assert_eq!(picked.len(), 40);
            let picked_named: Vec<i64> = picked.iter().copied().filter(|e| is_named(e)).collect();
            assert_eq!(
                picked_named.len(),
                named.len(),
                "seed {seed:#x}: every named entry must be retained"
            );
            assert_eq!(
                picked.iter().filter(|e| **e >= 0).count(),
                35,
                "the synthetic remainder fills the rest of the budget"
            );
        }
    }

    // ── The objective the head was trained on (Codex R6) ────────────────────

    #[test]
    fn the_gate_measures_the_mode_the_trainer_mints_in() {
        // The trainer mints with BenchMode::Latency; measuring throughput here
        // would score the head on a ranking it never saw. `require_mode`
        // enforces the pair at runtime; this pins the constant.
        assert_eq!(BENCH_MODE, BenchMode::Latency);
        assert_eq!(bench_mode_slug(BENCH_MODE), "latency");
    }

    // ── Same-form gate: the compositional bound (Codex R1 / smoke B1) ───────

    #[test]
    fn noswap_gate_passes_on_named_kernels() {
        let grid = CheckGrid::build();
        for (name, arena, root) in named_kernel_list() {
            let (_code, check) = check_policy(
                &format!("{name}/noswap"),
                (&arena, root),
                (&arena, root),
                &grid,
            )
            .unwrap_or_else(|f| panic!("{}: {}", f.kind(), f.detail()));
            assert!(
                check.checked > 0,
                "{name}: gate must check at least one point"
            );
        }
    }

    #[test]
    fn amplified_estimate_error_is_not_a_miscompile() {
        // Codex R1 / smoke B1: `recip(x)^16` and `sin(recip(x)·y·y)` are valid
        // kernels whose 12-bit estimate error is amplified far past any
        // per-op tolerance row. The max-over-ops fold called them compiler
        // bugs — 14.4% of a real corpus. The compositional bound admits them.
        let grid = CheckGrid::build();

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let r = a.push_unary(OpKind::Recip, x);
        let r2 = a.push_binary(OpKind::Mul, r, r);
        let r4 = a.push_binary(OpKind::Mul, r2, r2);
        let r8 = a.push_binary(OpKind::Mul, r4, r4);
        let r16 = a.push_binary(OpKind::Mul, r8, r8);
        check_policy("test/recip_pow16", (&a, r16), (&a, r16), &grid)
            .unwrap_or_else(|f| panic!("recip(x)^16 must pass the same-form gate: {}", f.detail()));

        let mut b = ExprArena::new();
        let bx = b.push_var(0);
        let by = b.push_var(1);
        let k = b.push_unary(OpKind::Recip, bx);
        let yy = b.push_binary(OpKind::Mul, by, by);
        let arg = b.push_binary(OpKind::Mul, k, yy);
        let sn = b.push_unary(OpKind::Sin, arg);
        check_policy("test/amplified_sin", (&b, sn), (&b, sn), &grid).unwrap_or_else(|f| {
            panic!(
                "sin(recip(x)*y*y) must pass the same-form gate: {}",
                f.detail()
            )
        });
    }

    #[test]
    fn benign_low_gain_multiply_stays_well_conditioned() {
        // Codex 3741889906: the deleted cancellation-ratio heuristic condemned
        // `x·1e-4` at every point because the output is far below the largest
        // intermediate. Every point must stay eligible for the cross-form
        // comparison — otherwise a broken extraction hides inside "metadata".
        let grid = CheckGrid::build();
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let k = a.push_const(1e-4);
        let root = a.push_binary(OpKind::Mul, x, k);
        let (_code, check) = check_policy("test/lowgain", (&a, root), (&a, root), &grid)
            .unwrap_or_else(|f| panic!("x*1e-4 must pass: {}", f.detail()));
        assert_eq!(
            check.cross_ill_points, 0,
            "x*1e-4 is perfectly determined at every point"
        );
        assert!(check.cross_compared > 0);
    }

    #[test]
    fn mask_valued_root_passes_the_gate_bit_exactly() {
        // Lt(x, y): the root lane is a mask, and an all-ones TRUE lane reads
        // as NaN. A numeric bound would judge it by a tolerance whose
        // NaN-vs-NaN arm accepts any payload; the mask path compares bit
        // patterns, so the gate is both stricter and free of false
        // ill-conditioning.
        let grid = CheckGrid::build();
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Lt, x, y);
        let (_code, check) = check_policy("test/mask", (&arena, root), (&arena, root), &grid)
            .unwrap_or_else(|f| panic!("Lt(x, y) must pass the same-form gate: {}", f.detail()));
        assert!(check.mask_root, "the check must take the mask path");
        assert!(check.checked > 0);
        assert_eq!(
            check.mask_near_threshold, 0,
            "JIT and oracle agree bit-exactly on Lt away from exact ties"
        );
        // Every retained expected lane is a valid mask pattern — the thing a
        // folded numeric tolerance could never assert.
        for pc in check.expected.iter().flatten() {
            assert!(
                pixelflow_ir::is_valid_mask(pc.value().to_bits()),
                "expected lane {:#010x} is not a mask pattern",
                pc.value().to_bits()
            );
        }
    }

    #[test]
    fn static_extraction_passes_oracle_gate_on_poly() {
        // Exercises the full phase-A pipeline (saturate, extract, compile,
        // oracle gate) without NNUE weights.
        let (arena, root) = poly();
        let grid = CheckGrid::build();
        let mut eg = EGraph::with_rules(all_rules());
        let root_class = eg.add_arena(&arena, root);
        eg.saturate_with_limit(SATURATE_LIMIT);
        let costs = CostModel::latency_prior();
        let (static_arena, static_root, _cost) = extract(&eg, root_class, &costs);
        let (code, check) = check_policy(
            "poly/static",
            (&arena, root),
            (&static_arena, static_root),
            &grid,
        )
        .unwrap_or_else(|f| panic!("{}: {}", f.kind(), f.detail()));
        assert!(check.checked > 0);
        assert_eq!(check.expected.len(), INPUT_TUPLES);

        // The gated code object is returned so phase B can time THAT artifact.
        // Timing it must reproduce the phase-A verdicts — which is exactly
        // what verify_repeat_outputs asserts per repeat.
        let mut session = BenchSession::new();
        let bench = session
            .benchmark_compiled(&code, &static_arena, static_root, BENCH_MODE)
            .expect("the gated code object must benchmark");
        verify_repeat_outputs("poly/static", 0, &check, &bench.outputs);
    }

    // ── Cross-form tier: exclude and count, never abort (smoke B4) ──────────

    #[test]
    fn cross_form_well_conditioned_disagreement_excludes_and_names_both_forms() {
        // A candidate that is NOT algebraically equal to the original
        // (`x·2e-4` against `x·1e-4`) is exactly the broken extraction the
        // deleted ratio heuristic used to hide. It must be caught — and
        // reported as an EXCLUSION carrying both forms and the point tallies,
        // not as a process-killing panic (smoke B4: one bad expression used to
        // destroy a multi-hour run whose other kernels had already passed).
        let grid = CheckGrid::build();
        let mut orig = ExprArena::new();
        let ox = orig.push_var(0);
        let ok = orig.push_const(1e-4);
        let oroot = orig.push_binary(OpKind::Mul, ox, ok);
        let mut cand = ExprArena::new();
        let cx = cand.push_var(0);
        let ck = cand.push_const(2e-4);
        let croot = cand.push_binary(OpKind::Mul, cx, ck);

        let failure = check_policy("test/broken", (&orig, oroot), (&cand, croot), &grid)
            .err()
            .expect("a form computing a different function must not pass");
        assert_eq!(failure.kind(), "cross_form_divergence");
        let detail = failure.detail();
        assert!(detail.contains("WELL-CONDITIONED"), "got: {detail}");
        // The smoke agent's diagnostic: both forms, plus the point population.
        assert!(detail.contains("ORIGINAL kernel:"), "got: {detail}");
        assert!(detail.contains("CANDIDATE kernel:"), "got: {detail}");
        assert!(detail.contains("well-conditioned"), "got: {detail}");
        assert!(detail.contains("ill-conditioned"), "got: {detail}");
        assert!(detail.contains("platform-divergent"), "got: {detail}");
    }

    #[test]
    fn cross_form_ill_conditioned_disagreement_is_metadata_not_failure() {
        // Plan 0.4 reframing: an algebraically-valid form that overflows where
        // the original does not must NOT fail the gate — the divergence is
        // recorded as metadata. original: x. candidate: (x * 1e35) * 1e-35,
        // algebraically x but ±inf wherever |x| * 1e35 exceeds f32::MAX.
        let grid = CheckGrid::build();
        let mut orig = ExprArena::new();
        let ox = orig.push_var(0);
        let mut cand = ExprArena::new();
        let cx = cand.push_var(0);
        let big = cand.push_const(1e35);
        let small = cand.push_const(1e-35);
        let scaled = cand.push_binary(OpKind::Mul, cx, big);
        let croot = cand.push_binary(OpKind::Mul, scaled, small);
        let (_code, check) = check_policy("test/illcond", (&orig, ox), (&cand, croot), &grid)
            .unwrap_or_else(|f| {
                panic!(
                    "an overflow-at-a-singularity rewrite is contract, not a gate failure: {}",
                    f.detail()
                )
            });
        assert!(
            check.cross_ill_points > 0,
            "the seeded grid reaches |x| ~ 1e4, which overflows the candidate to inf — those \
             points must be classified ill-conditioned"
        );
        assert!(
            check.cross_ill_disagreements > 0,
            "inf-vs-finite at overflow points must be recorded as metadata disagreements"
        );
        assert_eq!(
            check.cross_wc_disagreements, 0,
            "no WELL-CONDITIONED point may disagree for an algebraically valid rewrite"
        );
    }

    // ── Gate tallies and their alarms ───────────────────────────────────────

    #[test]
    fn gate_tally_counts_each_failure_kind_separately() {
        let mut t = GateTally::default();
        t.record_extracted(None);
        t.record_extracted(Some(&GateFailure::SameForm("m".into())));
        t.record_extracted(Some(&GateFailure::CrossForm("c".into())));
        t.record_extracted(Some(&GateFailure::CompileFailed("j".into())));
        t.record_extracted(Some(&GateFailure::OracleUnsupported("o".into())));
        assert_eq!(t.gates, 5);
        assert_eq!(t.same_form, 1);
        assert_eq!(t.cross_form, 1);
        assert_eq!(t.compile_failed, 1);
        assert_eq!(t.oracle_unsupported, 1);
        assert!((t.same_form_rate() - 0.2).abs() < 1e-12);
        assert!(t.describe().contains("same-form miscompiles"));
    }

    #[test]
    fn one_cross_form_divergence_in_a_large_run_is_not_a_failure() {
        // The smoke B4 shape: a single synthetic expression diverged and the
        // whole run died, taking five already-passing named kernels with it.
        let mut t = GateTally::default();
        for _ in 0..99 {
            t.record_extracted(None);
        }
        t.record_extracted(Some(&GateFailure::CrossForm("one".into())));
        assert!(
            t.alarms().is_empty(),
            "1% cross-form divergence is inside the documented threshold"
        );
    }

    #[test]
    fn a_systematic_cross_form_divergence_rate_fails_the_run() {
        let mut t = GateTally::default();
        for _ in 0..80 {
            t.record_extracted(None);
        }
        for _ in 0..20 {
            t.record_extracted(Some(&GateFailure::CrossForm("x".into())));
        }
        let alarms = t.alarms();
        assert_eq!(alarms.len(), 1, "got: {alarms:?}");
        assert!(alarms[0].starts_with("CROSS-FORM"), "got: {}", alarms[0]);
    }

    #[test]
    fn a_single_same_form_miscompile_in_a_small_run_fails_it() {
        // The same-form bar is tight on purpose: both tiers evaluate the SAME
        // arena, so a disagreement is a compiler bug, not corpus noise.
        let mut t = GateTally::default();
        for _ in 0..49 {
            t.record_extracted(None);
        }
        t.record_extracted(Some(&GateFailure::SameForm("miscompile".into())));
        let alarms = t.alarms();
        assert_eq!(alarms.len(), 1, "got: {alarms:?}");
        assert!(alarms[0].starts_with("SAME-FORM"), "got: {}", alarms[0]);
    }

    #[test]
    #[should_panic(expected = "before any policy was gated")]
    fn gate_rates_refuse_an_empty_denominator() {
        let _ = GateTally::default().same_form_rate();
    }

    // ── Verdict censoring (Codex P1) ────────────────────────────────────────

    #[test]
    fn failure_rates_count_exclusions_against_every_policy() {
        // 100 kernels; 5 excluded outright (no arm survives), plus 2 static
        // and 20 nnue phase-A failures among the survivors.
        let rates = PolicyFailureRates::compute(100, 5, 2, 20);
        assert!((rates.noswap - 0.05).abs() < 1e-12);
        assert!((rates.static_ - 0.07).abs() < 1e-12);
        assert!((rates.nnue - 0.25).abs() < 1e-12);
        assert_eq!(
            rates
                .censoring()
                .iter()
                .map(|&(n, _)| n)
                .collect::<Vec<_>>(),
            vec!["nnue"]
        );
    }

    /// A selection run with no correctness alarms — the baseline every verdict
    /// test varies one thing from. The CI defaults to a tight band around the
    /// geomean, so tests exercising other branches stay valid.
    fn selection_verdict_ci(
        geomean_nnue_static: f64,
        ci: Option<(f64, f64)>,
        noise_floor_pct: f64,
        rates: &PolicyFailureRates,
    ) -> String {
        verdict_text(&VerdictInputs {
            final_eval: false,
            gate_alarms: &[],
            geomean_nnue_static,
            ci,
            nnue_static_pairs: 100,
            noise_floor_pct,
            rates,
        })
    }

    fn selection_verdict(
        geomean_nnue_static: f64,
        noise_floor_pct: f64,
        rates: &PolicyFailureRates,
    ) -> String {
        selection_verdict_ci(
            geomean_nnue_static,
            Some((geomean_nnue_static * 0.99, geomean_nnue_static * 1.01)),
            noise_floor_pct,
            rates,
        )
    }

    #[test]
    fn censored_verdict_beats_a_large_favorable_margin() {
        // The exact shape Codex flagged: NNUE fails most kernels but wins big
        // on the survivors. A 50% speedup must NOT print as a PASS.
        let rates = PolicyFailureRates::compute(100, 0, 0, 60);
        let verdict = selection_verdict(0.5, 1.0, &rates);
        assert!(
            verdict.starts_with("INCONCLUSIVE-CENSORED"),
            "got: {verdict}"
        );
        assert!(verdict.contains("nnue 60.0%"), "got: {verdict}");
        // The full rate table is stated, not just the offenders.
        assert!(verdict.contains("noswap 0.0%"), "got: {verdict}");
    }

    #[test]
    fn censoring_vetoes_a_publication_run_too() {
        // Codex 3758853801: the publication return used to sit ABOVE the
        // censoring gate, so a FINAL run whose policies failed >10% of kernels
        // still printed "may be published" over a survivor-selected geomean —
        // and a FINAL run with zero pairs published a NaN.
        let rates = PolicyFailureRates::compute(100, 0, 0, 60);
        let verdict = verdict_text(&VerdictInputs {
            final_eval: true,
            gate_alarms: &[],
            geomean_nnue_static: 0.5,
            ci: Some((0.45, 0.55)),
            nnue_static_pairs: 40,
            noise_floor_pct: 0.3,
            rates: &rates,
        });
        assert!(
            verdict.starts_with("INCONCLUSIVE-CENSORED"),
            "got: {verdict}"
        );
        assert!(!verdict.contains("may be published"), "got: {verdict}");

        let clean = PolicyFailureRates::compute(100, 0, 0, 0);
        let verdict = verdict_text(&VerdictInputs {
            final_eval: true,
            gate_alarms: &[],
            geomean_nnue_static: f64::NAN,
            ci: None,
            nnue_static_pairs: 0,
            noise_floor_pct: f64::NAN,
            rates: &clean,
        });
        assert!(verdict.starts_with("INCONCLUSIVE"), "got: {verdict}");
        assert!(!verdict.contains("may be published"), "got: {verdict}");
    }

    #[test]
    fn clean_run_passes_and_states_the_failure_rates_it_survived() {
        let rates = PolicyFailureRates::compute(100, 1, 2, 3);
        assert!(rates.censoring().is_empty(), "1-3% is under the 10% bar");
        let verdict = selection_verdict(0.5, 1.0, &rates);
        assert!(verdict.contains("PASSES"), "got: {verdict}");
        assert!(
            verdict.contains("noswap 1.0%")
                && verdict.contains("static 3.0%")
                && verdict.contains("nnue 4.0%"),
            "a PASS must state the failure rates it survived; got: {verdict}"
        );
    }

    // ── The verdict compares the CI against the gate, not a point estimate ──

    #[test]
    fn a_pass_requires_the_whole_interval_to_clear_the_gate() {
        let rates = PolicyFailureRates::compute(100, 0, 0, 0);
        // Point estimate says +12% win, but the CI's slow end (ratio 0.97 →
        // +3%) does not clear +5%: round 0's exact failure mode, where a
        // favorable point estimate was one kernel away from flipping.
        let verdict = selection_verdict_ci(0.88, Some((0.80, 0.97)), 0.3, &rates);
        assert!(
            verdict.starts_with("INCONCLUSIVE-UNDERPOWERED"),
            "got: {verdict}"
        );
        assert!(verdict.contains("--max-kernels"), "got: {verdict}");
        // Move the whole interval past the gate and it passes.
        let verdict = selection_verdict_ci(0.88, Some((0.80, 0.94)), 0.3, &rates);
        assert!(verdict.contains("PASSES"), "got: {verdict}");
        assert!(verdict.contains("ENTIRE 95% CI"), "got: {verdict}");
    }

    #[test]
    fn a_decisive_loss_requires_the_whole_interval_below_the_line() {
        let rates = PolicyFailureRates::compute(100, 0, 0, 0);
        // Whole CI above ratio 1.05 (slower than -5%): decisive FAIL.
        let verdict = selection_verdict_ci(1.10, Some((1.06, 1.15)), 0.3, &rates);
        assert!(verdict.contains("FAILS"), "got: {verdict}");
        assert!(verdict.contains("ENTIRE 95% CI"), "got: {verdict}");
        // CI reaching back inside the band: underpowered, not a verdict.
        let verdict = selection_verdict_ci(1.10, Some((1.02, 1.19)), 0.3, &rates);
        assert!(
            verdict.starts_with("INCONCLUSIVE-UNDERPOWERED"),
            "got: {verdict}"
        );
    }

    #[test]
    fn an_interval_inside_the_band_is_a_resolved_tie_not_underpowered() {
        let rates = PolicyFailureRates::compute(100, 0, 0, 0);
        // CI wholly inside ±5%: the estimator RESOLVED "no meaningful win" —
        // plan 4.3 says iterate, and the static prior stays default.
        let verdict = selection_verdict_ci(1.01, Some((0.98, 1.04)), 0.3, &rates);
        assert!(verdict.contains("approx-ties"), "got: {verdict}");
        assert!(verdict.contains("FAILS"), "got: {verdict}");
        assert!(verdict.contains("iterate"), "got: {verdict}");
    }

    #[test]
    fn no_interval_means_no_verdict() {
        let rates = PolicyFailureRates::compute(100, 0, 0, 0);
        // A 50% point-estimate win with no CI must not pass: refusing the
        // point estimate is the entire reason the interval exists.
        let verdict = selection_verdict_ci(0.5, None, 0.3, &rates);
        assert!(verdict.starts_with("INCONCLUSIVE"), "got: {verdict}");
        assert!(!verdict.contains("PASSES"), "got: {verdict}");
    }

    // ── Bootstrap CI / ratio distribution / leave-one-out ───────────────────

    #[test]
    fn bootstrap_ci_is_deterministic_and_brackets_the_geomean() {
        let ratios: Vec<f64> = (0..40).map(|i| 0.8 + 0.01 * i as f64).collect();
        let a = bootstrap_geomean_ci(&ratios, 2000, BOOTSTRAP_SEED).expect("n=40 has a CI");
        let b = bootstrap_geomean_ci(&ratios, 2000, BOOTSTRAP_SEED).expect("n=40 has a CI");
        assert_eq!(a, b, "same seed must give the same interval");
        let g = geomean(&ratios, "test ratios");
        assert!(a.0 < g && g < a.1, "CI {a:?} must bracket the geomean {g}");
        assert!(a.0 < a.1);
        // More dispersion → wider interval.
        let tight: Vec<f64> = vec![1.0; 40];
        let t = bootstrap_geomean_ci(&tight, 2000, BOOTSTRAP_SEED).expect("CI");
        assert_eq!(t, (1.0, 1.0), "zero dispersion is a zero-width interval");
    }

    #[test]
    fn bootstrap_ci_refuses_fewer_than_two_kernels() {
        assert!(bootstrap_geomean_ci(&[], 100, 1).is_none());
        assert!(bootstrap_geomean_ci(&[0.9], 100, 1).is_none());
    }

    #[test]
    #[should_panic(expected = "non-positive ratio")]
    fn bootstrap_ci_panics_on_nonpositive_ratios() {
        let _ = bootstrap_geomean_ci(&[1.0, 0.0], 100, 1);
    }

    #[test]
    fn ratio_distribution_counts_wins_and_losses() {
        let d = RatioDistribution::of(&[0.5, 0.9, 1.0, 1.1, 1.3]).expect("nonempty");
        assert_eq!(d.wins, 2);
        assert_eq!(d.losses, 2);
        assert_eq!(d.ties, 1);
        assert_eq!(d.median, 1.0);
        assert_eq!(d.q1, 0.9);
        assert_eq!(d.q3, 1.1);
        assert!(RatioDistribution::of(&[]).is_none());
    }

    #[test]
    fn leave_one_out_finds_the_dominating_kernel() {
        // Round 0: a 12-kernel geomean of 0.9128 flipped to 1.1031 by dropping
        // ONE kernel. The LOO figure exists to make that visible every run.
        let pairs: Vec<(String, f64)> = vec![
            ("a".into(), 1.0),
            ("b".into(), 1.02),
            ("outlier".into(), 0.354),
            ("c".into(), 0.98),
        ];
        let loo = LooSensitivity::of(&pairs).expect("n=4");
        assert_eq!(loo.most_influential, "outlier");
        assert!(loo.min < loo.max);
        assert!(
            loo.shift_pct > 20.0,
            "dropping the outlier must shift the geomean hard, got {}%",
            loo.shift_pct
        );
        assert!(LooSensitivity::of(&pairs[..1]).is_none(), "n=1 has no LOO");
    }

    #[test]
    fn censoring_threshold_is_strict_inequality_at_the_bar() {
        // Exactly 10% is tolerated; one kernel more is not.
        let at_bar = PolicyFailureRates::compute(100, 0, 0, 10);
        assert!(at_bar.censoring().is_empty());
        let over_bar = PolicyFailureRates::compute(100, 0, 0, 11);
        assert_eq!(over_bar.censoring().len(), 1);
    }

    #[test]
    #[should_panic(expected = "no kernels entered phase A")]
    fn failure_rates_refuse_an_empty_denominator() {
        let _ = PolicyFailureRates::compute(0, 0, 0, 0);
    }

    // ── Correctness alarms precede the verdict (Codex 3744586360) ───────────

    /// A tally with `same_form` miscompiles and `cross_form` divergences out
    /// of 100 gates, all of them extracted-policy gates (so both rates share
    /// the denominator and read as plain percentages).
    fn tally_with(same_form: usize, cross_form: usize) -> GateTally {
        GateTally {
            gates: 100,
            extracted_gates: 100,
            same_form,
            cross_form,
            ..GateTally::default()
        }
    }

    #[test]
    fn cross_form_rate_is_over_extracted_gates_only() {
        // Codex 3759014680: the healthy three-gate shape is one no-swap gate
        // per kernel beside two extracted gates. 12 divergences over 100
        // extracted attempts is 12% — over the old 150-gate denominator it
        // read 8% and slid under the 10% alarm.
        let mut tally = GateTally::default();
        for i in 0..50 {
            tally.record_noswap(None);
            let cross = GateFailure::CrossForm("test".into());
            for arm in 0..2 {
                let diverged = i * 2 + arm < 12;
                tally.record_extracted(diverged.then_some(&cross));
            }
        }
        assert_eq!(tally.gates, 150);
        assert_eq!(tally.extracted_gates, 100);
        assert!((tally.cross_form_rate() - 0.12).abs() < 1e-12);
        assert_eq!(
            tally.alarms().len(),
            1,
            "12% over extracted gates must trip the 10% alarm"
        );
    }

    #[test]
    #[should_panic(expected = "no-swap gate produced CrossForm")]
    fn a_noswap_cross_form_is_a_harness_bug() {
        let mut tally = GateTally::default();
        tally.record_noswap(Some(&GateFailure::CrossForm("impossible".into())));
    }

    #[test]
    fn same_form_alarm_forces_an_invalid_verdict_over_a_clean_pass() {
        // Everything else says PASS: a huge margin, no censoring. The 2%
        // same-form miscompile rate must still take the verdict.
        let tally = tally_with(2, 0);
        let alarms = tally.alarms();
        assert_eq!(alarms.len(), 1, "2% > 1% must trip exactly one alarm");
        let rates = PolicyFailureRates::compute(100, 1, 2, 3);
        let verdict = verdict_text(&VerdictInputs {
            final_eval: false,
            gate_alarms: &alarms,
            geomean_nnue_static: 0.5,
            ci: Some((0.45, 0.55)),
            nnue_static_pairs: 100,
            noise_floor_pct: 1.0,
            rates: &rates,
        });
        assert!(
            verdict.starts_with("INVALID-CORRECTNESS-GATE"),
            "got: {verdict}"
        );
        assert!(!verdict.contains("PASSES"), "got: {verdict}");
        assert!(
            verdict.contains("SAME-FORM miscompile rate"),
            "got: {verdict}"
        );
    }

    #[test]
    fn cross_form_alarm_forces_an_invalid_verdict_over_a_publication_run() {
        // The run that actually happened: 0 same-form, 19.44% cross-form. A
        // FINAL-tier run must not say these numbers may be published.
        let tally = tally_with(0, 20);
        let alarms = tally.alarms();
        assert_eq!(alarms.len(), 1, "20% > 10% must trip exactly one alarm");
        let rates = PolicyFailureRates::compute(100, 0, 0, 0);
        let verdict = verdict_text(&VerdictInputs {
            final_eval: true,
            gate_alarms: &alarms,
            geomean_nnue_static: 0.93,
            ci: Some((0.90, 0.96)),
            nnue_static_pairs: 100,
            noise_floor_pct: 0.44,
            rates: &rates,
        });
        assert!(
            verdict.starts_with("INVALID-CORRECTNESS-GATE"),
            "got: {verdict}"
        );
        assert!(
            !verdict.contains("may be published"),
            "a failed correctness gate must not authorize publication; got: {verdict}"
        );
        assert!(
            verdict.contains("CROSS-FORM divergence rate"),
            "the verdict must name the alarm it failed on; got: {verdict}"
        );
    }

    #[test]
    fn both_alarms_are_named_in_the_verdict() {
        let alarms = tally_with(5, 20).alarms();
        assert_eq!(alarms.len(), 2);
        let rates = PolicyFailureRates::compute(100, 0, 0, 0);
        let verdict = verdict_text(&VerdictInputs {
            final_eval: false,
            gate_alarms: &alarms,
            geomean_nnue_static: 0.93,
            ci: Some((0.90, 0.96)),
            nnue_static_pairs: 100,
            noise_floor_pct: 0.44,
            rates: &rates,
        });
        assert!(verdict.contains("2 correctness alarm(s)"), "got: {verdict}");
        assert!(verdict.contains("SAME-FORM") && verdict.contains("CROSS-FORM"));
    }

    #[test]
    fn a_clean_gate_leaves_the_publication_verdict_alone() {
        // The alarms are a veto, not a rewrite: with none tripped, a FINAL run
        // still reports exactly what it reported before.
        assert!(tally_with(0, 0).alarms().is_empty());
        let rates = PolicyFailureRates::compute(100, 0, 0, 0);
        let verdict = verdict_text(&VerdictInputs {
            final_eval: true,
            gate_alarms: &[],
            geomean_nnue_static: 0.93,
            ci: Some((0.90, 0.96)),
            nnue_static_pairs: 42,
            noise_floor_pct: 0.44,
            rates: &rates,
        });
        assert!(verdict.starts_with("PUBLICATION RUN"), "got: {verdict}");
        assert!(verdict.contains("over 42 pairs"), "got: {verdict}");
    }

    #[test]
    fn a_gate_that_bounded_nothing_is_its_own_slug() {
        // Vacuity is counted apart from an unquarantinable shape (Codex
        // 3744586366): they say different things about a run.
        let failure = GateFailure::NoBoundedPoints("k/nnue: all skipped".into());
        assert_eq!(failure.kind(), "no_bounded_points");
        let mut tally = GateTally::default();
        tally.record_extracted(Some(&failure));
        tally.record_extracted(None);
        assert_eq!(tally.no_bounded_points, 1);
        assert_eq!(
            tally.oracle_unsupported, 0,
            "vacuity must not hide inside oracle_unsupported"
        );
        assert!(
            tally.alarms().is_empty(),
            "an unverifiable form censors the comparison set; it does not accuse the compiler"
        );
        assert!(tally.describe().contains("1 bounded nothing"));
    }

    // ── Corpus identity in the config hash (Codex 3744586352) ───────────────

    #[test]
    fn corpus_identity_distinguishes_tier_contents() {
        let a = corpus_identity(b"tier built from seed 1");
        let b = corpus_identity(b"tier built from seed 2");
        assert_ne!(a, b, "different corpora must not share an identity");
        assert_eq!(
            a,
            corpus_identity(b"tier built from seed 1"),
            "identity must be a function of the bytes alone"
        );
        assert_eq!(a.len(), 16, "fixed-width hex, like the other hashes");
    }

    #[test]
    fn config_hash_changes_when_only_the_corpus_changed() {
        // The regression: same executable, same flags, same revision, a tier
        // regenerated from a different seed. Before the fix these two runs
        // were one `config_hash`, and the journal called them the same setup.
        // `config_params` now supplies only the protocol half; the corpus
        // identity is a `ConfigHash::compose` argument in its own right.
        let args = Args {
            corpus_dir: "pixelflow-pipeline/data".into(),
            final_eval: false,
            max_kernels: 40,
            sample_seed: 0x5A11_2026_0805,
        };
        let source = SourceVersion {
            rev: "0".repeat(40),
            diff_hash: None,
        };
        let protocol = config_params(Tier::Dev, &args);
        let before = ConfigHash::compose(&source, "1111111111111111", "weights", &protocol);
        let after = ConfigHash::compose(&source, "2222222222222222", "weights", &protocol);
        assert_ne!(before.value, after.value);
        assert_eq!(before.corpus_identity, "1111111111111111");
        assert_eq!(after.corpus_identity, "2222222222222222");
    }

    // ── Source version in the config hash (Codex P2 + 3741889899) ───────────

    #[test]
    fn source_revision_is_a_sha_or_a_loud_unversioned() {
        let source = SourceVersion::current(&DIFF_HASH_EXCLUDES);
        if source.rev == "unversioned" {
            return; // git unavailable in this environment; the warning printed.
        }
        let sha = source.rev.strip_suffix("-dirty").unwrap_or(&source.rev);
        assert_eq!(sha.len(), 40, "expected a full sha, got {:?}", source.rev);
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "expected hex, got {:?}",
            source.rev
        );
        assert_eq!(
            source.diff_hash.is_some(),
            source.rev.ends_with("-dirty"),
            "a dirty tree must carry a diff hash and a clean tree must not"
        );
    }

    #[test]
    fn two_dirty_trees_at_one_commit_hash_differently() {
        // Codex 3741889899: "-dirty" alone collapsed every uncommitted variant
        // onto one identity, so two working trees measuring different code
        // produced identical config hashes.
        let clean = SourceVersion {
            rev: "abc".into(),
            diff_hash: None,
        };
        let dirty_a = SourceVersion {
            rev: "abc-dirty".into(),
            diff_hash: Some("1111".into()),
        };
        let dirty_b = SourceVersion {
            rev: "abc-dirty".into(),
            diff_hash: Some("2222".into()),
        };
        let hash =
            |s: &SourceVersion| ConfigHash::compose(s, "corpus", "weights", "protocol").value;
        assert_ne!(hash(&dirty_a), hash(&dirty_b));
        assert_ne!(hash(&clean), hash(&dirty_a));
        assert_eq!(hash(&dirty_a), hash(&dirty_a));
    }

    #[test]
    fn the_diff_hash_ignores_the_run_s_own_outputs() {
        // A run writes its JSONL under pixelflow-pipeline/data and appends to
        // docs/results/journal.jsonl. If those counted, a second run of an
        // otherwise unchanged tree would hash differently from the first.
        //
        // `top,` is load-bearing, not decoration: `git()` runs with
        // current_dir set to this crate's own manifest directory, so a bare
        // `:(exclude)pixelflow-pipeline/data` resolves relative to THAT
        // directory (i.e. `pixelflow-pipeline/pixelflow-pipeline/data`,
        // which never exists) and excludes nothing — see the real
        // behavioral regression in journal.rs's own test module
        // (`diff_hash_exclusions_survive_running_from_a_subdirectory`).
        assert!(DIFF_HASH_EXCLUDES.contains(&":(top,exclude)pixelflow-pipeline/data"));
        assert!(DIFF_HASH_EXCLUDES.contains(&":(top,exclude)docs/results"));
    }

    #[test]
    fn config_hash_changes_with_the_revision() {
        let source_a = SourceVersion {
            rev: "aaaaaaa".into(),
            diff_hash: None,
        };
        let source_b = SourceVersion {
            rev: "bbbbbbb".into(),
            diff_hash: None,
        };
        let hash =
            |s: &SourceVersion| ConfigHash::compose(s, "corpus", "weights", "mode=latency").value;
        assert_ne!(hash(&source_a), hash(&source_b));
        assert_eq!(hash(&source_a), hash(&source_a));
    }

    #[test]
    fn shuffle_is_deterministic_and_permutes() {
        let mut a = SeededRng::new(ORDER_SEED);
        let mut b = SeededRng::new(ORDER_SEED);
        let mut xs = ARMS;
        let mut ys = ARMS;
        for _ in 0..100 {
            a.shuffle(&mut xs);
            b.shuffle(&mut ys);
            assert_eq!(xs, ys, "same seed must give the same order");
            let mut sorted = xs.map(Arm::idx);
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2, 3], "shuffle must stay a permutation");
        }
    }

    #[test]
    fn family_labels_split_named_production_kernels_from_synthetics() {
        assert_eq!(family_of("swirl"), "named");
        assert_eq!(family_of("dev_b03_f07_00012"), "synthetic");
    }

    /// Registration check for the withheld ShaderToy set (--final-eval path):
    /// every name `corpus_split.toml`'s `[final].kernels` adds beyond the
    /// five original production kernels must report `family_of == "named"`
    /// here too, or a `corpus_final.bin` row for one would be silently
    /// mislabeled `synthetic` in the report `--final-eval` prints.
    #[test]
    fn family_labels_cover_the_withheld_shadertoy_set() {
        use pixelflow_pipeline::shader_bench::SHADERTOY_KERNEL_NAMES;
        for name in SHADERTOY_KERNEL_NAMES {
            assert_eq!(
                family_of(name),
                "named",
                "{name} is in SHADERTOY_KERNEL_NAMES but NAMED_KERNELS does not know it"
            );
        }
        assert_eq!(NAMED_KERNELS.len(), 5 + SHADERTOY_KERNEL_NAMES.len());
    }
}
