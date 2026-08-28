//! Round 2b extraction-overhead investigation
//! (docs/plans/2026-08-17-cost-model-domain.md, anti-row A6): determinism
//! digest + phase-timing attribution over
//! `IncrementalExtractor::extract_choices_only`.
//!
//! Two things, one binary, because both need the same per-kernel setup (load
//! a DEV corpus expression, build + saturate its e-graph, run NNUE
//! extraction under the canonical weights):
//!
//! 1. **Determinism digest** — the acceptance test the later Fix phase must
//!    hold to. Runs NNUE extraction over every sampled DEV expression and
//!    folds every chosen form (the exact `choices_to_arena` output, node for
//!    node — op kinds, constants, child ids, root id) plus every predicted
//!    log-cost into one FNV-1a digest. Two runs against the same corpus and
//!    weights on the same source must print the identical digest; a changed
//!    digest after a "performance-only" change means extraction chose a
//!    different form or scored one differently, which is a semantics
//!    regression under HARD CONSTRAINT (1), not a speed win.
//!
//! 2. **Phase attribution** — only meaningful under
//!    `--features extraction-profile` (this crate passes the flag through to
//!    `pixelflow-search/extraction-profile`; see
//!    `pixelflow_search::egraph::profile`). Resets the profiler's
//!    thread-local buckets before each kernel's extraction call and reads
//!    them back after, so the aggregated report is mean us/kernel and call
//!    counts per phase over the whole run — not a cumulative total that
//!    would hide per-kernel variance.
//!
//! Run:
//! ```bash
//! # digest only (any build)
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin profile_extraction -- --max-kernels 280
//!
//! # digest + phase attribution
//! cargo run --release -p pixelflow-pipeline --features training,extraction-profile \
//!     --bin profile_extraction -- --max-kernels 280
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use pixelflow_ir::{ExprArena, ExprId, ExprNode};
use pixelflow_search::egraph::{EGraph, IncrementalExtractor, all_rules, choices_to_arena};
use pixelflow_search::nnue::ExprNnue;

use pixelflow_pipeline::extraction_head_weights_path;
use pixelflow_pipeline::training::corpus::read_corpus;

/// E-graph saturation budget — matches `prod_kernel_jit.rs` and
/// `bench_extraction_3way`'s `SATURATE_LIMIT`. Deliberately the same value:
/// a digest minted under a different saturation budget is not comparable.
const SATURATE_LIMIT: usize = 40;

/// Alternatives considered per e-class during NNUE refinement — matches
/// `bench_extraction_3way`'s `EXTRACT_TOP_K` and `pixelflow-compiler`'s.
const EXTRACT_TOP_K: usize = 8;

#[derive(Parser)]
#[command(name = "profile_extraction")]
#[command(about = "Determinism digest + phase-timing attribution for NNUE extraction (Round 2b)")]
struct Args {
    /// Directory holding the tiered corpus written by `gen_bench_corpus`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Maximum DEV-tier expressions to run (0 = the whole tier).
    #[arg(long, default_value_t = 0)]
    max_kernels: usize,

    /// NNUE weights to extract with. Defaults to the canonical
    /// `extraction_head_weights_path()` — pass an explicit throwaway
    /// checkpoint's path only when the canonical one isn't what you want to
    /// pin the digest to.
    #[arg(long)]
    weights: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// FNV-1a digest over the exact bytes `choices_to_arena` would materialise.
// ---------------------------------------------------------------------------

/// Deterministic across runs and platforms (no hashmap-seed randomness,
/// unlike `std::hash::DefaultHasher`) — the entire point of a cross-run
/// digest is that it does NOT change unless the inputs do.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn write_u8(&mut self, b: u8) {
        self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn write_bytes(&mut self, bs: &[u8]) {
        for &b in bs {
            self.write_u8(b);
        }
    }

    fn write_u32(&mut self, v: u32) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn write_f32(&mut self, v: f32) {
        // Bit pattern, not the float value: two NaN payloads with the same
        // meaning must still show up as a digest change if the bits differ,
        // because the JIT compiles the bit pattern, not the mathematical
        // value.
        self.write_bytes(&v.to_bits().to_le_bytes());
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Fold every node of `arena` (in arena order — `choices_to_arena` builds
/// deterministically, so identical choices always produce byte-identical
/// arenas in the same order) plus the root id into `h`.
fn hash_arena(h: &mut Fnv1a, arena: &ExprArena, root: ExprId) {
    h.write_u32(arena.len() as u32);
    h.write_u32(root.0);
    for i in 0..arena.len() {
        let id = ExprId(i as u32);
        match arena.node(id) {
            ExprNode::Var(v) => {
                h.write_u8(0);
                h.write_u8(*v);
            }
            ExprNode::Const(c) => {
                h.write_u8(1);
                h.write_f32(*c);
            }
            ExprNode::Param(p) => {
                h.write_u8(2);
                h.write_u8(*p);
            }
            ExprNode::Buffer(b) => {
                h.write_u8(3);
                h.write_u8(0); // padding tag byte, kept stable across variants
                h.write_bytes(&b.0.to_le_bytes());
            }
            ExprNode::Unary(op, a) => {
                h.write_u8(4);
                h.write_u8(*op as u8);
                h.write_u32(a.0);
            }
            ExprNode::Binary(op, a, b) => {
                h.write_u8(5);
                h.write_u8(*op as u8);
                h.write_u32(a.0);
                h.write_u32(b.0);
            }
            ExprNode::Ternary(op, a, b, c) => {
                h.write_u8(6);
                h.write_u8(*op as u8);
                h.write_u32(a.0);
                h.write_u32(b.0);
                h.write_u32(c.0);
            }
            ExprNode::Nary(op, start, len) => {
                h.write_u8(7);
                h.write_u8(*op as u8);
                h.write_u32(u32::from(*len));
                for child in arena.nary_children_slice(*start, *len) {
                    h.write_u32(child.0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase attribution (only does anything under `extraction-profile`)
// ---------------------------------------------------------------------------

#[cfg(feature = "extraction-profile")]
mod attribution {
    use pixelflow_search::egraph::profile::{self, Bucket};

    const BUCKETS: [Bucket; 7] = [
        Bucket::CandidateEnumeration,
        Bucket::TrySwapBackfill,
        Bucket::AcyclicityCheck,
        Bucket::ChosenVariance,
        Bucket::PinnedChoices,
        Bucket::AccumulatorRebuild,
        Bucket::NnueForward,
    ];

    #[derive(Clone, Copy, Default)]
    pub struct Totals {
        nanos: u64,
        count: u64,
    }

    pub struct Attribution {
        per_bucket: [Totals; 7],
        kernels: u64,
    }

    impl Attribution {
        pub fn new() -> Self {
            Self {
                per_bucket: [Totals::default(); 7],
                kernels: 0,
            }
        }

        /// Call once per kernel, immediately before the timed extraction
        /// call.
        pub fn reset_for_kernel(&self) {
            profile::reset();
        }

        /// Call once per kernel, immediately after the timed extraction
        /// call, before the next kernel's `reset_for_kernel`.
        pub fn absorb_kernel(&mut self) {
            self.kernels += 1;
            for (i, (_bucket, stats)) in profile::snapshot().into_iter().enumerate() {
                self.per_bucket[i].nanos += stats.nanos;
                self.per_bucket[i].count += stats.count;
            }
        }

        /// `(bucket name, mean us/kernel, mean calls/kernel, total calls)`.
        pub fn report(&self) -> Vec<(&'static str, f64, f64, u64)> {
            let n = self.kernels.max(1) as f64;
            BUCKETS
                .iter()
                .zip(self.per_bucket.iter())
                .map(|(b, t)| {
                    (
                        b.name(),
                        (t.nanos as f64 / 1000.0) / n,
                        t.count as f64 / n,
                        t.count,
                    )
                })
                .collect()
        }

        pub fn kernels(&self) -> u64 {
            self.kernels
        }
    }
}

#[cfg(not(feature = "extraction-profile"))]
mod attribution {
    pub struct Attribution;
    impl Attribution {
        pub fn new() -> Self {
            Self
        }
        pub fn reset_for_kernel(&self) {}
        pub fn absorb_kernel(&mut self) {}
        pub fn report(&self) -> Vec<(&'static str, f64, f64, u64)> {
            Vec::new()
        }
        pub fn kernels(&self) -> u64 {
            0
        }
    }
}

fn main() {
    let args = Args::parse();

    let weights_path = args.weights.unwrap_or_else(extraction_head_weights_path);
    eprintln!("Loading NNUE weights from {}", weights_path.display());
    let weights_bytes = std::fs::read(&weights_path).unwrap_or_else(|e| {
        panic!(
            "profile_extraction: failed to read weights at {}: {e}. Train a checkpoint first \
             (bootstrap_extraction_head), even a throwaway one — determinism is what's under \
             test here, not quality.",
            weights_path.display()
        )
    });
    let nnue = ExprNnue::from_bytes(&weights_bytes).unwrap_or_else(|e| {
        panic!(
            "profile_extraction: ExprNnue::from_bytes rejected {}: {e}",
            weights_path.display()
        )
    });
    eprintln!("NNUE weights loaded OK ({} bytes).", weights_bytes.len());

    let corpus_path = PathBuf::from(&args.corpus_dir).join("corpus_dev.bin");
    eprintln!("Loading DEV corpus from {}", corpus_path.display());
    let mut entries = read_corpus(&corpus_path).unwrap_or_else(|e| {
        panic!(
            "profile_extraction: failed to read corpus at {}: {e}. Generate one first: \
             cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus \
             -- --target 1500",
            corpus_path.display()
        )
    });
    if args.max_kernels > 0 && entries.len() > args.max_kernels {
        entries.truncate(args.max_kernels);
    }
    eprintln!("Running {} DEV expressions.", entries.len());
    if entries.len() < 200 {
        eprintln!(
            "WARNING: only {} expressions — the task's acceptance bar for the phase-attribution \
             report is >= 200 DEV expressions. Regenerate the corpus with a larger --target if \
             this run is meant to satisfy that bar.",
            entries.len()
        );
    }

    #[cfg(feature = "extraction-profile")]
    eprintln!("extraction-profile feature: ON — phase attribution will be reported.");
    #[cfg(not(feature = "extraction-profile"))]
    eprintln!(
        "extraction-profile feature: OFF — digest only. Rebuild with \
         `--features training,extraction-profile` for phase attribution."
    );

    let mut digest = Fnv1a::new();
    let mut attribution = attribution::Attribution::new();
    let mut total_us = 0.0f64;
    let mut kernels_run = 0u64;
    let mut num_classes_sum = 0u64;
    let mut num_classes_max = 0usize;
    let wall_start = Instant::now();

    for (name, arena, root) in &entries {
        let mut eg = EGraph::with_rules(all_rules());
        let root_class = eg.add_arena(arena, *root);
        eg.saturate_with_limit(SATURATE_LIMIT);
        let num_classes = eg.num_classes();
        num_classes_sum += num_classes as u64;
        num_classes_max = num_classes_max.max(num_classes);

        let extractor = IncrementalExtractor::new(&nnue, EXTRACT_TOP_K);

        attribution.reset_for_kernel();
        let t0 = Instant::now();
        let (cost, extraction) = extractor.extract_choices_only(&eg, root_class);
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
        attribution.absorb_kernel();

        let (out_arena, out_root) = choices_to_arena(&extraction);

        hash_arena(&mut digest, &out_arena, out_root);
        digest.write_f32(cost);

        total_us += elapsed_us;
        kernels_run += 1;

        let _ = name; // kernel name only matters for future per-kernel debugging output
    }

    let wall_us = wall_start.elapsed().as_secs_f64() * 1e6;

    println!("=== profile_extraction: Round 2b determinism + attribution ===");
    println!("kernels run:        {kernels_run}");
    println!("weights:            {}", weights_path.display());
    println!("corpus:             {}", corpus_path.display());
    println!("saturate_limit:     {SATURATE_LIMIT}");
    println!("extract_top_k:      {EXTRACT_TOP_K}");
    println!(
        "e-graph size:       mean {:.0} classes/kernel, max {} classes — the O(N) evidence for \
         the buckets below is this number, not the reachable-subtree size",
        num_classes_sum as f64 / kernels_run.max(1) as f64,
        num_classes_max
    );
    println!();
    println!("DETERMINISM DIGEST (FNV-1a over every chosen form + predicted cost):");
    println!("  digest = {:016x}", digest.finish());
    println!();
    println!(
        "MEAN EXTRACTION WALL TIME: {:.2} us/kernel (sum {:.2} ms over {} kernels; \
         includes egraph build+saturate is NOT counted — only extract_choices_only)",
        total_us / kernels_run.max(1) as f64,
        total_us / 1000.0,
        kernels_run
    );
    println!(
        "(run wall clock including corpus/egraph setup: {:.2} ms total)",
        wall_us / 1000.0
    );
    println!();

    let report = attribution.report();
    if report.is_empty() {
        println!(
            "No phase attribution — rebuild with --features training,extraction-profile to get \
             per-bucket timing."
        );
    } else {
        println!(
            "PHASE ATTRIBUTION over {} kernels (mean us/kernel, mean calls/kernel, total calls):",
            attribution.kernels()
        );
        let mut bucket_us_sum = 0.0f64;
        for (name, mean_us, mean_calls, total_calls) in &report {
            println!(
                "  {name:<24} {mean_us:>10.2} us/kernel   {mean_calls:>8.2} calls/kernel   \
                 {total_calls:>10} total calls"
            );
            bucket_us_sum += mean_us;
        }
        let residual = (total_us / kernels_run.max(1) as f64) - bucket_us_sum;
        println!(
            "  {:<24} {residual:>10.2} us/kernel   (total {:.2} - sum of buckets above {:.2})",
            "everything_else",
            total_us / kernels_run.max(1) as f64,
            bucket_us_sum
        );
    }
}
