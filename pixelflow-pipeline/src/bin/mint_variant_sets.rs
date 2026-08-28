//! Mint the contrastive `VariantSet` dataset (J6,
//! docs/plans/2026-08-17-cost-model-domain.md; Round-2 contrastive plan,
//! docs/plans/2026-08-05-egraph-nnue-research-workflow.md §4.2 item 2).
//!
//! For each of `--sets` TRAIN-family base expressions (read from
//! `corpus_train.bin`, per `gen_bench_corpus`'s family-holdout tiering):
//! saturate a fresh e-graph, enumerate up to `--k` diverse candidate
//! extractions (`pixelflow_search::egraph::enumerate_candidates` —
//! static-optimal, NNUE, and estimate-op swaps of both), fence EVERY
//! candidate against DEV/FINAL (not just the base), quarantine every
//! surviving candidate against the independent scalar oracle, and mint a
//! `CostLabel` for every candidate that passes both under one
//! `BenchSession` in `BenchMode::Latency`.
//!
//! Output is one JSON `VariantSet` per line
//! (`pixelflow-pipeline/data/variant_sets.jsonl` by default) plus a summary
//! printed to stdout and a `variant_set_mint` line appended to
//! `docs/results/journal.jsonl` — sets minted, candidates/set, exclusion
//! counts by reason, the fraction of within-set pairs whose measured delta
//! clears the run's noise floor (the key dataset-quality number: if most
//! pairs are unorderable, training has nothing to learn from them), and the
//! distribution of within-set spreads.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin mint_variant_sets -- \
//!     --sets 400 --k 8
//! ```

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;

use pixelflow_search::egraph::CostModel;
use pixelflow_search::nnue::ExprNnue;

use pixelflow_pipeline::extraction_head_weights_path;
use pixelflow_pipeline::jit_bench::{BenchMode, BenchSession};
use pixelflow_pipeline::journal::{
    ArtifactId, ConfigHash, DIFF_HASH_EXCLUDES, JournalEntry, SourceVersion, fnv1a64_hex,
};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::mint::{MintMetadata, normalized_label_ns};
use pixelflow_pipeline::training::quarantine::Quarantine;
use pixelflow_pipeline::training::split::{DevSide, Fence, FinalSide};
use pixelflow_pipeline::training::stats::{median, noise_floor_pct, noise_probe_arena, percentile};
use pixelflow_pipeline::training::variant_set::{MintTally, VariantSet, mint_variant_set};

/// E-graph saturation budget — matches `bench_extraction_3way`/`prod_kernel_jit`.
const SATURATE_LIMIT: usize = 40;

#[derive(Parser, Debug)]
#[command(name = "mint_variant_sets")]
#[command(
    about = "Mint the contrastive VariantSet dataset (J6): k labeled rewrite-variant \
                    extractions per TRAIN-family base expression"
)]
struct Args {
    /// Number of TRAIN-family base expressions to attempt (fewer sets are
    /// minted whenever a base or its candidates are excluded).
    #[arg(long, default_value_t = 400)]
    sets: usize,

    /// Candidate extractions requested per base (diversity, not optimality —
    /// see `pixelflow_search::egraph::enumerate_candidates`).
    #[arg(long, default_value_t = 8)]
    k: usize,

    /// Directory holding the tiered corpus files written by `gen_bench_corpus`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: PathBuf,

    /// Trained extraction-head weights (the NNUE arm of the enumeration).
    #[arg(long, default_value_os_t = extraction_head_weights_path())]
    weights: PathBuf,

    /// Output JSONL path: one `VariantSet` per line.
    #[arg(long, default_value = "pixelflow-pipeline/data/variant_sets.jsonl")]
    output: PathBuf,

    /// Quarantine sidecar JSONL (overwritten each run).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/variant_set_quarantine.jsonl"
    )]
    quarantine_log: PathBuf,

    /// Seed for the TRAIN-base attempt order shuffle.
    #[arg(long, default_value_t = 20260827)]
    order_seed: u64,

    /// How many bases between noise-floor probe measurements (a fixed
    /// reference kernel re-measured under the same session/mode, A/A style).
    #[arg(long, default_value_t = 20)]
    noise_probe_interval: usize,
}

// ---------------------------------------------------------------------------
// Seeded RNG for the base-attempt order shuffle. xorshift64* — same family
// `bench_extraction_3way`'s arm-order shuffle uses; the shuffle only needs
// determinism, not cryptographic quality, and `rand`/`StdRng` is not a
// dependency of this crate.
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

    fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            s.swap(i, j);
        }
    }
}

#[derive(Serialize)]
struct RunMetrics {
    sets_requested: usize,
    k_requested: usize,
    sets_attempted: usize,
    sets_minted: usize,
    base_quarantine_failed: usize,
    base_fence_blocked: usize,
    sets_insufficient_members: usize,
    candidates_attempted: usize,
    candidates_fence_blocked: usize,
    candidates_quarantine_failed: usize,
    candidates_bench_failed: usize,
    candidates_minted: usize,
    mean_candidates_per_set: f64,
    median_candidates_per_set: f64,
    noise_probe_samples: usize,
    noise_floor_pct: Option<f64>,
    total_pairs: usize,
    orderable_pairs: usize,
    orderable_pair_fraction: Option<f64>,
    spread_median_pct: Option<f64>,
    spread_p90_pct: Option<f64>,
    spread_max_pct: Option<f64>,
    wall_s: f64,
    wall_s_per_set_minted: Option<f64>,
    wall_s_per_base_attempted: f64,
}

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();

    eprintln!(
        "mint_variant_sets: requesting {} sets, k={}, corpus_dir={}",
        args.sets,
        args.k,
        args.corpus_dir.display()
    );

    // Holdout fence, built from DEV/FINAL on disk — applied to the base AND
    // every candidate (mint_variant_set's contract; plan task 3).
    let dev_fence = Fence::<DevSide>::build(&args.corpus_dir);
    let final_fence = Fence::<FinalSide>::build(&args.corpus_dir);
    eprintln!(
        "Holdout fence: {} DEV keys ({} entries), {} FINAL keys ({} entries)",
        dev_fence.len(),
        dev_fence.entries(),
        final_fence.len(),
        final_fence.entries()
    );

    // Extraction-head weights: the NNUE arm of `enumerate_candidates`.
    let weights_bytes = fs::read(&args.weights).unwrap_or_else(|e| {
        panic!(
            "failed to read extraction-head weights {}: {e}\n\
             Train them first:\n  cargo run --release -p pixelflow-pipeline --features \
             training --bin bootstrap_extraction_head",
            args.weights.display()
        )
    });
    let nnue = ExprNnue::from_bytes(&weights_bytes).unwrap_or_else(|e| {
        panic!(
            "ExprNnue::from_bytes rejected {}: {e}",
            args.weights.display()
        )
    });
    let mint = MintMetadata::read_for(&args.weights).unwrap_or_else(|e| {
        panic!(
            "failed to read mint sidecar for {}: {e}",
            args.weights.display()
        )
    });
    mint.require_mode(BenchMode::Latency);
    mint.require_weights(&weights_bytes);
    eprintln!(
        "Extraction head: {} ({} training samples, bench_mode={})",
        args.weights.display(),
        mint.samples,
        mint.bench_mode
    );

    let static_costs = CostModel::latency_prior();

    // TRAIN-family bases, read from the tiered corpus (already family-held-
    // out from DEV/FINAL by `gen_bench_corpus`) and shuffled deterministically
    // so a `--sets` smaller than the corpus samples across bands/families
    // rather than exhausting one band first.
    let train_path = args.corpus_dir.join("corpus_train.bin");
    let mut bases = read_corpus(&train_path).unwrap_or_else(|e| {
        panic!(
            "failed to read TRAIN corpus {}: {e}\n\
             Build the tiered corpus first:\n  cargo run --release -p pixelflow-pipeline \
             --features training --bin gen_bench_corpus",
            train_path.display()
        )
    });
    assert!(
        !bases.is_empty(),
        "TRAIN corpus {} is empty — nothing to mint variant sets from",
        train_path.display()
    );
    SeededRng::new(args.order_seed).shuffle(&mut bases);
    let attempt_budget = args.sets.min(bases.len());
    eprintln!(
        "TRAIN corpus: {} bases available, attempting {}",
        bases.len(),
        attempt_budget
    );

    let mut session = BenchSession::new();
    let mut quarantine = Quarantine::new(
        args.quarantine_log
            .to_str()
            .expect("quarantine log path must be utf8"),
    );
    let mut position = 0usize;
    let mut tally = MintTally::default();

    let (probe_arena, probe_root) = noise_probe_arena();
    let mut probe_ns: Vec<f64> = Vec::new();

    fs::create_dir_all(
        args.output
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )
    .unwrap_or_else(|e| panic!("failed to create output directory: {e}"));
    let out_file = fs::File::create(&args.output)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", args.output.display()));
    let mut out = std::io::BufWriter::new(out_file);

    let mut minted_sets: Vec<VariantSet> = Vec::new();
    let mut per_set_wall_s: Vec<f64> = Vec::new();

    for (attempt, (name, arena, root)) in bases.iter().take(attempt_budget).enumerate() {
        if attempt % args.noise_probe_interval == 0 {
            match session.benchmark_arena(&probe_arena, probe_root, BenchMode::Latency) {
                Ok(bench) => probe_ns.push(normalized_label_ns(&bench, "noise_probe")),
                Err(e) => eprintln!("noise probe measurement failed at attempt {attempt}: {e}"),
            }
        }

        let set_start = std::time::Instant::now();
        match mint_variant_set(
            &mut session,
            &mut quarantine,
            &dev_fence,
            &final_fence,
            &nnue,
            &static_costs,
            name,
            arena,
            *root,
            args.k,
            SATURATE_LIMIT,
            &mut position,
            &mut tally,
        ) {
            Ok(set) => {
                per_set_wall_s.push(set_start.elapsed().as_secs_f64());
                let line = serde_json::to_string(&set).expect("VariantSet must serialize");
                writeln!(out, "{line}")
                    .unwrap_or_else(|e| panic!("failed to write {}: {e}", args.output.display()));
                minted_sets.push(set);
            }
            Err(reason) => {
                eprintln!("[{name}] not minted: {}", reason.reason());
            }
        }

        if (attempt + 1) % 50 == 0 || attempt + 1 == attempt_budget {
            eprintln!(
                "  progress [{}/{}]: {} sets minted, {} candidates minted, elapsed {:.1}s",
                attempt + 1,
                attempt_budget,
                tally.sets_minted,
                tally.candidates_minted,
                start.elapsed().as_secs_f64()
            );
        }
    }
    out.flush()
        .unwrap_or_else(|e| panic!("failed to flush {}: {e}", args.output.display()));

    quarantine.finish();

    // ---- Dataset-quality report (plan task 4) -----------------------------

    let candidates_per_set: Vec<f64> = minted_sets.iter().map(|s| s.members.len() as f64).collect();
    let mean_candidates_per_set = if candidates_per_set.is_empty() {
        0.0
    } else {
        candidates_per_set.iter().sum::<f64>() / candidates_per_set.len() as f64
    };
    let median_candidates_per_set = if candidates_per_set.is_empty() {
        0.0
    } else {
        median(candidates_per_set.clone())
    };

    let floor = noise_floor_pct(&probe_ns);

    let all_pair_deltas: Vec<f64> = minted_sets
        .iter()
        .flat_map(|s| s.pair_deltas_pct())
        .collect();
    let total_pairs = all_pair_deltas.len();
    let orderable_pairs = match floor {
        Some(f) => all_pair_deltas.iter().filter(|&&d| d >= f).count(),
        None => 0,
    };
    let orderable_fraction =
        (total_pairs > 0 && floor.is_some()).then(|| orderable_pairs as f64 / total_pairs as f64);

    // Per-set spread: max/min member session_ns as a percent delta — "how
    // much daylight is there between the best and worst candidate in one
    // set" as a distribution across sets (distinct from all-pair deltas,
    // which mixes adjacent and far-apart candidates together).
    let spreads_pct: Vec<f64> = minted_sets
        .iter()
        .filter_map(|s| {
            let vals: Vec<f64> = s.members.iter().map(|m| m.session_ns).collect();
            let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (lo > 0.0).then(|| (hi - lo) / lo * 100.0)
        })
        .collect();

    let wall_s = start.elapsed().as_secs_f64();
    let wall_s_per_set_minted = (!per_set_wall_s.is_empty())
        .then(|| per_set_wall_s.iter().sum::<f64>() / per_set_wall_s.len() as f64);

    let metrics = RunMetrics {
        sets_requested: args.sets,
        k_requested: args.k,
        sets_attempted: tally.sets_attempted,
        sets_minted: tally.sets_minted,
        base_quarantine_failed: tally.base_quarantine_failed,
        base_fence_blocked: tally.base_fence_blocked,
        sets_insufficient_members: tally.sets_insufficient_members,
        candidates_attempted: tally.candidates_attempted,
        candidates_fence_blocked: tally.candidates_fence_blocked,
        candidates_quarantine_failed: tally.candidates_quarantine_failed,
        candidates_bench_failed: tally.candidates_bench_failed,
        candidates_minted: tally.candidates_minted,
        mean_candidates_per_set,
        median_candidates_per_set,
        noise_probe_samples: probe_ns.len(),
        noise_floor_pct: floor,
        total_pairs,
        orderable_pairs,
        orderable_pair_fraction: orderable_fraction,
        spread_median_pct: (!spreads_pct.is_empty()).then(|| median(spreads_pct.clone())),
        spread_p90_pct: (!spreads_pct.is_empty()).then(|| percentile(spreads_pct.clone(), 0.9)),
        spread_max_pct: spreads_pct
            .iter()
            .cloned()
            .fold(None, |acc, x| Some(acc.map_or(x, |a: f64| a.max(x)))),
        wall_s,
        wall_s_per_set_minted,
        wall_s_per_base_attempted: wall_s / attempt_budget.max(1) as f64,
    };

    println!("\n=== VariantSet minting summary ===");
    println!(
        "Sets: {} attempted, {} minted ({} base-quarantine-failed, {} base-fence-blocked, {} \
         insufficient-members)",
        metrics.sets_attempted,
        metrics.sets_minted,
        metrics.base_quarantine_failed,
        metrics.base_fence_blocked,
        metrics.sets_insufficient_members
    );
    println!(
        "Candidates: {} attempted, {} minted ({} fence-blocked, {} quarantine-failed, {} \
         bench-failed)",
        metrics.candidates_attempted,
        metrics.candidates_minted,
        metrics.candidates_fence_blocked,
        metrics.candidates_quarantine_failed,
        metrics.candidates_bench_failed
    );
    println!(
        "Candidates/set: mean {:.2}, median {:.1}",
        metrics.mean_candidates_per_set, metrics.median_candidates_per_set
    );
    match floor {
        Some(f) => println!(
            "Noise floor: {:.3}% (from {} probe re-measurements of one fixed kernel)",
            f, metrics.noise_probe_samples
        ),
        None => println!(
            "Noise floor: UNKNOWN (only {} probe samples — need >= 2)",
            metrics.noise_probe_samples
        ),
    }
    match orderable_fraction {
        Some(frac) => println!(
            "Orderable pairs: {}/{} ({:.1}%) clear the noise floor",
            orderable_pairs,
            total_pairs,
            frac * 100.0
        ),
        None => println!("Orderable pairs: UNKNOWN (no noise floor or no pairs)"),
    }
    if let (Some(med), Some(p90), Some(mx)) = (
        metrics.spread_median_pct,
        metrics.spread_p90_pct,
        metrics.spread_max_pct,
    ) {
        println!(
            "Within-set spread (max/min member, % of min): median {med:.2}%, p90 {p90:.2}%, max {mx:.2}%"
        );
    }
    println!(
        "Wall time: {:.1}s total, {:.3}s/set minted, {:.3}s/base attempted",
        wall_s,
        wall_s_per_set_minted.unwrap_or(f64::NAN),
        metrics.wall_s_per_base_attempted
    );
    println!("Output: {}", args.output.display());

    // ---- Journal (J15) ------------------------------------------------

    let source = SourceVersion::current(&DIFF_HASH_EXCLUDES);
    let train_bytes = fs::read(&train_path).unwrap_or_default();
    let corpus_hash = fnv1a64_hex(&train_bytes);
    let protocol = format!(
        "sets={};k={};order_seed={};noise_probe_interval={};saturate_limit={SATURATE_LIMIT}",
        args.sets, args.k, args.order_seed, args.noise_probe_interval
    );
    let config = ConfigHash::compose(&source, &corpus_hash, &mint.weights_fnv64, &protocol);
    let weights_artifact = ArtifactId::with_identity(
        args.weights.display().to_string(),
        mint.weights_fnv64.clone(),
    );
    let journal_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/results/journal.jsonl"
    ));
    JournalEntry::new("variant_set_mint", config, weights_artifact, metrics).append(&journal_path);
}
