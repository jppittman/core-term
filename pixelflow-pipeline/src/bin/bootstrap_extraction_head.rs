//! Bootstrap the extraction head (Judge value head) on a large diverse corpus.
//!
//! Generates expressions with BwdGenerator, JIT-benchmarks them through a
//! [`BenchSession`] (QoS pinning, DVFS burn-in, drift sentinel, tick-floor
//! autoscaling, overhead subtraction — audit H4/H5/M1/L5), and trains the
//! value MLP of ExprNnue on (expression → log-ns) pairs. No policy, no
//! critic, no egraph — just pure value head training.
//!
//! Labels are overhead-adjusted, sentinel-normalized [`BenchMode::Latency`]
//! measurements: the plan prefers chain-serialized latency for production
//! fidelity (per-pixel varying coordinates, audit H3), and the ephemeral
//! training-sample struct carries a single target, so only one mode is
//! minted. The mode is recorded in a
//! [`MintMetadata`](pixelflow_pipeline::training::mint::MintMetadata) sidecar
//! next to the weights so a downstream gate can *assert* that the objective
//! it measures is the objective the head was trained on, instead of assuming
//! it (latency and throughput rank expression shapes differently).
//!
//! Every genuine exclusion (compile failure, harness error) is written to a
//! structured sidecar JSONL and the run panics if the exclusion rate exceeds
//! [`MAX_EXCLUSION_RATE`] (audit M2). Zero-ns measurements are VALID labels
//! (constant-folded expressions) and are kept at the `log_ns` floor.
//!
//! The DEV tier (`corpus_dev.bin`, held out by generator family — see
//! `training::split`) is *measured* alongside everything else and *predicted*
//! after training, then reported as held-out MAE + Spearman rank correlation
//! (audit L4, plan 0.3b). Its ground truth does not depend on the model, so
//! there is no reason to measure it an hour later on a different part of the
//! machine's drift curve. The DEV corpus being absent is a hard error, not a
//! skip. Weights are written only *after* that evaluation completes: a
//! checkpoint left behind by an aborted run is silently warm-started from by
//! the next supposedly-fresh run, so losing the training is strictly better
//! than laundering unvalidated weights into the next experiment.
//!
//! # One clock, one order
//!
//! Every label — synthetic, TRAIN corpus, DEV — is minted through
//! [`normalized_label_ns`], which re-expresses the measurement in the
//! session's opening clock using the sentinel's measured drift. And all three
//! sources are pooled into one queue and benchmarked in a seeded shuffle of
//! it rather than source-by-source in generation order. The two are
//! complementary:
//! normalization removes slow drift, shuffling decorrelates whatever it
//! cannot remove from expression structure. Without the shuffle, generation
//! order emits size and family bands sequentially, so residual drift is
//! perfectly confounded with structure — learnable spurious signal, not
//! noise. The shuffle seed is printed and recorded in the weights sidecar;
//! stored corpus order is untouched.
//!
//! # The holdout fence
//!
//! The TRAIN corpus is fenced off from DEV/FINAL by `gen_bench_corpus`'s
//! cross-tier structural ledger, but the live [`BwdGenerator`] stream in
//! Phase 1 consults nothing: a shallow motif it happens to redraw may already
//! live in `corpus_dev.bin`, in which case the same structure is trained on
//! and then "held out", and the DEV MAE/Spearman stop measuring a holdout.
//! So the DEV and FINAL tiers are loaded up front, hashed with the *same*
//! [`arena_structural_key`](pixelflow_pipeline::training::structural) the
//! corpus ledger uses, and every colliding live expression is dropped —
//! counted, written to the exclusion sidecar, and reported loudly.
//!
//! The fence reports the node-size distribution of BOTH populations, because
//! its headline number is otherwise unreadable: the first end-to-end run saw
//! "dropped 0/100 (0.00%)" against a live stream whose median was 147 nodes
//! and fenced structures of at most 40. Zero collisions there was arithmetic,
//! not evidence — the populations could not overlap.
//!
//! # The numeric quarantine covers the live stream too
//!
//! `gen_bench_corpus` gates corpus-tier expressions through
//! [`training::quarantine`](pixelflow_pipeline::training::quarantine), but at
//! default flags the live synthetic stream is the *dominant* label source
//! (50000 against a few thousand) and used to reach the model unchecked. It
//! now passes the same predicate — literally the same module, so there is one
//! notion of "numerically trustworthy", not two — with its own sidecar, its
//! own rate line, and its own alarm.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin bootstrap_extraction_head -- \
//!   --synthetic 50000 --epochs 100 --lr 0.001
//! ```

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;
use serde::Serialize;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_search::egraph::all_rules;
use pixelflow_search::nnue::factored::{EMBED_DIM, EdgeAccumulator, ExprNnue, GraphAccumulator};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};

use pixelflow_pipeline::extraction_head_weights_path;
use pixelflow_pipeline::jit_bench::{BenchError, BenchMode, BenchResult, BenchSession, log_ns};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::episodes::{build_rule_templates, load_corpus_exprs};
use pixelflow_pipeline::training::factored::arena_to_kernel_code;
use pixelflow_pipeline::training::mint::{
    MintMetadata, NormalizationStats, bench_mode_slug, file_mtime_report, label_normalization,
    normalized_label_ns, unix_now_s, weights_identity,
};
use pixelflow_pipeline::training::quarantine::Quarantine;
use pixelflow_pipeline::training::split::Tier;
use pixelflow_pipeline::training::structural::StructuralKeySet;
use pixelflow_pipeline::training::unified_backward::{
    UnifiedGradients, apply_unified_sgd, backward_value, forward_cached,
};

/// The mode every target in this binary is minted with, recorded in the
/// weights sidecar so consumers can assert instead of assume (R6).
const MINT_MODE: BenchMode = BenchMode::Latency;

/// Censoring alarm threshold (audit M2): a run excluding more than this
/// fraction of its expressions is minting a non-randomly censored label set
/// and must not be trusted silently.
const MAX_EXCLUSION_RATE: f64 = 0.05;

#[derive(Parser, Debug)]
#[command(name = "bootstrap_extraction_head")]
#[command(about = "Bootstrap the extraction head on (expression, JIT timing) pairs")]
struct Args {
    /// Number of synthetic expressions to generate.
    #[arg(long, default_value_t = 50000)]
    synthetic: usize,

    /// Training epochs.
    #[arg(long, default_value_t = 100)]
    epochs: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Momentum.
    #[arg(long, default_value_t = 0.9)]
    momentum: f32,

    /// Weight decay.
    #[arg(long, default_value_t = 1e-5)]
    weight_decay: f32,

    /// Gradient clipping.
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,

    /// Value loss coefficient.
    #[arg(long, default_value_t = 1.0)]
    value_coeff: f32,

    /// Mini-batch size.
    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    /// Input model weights: loaded if the file exists, else random init.
    #[arg(long, default_value_os_t = extraction_head_weights_path())]
    model: PathBuf,

    /// Output path for the trained weights.
    #[arg(long, default_value_os_t = extraction_head_weights_path())]
    output: PathBuf,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Seed for the *benchmark order* shuffle (not storage order, which is
    /// never rewritten). Defaults to a fixed derivation of `--seed`, and the
    /// value actually used is printed and recorded in the weights sidecar so
    /// a run's drift-vs-structure confounding is reproducible.
    #[arg(long)]
    order_seed: Option<u64>,

    /// Directory holding the tiered corpus files written by gen_bench_corpus
    /// (`corpus_train.bin` / `corpus_dev.bin` / `corpus_final.bin`).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: PathBuf,

    /// Skip TRAIN-tier corpus expressions (train on synthetic only).
    #[arg(long, default_value_t = false)]
    skip_corpus: bool,

    /// Exclusion sidecar JSONL (overwritten each run): one structured record
    /// per excluded expression (audit M2).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/bootstrap_exclusions.jsonl"
    )]
    exclusions: PathBuf,

    /// Numeric-quarantine sidecar JSONL for the LIVE synthetic stream
    /// (overwritten each run). Deliberately a different file from the corpus
    /// build's `corpus_quarantine.jsonl`: two label sources, two records of
    /// what each one refused, neither clobbering the other.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/bootstrap_quarantine.jsonl"
    )]
    quarantine: PathBuf,

    /// Corpus-revalidation sidecar JSONL (overwritten each run): the numeric
    /// quarantine re-run on every loaded TRAIN/DEV corpus entry with THIS
    /// build's JIT and oracle (Codex 3759014678). "Quarantined at generation
    /// time" is a claim about the executable that generated the corpus — the
    /// v3 format records neither source revision nor target, so an entry
    /// minted by another ISA/build, or before an emitter regression, could
    /// otherwise hand a platform-specific miscompile a timing target.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/bootstrap_corpus_revalidation.jsonl"
    )]
    corpus_revalidation: PathBuf,

    /// Maximum DEV-tier expressions to benchmark for held-out metrics
    /// (sampled deterministically from `corpus_dev.bin`).
    #[arg(long, default_value_t = 2000)]
    dev_eval_max: usize,

    /// Progress print interval.
    #[arg(long, default_value_t = 1000)]
    progress_every: usize,
}

/// A single training sample: accumulator state + ground-truth log-ns cost
/// (overhead-adjusted, sentinel-normalized latency; see module docs for the
/// mode choice).
struct ValueSample {
    acc: EdgeAccumulator,
    target_log_ns: f32,
    /// The sentinel factor that produced `target_log_ns` from the raw
    /// measurement. Kept per sample, not just in aggregate: a label corrected
    /// by 1.6x leans on the correction far harder than one corrected by
    /// 1.001x, and a training loop that wants to down-weight the former needs
    /// the number attached to the sample rather than a run-level summary.
    normalization: f32,
}

/// Which label source a queued expression came from. Fieldless and ordered so
/// it can index the per-phase attempt/exclusion tallies directly — the audit-M2
/// censoring alarm is per phase, and a shuffled benchmark queue interleaves all
/// three.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    Synthetic = 0,
    TrainCorpus = 1,
    Dev = 2,
}

/// Every source, for iterating tallies.
const SOURCES: [Source; 3] = [Source::Synthetic, Source::TrainCorpus, Source::Dev];

impl Source {
    /// The `phase` slug written into exclusion records.
    fn phase(self) -> &'static str {
        match self {
            Source::Synthetic => "synthetic",
            Source::TrainCorpus => "train_corpus",
            Source::Dev => "dev",
        }
    }
}

/// One expression waiting to be benchmarked.
///
/// Everything that will be measured in this run — live synthetic, TRAIN
/// corpus, DEV — is pooled into one queue and benchmarked in a seeded shuffle
/// of it. Pooling is not a convenience: benchmarking each source in its own
/// contiguous block puts every source at a different point on the machine's
/// drift curve, so a source's mean timing carries an offset that has nothing
/// to do with its expressions. Interleaving them puts all three on one clock,
/// which is the same reason every label goes through the sentinel correction.
struct WorkItem {
    source: Source,
    name: String,
    /// Released (set to `None`) once benchmarked, for every source except
    /// DEV. DEV arenas outlive training because their predictions must be
    /// computed with the *trained* embeddings; the rest are dead weight the
    /// moment their label exists, and at `--synthetic 50000` that weight is
    /// hundreds of megabytes.
    arena: Option<ExprArena>,
    root: ExprId,
}

impl WorkItem {
    /// # Panics
    ///
    /// Panics if the arena was already released. A released item has no
    /// expression left to read, so returning a placeholder would silently
    /// produce a label or a prediction for the empty expression.
    fn arena(&self) -> &ExprArena {
        self.arena.as_ref().unwrap_or_else(|| {
            panic!(
                "work item '{}' was released after benchmarking — only DEV items are retained \
                 for post-training prediction",
                self.name
            )
        })
    }
}

// ── Label minting ───────────────────────────────────────────────────────────

/// A benchmark measurement turned into a training target.
struct MintedLabel {
    /// `log_ns` of the drift-corrected measurement.
    target_log_ns: f32,
    /// The correction factor that was applied, kept for the sample.
    normalization: f64,
}

/// Turn a measurement into a label, in the session's opening clock.
///
/// This is the *only* way a target is produced in this binary. `BenchSession`
/// records the sentinel's correction on every result, but recording is not
/// applying: feeding `adjusted_ns` straight to `log_ns` leaves timing drift
/// correlated with collection position, and collection position correlates
/// with expression structure unless something breaks the correlation. Both
/// halves are here — this function removes the drift the sentinel measured,
/// and [`shuffled_order`] decorrelates whatever is left.
///
/// All three label sources (the synthetic stream, the TRAIN corpus, and DEV)
/// are minted here, so they sit on one clock by construction — including the
/// order in which [`normalized_label_ns`] applies the factor and subtracts the
/// opening-clock call overhead, which is not commutative and biases small
/// kernels if swapped.
///
/// # Panics
///
/// Panics when the measurement carries no sentinel context (see
/// [`normalized_label_ns`]).
fn mint_label(bench: &BenchResult, context: &str) -> MintedLabel {
    MintedLabel {
        target_log_ns: log_ns(normalized_label_ns(bench, context)),
        normalization: label_normalization(bench, context),
    }
}

// ── Research journal (plan 0.3b) ────────────────────────────────────────────

/// One line appended to `docs/results/journal.jsonl` per completed training
/// run.
///
/// Until this record existed, `bench_extraction_3way` was the journal's ONLY
/// writer, so DEV model quality (Spearman, MAE) was printed to a scrollback
/// and lost — model-quality history across rounds was invisible, and the
/// §2 loop's "read journal first" step could not see whether intrinsic
/// quality moved between rounds. The record is written only after the weights
/// AND their mint sidecar are on disk, so every journal line names a
/// checkpoint that actually exists.
#[derive(Serialize)]
struct TrainerJournalRecord {
    record: &'static str,
    ts_unix: u64,
    weights_path: String,
    corpus_dir: String,
    bench_mode: &'static str,
    /// Training samples the head was actually fit on (post-exclusion).
    train_samples: usize,
    synthetic_requested: usize,
    epochs: usize,
    lr: f32,
    seed: u64,
    order_seed: u64,
    /// DEV held-out intrinsic quality — the trainer's side of the plan §4.1
    /// eval-intrinsic step.
    dev_samples: usize,
    dev_mae_log_ns: f64,
    /// `None` (JSON `null`) when rank variance is zero — recorded, never
    /// dropped.
    dev_spearman_rho: Option<f64>,
    /// Mean (predicted − measured) log-ns on DEV through the DEPLOYED
    /// prediction path. Round-0 skew evidence was −0.291; ~0 means unbiased.
    dev_pred_bias_log_ns: f64,
    /// std(predicted)/std(measured) on DEV. Round-0 skew evidence was 0.45
    /// (range compressed to 45% of true); ~1 means calibrated spread.
    dev_pred_std_ratio: f64,
    /// (p90−p10 predicted)/(p90−p10 measured) on DEV — the same story as the
    /// std ratio but robust to tails.
    dev_pred_p90p10_ratio: f64,
}

/// Append one line to the research journal, loudly on any failure.
fn append_journal(record: &TrainerJournalRecord) {
    let journal_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/results/journal.jsonl"
    ));
    let line = serde_json::to_string(record)
        .unwrap_or_else(|e| panic!("failed to serialize the trainer journal record: {e}"));
    if let Some(parent) = journal_path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", parent.display()));
    }
    let mut journal = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&journal_path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", journal_path.display()));
    writeln!(journal, "{line}")
        .unwrap_or_else(|e| panic!("failed to append to {}: {e}", journal_path.display()));
    eprintln!("Journal appended: {}", journal_path.display());
}

/// A seeded permutation of `0..len` (LCG Fisher-Yates).
///
/// Used for *benchmark* order only. Stored corpus order is never rewritten —
/// the shuffle exists so that residual timing drift lands on expressions
/// independently of their size and family, not to change what is on disk.
fn shuffled_order(len: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut state = seed;
    let mut next = || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    for i in (1..order.len()).rev() {
        let j = next() as usize % (i + 1);
        order.swap(i, j);
    }
    order
}

// ── Population size reporting (the "dropped 0/100" misreading) ──────────────

/// Node-count summary of one population of expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SizeDistribution {
    count: usize,
    min: usize,
    p25: usize,
    median: usize,
    p75: usize,
    max: usize,
}

impl SizeDistribution {
    /// `None` for an empty population: a distribution over nothing is not a
    /// distribution, and rendering zeros would read as "every expression had
    /// 0 nodes".
    fn of(counts: &[usize]) -> Option<Self> {
        if counts.is_empty() {
            return None;
        }
        let mut sorted = counts.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        Some(Self {
            count: n,
            min: sorted[0],
            p25: sorted[n / 4],
            median: sorted[n / 2],
            p75: sorted[3 * n / 4],
            max: sorted[n - 1],
        })
    }

    fn render(&self) -> String {
        format!(
            "n={} min={} p25={} median={} p75={} max={}",
            self.count, self.min, self.p25, self.median, self.p75, self.max
        )
    }
}

/// Why a collision count of zero may carry no information.
///
/// The first end-to-end run reported the fence "dropped 0/100 (0.00%)" and the
/// number was read as evidence the fence worked. It was not: the live stream's
/// median was 147 nodes and the fenced structures were all at most 40, so no
/// live expression *could* have collided. Zero was arithmetic. Whenever the
/// two node-count ranges fail to overlap at all, the fence's headline number
/// must say so in the same breath.
fn disjointness_note(live: &SizeDistribution, fenced: &SizeDistribution) -> Option<String> {
    if live.min > fenced.max {
        return Some(format!(
            "the populations are DISJOINT BY CONSTRUCTION: the smallest live expression has {} \
             nodes and the largest fenced structure has {}. A collision count of zero here is \
             arithmetic, not evidence that the fence is working — nothing could have collided",
            live.min, fenced.max
        ));
    }
    if live.max < fenced.min {
        return Some(format!(
            "the populations are DISJOINT BY CONSTRUCTION: the largest live expression has {} \
             nodes and the smallest fenced structure has {}. A collision count of zero here is \
             arithmetic, not evidence that the fence is working — nothing could have collided",
            live.max, fenced.min
        ));
    }
    None
}

// ── Label sources ───────────────────────────────────────────────────────────

/// Refuse a run only when it has *no* label source at all.
///
/// An empty live stream is legitimate: `--synthetic 0` is a TRAIN-corpus-only
/// ablation, and a fence that happens to drop every live expression is the
/// fence doing its job. The corpus is a nonempty, fully-fenced, fully-
/// quarantined training source in its own right. What is never legitimate is
/// both sources being empty, which is a run that cannot train and must say so
/// instead of failing later with something vaguer.
///
/// # Panics
///
/// Panics when the live stream yielded nothing *and* `--skip-corpus` removed
/// the other source.
fn assert_label_sources(live_labelable: usize, corpus_enabled: bool, requested_synthetic: usize) {
    if live_labelable > 0 {
        return;
    }
    assert!(
        corpus_enabled,
        "no label source: the live synthetic stream yielded 0 usable expressions (--synthetic \
         {requested_synthetic}) and --skip-corpus removed the TRAIN corpus, so this run has \
         nothing to train on. Raise --synthetic or drop --skip-corpus"
    );
    eprintln!(
        "The live synthetic stream yielded 0 usable expressions (--synthetic \
         {requested_synthetic}); this run trains on the TRAIN corpus alone. That is a valid \
         configuration — the corpus is fenced and quarantined at build time — and is reported \
         here so it is a choice rather than a surprise."
    );
}

/// One excluded expression, written to the structured sidecar (audit M2):
/// never a bare counter, never a swallowed error.
#[derive(Serialize)]
struct ExclusionRecord<'a> {
    /// Which pipeline phase excluded it: `synthetic`, `train_corpus`, `dev`.
    phase: &'a str,
    name: &'a str,
    expression: &'a str,
    reason: &'a str,
    error: &'a str,
}

/// One live-generated expression refused by the holdout fence: its structure
/// already belongs to DEV or FINAL. Not a [`BenchError`] and not an exclusion
/// in the audit-M2 sense (nothing failed to measure), so it carries its own
/// record shape and its own counter — folding it into the exclusion rate
/// would make a *correctly working* fence look like a censoring alarm.
#[derive(Serialize)]
struct HoldoutDropRecord<'a> {
    phase: &'a str,
    name: &'a str,
    expression: &'a str,
    reason: &'a str,
    /// Which holdout tier owns the colliding structure, when known;
    /// `"dev_or_final"` when the fence holds both tiers in one set.
    owner: &'a str,
}

/// Sidecar writer for exclusion records. Every write failure panics — an
/// exclusion log that silently drops records defeats its purpose.
///
/// Records go through the `File` directly, one line per record — no
/// `BufWriter`. The workspace compiles with `panic = "abort"`, so the
/// designed mid-run loud failures (sentinel drift, plausibility floor) skip
/// `Drop` and would discard everything a buffer still held, leaving the
/// audit-M2 structured artifact empty exactly when it matters most.
/// Exclusions are asserted rare (<= 5%), so per-record writes cost nothing.
struct ExclusionLog {
    out: File,
    path: PathBuf,
    count: usize,
    /// Holdout-fence drops, counted separately from `count`: see
    /// [`HoldoutDropRecord`].
    holdout_dropped: usize,
}

impl ExclusionLog {
    fn create(path: &Path) -> Self {
        let file = File::create(path).unwrap_or_else(|e| {
            panic!("failed to create exclusion sidecar {}: {e}", path.display())
        });
        Self {
            out: file,
            path: path.to_path_buf(),
            count: 0,
            holdout_dropped: 0,
        }
    }

    /// Record one holdout-fence drop. Written per record (greppable, never
    /// buffered away by a mid-run abort) but NOT echoed per line to stderr:
    /// drops are expected in the hundreds on a 50k stream, and the loud
    /// signal is the phase-1 summary plus this sidecar.
    fn record_holdout_drop(&mut self, name: &str, expression: &str) {
        let record = HoldoutDropRecord {
            phase: "synthetic",
            name,
            expression,
            reason: "holdout_structural_collision",
            owner: "dev_or_final",
        };
        let json = serde_json::to_string(&record).unwrap_or_else(|e| {
            panic!("failed to serialize holdout drop record for '{name}': {e}")
        });
        writeln!(self.out, "{json}").unwrap_or_else(|e| {
            panic!(
                "write to exclusion sidecar {} failed: {e}",
                self.path.display()
            )
        });
        self.holdout_dropped += 1;
    }

    fn record(&mut self, phase: &str, name: &str, expression: &str, error: &BenchError) {
        let record = ExclusionRecord {
            phase,
            name,
            expression,
            reason: bench_error_reason(error),
            error: &error.to_string(),
        };
        let json = serde_json::to_string(&record)
            .unwrap_or_else(|e| panic!("failed to serialize exclusion record for '{name}': {e}"));
        writeln!(self.out, "{json}").unwrap_or_else(|e| {
            panic!(
                "write to exclusion sidecar {} failed: {e}",
                self.path.display()
            )
        });
        eprintln!("  EXCLUDED [{phase}] '{name}': {error}");
        self.count += 1;
    }
}

// ── Holdout fence (DEV/FINAL leakage, plan 0.2) ─────────────────────────────

/// The path of a tier's corpus file inside `corpus_dir`.
fn tier_corpus_path(corpus_dir: &Path, tier: Tier) -> PathBuf {
    corpus_dir.join(format!("corpus_{}.bin", tier.name()))
}

/// Every structure owned by the held-out tiers, so the live generator stream
/// can be filtered against it.
///
/// Built from the *complete* tier files (`read_corpus`), not the sampled and
/// size-filtered [`load_corpus_exprs`]: a structure that exists anywhere in
/// DEV or FINAL is holdout, whether or not this run's DEV evaluation happens
/// to draw it. Sampling the fence would leak exactly the expressions the
/// sampler skipped.
struct HoldoutFence {
    keys: StructuralKeySet,
    dev_entries: usize,
    final_entries: usize,
    /// Node count of every fenced structure, so the fence's drop count can be
    /// read against the size range it could possibly have matched (see
    /// [`disjointness_note`]).
    node_counts: Vec<usize>,
}

impl HoldoutFence {
    /// Load DEV and FINAL and hash every expression.
    ///
    /// # Panics
    ///
    /// Panics when either tier file is missing. `gen_bench_corpus` writes all
    /// three tiers in one loop, so a directory holding DEV but not FINAL is a
    /// partial artifact; and DEV's absence is already fatal at the
    /// held-out-evaluation phase — failing here rather than after an hour of
    /// benchmarking is the same verdict, delivered sooner.
    fn build(corpus_dir: &Path) -> Self {
        let mut keys = StructuralKeySet::new();
        let mut counts = [0usize; 2];
        let mut node_counts: Vec<usize> = Vec::new();
        for (slot, tier) in [Tier::Dev, Tier::Final].into_iter().enumerate() {
            let path = tier_corpus_path(corpus_dir, tier);
            assert!(
                path.exists(),
                "{} tier corpus not found at {} — the live-generation holdout fence cannot be \
                 built without it, and training an unfenced stream would put DEV/FINAL \
                 structures into the training set (the DEV metrics would stop being held out). \
                 Run gen_bench_corpus (--output {}), which writes all three tiers together",
                tier,
                path.display(),
                corpus_dir.display()
            );
            let entries = read_corpus(&path).unwrap_or_else(|e| {
                panic!("failed to read {} corpus {}: {e}", tier, path.display())
            });
            counts[slot] = entries.len();
            for (_name, arena, root) in &entries {
                keys.insert(arena, *root);
                node_counts.push(arena.node_count_subtree(*root));
            }
        }
        Self {
            keys,
            dev_entries: counts[0],
            final_entries: counts[1],
            node_counts,
        }
    }

    /// Whether this structure is already owned by a holdout tier.
    fn blocks(&self, arena: &ExprArena, root: ExprId) -> bool {
        self.keys.contains(arena, root)
    }
}

/// Report what the fence did, with enough context that its headline number
/// cannot be misread (see [`disjointness_note`]).
fn report_holdout_fence(
    fence: &HoldoutFence,
    live_counts: &[usize],
    dropped: usize,
    requested_synthetic: usize,
    sidecar: &Path,
) {
    let fenced = SizeDistribution::of(&fence.node_counts);
    let live = SizeDistribution::of(live_counts);

    eprintln!(
        "\nHoldout fence: dropped {dropped}/{requested_synthetic} live-generated expressions \
         ({:.2}%) whose structure already lives in DEV or FINAL — every one is a \
         `holdout_structural_collision` record in {}. Without this the DEV metrics would not \
         measure a holdout.",
        100.0 * dropped as f64 / requested_synthetic.max(1) as f64,
        sidecar.display()
    );

    match (&live, &fenced) {
        (Some(live), Some(fenced)) => {
            eprintln!(
                "  fenced structures (DEV+FINAL) node counts: {}",
                fenced.render()
            );
            eprintln!(
                "  live stream (kept) node counts:             {}",
                live.render()
            );
            match disjointness_note(live, fenced) {
                Some(note) => eprintln!("  WARNING: {note}."),
                None => eprintln!(
                    "  The two node-count ranges overlap ([{}, {}] vs [{}, {}]), so the drop \
                     count above is informative.",
                    live.min, live.max, fenced.min, fenced.max
                ),
            }
        }
        (None, Some(fenced)) => eprintln!(
            "  No live expressions survived to be sized (--synthetic {requested_synthetic}), so \
             the drop count is 0 by construction and says nothing about the fence. Fenced \
             structures: {}",
            fenced.render()
        ),
        (_, None) => eprintln!(
            "  The fence holds ZERO structures: DEV and FINAL are both empty, so nothing could \
             ever be dropped and the DEV metrics below have no holdout behind them."
        ),
    }
}

/// Stable category slug for a [`BenchError`], for the sidecar's `reason` field.
fn bench_error_reason(e: &BenchError) -> &'static str {
    match e {
        BenchError::CompileFailed(_) => "compile_failed",
        BenchError::UnsupportedArch => "unsupported_arch",
        BenchError::InvalidMeasurement(_) => "invalid_measurement",
    }
}

/// Print the exclusion rate for a phase and panic above the censoring alarm
/// threshold (audit M2).
fn assert_exclusion_rate(excluded: usize, attempted: usize, context: &str, sidecar: &Path) {
    assert!(
        attempted > 0,
        "{context}: zero expressions attempted — nothing was benchmarked"
    );
    let rate = excluded as f64 / attempted as f64;
    eprintln!(
        "{context}: excluded {excluded}/{attempted} ({:.2}%)",
        rate * 100.0
    );
    assert!(
        rate <= MAX_EXCLUSION_RATE,
        "censoring alarm (audit M2): {context} excluded {excluded}/{attempted} expressions \
         ({:.2}% > {:.0}%) — the label set is non-randomly censored; see {} for every exclusion",
        rate * 100.0,
        MAX_EXCLUSION_RATE * 100.0,
        sidecar.display()
    );
}

// ── Ranking metrics (plan 0.3b) ─────────────────────────────────────────────

/// Ranks (1-based) with ties assigned their average rank.
///
/// # Panics
///
/// Panics on NaN values — a NaN prediction or measurement is a bug upstream,
/// not something a metric should order arbitrarily.
fn average_ranks(values: &[f32]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        values[i].partial_cmp(&values[j]).unwrap_or_else(|| {
            panic!(
                "average_ranks: non-comparable (NaN) values at indices {i} ({}) and {j} ({})",
                values[i], values[j]
            )
        })
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        // Average of 1-based ranks i+1 ..= j+1.
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Spearman rank correlation: Pearson correlation of average ranks (exact
/// under ties, unlike the 6Σd²/n(n²−1) shortcut).
///
/// Returns `None` when either side has zero rank variance (all values tied) —
/// the coefficient is undefined there, and the caller must report that
/// explicitly rather than receive a fabricated number.
///
/// # Panics
///
/// Panics on length mismatch, fewer than 2 samples, or NaN values.
fn spearman_rho(a: &[f32], b: &[f32]) -> Option<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "spearman_rho: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    assert!(
        a.len() >= 2,
        "spearman_rho: need at least 2 samples, got {}",
        a.len()
    );
    let ra = average_ranks(a);
    let rb = average_ranks(b);
    // With average ranks the rank sum is always n(n+1)/2, so the mean rank
    // is (n+1)/2 regardless of ties.
    let mean = (a.len() as f64 + 1.0) / 2.0;
    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for (x, y) in ra.iter().zip(&rb) {
        let dx = x - mean;
        let dy = y - mean;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a == 0.0 || var_b == 0.0 {
        return None;
    }
    Some(cov / (var_a.sqrt() * var_b.sqrt()))
}

/// Load the starting weights, stating unmissably which of the two very
/// different experiments this run is.
///
/// Load-if-exists is convenient and dangerous in equal measure: an aborted run
/// leaves a checkpoint behind, and the next run — believed to be fresh — picks
/// it up. The smoke run's log opened with a bare `Loaded model from ...` line
/// on what was meant to be a clean start, and nobody read it as the warning it
/// was. Phase 4 now writes weights only after DEV evaluation so the stale
/// checkpoint is rarer; this banner makes the remaining case impossible to
/// skim past, and names the file's mtime so "which run left this?" is
/// answerable on the spot.
///
/// A file that exists but fails to parse is corrupt and fails loudly rather
/// than silently retraining from scratch.
fn load_model(path: &Path, seed: u64) -> ExprNnue {
    if !path.exists() {
        eprintln!(
            "\n########## COLD START ##########\n\
             No weights at {} — initializing randomly from seed {seed}.\n\
             ################################\n",
            path.display()
        );
        return ExprNnue::new_random(seed);
    }

    let model = ExprNnue::load(path).unwrap_or_else(|e| {
        panic!(
            "model file {} exists but failed to load: {e}",
            path.display()
        )
    });
    let mtime = file_mtime_report(path).unwrap_or_else(|reason| format!("mtime unknown: {reason}"));
    eprintln!(
        "\n########## WARM START — THIS RUN IS NOT FRESH ##########\n\
         Loaded existing weights from {} ({mtime}).\n\
         Training continues FROM those weights, not from random initialization. If you meant a\n\
         clean experiment, delete that file (or pass --model to a path that does not exist)\n\
         before rerunning: a checkpoint left behind by an earlier — possibly aborted — run\n\
         silently becomes this run's prior.\n\
         #######################################################\n",
        path.display()
    );
    model
}

fn main() {
    let args = Args::parse();

    let mut model = load_model(&args.model, args.seed);

    let rules = all_rules();
    let templates = build_rule_templates(&rules);

    let mut exclusions = ExclusionLog::create(&args.exclusions);

    // Holdout fence FIRST: built before a single expression is generated, so
    // a missing tier costs a second instead of an hour of benchmarking, and
    // so no unfenced expression can reach the training set.
    let fence = HoldoutFence::build(&args.corpus_dir);
    eprintln!(
        "Holdout fence: {} distinct structures from DEV ({} entries) + FINAL ({} entries) in {}",
        fence.keys.len(),
        fence.dev_entries,
        fence.final_entries,
        args.corpus_dir.display()
    );

    // One session for every label minted in this run: same QoS pin, same
    // sentinel calibration, same overhead baseline (audit H4/M1).
    let mut session = BenchSession::new();

    // The live stream's numeric quarantine (B8). Its own sidecar, so the
    // corpus build's `corpus_quarantine.jsonl` is never clobbered and each
    // label source keeps its own record of what it refused.
    let quarantine_path = args.quarantine.to_str().unwrap_or_else(|| {
        panic!(
            "--quarantine path {} is not valid UTF-8",
            args.quarantine.display()
        )
    });
    let mut quarantine = Quarantine::new(quarantine_path);

    // ── PHASE 1: Generate synthetic expressions (fence + quarantine) ──
    //
    // Generation and benchmarking are deliberately separated: everything this
    // run will measure is pooled first, then measured in one shuffled pass
    // (Phase 1d).

    eprintln!(
        "\n=== Phase 1: Generating {} synthetic expressions ===",
        args.synthetic
    );

    let config = BwdGenConfig::default();
    let mut generator = BwdGenerator::new(args.seed, config, templates.clone());

    let mut queue: Vec<WorkItem> = Vec::new();
    let mut live_counts: Vec<usize> = Vec::new();
    let mut live_ill_conditioned = 0usize;

    for i in 0..args.synthetic {
        let pair = generator.generate_arena();
        let name = format!("synthetic_{i}");

        // Holdout fence: a live redraw of a DEV/FINAL structure would be
        // trained on and then evaluated as if held out. Dropped before it can
        // become a label, and recorded structurally in the sidecar.
        if fence.blocks(&pair.arena, pair.unoptimized) {
            let expression = arena_to_kernel_code(&pair.arena, pair.unoptimized);
            exclusions.record_holdout_drop(&name, &expression);
            continue;
        }

        // Numeric quarantine (B8, plan 0.4): the SAME predicate the corpus
        // build applies, on the source that actually dominates the label set.
        // An expression the JIT and the scalar oracle disagree about on its
        // own arena is not a datum about how fast that expression runs.
        let (exclusion, conditioning) = quarantine.verdict(&name, &pair.arena, pair.unoptimized);
        if exclusion.is_some() {
            continue;
        }
        if !conditioning.is_clean() {
            live_ill_conditioned += 1;
        }

        live_counts.push(pair.arena.node_count_subtree(pair.unoptimized));
        queue.push(WorkItem {
            source: Source::Synthetic,
            name,
            arena: Some(pair.arena),
            root: pair.unoptimized,
        });

        if args.progress_every > 0 && (i + 1) % args.progress_every == 0 {
            let (checked, q_excluded, q_mismatched) = quarantine.tallies();
            eprintln!(
                "  [{}/{}] queued={} holdout_dropped={} quarantined={}/{} (miscompiles={})",
                i + 1,
                args.synthetic,
                queue.len(),
                exclusions.holdout_dropped,
                q_excluded,
                checked,
                q_mismatched
            );
        }
    }

    report_holdout_fence(
        &fence,
        &live_counts,
        exclusions.holdout_dropped,
        args.synthetic,
        &args.exclusions,
    );

    // The live stream's own quarantine rate line and alarm. `finish` prints
    // the tally and panics past the thresholds; it needs something to have
    // been checked, so a run with no live stream says so explicitly instead of
    // dividing by zero.
    let (q_checked, _, _) = quarantine.tallies();
    if q_checked > 0 {
        eprintln!(
            "\nLive-stream numeric quarantine ({live_ill_conditioned} of the survivors carry a \
             non-clean conditioning survey — recorded as metadata, never an exclusion):"
        );
        quarantine.finish();
    } else {
        eprintln!(
            "\nLive-stream numeric quarantine: nothing checked (--synthetic {}), so there is no \
             rate to report and no alarm to trip. Its sidecar {} is empty by construction.",
            args.synthetic, quarantine_path
        );
    }

    let live_labelable = queue.len();
    eprintln!(
        "Live stream: {live_labelable} of {} generated expressions survived the fence and the \
         quarantine",
        args.synthetic
    );
    assert_label_sources(live_labelable, !args.skip_corpus, args.synthetic);

    // ── PHASE 1b: Add TRAIN-tier corpus expressions ──────────────────
    //
    // Every loaded corpus entry — TRAIN here, DEV below — is revalidated
    // through the numeric quarantine before it can be minted (Codex
    // 3759014678). The corpus was quarantined at generation time, but that
    // verdict belongs to the executable that generated it: the v3 format
    // records neither source revision nor target, so a tier built by another
    // ISA/build — or before an emitter regression — could hand a
    // platform-specific miscompile a timing target (TRAIN) or measure a
    // broken backend as held-out truth (DEV). The same-form gate is re-run
    // with THIS build's JIT and oracle; drops are recorded in their own
    // sidecar, and the rate alarms in `finish` judge the corpus afterward.
    let revalidation_path = args.corpus_revalidation.to_str().unwrap_or_else(|| {
        panic!(
            "--corpus-revalidation path {} is not valid UTF-8",
            args.corpus_revalidation.display()
        )
    });
    let mut corpus_revalidation = Quarantine::new(revalidation_path);

    if !args.skip_corpus {
        let train_path = args
            .corpus_dir
            .join(format!("corpus_{}.bin", Tier::Train.name()));
        assert!(
            train_path.exists(),
            "TRAIN corpus not found at {} — run gen_bench_corpus (--output {}) to build the \
             tiered corpus, or pass --skip-corpus to train on synthetic expressions only",
            train_path.display(),
            args.corpus_dir.display()
        );
        let corpus_exprs = load_corpus_exprs(&train_path, 100_000, args.seed + 1);
        eprintln!(
            "\nLoading TRAIN corpus from {} ({} expressions)",
            train_path.display(),
            corpus_exprs.len()
        );
        let mut train_revalidation_drops = 0usize;
        for (name, arena, root) in corpus_exprs {
            if corpus_revalidation.verdict(&name, &arena, root).0.is_some() {
                train_revalidation_drops += 1;
                continue;
            }
            queue.push(WorkItem {
                source: Source::TrainCorpus,
                name,
                arena: Some(arena),
                root,
            });
        }
        if train_revalidation_drops > 0 {
            eprintln!(
                "  TRAIN revalidation dropped {train_revalidation_drops} stored entries this \
                 build cannot verify (recorded in {revalidation_path})"
            );
        }
    }

    // ── PHASE 1c: Add the DEV tier ───────────────────────────────────
    //
    // DEV is *measured* here, alongside everything else, and only *predicted*
    // after training. Its ground truth does not depend on the model, so there
    // is no reason to measure it an hour later on a different part of the
    // machine's drift curve — which is exactly what made held-out MAE
    // incomparable with training loss.

    let dev_path = args
        .corpus_dir
        .join(format!("corpus_{}.bin", Tier::Dev.name()));
    assert!(
        dev_path.exists(),
        "DEV corpus not found at {} — run gen_bench_corpus (--output {}) to build the tiered \
         corpus. Without the DEV tier there is no held-out validation signal (audit L4); \
         refusing to report training metrics as if they were one",
        dev_path.display(),
        args.corpus_dir.display()
    );
    let dev_exprs = load_corpus_exprs(&dev_path, args.dev_eval_max, args.seed + 2);
    eprintln!(
        "Loading DEV corpus from {} ({} expressions)",
        dev_path.display(),
        dev_exprs.len()
    );
    let mut dev_revalidation_drops = 0usize;
    for (name, arena, root) in dev_exprs {
        // Same revalidation gate as TRAIN (Codex 3759014678): a stored DEV
        // entry this build's JIT and oracle disagree about would be measured
        // as held-out TRUTH — worse than a bad training label, because every
        // reported DEV metric leans on it.
        if corpus_revalidation.verdict(&name, &arena, root).0.is_some() {
            dev_revalidation_drops += 1;
            continue;
        }
        queue.push(WorkItem {
            source: Source::Dev,
            name,
            arena: Some(arena),
            root,
        });
    }
    if dev_revalidation_drops > 0 {
        eprintln!(
            "  DEV revalidation dropped {dev_revalidation_drops} stored entries this build \
             cannot verify (recorded in {revalidation_path})"
        );
    }

    // The corpus revalidation tally and its rate alarms: past the thresholds
    // the corpus does not describe this build and must be regenerated, not
    // subsetted (Codex 3759014678).
    eprintln!(
        "\nCorpus revalidation (TRAIN+DEV entries re-checked against this build's JIT and \
         oracle):"
    );
    corpus_revalidation.finish();

    // ── PHASE 1d: Benchmark the pooled queue in seeded-shuffled order ─

    let order_seed = args
        .order_seed
        .unwrap_or_else(|| args.seed.rotate_left(32) ^ 0x00C0_FFEE);
    eprintln!(
        "\n=== Phase 1d: benchmarking {} pooled expressions in shuffled order ===\n\
         Benchmark order seed: {order_seed} (stored corpus order is unchanged). Shuffling is \
         what makes the sentinel correction sufficient: it removes slow drift, and this removes \
         the correlation between whatever drift remains and expression structure.",
        queue.len()
    );
    let order = shuffled_order(queue.len(), order_seed);

    let mut samples: Vec<ValueSample> = Vec::new();
    // (queue index, measured log-ns) for DEV; predictions wait for training.
    let mut dev_measurements: Vec<(usize, f32)> = Vec::new();
    let mut normalization_factors: Vec<f64> = Vec::new();
    let mut attempted = [0usize; 3];
    let mut excluded = [0usize; 3];

    for (position, &idx) in order.iter().enumerate() {
        let source = queue[idx].source;
        let root = queue[idx].root;
        attempted[source as usize] += 1;

        // Zero (or overhead-dominated ~0) is a VALID measurement for
        // constant-folded expressions (audit M2 — `validate_median` documents
        // it); log_ns floors it at 1e-3ns so the model sees the fast end of
        // the distribution instead of having it censored away.
        match session.benchmark_arena(queue[idx].arena(), root, MINT_MODE) {
            Ok(bench) => {
                let label = mint_label(&bench, source.phase());
                match source {
                    Source::Dev => dev_measurements.push((idx, label.target_log_ns)),
                    Source::Synthetic | Source::TrainCorpus => {
                        // Training labels only (Codex 3759014682): the sidecar
                        // summarizes the drift correction behind the TARGETS
                        // the weights were fit on. DEV is measured on a
                        // different part of the drift curve, so folding its
                        // factors in made the sidecar describe a population
                        // its own `samples` count does not.
                        normalization_factors.push(label.normalization);
                        // DEPLOYMENT representation (from_arena_dag): register-
                        // reload edges for shared nodes + populated variance
                        // histogram — the same accumulator semantics
                        // `IncrementalExtractor` builds from DAG choices, so
                        // the trained function is the deployed function
                        // (round-0 train/deploy skew fix).
                        let acc = EdgeAccumulator::from_arena_dag(
                            queue[idx].arena(),
                            root,
                            &model.embeddings,
                        );
                        samples.push(ValueSample {
                            acc,
                            target_log_ns: label.target_log_ns,
                            normalization: label.normalization as f32,
                        });
                    }
                }
            }
            Err(e) => {
                let expression = arena_to_kernel_code(queue[idx].arena(), root);
                exclusions.record(source.phase(), &queue[idx].name, &expression, &e);
                excluded[source as usize] += 1;
            }
        }

        // Only DEV arenas are needed after this loop (their predictions use
        // the trained embeddings). Releasing the rest keeps a 50k-expression
        // run from carrying hundreds of megabytes of dead arenas through
        // training.
        if source != Source::Dev {
            queue[idx].arena = None;
        }

        if args.progress_every > 0 && (position + 1) % args.progress_every == 0 {
            eprintln!(
                "  benchmarked [{}/{}] labels={} dev={} excluded={}",
                position + 1,
                order.len(),
                samples.len(),
                dev_measurements.len(),
                exclusions.count
            );
        }
    }

    for source in SOURCES {
        eprintln!(
            "  {}: {}/{} benchmarked",
            source.phase(),
            attempted[source as usize] - excluded[source as usize],
            attempted[source as usize]
        );
    }

    // Fence and quarantine drops are not exclusions in the audit-M2 sense:
    // nothing failed to measure, so they must inflate neither the censoring
    // rate's numerator nor its denominator.
    assert_exclusion_rate(
        excluded[Source::Synthetic as usize] + excluded[Source::TrainCorpus as usize],
        attempted[Source::Synthetic as usize] + attempted[Source::TrainCorpus as usize],
        "Label minting (synthetic + train corpus)",
        &args.exclusions,
    );
    assert_exclusion_rate(
        excluded[Source::Dev as usize],
        attempted[Source::Dev as usize],
        "DEV evaluation",
        &args.exclusions,
    );

    eprintln!("\nTotal training samples: {}", samples.len());
    assert!(
        !samples.is_empty(),
        "No samples — cannot train. Check JIT and BwdGenerator."
    );

    // Print target distribution
    let mut targets: Vec<f32> = samples.iter().map(|s| s.target_log_ns).collect();
    targets.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("NaN target log-ns: {a} vs {b}"))
    });
    eprintln!(
        "Target log-ns distribution: min={:.2} p25={:.2} median={:.2} p75={:.2} max={:.2}",
        targets[0],
        targets[targets.len() / 4],
        targets[targets.len() / 2],
        targets[3 * targets.len() / 4],
        targets[targets.len() - 1],
    );

    // What the drift correction actually did to those targets. A run whose
    // factors are all ~1.0 saw a steady machine; a wide spread means the
    // labels lean on the correction, and the per-sample factor is retained so
    // training can down-weight the ones that lean hardest.
    let training_factors: Vec<f64> = samples.iter().map(|s| f64::from(s.normalization)).collect();
    let training_norm = NormalizationStats::summarize(&training_factors);
    eprintln!(
        "Sentinel normalization applied to training targets: min={:.4} median={:.4} max={:.4} \
         (n={})",
        training_norm.min, training_norm.median, training_norm.max, training_norm.count
    );

    // ── PHASE 2: Train value head ────────────────────────────────────

    eprintln!(
        "\n=== Phase 2: Training value head ({} epochs, batch={}) ===",
        args.epochs, args.batch_size
    );

    let mut momentum_buf = UnifiedGradients::zero();

    // Simple LCG for shuffling
    let mut rng_state = args.seed;
    let mut rng_next = || -> u64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng_state >> 33
    };

    // We need a dummy rule embedding for forward_cached — value head doesn't use it
    let dummy_rule_embed = [0.0f32; EMBED_DIM];
    let dummy_gacc = GraphAccumulator::new();

    for epoch in 0..args.epochs {
        // Shuffle indices
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng_next() as usize % (i + 1);
            indices.swap(i, j);
        }

        let mut epoch_loss = 0.0f64;
        let mut epoch_count = 0usize;

        for chunk in indices.chunks(args.batch_size) {
            let mut grads = UnifiedGradients::zero();

            for &idx in chunk {
                let sample = &samples[idx];
                let cache = forward_cached(&model, &sample.acc, &dummy_gacc, &dummy_rule_embed);

                backward_value(
                    &model,
                    &cache,
                    sample.target_log_ns,
                    args.value_coeff,
                    &mut grads,
                );

                let err = cache.value_pred - sample.target_log_ns;
                epoch_loss += (err * err) as f64;
                epoch_count += 1;
            }

            let batch_size = chunk.len().max(1) as f32;
            grads.scale(1.0 / batch_size);

            apply_unified_sgd(
                &mut model,
                &grads,
                &mut momentum_buf,
                pixelflow_pipeline::training::unified_backward::SgdConfig {
                    lr: args.lr,
                    momentum: args.momentum,
                    weight_decay: args.weight_decay,
                    grad_clip: args.grad_clip,
                },
            );
        }

        let mse = epoch_loss / epoch_count.max(1) as f64;
        let rmse = mse.sqrt();

        if epoch == 0 || epoch == args.epochs - 1 || (epoch + 1) % 10 == 0 {
            // TRAIN metrics only. The old "tail MAE" over the last 1000
            // training samples was removed (audit L4): it was training data
            // presented as validation. Held-out metrics come from the DEV
            // tier in Phase 3.
            eprintln!(
                "Epoch {:>3}/{}: train_MSE={:.6} train_RMSE={:.4}",
                epoch + 1,
                args.epochs,
                mse,
                rmse,
            );
        }
    }

    // ── PHASE 3: DEV held-out evaluation (audit L4, plan 0.3b) ───────
    //
    // Before saving, not after (bug B7). The DEV measurements were taken in
    // Phase 1d on the same clock as the training targets; what happens here is
    // only the prediction side, which needs the trained embeddings.

    eprintln!(
        "\n=== Phase 3: DEV held-out evaluation ({} measured expressions from {}) ===",
        dev_measurements.len(),
        dev_path.display()
    );

    let mut predicted: Vec<f32> = Vec::with_capacity(dev_measurements.len());
    let mut measured: Vec<f32> = Vec::with_capacity(dev_measurements.len());
    for &(idx, measured_log_ns) in &dev_measurements {
        let item = &queue[idx];
        // The DEPLOYED prediction path, end to end: the same accumulator
        // semantics (from_arena_dag: reload edges + variance histogram) and
        // the same forward entry point (`predict_log_cost_with_features`)
        // that `IncrementalExtractor` calls. These metrics therefore describe
        // the function extraction actually uses, not a trainer-only shadow.
        let acc = EdgeAccumulator::from_arena_dag(item.arena(), item.root, &model.embeddings);
        predicted.push(model.predict_log_cost_with_features(&acc));
        measured.push(measured_log_ns);
    }

    assert!(
        measured.len() >= 2,
        "DEV evaluation produced {} usable samples — too few for held-out metrics",
        measured.len()
    );
    let dev_mae: f64 = predicted
        .iter()
        .zip(&measured)
        .map(|(p, m)| f64::from((p - m).abs()))
        .sum::<f64>()
        / measured.len() as f64;

    // Prediction bias and range compression (round-0 skew evidence): the
    // skewed head was biased -0.291 log-ns with its prediction spread
    // compressed to 45% of the measured spread. Bias ~0 and ratios ~1 are the
    // direct evidence the train/deploy skew is gone.
    let n = measured.len() as f64;
    let dev_bias: f64 = predicted
        .iter()
        .zip(&measured)
        .map(|(p, m)| f64::from(p - m))
        .sum::<f64>()
        / n;
    let mean_of = |xs: &[f32]| xs.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let std_of = |xs: &[f32]| {
        let mean = mean_of(xs);
        (xs.iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    };
    let p90p10 = |xs: &[f32]| {
        let mut s: Vec<f32> = xs.to_vec();
        s.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("NaN in DEV predictions/measurements"))
        });
        f64::from(s[(s.len() - 1) * 9 / 10] - s[(s.len() - 1) / 10])
    };
    let (pred_std, meas_std) = (std_of(&predicted), std_of(&measured));
    let (pred_spread, meas_spread) = (p90p10(&predicted), p90p10(&measured));

    eprintln!("\n=== DEV held-out metrics (families never trained on; audit L4 / plan 0.3b) ===");
    eprintln!("  DEV samples: {}", measured.len());
    eprintln!("  DEV MAE (log-ns): {dev_mae:.4}");
    let dev_spearman = spearman_rho(&predicted, &measured);
    match dev_spearman {
        Some(rho) => eprintln!("  DEV Spearman rho (predicted vs measured log-ns): {rho:.4}"),
        None => eprintln!(
            "  DEV Spearman rho: UNDEFINED — zero rank variance (all predictions or all \
             measurements tied); the model or the corpus tier is degenerate"
        ),
    }
    eprintln!("  DEV prediction bias (mean pred-measured, log-ns): {dev_bias:+.4}");
    if meas_std > 0.0 && meas_spread > 0.0 {
        eprintln!(
            "  DEV prediction range vs true: std ratio {:.4} ({:.4}/{:.4}), p90-p10 ratio {:.4} \
             ({:.4}/{:.4})",
            pred_std / meas_std,
            pred_std,
            meas_std,
            pred_spread / meas_spread,
            pred_spread,
            meas_spread
        );
    } else {
        eprintln!(
            "  DEV prediction range vs true: UNDEFINED — measured spread is zero (std {meas_std}, \
             p90-p10 {meas_spread}); the DEV tier is degenerate"
        );
    }

    // Quick eval: predict a few TRAIN samples (in-sample sanity check only).
    eprintln!("\n=== Sample predictions (TRAIN samples, in-sample) ===");
    for i in [
        0,
        samples.len() / 4,
        samples.len() / 2,
        3 * samples.len() / 4,
        samples.len() - 1,
    ] {
        let sample = &samples[i];
        let cache = forward_cached(&model, &sample.acc, &dummy_gacc, &dummy_rule_embed);
        let pred_ns = libm::expf(cache.value_pred) as f64;
        let actual_ns = libm::expf(sample.target_log_ns) as f64;
        eprintln!(
            "  sample {i}: pred={:.1}ns actual={:.1}ns (log pred={:.3} actual={:.3})",
            pred_ns, actual_ns, cache.value_pred, sample.target_log_ns,
        );
    }

    // ── PHASE 4: Save weights + mint metadata ────────────────────────
    //
    // Last, deliberately (bug B7). Writing weights before the held-out
    // evaluation means an aborted run leaves a checkpoint on disk, and the
    // next run — believed fresh — warm-starts from unvalidated weights while
    // announcing itself with one easily-skimmed "Loaded model from" line.
    // Losing a run's training to a Phase 3 panic is the cheaper failure: it is
    // recoverable by rerunning, whereas a laundered checkpoint silently
    // contaminates every experiment after it.

    model
        .save(&args.output)
        .unwrap_or_else(|e| panic!("Failed to save model to {}: {e}", args.output.display()));
    eprintln!(
        "\nSaved bootstrapped extraction head to {}",
        args.output.display()
    );

    // The objective these weights were fit against, recorded so a consumer can
    // assert it rather than assume it (R6): the gate measures one BenchMode
    // and the trainer minted another would otherwise be an invisible mismatch.
    // A failed sidecar write fails the run — weights whose objective is
    // unrecorded cannot be gate-checked, so they are worse than no weights.
    //
    // The sidecar records the hash of the bytes that actually landed on disk
    // (Codex 3758853787): weights and sidecar are two files with a crash
    // window between the writes, and a stale sidecar beside replaced weights
    // used to pass every mode check. Consumers assert `require_weights`.
    let saved_bytes = std::fs::read(&args.output).unwrap_or_else(|e| {
        panic!(
            "failed to read back the just-saved weights at {}: {e} — cannot record their \
             identity in the mint sidecar",
            args.output.display()
        )
    });
    let metadata = MintMetadata {
        bench_mode: bench_mode_slug(MINT_MODE).to_string(),
        trainer: "bootstrap_extraction_head".to_string(),
        samples: samples.len(),
        order_shuffle_seed: order_seed,
        sentinel_calibration_ns: session.calibration_ns(),
        // Training labels only (Codex 3759014682): DEV factors describe the
        // evaluation population, not the targets these weights were fit on.
        normalization: NormalizationStats::summarize(&normalization_factors),
        weights_fnv64: weights_identity(&saved_bytes),
        written_at_unix_s: unix_now_s(),
    };
    let metadata_path = metadata.write_for(&args.output).unwrap_or_else(|e| {
        panic!(
            "failed to write the mint-metadata sidecar for {}: {e}. The weights on disk now have \
             no recorded objective, so no gate can verify it measures what they were trained on \
             — delete them and rerun",
            args.output.display()
        )
    });
    eprintln!(
        "Recorded mint metadata in {} (bench_mode={}, order_shuffle_seed={order_seed}, \
         samples={})",
        metadata_path.display(),
        metadata.bench_mode,
        metadata.samples
    );

    // DEV model quality into the research journal (plan 0.3b): the 3-way
    // bench was the only journal writer, so intrinsic quality never had a
    // cross-round history. Written last — after the weights and sidecar — so
    // the line always names a checkpoint that exists.
    append_journal(&TrainerJournalRecord {
        record: "bootstrap_extraction_head",
        ts_unix: unix_now_s(),
        weights_path: args.output.display().to_string(),
        corpus_dir: args.corpus_dir.display().to_string(),
        bench_mode: bench_mode_slug(MINT_MODE),
        train_samples: samples.len(),
        synthetic_requested: args.synthetic,
        epochs: args.epochs,
        lr: args.lr,
        seed: args.seed,
        order_seed,
        dev_samples: measured.len(),
        dev_mae_log_ns: dev_mae,
        dev_spearman_rho: dev_spearman,
        dev_pred_bias_log_ns: dev_bias,
        dev_pred_std_ratio: if meas_std > 0.0 {
            pred_std / meas_std
        } else {
            f64::NAN
        },
        dev_pred_p90p10_ratio: if meas_spread > 0.0 {
            pred_spread / meas_spread
        } else {
            f64::NAN
        },
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    use pixelflow_ir::OpKind;
    use pixelflow_pipeline::training::corpus::write_corpus;

    // ── Holdout fence ───────────────────────────────────────────────────────

    /// A unique scratch directory for one test (no temp-dir dependency).
    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before the unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("bootstrap_fence_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", dir.display()));
        dir
    }

    /// `x * k` — the motif a live generator can plausibly redraw.
    fn scaled_var(k: f32) -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let c = arena.push_const(k);
        let root = arena.push_binary(OpKind::Mul, x, c);
        (arena, root)
    }

    fn write_tier(dir: &Path, tier: Tier, exprs: &[(ExprArena, ExprId)]) {
        let entries: Vec<(String, ExprArena, ExprId)> = exprs
            .iter()
            .enumerate()
            .map(|(i, (a, r))| (format!("{}_{i}", tier.name()), a.clone(), *r))
            .collect();
        write_corpus(&tier_corpus_path(dir, tier), &entries)
            .unwrap_or_else(|e| panic!("failed to write {} corpus: {e}", tier.name()));
    }

    #[test]
    fn fence_blocks_structures_owned_by_dev_or_final() {
        let dir = scratch_dir("blocks");
        write_tier(&dir, Tier::Dev, &[scaled_var(2.0)]);
        write_tier(&dir, Tier::Final, &[scaled_var(3.0)]);

        let fence = HoldoutFence::build(&dir);
        assert_eq!(fence.dev_entries, 1);
        assert_eq!(fence.final_entries, 1);
        assert_eq!(fence.keys.len(), 2);

        // A freshly built arena with the SAME structure as a DEV expression
        // is the leak: different arena, different ids, same computation.
        let (dev_shape, dev_root) = scaled_var(2.0);
        assert!(fence.blocks(&dev_shape, dev_root));
        // FINAL is fenced identically.
        let (final_shape, final_root) = scaled_var(3.0);
        assert!(fence.blocks(&final_shape, final_root));
        // A different constant is a different structure (Const compares by
        // bits) and must survive.
        let (other, other_root) = scaled_var(4.0);
        assert!(!fence.blocks(&other, other_root));
        // So must a different operator over the same leaves.
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let c = arena.push_const(2.0);
        let add = arena.push_binary(OpKind::Add, x, c);
        assert!(!fence.blocks(&arena, add));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "holdout fence cannot be built")]
    fn fence_refuses_to_build_without_final() {
        let dir = scratch_dir("no_final");
        write_tier(&dir, Tier::Dev, &[scaled_var(2.0)]);
        // FINAL missing: gen_bench_corpus writes all three tiers together, so
        // this directory is a partial artifact and the fence would be blind
        // to the most sacred tier. Refuse rather than train unfenced.
        let _ = HoldoutFence::build(&dir);
    }

    #[test]
    #[should_panic(expected = "holdout fence cannot be built")]
    fn fence_refuses_to_build_without_dev() {
        let dir = scratch_dir("no_dev");
        write_tier(&dir, Tier::Final, &[scaled_var(3.0)]);
        let _ = HoldoutFence::build(&dir);
    }

    #[test]
    fn holdout_drops_are_logged_and_counted_apart_from_exclusions() {
        let dir = scratch_dir("log");
        let sidecar = dir.join("exclusions.jsonl");
        let mut log = ExclusionLog::create(&sidecar);
        log.record_holdout_drop("synthetic_7", "X * 2.0");
        log.record_holdout_drop("synthetic_9", "X * 3.0");

        // A fence drop is not a measurement exclusion: the censoring alarm's
        // counter must be untouched, or a working fence would look like a
        // broken JIT.
        assert_eq!(log.holdout_dropped, 2);
        assert_eq!(log.count, 0);

        let text = std::fs::read_to_string(&sidecar).expect("sidecar readable");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).expect("valid JSON record");
            assert_eq!(v["reason"], "holdout_structural_collision");
            assert_eq!(v["phase"], "synthetic");
            assert!(v["expression"].as_str().expect("expression").contains('X'));
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn average_ranks_no_ties() {
        assert_eq!(average_ranks(&[10.0, 30.0, 20.0]), vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn average_ranks_with_ties() {
        // 2.0 occupies ranks 2 and 3 → both get 2.5.
        assert_eq!(
            average_ranks(&[1.0, 2.0, 2.0, 4.0]),
            vec![1.0, 2.5, 2.5, 4.0]
        );
    }

    #[test]
    fn spearman_perfect_monotone() {
        // Monotone but non-linear: rank correlation is exactly 1.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 4.0, 9.0, 16.0, 25.0];
        let rho = spearman_rho(&a, &b).expect("defined");
        assert!((rho - 1.0).abs() < 1e-12, "got {rho}");

        let rev: Vec<f32> = b.iter().rev().copied().collect();
        let rho = spearman_rho(&a, &rev).expect("defined");
        assert!((rho + 1.0).abs() < 1e-12, "got {rho}");
    }

    #[test]
    fn spearman_known_small_example() {
        // Ranks: a=[1,2,3], b=[3,1,2]; d²=[4,1,1] → ρ = 1 − 6·6/(3·8) = −0.5.
        let rho = spearman_rho(&[1.0, 2.0, 3.0], &[3.0, 1.0, 2.0]).expect("defined");
        assert!((rho + 0.5).abs() < 1e-12, "got {rho}");
    }

    #[test]
    fn spearman_with_ties_uses_average_ranks() {
        // a ranks: [1, 2.5, 2.5, 4]; b ranks: [1, 2, 3, 4].
        // Pearson on those ranks: 4.5 / sqrt(4.5 * 5) = 0.9486832...
        let rho = spearman_rho(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0, 3.0, 4.0]).expect("defined");
        let expected = 4.5 / (4.5f64 * 5.0).sqrt();
        assert!((rho - expected).abs() < 1e-12, "got {rho}, want {expected}");
    }

    #[test]
    fn spearman_zero_variance_is_undefined() {
        assert!(spearman_rho(&[7.0, 7.0, 7.0], &[1.0, 2.0, 3.0]).is_none());
        assert!(spearman_rho(&[1.0, 2.0, 3.0], &[7.0, 7.0, 7.0]).is_none());
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn spearman_length_mismatch_panics() {
        let _ = spearman_rho(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "non-comparable (NaN)")]
    fn average_ranks_nan_panics() {
        let _ = average_ranks(&[1.0, f32::NAN, 3.0]);
    }

    // ── Benchmark order (item 6b) ───────────────────────────────────────────

    #[test]
    fn shuffled_order_is_a_permutation() {
        for len in [0usize, 1, 2, 7, 64, 1000] {
            let order = shuffled_order(len, 12345);
            assert_eq!(order.len(), len);
            let mut seen = order.clone();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..len).collect::<Vec<_>>(),
                "every index must appear exactly once (len {len}) — a shuffle that drops or \
                 duplicates work would silently change what got benchmarked"
            );
        }
    }

    #[test]
    fn shuffled_order_actually_reorders_and_follows_its_seed() {
        let identity: Vec<usize> = (0..256).collect();
        let a = shuffled_order(256, 1);
        let b = shuffled_order(256, 2);
        assert_ne!(
            a, identity,
            "a shuffle that returns generation order is a no-op"
        );
        assert_ne!(
            a, b,
            "the order must depend on the seed to be reproducible by it"
        );
        assert_eq!(a, shuffled_order(256, 1), "same seed, same order");
    }

    // ── Sentinel normalization at the point of consumption (item 6a) ────────

    /// A `BenchResult` for a kernel whose true cost is `true_ns`, measured
    /// under a machine running `drift`x slower than the session's opening
    /// calibration.
    ///
    /// The overhead is deliberately nonzero (~the real identity-call figure):
    /// the whole clock inflates, so the raw reading covers kernel *and*
    /// overhead, while `call_overhead_ns` stays denominated in the opening
    /// clock. With a zero overhead the two possible orderings of
    /// normalize-and-subtract coincide, and these tests would pass under
    /// either — including the one that biases small kernels.
    fn drifted(true_ns: f64, drift: f64) -> BenchResult {
        const CALIBRATION_NS: f64 = 8.0;
        const OVERHEAD_NS: f64 = 4.0;
        BenchResult {
            ns: (true_ns + OVERHEAD_NS) * drift,
            adjusted_ns: (true_ns + OVERHEAD_NS) * drift - OVERHEAD_NS,
            call_overhead_ns: OVERHEAD_NS,
            iqr_ns: 0.0,
            mode: MINT_MODE,
            repeat_batches: 1,
            output: [0.0; 4],
            outputs: Vec::new(),
            sentinel: Some(pixelflow_pipeline::jit_bench::SentinelContext {
                calibration_ns: CALIBRATION_NS,
                local_sentinel_ns: CALIBRATION_NS * drift,
            }),
        }
    }

    /// Mint labels for `truth`, benchmarked in `order`, under a drift profile
    /// that depends only on *collection position*.
    fn mint_in_order(truth: &[f64], order: &[usize], drift: &[f64]) -> Vec<(usize, f32)> {
        order
            .iter()
            .enumerate()
            .map(|(position, &expr)| {
                let bench = drifted(truth[expr], drift[position]);
                (expr, mint_label(&bench, "test").target_log_ns)
            })
            .collect()
    }

    #[test]
    fn minted_targets_are_invariant_to_benchmark_order_under_drift() {
        // This is the whole point of item 6: the sentinel correction has to be
        // APPLIED at the point the target is constructed, not merely recorded
        // on the BenchResult. Under a monotone slowdown, an expression
        // measured last must mint the same target as one measured first.
        let truth = [12.0f64, 47.0, 310.0, 900.0];
        let drift = [1.0f64, 1.2, 1.55, 2.0];

        let forward = mint_in_order(&truth, &[0, 1, 2, 3], &drift);
        let reversed = mint_in_order(&truth, &[3, 2, 1, 0], &drift);
        let shuffled = mint_in_order(&truth, &shuffled_order(4, 99), &drift);

        for other in [&reversed, &shuffled] {
            for &(expr, target) in other {
                let (_, base) = forward
                    .iter()
                    .find(|(e, _)| *e == expr)
                    .expect("the same expressions in every ordering");
                assert!(
                    (target - base).abs() < 1e-6,
                    "expression {expr} minted {target} in one benchmark order and {base} in \
                     another — drift is still correlated with collection order"
                );
                assert!(
                    (target - log_ns(truth[expr])).abs() < 1e-6,
                    "the normalized target must be the undrifted truth"
                );
            }
        }
    }

    #[test]
    fn unnormalized_targets_would_have_been_order_dependent() {
        // The control the test above needs to mean anything: without the
        // correction the same expression at two collection positions produces
        // two different targets.
        let early = drifted(100.0, 1.0);
        let late = drifted(100.0, 2.0);
        assert!(
            (log_ns(early.adjusted_ns) - log_ns(late.adjusted_ns)).abs() > 0.5,
            "the drift profile must actually move an uncorrected target"
        );
        assert!(
            (mint_label(&early, "t").target_log_ns - mint_label(&late, "t").target_log_ns).abs()
                < 1e-6
        );
    }

    #[test]
    fn mint_label_reports_the_factor_it_applied() {
        let label = mint_label(&drifted(100.0, 1.25), "t");
        assert!(
            (label.normalization - 0.8).abs() < 1e-9,
            "a 1.25x slowdown is corrected by 0.8, got {}",
            label.normalization
        );
    }

    #[test]
    #[should_panic(expected = "no sentinel context")]
    fn mint_label_refuses_an_unprotected_measurement() {
        // A sessionless measurement carries no drift information, so its
        // correction cannot be skipped quietly — the label would silently join
        // a normalized set on a different clock.
        let mut bench = drifted(100.0, 1.0);
        bench.sentinel = None;
        let _ = mint_label(&bench, "unit test");
    }

    // ── Population sizes (item 5) ───────────────────────────────────────────

    #[test]
    fn size_distribution_of_nothing_is_nothing() {
        assert_eq!(SizeDistribution::of(&[]), None);
    }

    #[test]
    fn size_distribution_quartiles() {
        let d = SizeDistribution::of(&[7, 1, 3, 9, 5]).expect("non-empty");
        assert_eq!(d.count, 5);
        assert_eq!(d.min, 1);
        assert_eq!(d.median, 5);
        assert_eq!(d.max, 9);
        assert!(d.render().contains("median=5"));
    }

    #[test]
    fn disjointness_note_fires_when_the_live_stream_is_all_larger() {
        // Exactly the smoke run's shape: live median 147, fence capped at 40.
        let live = SizeDistribution::of(&[120, 147, 300]).expect("non-empty");
        let fenced = SizeDistribution::of(&[8, 20, 40]).expect("non-empty");
        let note = disjointness_note(&live, &fenced).expect("disjoint populations must be named");
        assert!(note.contains("DISJOINT"), "{note}");
        assert!(
            note.contains("arithmetic, not evidence"),
            "the note must say what the zero means: {note}"
        );
    }

    #[test]
    fn disjointness_note_fires_when_the_live_stream_is_all_smaller() {
        let live = SizeDistribution::of(&[3, 4, 5]).expect("non-empty");
        let fenced = SizeDistribution::of(&[80, 400]).expect("non-empty");
        assert!(disjointness_note(&live, &fenced).is_some());
    }

    #[test]
    fn overlapping_populations_get_no_disjointness_note() {
        // Here a zero drop count IS evidence, so the report must not cry wolf.
        let live = SizeDistribution::of(&[5, 30, 147]).expect("non-empty");
        let fenced = SizeDistribution::of(&[8, 20, 40]).expect("non-empty");
        assert_eq!(disjointness_note(&live, &fenced), None);
    }

    #[test]
    fn touching_ranges_are_not_disjoint() {
        // live.min == fenced.max: a collision is possible at that size.
        let live = SizeDistribution::of(&[40, 90]).expect("non-empty");
        let fenced = SizeDistribution::of(&[8, 40]).expect("non-empty");
        assert_eq!(disjointness_note(&live, &fenced), None);
    }

    // ── --synthetic 0 (item 7) ──────────────────────────────────────────────

    #[test]
    fn corpus_only_run_is_allowed_with_no_live_expressions() {
        // `--synthetic 0` is a legitimate TRAIN-corpus-only ablation, and so
        // is a fence that happens to drop every live expression.
        assert_label_sources(0, true, 0);
        assert_label_sources(0, true, 100);
    }

    #[test]
    fn a_live_stream_alone_is_a_label_source() {
        assert_label_sources(5, false, 5);
    }

    #[test]
    #[should_panic(expected = "no label source")]
    fn no_synthetic_and_no_corpus_is_refused() {
        assert_label_sources(0, false, 0);
    }
}
