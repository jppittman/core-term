//! Label minting: turning a [`BenchResult`] into a training target, and
//! recording the conditions the targets were minted under.
//!
//! Two things live here because they are the same decision seen from two
//! sides — *what a label means*:
//!
//! 1. [`normalized_label_ns`] applies the drift correction the sentinel
//!    measured — to the raw reading, before the opening-clock call overhead
//!    comes off, because those two numbers are denominated in different
//!    clocks (see that function). `BenchSession` records `calibration_ns /
//!    local_sentinel_ns` on every [`BenchResult`], but recording a correction
//!    is not applying one: feeding `adjusted_ns` to `log_ns` leaves timing drift
//!    correlated with *collection order*, and collection order correlates
//!    with expression structure (generation emits size/family bands
//!    sequentially). Drift then stops being noise and becomes learnable
//!    spurious signal. Every label-minting path must go through this
//!    function, and a [`BenchResult`] with no sentinel (the sessionless
//!    wrappers) is a hard error there — an unprotected measurement silently
//!    accepted as a label is exactly the failure the sentinel exists to make
//!    impossible.
//!
//! 2. [`MintMetadata`] is the sidecar written next to a weights file naming
//!    the [`BenchMode`] the targets were minted with, plus the benchmark
//!    order seed and the drift the run actually saw. The extraction head is
//!    trained against `BenchMode::Latency` targets; a gate that measures
//!    `BenchMode::Throughput` is scoring the model on an objective it was
//!    never trained to predict, and those two modes rank expression shapes
//!    differently. The sidecar exists so the gate can *assert* the match
//!    instead of assuming it.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::jit_bench::{BenchMode, BenchResult};

// ── Sentinel normalization ──────────────────────────────────────────────────

/// The drift-corrected nanosecond value of a measurement, in the run's
/// *opening* clock.
///
/// # Order of operations
///
/// Normalize the **raw** reading, *then* subtract the call overhead:
///
/// ```text
/// label = ns * (calibration_ns / local_sentinel_ns) - call_overhead_ns
/// ```
///
/// The two operands live on different clocks and only one order puts them on
/// the same one. `call_overhead_ns` is the identity-kernel cost measured once
/// at session open, so it is denominated in the *opening* clock;
/// `BenchResult::ns` is denominated in the clock that was running when the
/// kernel was timed. The sentinel factor converts the second into the first,
/// which makes it a conversion on `ns` alone — `adjusted_ns` is a mixture of
/// the two clocks and scaling it converts neither correctly.
///
/// Scaling `adjusted_ns` instead (the original form) leaves a residue. With a
/// slowdown factor `d`, the raw reading is `d * (kernel + overhead)` and the
/// factor is `1/d`, so
///
/// ```text
/// (d * (kernel + overhead) - overhead) / d  =  kernel + overhead * (1 - 1/d)
/// ```
///
/// — a term that depends on `d`, i.e. on *when in the run* the expression was
/// benchmarked. That is precisely the order-dependent contamination the
/// sentinel exists to remove, and at the accepted sub-50% drift band with a
/// ~4ns call overhead it is >1ns, which is a large fraction of the smallest
/// kernels — the bulk of the corpus.
///
/// The alternative, remeasuring the identity kernel next to every label and
/// normalizing that too, is rejected: it buys the same number at the price of
/// a benchmark per label plus that benchmark's own noise, when the sentinel
/// factor already carries exactly the clock-speed information needed to move
/// the one operand that is on the wrong clock.
///
/// The sign of the result is not clamped: a value `<= 0` is a documented
/// measurement condition for kernels cheaper than the identity call overhead.
///
/// # Panics
///
/// Panics when `bench.sentinel` is `None`. That means the measurement came
/// from a sessionless wrapper which runs no sentinel at all, so there is no
/// drift information to correct with and no drift protection behind the
/// number. Minting a label from it would silently mix an unprotected
/// measurement into a normalized set — the correction would be skipped, not
/// unnecessary. `context` names the call site in the panic.
#[must_use]
pub fn normalized_label_ns(bench: &BenchResult, context: &str) -> f64 {
    let sentinel = bench.sentinel.unwrap_or_else(|| {
        panic!(
            "{context}: label minted from a BenchResult with no sentinel context — the \
             measurement came from a sessionless wrapper (no QoS pin, no drift sentinel, no \
             overhead subtraction), so its drift is neither measured nor correctable. Mint \
             labels through a BenchSession"
        )
    });
    bench.ns * sentinel.normalization() - bench.call_overhead_ns
}

/// The sentinel normalization factor a measurement carries, for persisting
/// alongside the label so training can down-weight heavily-corrected samples.
///
/// # Panics
///
/// Same contract as [`normalized_label_ns`]: a missing sentinel is a hard
/// error in label-minting paths.
#[must_use]
pub fn label_normalization(bench: &BenchResult, context: &str) -> f64 {
    let sentinel = bench.sentinel.unwrap_or_else(|| {
        panic!(
            "{context}: label normalization requested for a BenchResult with no sentinel \
             context — mint labels through a BenchSession"
        )
    });
    sentinel.normalization()
}

/// Summary statistics of the per-label normalization factors a run applied.
///
/// A run whose factors are all ~1.0 saw no drift; a run whose factors span a
/// wide range drifted a lot and its labels lean hard on the correction. Both
/// are reportable facts, so both go in the weights sidecar.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct NormalizationStats {
    pub count: usize,
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

impl NormalizationStats {
    /// Summarize the factors applied across a run.
    ///
    /// # Panics
    ///
    /// Panics on an empty slice (a run that minted no labels has no stats to
    /// report and must not fabricate any) or on a non-finite factor.
    #[must_use]
    pub fn summarize(factors: &[f64]) -> Self {
        assert!(
            !factors.is_empty(),
            "NormalizationStats::summarize: no normalization factors — a run that minted zero \
             labels has no drift summary"
        );
        let mut sorted = factors.to_vec();
        sorted.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("non-finite normalization factor: {a} vs {b}"))
        });
        Self {
            count: sorted.len(),
            min: sorted[0],
            median: sorted[sorted.len() / 2],
            max: sorted[sorted.len() - 1],
        }
    }
}

// ── Weights sidecar ─────────────────────────────────────────────────────────

/// Stable slug for a [`BenchMode`]. The sidecar is read by other binaries, so
/// the spelling is part of the contract, not a `Debug` rendering.
#[must_use]
pub fn bench_mode_slug(mode: BenchMode) -> &'static str {
    match mode {
        BenchMode::Latency => "latency",
        BenchMode::Throughput => "throughput",
    }
}

/// The sidecar path for a weights file: the same path with `.mint.json`
/// appended (`extraction_head.bin` → `extraction_head.bin.mint.json`).
///
/// Appended rather than extension-replaced so two weights files that differ
/// only by extension can never share one sidecar.
#[must_use]
pub fn mint_metadata_path(weights: &Path) -> PathBuf {
    let mut s = weights.as_os_str().to_os_string();
    s.push(".mint.json");
    PathBuf::from(s)
}

/// What a weights file's training targets mean.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MintMetadata {
    /// [`bench_mode_slug`] of the mode every target was measured in.
    pub bench_mode: String,
    /// Which binary wrote these weights.
    pub trainer: String,
    /// Number of training samples the weights were fit on.
    pub samples: usize,
    /// Seed of the shuffle that decided *benchmark* order (not storage
    /// order), so a run's drift-vs-structure correlation is reproducible.
    pub order_shuffle_seed: u64,
    /// The session's opening sentinel calibration (ns/eval).
    pub sentinel_calibration_ns: f64,
    /// Drift correction actually applied across the run's labels.
    pub normalization: NormalizationStats,
    /// Unix seconds at write time.
    pub written_at_unix_s: u64,
}

impl MintMetadata {
    /// Write the sidecar for `weights`.
    ///
    /// # Errors
    ///
    /// Returns the underlying I/O or serialization error; callers must treat
    /// a failed sidecar write as a failed run — weights whose objective is
    /// unrecorded cannot be gate-checked.
    pub fn write_for(&self, weights: &Path) -> io::Result<PathBuf> {
        let path = mint_metadata_path(weights);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, json)?;
        Ok(path)
    }

    /// Read the sidecar for `weights`.
    ///
    /// # Errors
    ///
    /// Returns an error when the sidecar is missing or unparseable. A missing
    /// sidecar means the weights predate this contract: their objective is
    /// unknown, which a consumer must refuse rather than assume.
    pub fn read_for(weights: &Path) -> io::Result<Self> {
        let path = mint_metadata_path(weights);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "mint metadata sidecar {} is unreadable: {e}. The objective these weights \
                     were trained against is therefore unknown — retrain with \
                     bootstrap_extraction_head rather than evaluating on a guessed mode",
                    path.display()
                ),
            )
        })?;
        serde_json::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("mint metadata sidecar {} is malformed: {e}", path.display()),
            )
        })
    }

    /// Assert these weights were minted in `expected`.
    ///
    /// # Panics
    ///
    /// Panics on a mismatch. Latency and throughput rank expression shapes
    /// differently, so a gate measuring one against a head trained on the
    /// other is scoring a model on an objective it never saw.
    pub fn require_mode(&self, expected: BenchMode) {
        let want = bench_mode_slug(expected);
        assert_eq!(
            self.bench_mode, want,
            "extraction-head objective mismatch: the weights were minted with BenchMode::{} \
             targets by {}, but this consumer measures BenchMode::{want}. Those modes rank \
             expression shapes differently, so the verdict would score the model on an \
             objective it was never trained to predict",
            self.bench_mode, self.trainer
        );
    }
}

/// Unix seconds now.
///
/// # Panics
///
/// Panics if the system clock is before the Unix epoch.
#[must_use]
pub fn unix_now_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_secs()
}

/// Human-readable modification time of `path`, as an age in seconds plus the
/// raw unix timestamp — enough for a log line to say *which* file a warm
/// start came from and how stale it is.
///
/// Returns `Err` with the reason when the metadata cannot be read; callers
/// report it rather than dropping the detail.
pub fn file_mtime_report(path: &Path) -> Result<String, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("metadata unavailable: {e}"))?;
    let modified = meta
        .modified()
        .map_err(|e| format!("mtime unavailable on this platform: {e}"))?;
    let unix_s = modified
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("mtime is before the unix epoch: {e}"))?
        .as_secs();
    let age_s = unix_now_s().saturating_sub(unix_s);
    Ok(format!(
        "mtime unix {unix_s} ({age_s}s ago), {} bytes",
        meta.len()
    ))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_bench::SentinelContext;

    /// A `BenchResult` shaped only for the fields these tests exercise, with
    /// no call overhead (`adjusted_ns == ns`).
    fn bench(adjusted_ns: f64, sentinel: Option<SentinelContext>) -> BenchResult {
        bench_raw(adjusted_ns, 0.0, sentinel)
    }

    /// A `BenchResult` as a `BenchSession` builds one: `ns` is the raw reading
    /// on whatever clock was running, `call_overhead_ns` is the identity
    /// measurement taken at session open, and `adjusted_ns` is their
    /// difference — the cross-clock mixture that must not be scaled.
    fn bench_raw(ns: f64, call_overhead_ns: f64, sentinel: Option<SentinelContext>) -> BenchResult {
        BenchResult {
            ns,
            adjusted_ns: ns - call_overhead_ns,
            call_overhead_ns,
            iqr_ns: 0.0,
            mode: BenchMode::Latency,
            repeat_batches: 1,
            output: [0.0; 4],
            outputs: Vec::new(),
            sentinel,
        }
    }

    /// The measurement a machine running `drift`x slower than the opening
    /// clock produces for a kernel whose true cost is `kernel_ns`, given an
    /// identity call overhead of `overhead_ns` measured at session open.
    ///
    /// The whole clock inflates, so the raw reading covers kernel *and*
    /// overhead: `drift * (kernel + overhead)`.
    fn drifted(kernel_ns: f64, overhead_ns: f64, drift: f64) -> BenchResult {
        const CALIBRATION_NS: f64 = 8.0;
        bench_raw(
            (kernel_ns + overhead_ns) * drift,
            overhead_ns,
            Some(SentinelContext {
                calibration_ns: CALIBRATION_NS,
                local_sentinel_ns: CALIBRATION_NS * drift,
            }),
        )
    }

    /// The property the whole normalization design exists for: with a nonzero
    /// call overhead in play, the minted label is the true kernel time and
    /// does not depend on the drift factor at all.
    ///
    /// `d = 1.0` is the identity case; the rest sit inside the accepted
    /// ±50% band (`SENTINEL_REGIME_CHANGE_FRAC`), on both sides of it.
    #[test]
    fn label_is_the_true_kernel_time_independent_of_drift() {
        // A ~4ns call overhead against a small kernel is the regime where the
        // order-of-operations residue was largest, and small kernels are most
        // of the corpus.
        const OVERHEAD_NS: f64 = 4.0;
        for kernel_ns in [0.5f64, 3.0, 120.0] {
            for drift in [1.0f64, 0.62, 0.85, 1.2, 1.49] {
                let label = normalized_label_ns(&drifted(kernel_ns, OVERHEAD_NS, drift), "test");
                assert!(
                    (label - kernel_ns).abs() < 1e-9,
                    "kernel {kernel_ns}ns measured under {drift}x drift minted {label}ns — the \
                     label must be the drift-free kernel time, not a function of when in the \
                     run it was benchmarked"
                );
            }
        }
    }

    /// The bias this ordering was chosen to remove, pinned as arithmetic so
    /// the swap cannot silently come back: scaling `adjusted_ns` leaves
    /// `overhead * (1 - 1/d)`, which is >1ns for a 4ns overhead well inside
    /// the accepted drift band.
    #[test]
    fn scaling_the_adjusted_value_would_leave_a_drift_dependent_residue() {
        const OVERHEAD_NS: f64 = 4.0;
        const KERNEL_NS: f64 = 3.0;
        let drift = 1.45f64;
        let b = drifted(KERNEL_NS, OVERHEAD_NS, drift);
        let factor = b.sentinel.expect("sentinel").normalization();

        let wrong = b.adjusted_ns * factor;
        let residue = wrong - KERNEL_NS;
        assert!(
            (residue - OVERHEAD_NS * (1.0 - 1.0 / drift)).abs() < 1e-9,
            "the old form's error must be exactly overhead*(1 - 1/d), got {residue}"
        );
        assert!(
            residue > 1.0,
            "the residue must be big enough to matter against a {KERNEL_NS}ns kernel, got \
             {residue}ns"
        );
        assert!((normalized_label_ns(&b, "test") - KERNEL_NS).abs() < 1e-9);
    }

    /// Same expression, two collection positions, nonzero overhead: the two
    /// labels must agree. Order invariance is what the correction buys, and
    /// the previous ordering broke it precisely when overhead was nonzero.
    #[test]
    fn labels_agree_across_collection_positions_with_overhead() {
        const OVERHEAD_NS: f64 = 4.0;
        let early = normalized_label_ns(&drifted(2.0, OVERHEAD_NS, 1.0), "test");
        let late = normalized_label_ns(&drifted(2.0, OVERHEAD_NS, 1.4), "test");
        assert!((early - late).abs() < 1e-9, "{early} vs {late}");
    }

    #[test]
    fn normalization_rescales_into_the_opening_clock() {
        // The machine slowed to 1.25x its opening speed: a 125ns reading is
        // a 100ns expression.
        let b = bench(
            125.0,
            Some(SentinelContext {
                calibration_ns: 10.0,
                local_sentinel_ns: 12.5,
            }),
        );
        assert!((normalized_label_ns(&b, "test") - 100.0).abs() < 1e-9);
    }

    #[test]
    fn normalization_preserves_nonpositive_adjusted() {
        // A kernel whose ILP hides inside the identity call overhead: the raw
        // reading is a physical 16ns on a 2x-slowed clock (8ns in the opening
        // clock) against a 10ns opening overhead. The documented `<= 0`
        // condition must survive as `<= 0` rather than being clamped, flipped,
        // or turned into a different negative number by the ordering.
        let b = bench_raw(
            16.0,
            10.0,
            Some(SentinelContext {
                calibration_ns: 10.0,
                local_sentinel_ns: 20.0,
            }),
        );
        assert!((normalized_label_ns(&b, "test") - (-2.0)).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "no sentinel context")]
    fn missing_sentinel_is_a_loud_error_not_a_skipped_correction() {
        let _ = normalized_label_ns(&bench(42.0, None), "unit test");
    }

    /// The point of item 6: with a drift profile applied to the measurements,
    /// the *normalized* targets must not depend on which order the
    /// expressions were benchmarked in.
    #[test]
    fn normalized_targets_are_order_invariant_under_drift() {
        // Three expressions with distinct true costs.
        let truth = [10.0f64, 40.0, 90.0];
        const OVERHEAD_NS: f64 = 4.0;
        // A monotone slowdown over the run: the machine gets 1.0x, 1.3x,
        // 1.45x slower at successive collection positions.
        let drift = [1.0f64, 1.3, 1.45];

        let measure_in_order = |order: [usize; 3]| -> Vec<(usize, f64)> {
            order
                .iter()
                .enumerate()
                .map(|(position, &expr)| {
                    let b = drifted(truth[expr], OVERHEAD_NS, drift[position]);
                    (expr, normalized_label_ns(&b, "test"))
                })
                .collect()
        };

        let forward = measure_in_order([0, 1, 2]);
        let reversed = measure_in_order([2, 1, 0]);
        let interleaved = measure_in_order([1, 2, 0]);

        for orderings in [&reversed, &interleaved] {
            for &(expr, value) in orderings.iter() {
                let (_, base) = forward
                    .iter()
                    .find(|(e, _)| *e == expr)
                    .expect("same expressions in every ordering");
                assert!(
                    (value - base).abs() < 1e-9,
                    "expression {expr} minted {value} in one order and {base} in another — \
                     drift is still correlated with collection order"
                );
                assert!((value - truth[expr]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn unnormalized_targets_are_order_dependent() {
        // The control: without the correction, the same expression measured
        // early and late disagrees. This is what item 6 is fixing, so it is
        // pinned rather than assumed.
        let early = drifted(10.0, 4.0, 1.0);
        let late = drifted(10.0, 4.0, 1.45);
        assert!((early.adjusted_ns - late.adjusted_ns).abs() > 1.0);
        assert!(
            (normalized_label_ns(&early, "t") - normalized_label_ns(&late, "t")).abs() < 1e-9,
            "normalization must reconcile them"
        );
    }

    #[test]
    fn normalization_stats_summarize() {
        let s = NormalizationStats::summarize(&[1.0, 0.5, 2.0]);
        assert_eq!(s.count, 3);
        assert!((s.min - 0.5).abs() < 1e-12);
        assert!((s.median - 1.0).abs() < 1e-12);
        assert!((s.max - 2.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "no normalization factors")]
    fn normalization_stats_refuse_empty() {
        let _ = NormalizationStats::summarize(&[]);
    }

    #[test]
    fn sidecar_round_trips_next_to_the_weights() {
        let dir = std::env::temp_dir().join(format!("mint_meta_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let weights = dir.join("extraction_head.bin");

        let meta = MintMetadata {
            bench_mode: bench_mode_slug(BenchMode::Latency).to_string(),
            trainer: "bootstrap_extraction_head".to_string(),
            samples: 7,
            order_shuffle_seed: 99,
            sentinel_calibration_ns: 12.5,
            normalization: NormalizationStats::summarize(&[1.0]),
            written_at_unix_s: unix_now_s(),
        };
        let path = meta.write_for(&weights).expect("write sidecar");
        assert_eq!(path, mint_metadata_path(&weights));
        assert_eq!(
            path.file_name().expect("name"),
            "extraction_head.bin.mint.json"
        );

        let read = MintMetadata::read_for(&weights).expect("read sidecar");
        assert_eq!(read, meta);
        read.require_mode(BenchMode::Latency);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "objective mismatch")]
    fn require_mode_rejects_the_other_objective() {
        let meta = MintMetadata {
            bench_mode: bench_mode_slug(BenchMode::Latency).to_string(),
            trainer: "bootstrap_extraction_head".to_string(),
            samples: 1,
            order_shuffle_seed: 0,
            sentinel_calibration_ns: 1.0,
            normalization: NormalizationStats::summarize(&[1.0]),
            written_at_unix_s: 0,
        };
        meta.require_mode(BenchMode::Throughput);
    }

    #[test]
    fn missing_sidecar_is_an_error_naming_the_path() {
        let weights = std::env::temp_dir().join("definitely_absent_weights_xyz.bin");
        let err = MintMetadata::read_for(&weights).expect_err("must not succeed");
        let msg = err.to_string();
        assert!(msg.contains("mint.json"), "{msg}");
        assert!(msg.contains("unknown"), "{msg}");
    }
}
