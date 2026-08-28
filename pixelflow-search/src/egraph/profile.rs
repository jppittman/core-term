//! Extraction-loop phase profiling — OPT-IN via the `extraction-profile`
//! Cargo feature, off by default and not part of this crate's public API
//! contract. It exists to attribute wall time inside
//! [`super::extract::IncrementalExtractor::extract_choices_only`] to
//! specific phases (candidate enumeration, `try_swap` backfill, the
//! acyclicity check, the `chosen_variance` walk, `pinned_choices`, the
//! `EdgeAccumulator` rebuild, and the NNUE forward pass) for the Round 2b
//! extraction-overhead investigation
//! (docs/plans/2026-08-17-cost-model-domain.md, anti-row A6).
//!
//! [`Bucket`] itself is always compiled (call sites pass it unconditionally
//! so instrumentation and non-instrumentation builds share one call shape),
//! but every stateful piece — the thread-local counters, [`record`],
//! [`reset`], [`snapshot`] — exists only under the feature. With the feature
//! off, [`timed`] is `f()` and nothing else: no thread-local, no `Instant`,
//! no branch.

/// One phase of the extraction refinement loop. See the module doc for what
/// each variant covers; the exact wrap points live in `extract.rs` (for
/// `TrySwapBackfill`/`AcyclicityCheck`/`CandidateEnumeration`/`NnueForward`)
/// and `nnue/factored.rs` (for `ChosenVariance`/`PinnedChoices`/
/// `AccumulatorRebuild`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // constructed only under the feature's call sites
pub enum Bucket {
    /// Building the per-pass active-class list
    /// ([`super::extract::IncrementalExtractor::get_active_classes`]) — the
    /// literal "which e-classes are candidates this pass" walk.
    CandidateEnumeration,
    /// `backfill_well_founded` calls made inside
    /// [`super::extract::Extraction::try_swap`] to complete a swapped
    /// node's newly-reachable children.
    TrySwapBackfill,
    /// `choices_have_cycle_from` call inside
    /// [`super::extract::Extraction::try_swap`].
    AcyclicityCheck,
    /// [`super::extract::Extraction::chosen_variance`].
    ChosenVariance,
    /// [`super::extract::Extraction::pinned_choices`] (`pin_shift_counts`).
    PinnedChoices,
    /// `EdgeAccumulator::from_cost_dag`'s structural walk, excluding the
    /// `pinned_choices`/`chosen_variance` sub-calls (each has its own
    /// bucket).
    AccumulatorRebuild,
    /// `ExprNnue::predict_log_cost_with_features`.
    NnueForward,
}

const BUCKET_COUNT: usize = 7;
const ALL_BUCKETS: [Bucket; BUCKET_COUNT] = [
    Bucket::CandidateEnumeration,
    Bucket::TrySwapBackfill,
    Bucket::AcyclicityCheck,
    Bucket::ChosenVariance,
    Bucket::PinnedChoices,
    Bucket::AccumulatorRebuild,
    Bucket::NnueForward,
];

impl Bucket {
    fn idx(self) -> usize {
        match self {
            Bucket::CandidateEnumeration => 0,
            Bucket::TrySwapBackfill => 1,
            Bucket::AcyclicityCheck => 2,
            Bucket::ChosenVariance => 3,
            Bucket::PinnedChoices => 4,
            Bucket::AccumulatorRebuild => 5,
            Bucket::NnueForward => 6,
        }
    }

    /// Stable snake_case name for reports (JSONL keys, printed tables).
    pub fn name(self) -> &'static str {
        match self {
            Bucket::CandidateEnumeration => "candidate_enumeration",
            Bucket::TrySwapBackfill => "try_swap_backfill",
            Bucket::AcyclicityCheck => "acyclicity_check",
            Bucket::ChosenVariance => "chosen_variance",
            Bucket::PinnedChoices => "pinned_choices",
            Bucket::AccumulatorRebuild => "accumulator_rebuild",
            Bucket::NnueForward => "nnue_forward",
        }
    }
}

#[cfg(feature = "extraction-profile")]
mod imp {
    use super::{ALL_BUCKETS, BUCKET_COUNT, Bucket};
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    #[derive(Clone, Copy, Default)]
    pub struct BucketStats {
        pub nanos: u64,
        pub count: u64,
    }

    std::thread_local! {
        static COUNTERS: RefCell<[BucketStats; BUCKET_COUNT]> =
            RefCell::new([BucketStats { nanos: 0, count: 0 }; BUCKET_COUNT]);
    }

    /// Zero every bucket on the current thread. Call once per measured unit
    /// (e.g. once per corpus expression) so `snapshot` reports that unit's
    /// cost, not a running total.
    pub fn reset() {
        COUNTERS.with(|c| *c.borrow_mut() = [BucketStats::default(); BUCKET_COUNT]);
    }

    /// Add one observation to `bucket`.
    pub fn record(bucket: Bucket, dur: Duration) {
        COUNTERS.with(|c| {
            let mut c = c.borrow_mut();
            let s = &mut c[bucket.idx()];
            s.nanos += dur.as_nanos() as u64;
            s.count += 1;
        });
    }

    /// `(bucket, stats)` for every bucket, in [`super::ALL_BUCKETS`] order.
    pub fn snapshot() -> [(Bucket, BucketStats); BUCKET_COUNT] {
        COUNTERS.with(|c| {
            let c = *c.borrow();
            let mut out = [(Bucket::CandidateEnumeration, BucketStats::default()); BUCKET_COUNT];
            for (i, b) in ALL_BUCKETS.into_iter().enumerate() {
                out[i] = (b, c[i]);
            }
            out
        })
    }

    /// Run `f`, attributing its wall time to `bucket`.
    #[inline]
    pub fn timed<T>(bucket: Bucket, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        record(bucket, start.elapsed());
        out
    }
}

#[cfg(not(feature = "extraction-profile"))]
mod imp {
    use super::Bucket;

    /// Feature off: run `f` with no timing, no thread-local touch.
    #[inline(always)]
    pub fn timed<T>(_bucket: Bucket, f: impl FnOnce() -> T) -> T {
        f()
    }
}

pub use imp::timed;
#[cfg(feature = "extraction-profile")]
pub use imp::{BucketStats, record, reset, snapshot};
