//! Production's application-budget calibration
//! (`docs/plans/2026-09-01-production-budget-determinism.md`), pinned as
//! tests rather than left as prose two commits could quietly drift apart
//! from.

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_search::egraph::{Optimizer, SaturationConfig};

/// A mid-sized expression with sharing and several rule families in reach —
/// the same shape `optimizer_laws.rs`'s fixture uses, duplicated here rather
/// than shared across `tests/` binaries (each integration test file is its
/// own crate).
fn fixture() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);
    let one = a.push_const(1.0);
    let two = a.push_const(2.0);

    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let sum = a.push_binary(OpKind::Add, xx, yy);
    let scaled = a.push_binary(OpKind::Mul, sum, two);
    let offset = a.push_binary(OpKind::Add, sum, one);
    let prod = a.push_binary(OpKind::Mul, scaled, offset);
    let with_z = a.push_binary(OpKind::Add, prod, z);
    let neg = a.push_unary(OpKind::Neg, with_z);
    let root = a.push_binary(OpKind::Sub, with_z, neg);
    (a, root)
}

/// A structural rendering of the extracted DAG, for equality comparisons —
/// the arena is append-only and extraction emits children before parents,
/// so this is already canonical for a given configuration.
fn arena_shape(arena: &ExprArena, root: ExprId) -> String {
    format!("{root:?}|{:?}", arena.nodes_raw())
}

// ---------------------------------------------------------------------------
// Calibration: the application budget must not newly truncate anything
// observed in either 2026-09-01 corpus.
// ---------------------------------------------------------------------------

/// One row of the calibration table in
/// `docs/plans/2026-09-01-production-budget-determinism.md`'s "Applications
/// at a deterministic stop, per tier" section: the highest application count
/// either corpus observed at a deterministic stop (class cap or quiescence,
/// never a clock) for that tier.
struct CalibrationRow {
    tier: &'static str,
    corpus: &'static str,
    /// Applications at the deterministic stop — `p100` (the observed
    /// maximum) where the source table has one, else the largest
    /// non-timeout figure recorded for that row.
    p100_applications: u64,
}

/// Every row the calibration doc measured 2026-09-01. Transcribed from the
/// doc's own table, not re-derived — this file is the fixture the doc's
/// "Decision: the application budgets" section promises, not a second
/// measurement.
const CALIBRATION: &[CalibrationRow] = &[
    CalibrationRow {
        tier: "blitz",
        corpus: "#1084 DEV, quiesced only",
        p100_applications: 45,
    },
    CalibrationRow {
        tier: "blitz",
        corpus: "#1087 real (space glyph)",
        p100_applications: 0,
    },
    CalibrationRow {
        tier: "rapid",
        corpus: "#1084 DEV",
        p100_applications: 1_862,
    },
    CalibrationRow {
        tier: "classical",
        corpus: "#1087 real, ref (no clock, ClassCap)",
        p100_applications: 55_242,
    },
    CalibrationRow {
        tier: "classical",
        corpus: "#1084 DEV, quiesced only",
        p100_applications: 38_645,
    },
];

fn max_applications_for_tier(tier: &str) -> u64 {
    match tier {
        "blitz" => SaturationConfig::blitz().max_applications,
        "rapid" => SaturationConfig::rapid().max_applications,
        "classical" => SaturationConfig::classical().max_applications,
        other => panic!("no such tier: {other}"),
    }
}

/// The application budget must sit strictly above every deterministic-stop
/// application count either corpus measured, at that row's own tier — the
/// property the calibration doc's "gate this satisfies" section states in
/// prose. A future change to the constants that drops below any of these
/// would newly truncate a kernel that used to reach its class cap or
/// quiesce cleanly, silently making that kernel's extraction worse.
#[test]
fn the_application_budget_exceeds_every_calibrated_deterministic_stop() {
    for row in CALIBRATION {
        let budget = max_applications_for_tier(row.tier);
        assert!(
            budget > row.p100_applications,
            "{} tier's max_applications ({budget}) does not exceed the {} corpus's \
             observed deterministic-stop application count ({}) — this would newly \
             truncate a kernel that used to reach its class cap or quiesce cleanly",
            row.tier,
            row.corpus,
            row.p100_applications,
        );
    }
}

/// The class and iteration caps the calibration doc says are unchanged.
#[test]
fn class_and_iteration_caps_are_unchanged() {
    let blitz = SaturationConfig::blitz();
    let rapid = SaturationConfig::rapid();
    let classical = SaturationConfig::classical();

    assert_eq!((blitz.max_iterations, blitz.max_classes), (20, 500));
    assert_eq!((rapid.max_iterations, rapid.max_classes), (50, 2_000));
    assert_eq!(
        (classical.max_iterations, classical.max_classes),
        (100, 5_000)
    );
}

// ---------------------------------------------------------------------------
// Determinism under CPU contention
// ---------------------------------------------------------------------------

/// Saturating the same arena under the same (deterministic, application-
/// counted) budget produces byte-identical extractions and identical stop
/// reasons even while the machine is under artificial load — the property
/// the whole change exists to buy back from what wall-clock timeouts made
/// host-speed-dependent. Contention is simulated with in-process spinner
/// threads only, never external processes.
#[test]
fn saturation_is_deterministic_under_cpu_contention() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let (arena, root) = fixture();

    let stop = Arc::new(AtomicBool::new(false));
    let spinners: Vec<_> = (0..8)
        .map(|_| {
            let stop = stop.clone();
            thread::spawn(move || {
                // A tight, side-effect-free loop that keeps a core busy
                // without allocating or touching shared state — pure CPU
                // contention, nothing that could itself race with the
                // saturation runs below.
                let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
                while !stop.load(Ordering::Relaxed) {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    std::hint::black_box(x);
                }
            })
        })
        .collect();

    let mut runs = Vec::new();
    for _ in 0..6 {
        let mut optimizer = Optimizer::production();
        let mut eg = optimizer.egraph();
        let root_class = eg.add_arena(&arena, root);
        let optimized = optimizer.run(&mut eg, root_class, arena.len());
        let stop_reason = optimized.stats.stop;
        let (out, out_root) = optimized.to_arena(&eg, root_class);
        runs.push((stop_reason, arena_shape(&out, out_root)));
    }

    stop.store(true, Ordering::Relaxed);
    for spinner in spinners {
        spinner.join().expect("spinner thread panicked");
    }

    let (reference_stop, reference_shape) = &runs[0];
    for (i, (stop_reason, shape)) in runs.iter().enumerate().skip(1) {
        assert_eq!(
            stop_reason, reference_stop,
            "run {i} stopped for a different reason under CPU contention than run 0"
        );
        assert_eq!(
            shape, reference_shape,
            "run {i} extracted a different term under CPU contention than run 0"
        );
    }
}
