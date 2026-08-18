//! Regression tests for the measured latency prior (2026-08-10 re-derivation,
//! `pixelflow-pipeline/examples/measure_latency_prior.rs`).
//!
//! The shipping bug this pins down: the prior priced `Pow` at 12, *cheaper*
//! than a hardware `Sqrt` at 15 — but `expand_transcendentals` lowers
//! `Pow(a,b)` to `exp2(b·log2 a)`, two polynomial bit-manipulation kernels
//! (measured 196 table-cycles, vs 15 for `Sqrt`). Under the old table the
//! DEFAULT extraction policy kept `Pow(x, 0.5)` instead of rewriting it to
//! `Sqrt(x)`, a measured 2.8x kernel slowdown on the affected kernel.

use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_search::runtime::optimize_runtime_arena;

/// Collect the OpKinds reachable from `root`.
fn reachable_kinds(arena: &ExprArena, root: ExprId) -> Vec<OpKind> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut kinds = Vec::new();
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if std::mem::replace(&mut seen[idx], true) {
            continue;
        }
        kinds.push(arena.kind(id));
        stack.extend(arena.children(id));
    }
    kinds
}

/// `Pow(x, 0.5)` must extract as the hardware `Sqrt`, not survive as `Pow`.
///
/// This is exactly the decision the mispriced table got wrong: the
/// `power-sqrt` rewrite put both forms in the e-graph, and extraction chose
/// `Pow` because 12 < 15. With the measured table (Pow 196, Sqrt 15) the
/// hardware primitive must win.
#[test]
fn pow_half_extracts_to_hardware_sqrt() {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let half = arena.push_const(0.5);
    let root = arena.push_binary(OpKind::Pow, x, half);

    let optimized = optimize_runtime_arena(&arena, root)
        .expect("pure arithmetic arena must be e-graph representable");
    let (opt_arena, opt_root) = (&optimized.0, optimized.1);

    let kinds = reachable_kinds(opt_arena, opt_root);
    assert!(
        kinds.contains(&OpKind::Sqrt),
        "Pow(x, 0.5) should extract to a hardware Sqrt; got {kinds:?}"
    );
    assert!(
        !kinds.contains(&OpKind::Pow),
        "Pow must not survive extraction when a Sqrt form exists; got {kinds:?}"
    );
}

/// Same decision one derivative over: `Pow(x, -0.5)` has an `Rsqrt` form via
/// the `power-rsqrt` rewrite. Rsqrt measures 21 — more than Sqrt's 15
/// (estimate + Newton chain), but still ~9x cheaper than the lowered `Pow`.
#[test]
fn pow_neg_half_does_not_survive_as_pow() {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let exp = arena.push_const(-0.5);
    let root = arena.push_binary(OpKind::Pow, x, exp);

    let optimized = optimize_runtime_arena(&arena, root)
        .expect("pure arithmetic arena must be e-graph representable");
    let kinds = reachable_kinds(&optimized.0, optimized.1);
    assert!(
        !kinds.contains(&OpKind::Pow),
        "Pow(x, -0.5) must lower to a cheaper primitive form; got {kinds:?}"
    );
}
