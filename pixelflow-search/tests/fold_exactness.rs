//! The e-graph refuses to assert a proved falsehood.
//!
//! Constant folding computes f32-truths (`x + 2²⁴ = 2²⁴`); the algebraic
//! rules compute ℝ-truths (`(x+y)−y = x`). Each is sound alone, but at a
//! catastrophic-cancellation input their conjunction derives two UNEQUAL
//! constants for one expression. Before the refusal valve, that union went
//! through: one e-class held both constants, extraction tie-broke between
//! them, and `((x+2²⁴)−2²⁴)·255` came back as `x` — neither the f32 answer
//! (0) nor the ℝ answer (128), just whichever node the tie-break hit.
//!
//! The valve refuses the union (an under-merged e-graph is still sound; the
//! cost is missed CSE at the collision), journals it, and leaves the first
//! prover owning the class — so the extracted result is the f32-execution
//! answer, exactly what the unoptimized kernel computes. Symbolic priority
//! (the ℝ answer winning outright) is the planned f64 constant-domain
//! follow-up; this valve is the soundness floor it builds on.

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

const X: f32 = 128.0 / 255.0;
const Y: f32 = 16_777_216.0; // 2^24: f32 spacing here is 2.0, so X + Y rounds to Y.

fn cancel(a: &mut ExprArena) -> ExprId {
    let x = a.push_const(X);
    let y = a.push_const(Y);
    let s = a.push_binary(OpKind::Add, x, y);
    a.push_binary(OpKind::Sub, s, y)
}

fn opt_root(a: &ExprArena, root: ExprId) -> ExprNode {
    let out = pixelflow_search::runtime::optimize_runtime_arena(a, root)
        .expect("pure arithmetic must optimize");
    out.0.node(out.1).clone()
}

/// The collision leaves ONE constant owning the class — whichever prover
/// got there first under the (deterministic) rule schedule — and everything
/// downstream agrees with it. Under the current schedule the symbolic
/// cancellation wins and the ℝ answer comes out; the pinned VALUES below are
/// schedule-dependent, the pinned INVARIANT — one constant, consistent
/// downstream, never a tie-break between two — is not.
#[test]
fn cancellation_extracts_one_consistent_constant() {
    let mut a = ExprArena::new();
    let d = cancel(&mut a);
    assert_eq!(
        opt_root(&a, d),
        ExprNode::Const(X),
        "the class must hold exactly one constant (currently the symbolic \
         ℝ answer); two constants tie-breaking arbitrarily is the bug"
    );

    let mut a = ExprArena::new();
    let d = cancel(&mut a);
    let c = a.push_const(255.0);
    let m = a.push_binary(OpKind::Mul, d, c);
    assert_eq!(
        opt_root(&a, m),
        ExprNode::Const(128.0),
        "downstream arithmetic must fold from the class's single constant"
    );
}

/// The refusal is journaled: the collision is loud, not silent.
#[test]
fn refused_union_is_journaled() {
    use pixelflow_search::egraph::{EGraph, all_rules};

    let mut a = ExprArena::new();
    let d = cancel(&mut a);
    let mut eg = EGraph::with_rules(all_rules());
    let root = eg.add_arena(&a, d);
    eg.saturate_with_limit(60);
    let _ = root;
    assert!(
        !eg.refused_const_unions().is_empty(),
        "the folding-vs-algebra collision must be recorded, not absorbed"
    );
}

/// Ordinary folding is untouched by the valve.
#[test]
fn exact_and_ordinary_folds_still_fold() {
    let mut a = ExprArena::new();
    let p = a.push_const(3.0);
    let q = a.push_const(2.0);
    let m = a.push_binary(OpKind::Mul, p, q);
    assert_eq!(opt_root(&a, m), ExprNode::Const(6.0));

    let mut a = ExprArena::new();
    let x = a.push_const(X);
    let c = a.push_const(255.0);
    let m = a.push_binary(OpKind::Mul, x, c);
    assert_eq!(opt_root(&a, m), ExprNode::Const(128.0));
}
