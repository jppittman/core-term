//! The one lowering pass [`pixelflow_ir::eval_scalar`] applies before it walks.
//!
//! The differential harnesses have to classify divergence and conditioning on
//! *the ops the oracle actually evaluates* — the `Min`/`Max`/`Round` inside an
//! expanded `sin`, not the `Sin` node the source arena spells. That means
//! expanding with **exactly** the pass `eval_scalar` uses and no other:
//! `pixelflow_ir::passes::legalize` is the compile pipeline (dwrt, reduce,
//! gather, then transcendentals) and would show the walker a different arena
//! than the oracle sees.
//!
//! `pixelflow_ir::passes::expand_transcendentals_owned` is `pub(crate)` to that
//! crate, so this reproduces it through the public mutating entry point. If the
//! two ever drift apart, the harnesses silently start judging the wrong arena —
//! so keep this a mirror of `passes.rs`, including the identity fast-path.

use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind, passes};

/// Whether `op` is a transcendental [`passes::expand_transcendentals`] rewrites.
///
/// Mirrors the private `is_transcendental_unary`/`is_transcendental_binary`
/// predicates in `pixelflow-ir/src/passes.rs`; it exists only to decide the
/// identity fast-path, so an over-broad answer costs a needless rebuild and
/// never a wrong result.
fn is_transcendental(node: &ExprNode) -> bool {
    match node {
        ExprNode::Unary(op, _) => matches!(
            op,
            OpKind::Sin
                | OpKind::Cos
                | OpKind::Tan
                | OpKind::Exp
                | OpKind::Exp2
                | OpKind::Ln
                | OpKind::Log2
                | OpKind::Log10
                | OpKind::Atan
                | OpKind::Asin
                | OpKind::Acos
        ),
        ExprNode::Binary(op, _, _) => matches!(op, OpKind::Atan2 | OpKind::Pow),
        _ => false,
    }
}

/// Expand every transcendental reachable from `root` into its primitive
/// arithmetic subgraph, in a clone, leaving the caller's arena untouched.
///
/// The identity fast-path matters for more than speed: the rebuild re-orders
/// and re-dedups nodes, so returning the arena unchanged when there is nothing
/// to lower keeps [`ExprId`]s comparable with the source arena's.
#[must_use]
pub fn expand_transcendentals_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    if !arena.nodes_raw().iter().any(is_transcendental) {
        return (arena.clone(), root);
    }
    let mut owned = arena.clone();
    let new_root = passes::expand_transcendentals(&mut owned, root);
    (owned, new_root)
}
