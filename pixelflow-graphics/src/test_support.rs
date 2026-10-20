//! Helpers that more than one module's tests share.

use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};

/// FNV-1a over every node of `arena` in arena order, then `root`: the
/// optimized program as a sequence, not as a canonical key. The key is
/// order-free, so two selections that differ only in a commutative
/// operand's position share one and emit different bytes; this digest does
/// not.
///
/// A node is hashed as what it *is* (its variant, operator, payload and
/// child ids) and not as its `Debug` text, so the pin moves when the
/// selection does and not when a field is added to a node's `Debug` output.
/// The match is exhaustive on purpose: a new kind of node fails the build
/// here until someone decides what distinguishes it.
pub(crate) fn program_digest(arena: &ExprArena, root: ExprId) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
    };
    for (id, node) in arena.nodes() {
        let (variant, op, payload): (u8, Option<OpKind>, u128) = match node {
            ExprNode::Var(i) => (0, None, u128::from(i)),
            ExprNode::Const(v) => (1, None, u128::from(v.to_bits())),
            ExprNode::Param(i) => (2, None, u128::from(i)),
            ExprNode::Buffer(b) => (3, None, u128::from(b.0)),
            ExprNode::Uniform(u) => (4, None, u128::from(u.0)),
            ExprNode::Ref(key) => (5, None, u128::from(key.bits())),
            ExprNode::Unary(op, _)
            | ExprNode::Binary(op, ..)
            | ExprNode::Ternary(op, ..)
            | ExprNode::Nary(op, _) => (6, Some(op), 0),
            ExprNode::Reduce { fold, .. } => (7, None, fold.to_bits()),
            ExprNode::Write { row, col, lane, .. } => (
                8,
                None,
                u128::from(row.var()) | u128::from(col.var()) << 8 | u128::from(lane.var()) << 16,
            ),
        };
        feed(&[variant]);
        feed(&op.map_or(0, |op| op as u64).to_le_bytes());
        feed(&payload.to_le_bytes());
        for child in arena.children(id) {
            feed(&child.0.to_le_bytes());
        }
    }
    feed(&root.0.to_le_bytes());
    hash
}
