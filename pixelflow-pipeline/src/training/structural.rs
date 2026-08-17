//! Structural hashing of arena expressions.
//!
//! One structural key per expression DAG, shared by every stage that must
//! agree on "same structure": `gen_bench_corpus`'s cross-tier dedup ledger
//! (plan 0.2) and `bootstrap_extraction_head`'s live-stream holdout fence
//! (a live-generated training expression whose structure already lives in
//! the DEV or FINAL tier would be trained on and then evaluated, so DEV
//! metrics would no longer measure a holdout). Keeping the hash in one place
//! is the point: two divergent notions of "duplicate" would re-open exactly
//! that leak.

use std::collections::{HashMap, HashSet};

use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};

/// One node of a structural key: the node's operation plus the key-local ids
/// of its children, in deterministic post-order. `Const` compares by bit
/// pattern (`-0.0` and `+0.0` are structurally distinct, as are NaN
/// payloads).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SigNode {
    /// Coordinate variable.
    Var(u8),
    /// Constant, by exact bit pattern.
    Const(u32),
    /// Kernel parameter.
    Param(u8),
    /// Buffer reference.
    Buffer(u16),
    /// Unary op over a key-local child id.
    Unary(OpKind, u32),
    /// Binary op over key-local child ids.
    Binary(OpKind, u32, u32),
    /// Ternary op over key-local child ids.
    Ternary(OpKind, u32, u32, u32),
    /// N-ary op over key-local child ids.
    Nary(OpKind, Box<[u32]>),
}

/// The structural key of the DAG reachable from `root`: a deterministic
/// post-order listing of [`SigNode`]s with arena ids rewritten to key-local
/// ids, so two arenas holding the same structure produce equal keys
/// regardless of node numbering or dead nodes.
#[must_use]
pub fn arena_structural_key(arena: &ExprArena, root: ExprId) -> Vec<SigNode> {
    enum Task {
        Visit(ExprId),
        Emit(ExprId),
    }

    let mut work = vec![Task::Visit(root)];
    let mut visited = HashSet::new();
    let mut signatures = Vec::new();
    let mut ids = HashMap::<ExprId, u32>::new();

    while let Some(task) = work.pop() {
        match task {
            Task::Visit(id) => {
                if !visited.insert(id) {
                    continue;
                }
                work.push(Task::Emit(id));
                let children: Vec<ExprId> = arena.children(id).collect();
                for child in children.into_iter().rev() {
                    work.push(Task::Visit(child));
                }
            }
            Task::Emit(id) => {
                let sig = match arena.node(id) {
                    ExprNode::Var(i) => SigNode::Var(*i),
                    ExprNode::Const(v) => SigNode::Const(v.to_bits()),
                    ExprNode::Param(i) => SigNode::Param(*i),
                    ExprNode::Buffer(b) => SigNode::Buffer(b.0),
                    ExprNode::Unary(op, a) => SigNode::Unary(*op, ids[a]),
                    ExprNode::Binary(op, a, b) => SigNode::Binary(*op, ids[a], ids[b]),
                    ExprNode::Ternary(op, a, b, c) => SigNode::Ternary(*op, ids[a], ids[b], ids[c]),
                    ExprNode::Nary(op, start, len) => {
                        let children = arena.nary_children_slice(*start, *len);
                        let child_ids: Vec<u32> = children.iter().map(|child| ids[child]).collect();
                        SigNode::Nary(*op, child_ids.into_boxed_slice())
                    }
                };
                let sig_id = signatures.len() as u32;
                signatures.push(sig);
                ids.insert(id, sig_id);
            }
        }
    }

    signatures
}

/// A set of structural keys — the holdout fence used by
/// `bootstrap_extraction_head`: every DEV and FINAL expression is inserted,
/// then each live-generated TRAIN expression is probed and dropped on a hit.
///
/// Deliberately the same [`arena_structural_key`] the corpus builder's
/// cross-tier ledger uses. Two notions of "same structure" would let a shallow
/// motif slip the fence here while the ledger considered it a duplicate, which
/// is exactly the leak (train on it, then evaluate on it, and call the result
/// held-out).
#[derive(Default)]
pub struct StructuralKeySet {
    keys: HashSet<Vec<SigNode>>,
}

impl StructuralKeySet {
    /// An empty fence.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keys: HashSet::new(),
        }
    }

    /// Insert the structure of `root`. Returns `true` when it was not already
    /// present (mirroring [`HashSet::insert`]).
    pub fn insert(&mut self, arena: &ExprArena, root: ExprId) -> bool {
        self.keys.insert(arena_structural_key(arena, root))
    }

    /// Whether this structure is already fenced.
    #[must_use]
    pub fn contains(&self, arena: &ExprArena, root: ExprId) -> bool {
        self.keys.contains(&arena_structural_key(arena, root))
    }

    /// Number of distinct structures fenced.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the fence holds no structures at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_set_fences_by_structure_not_identity() {
        // The fence is populated from one arena and probed with another that
        // merely holds the same shape — the holdout-leak case.
        let mut holdout = ExprArena::new();
        let hx = holdout.push_var(0);
        let hc = holdout.push_const(2.0);
        let hroot = holdout.push_binary(OpKind::Mul, hx, hc);

        let mut fence = StructuralKeySet::new();
        assert!(fence.insert(&holdout, hroot));
        assert!(!fence.insert(&holdout, hroot), "second insert is a no-op");
        assert_eq!(fence.len(), 1);

        let mut live = ExprArena::new();
        let _dead = live.push_const(99.0);
        let lx = live.push_var(0);
        let lc = live.push_const(2.0);
        let lhit = live.push_binary(OpKind::Mul, lx, lc);
        assert!(fence.contains(&live, lhit), "same structure must be fenced");

        let lmiss = live.push_binary(OpKind::Add, lx, lc);
        assert!(
            !fence.contains(&live, lmiss),
            "a different structure must pass the fence"
        );
    }

    #[test]
    fn empty_key_set_fences_nothing() {
        let fence = StructuralKeySet::new();
        assert!(fence.is_empty());
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        assert!(!fence.contains(&arena, x));
    }

    #[test]
    fn identical_structures_in_different_arenas_share_a_key() {
        // Same structure, different arena layouts (dead node in the second).
        let mut a = ExprArena::new();
        let ax = a.push_var(0);
        let ac = a.push_const(2.0);
        let aroot = a.push_binary(OpKind::Mul, ax, ac);

        let mut b = ExprArena::new();
        let _dead = b.push_const(99.0);
        let bx = b.push_var(0);
        let bc = b.push_const(2.0);
        let broot = b.push_binary(OpKind::Mul, bx, bc);

        assert_eq!(
            arena_structural_key(&a, aroot),
            arena_structural_key(&b, broot)
        );
    }

    #[test]
    fn different_structures_get_different_keys() {
        let mut a = ExprArena::new();
        let ax = a.push_var(0);
        let ac = a.push_const(2.0);
        let mul = a.push_binary(OpKind::Mul, ax, ac);
        let add = a.push_binary(OpKind::Add, ax, ac);
        assert_ne!(arena_structural_key(&a, mul), arena_structural_key(&a, add));

        // Constants compare by bit pattern: 0.0 vs -0.0 are distinct.
        let mut p = ExprArena::new();
        let pz = p.push_const(0.0);
        let mut n = ExprArena::new();
        let nz = n.push_const(-0.0);
        assert_ne!(arena_structural_key(&p, pz), arena_structural_key(&n, nz));
    }
}
