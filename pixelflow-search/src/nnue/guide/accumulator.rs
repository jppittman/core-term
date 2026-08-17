//! `GraphAccumulator`: VSA encoding of e-graph state.
//!
//! Private implementation detail of the [`super::SaturationGuide`] contract —
//! nothing outside `nnue::guide` names this type. See the module doc on
//! [`super`] for why it exists and why it is inert today.
//!
//! `#![allow(dead_code)]`: every mutator below is exercised only by this
//! module's own characterization tests (see `docs/plans/2026-07-07-guided-
//! saturation-redesign.md`, Phase 3 — not started). That is a fact about the
//! roadmap, not a bug in this file; suppressing the lint here says exactly
//! that, rather than a `pub` visibility silently hiding the same truth the
//! way the pre-encapsulation code did.
#![allow(dead_code)]

use libm::sqrtf;

use super::{shift_by, shift1};
use crate::nnue::factored::{K, OpEmbeddings, OpKind};

/// Graph accumulator dimension: marginals (2K) + 1-hop VSA binding (K) + 2-hop VSA binding (K).
pub(crate) const GRAPH_ACC_DIM: usize = 4 * K; // 128

/// VSA accumulator for e-graph state (rebuilt each epoch).
///
/// Three-section encoding captures both marginal and joint op distributions:
///
/// | Section | Dim | Operation | Signal |
/// |---------|-----|-----------|--------|
/// | `[0..K]` | K | `Σ E[parent]` | Marginal: which ops appear as parents |
/// | `[K..2K]` | K | `Σ E[child]` | Marginal: which ops appear as children |
/// | `[2K..3K]` | K | `Σ E[parent] ⊙ shift₁(E[child])` | **1-hop VSA binding**: which ops are connected |
/// | `[3K..4K]` | K | `Σ E[gp] ⊙ shift₁(E[par]) ⊙ shift²(E[child])` | **2-hop VSA binding**: 3-node path patterns |
///
/// The 1-hop binding section uses element-wise Hadamard product with a cyclic shift to
/// break commutativity (`Mul→Add ≠ Add→Mul`). This captures the **joint**
/// distribution of parent-child pairs — strictly more informative than marginals
/// alone.
///
/// The 2-hop binding section extends this to 3-node paths (grandparent→parent→child),
/// capturing patterns like "Mul feeding Add feeding Sqrt" which is exactly what
/// rewrite rules match on. This turns the accumulator from a 0-round GNN into
/// a 1-round GNN.
///
/// Shares `OpEmbeddings` with the extraction head — same learned op embeddings,
/// different downstream pathway.
#[derive(Clone)]
pub(crate) struct GraphAccumulator {
    /// `[0..K]`:    marginal parent sum     `Σ E[parent]`
    /// `[K..2K]`:   marginal child sum      `Σ E[child]`
    /// `[2K..3K]`:  1-hop VSA binding sum   `Σ E[parent] ⊙ shift_by(E[child], depth)`
    /// `[3K..4K]`:  2-hop VSA binding sum   `Σ E[gp] ⊙ shift₁(E[par]) ⊙ shift²(E[child])`
    pub(crate) values: [f32; GRAPH_ACC_DIM],
    /// Number of edges added to the accumulator.
    pub(crate) edge_count: u32,
    /// Number of nodes (ops + leaves) in the graph.
    pub(crate) node_count: u32,
    /// E-graph node budget for this trajectory (how many nodes the saturator may create).
    pub(crate) node_budget: u32,
    /// Epoch budget for this trajectory (max saturation epochs).
    pub(crate) epoch_budget: u32,
}

impl Default for GraphAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAccumulator {
    /// Create a zero-initialized graph accumulator.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            values: [0.0; GRAPH_ACC_DIM],
            edge_count: 0,
            node_count: 0,
            node_budget: 0,
            epoch_budget: 0,
        }
    }

    /// Reset to zero state.
    ///
    /// Budget fields are intentionally NOT reset — they are trajectory-level
    /// properties that should persist across epoch rebuilds.
    pub(crate) fn reset(&mut self) {
        self.values = [0.0; GRAPH_ACC_DIM];
        self.edge_count = 0;
        self.node_count = 0;
    }

    /// Add a single edge with depth-aware VSA encoding.
    ///
    /// Updates all three sections: marginal parent, marginal child, and
    /// VSA binding (`E[parent] ⊙ shift_by(E[child], depth % K)`).
    ///
    /// At `depth == 0` the shift is the identity (root edges), `depth == 1`
    /// shifts by 1 (matching the original `shift1` behavior), `depth == 2`
    /// shifts by 2, etc. This encodes hierarchical position into the binding
    /// without any extra parameters.
    #[inline]
    pub(crate) fn add_edge_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: usize,
    ) {
        let p = emb.get(parent_op);
        let c = emb.get(child_op);
        let c_shifted = shift_by(c, depth);
        for i in 0..K {
            self.values[i] += p[i]; // marginal parent
            self.values[K + i] += c[i]; // marginal child
            self.values[2 * K + i] += p[i] * c_shifted[i]; // VSA binding
        }
        self.edge_count += 1;
    }

    /// Add a single edge with VSA encoding (backward-compatible, depth = 1).
    ///
    /// Equivalent to `add_edge_at_depth(emb, parent_op, child_op, 1)`.
    /// Preserves the original `shift1` behavior for callers that do not
    /// track depth.
    #[inline]
    pub(crate) fn add_edge(&mut self, emb: &OpEmbeddings, parent_op: OpKind, child_op: OpKind) {
        self.add_edge_at_depth(emb, parent_op, child_op, 1);
    }

    /// Add a leaf node (Var/Const) — no edges, just increment node count.
    pub(crate) fn add_leaf(&mut self) {
        self.node_count += 1;
    }

    /// Add an Op node and all its edges to children, with depth-aware VSA.
    ///
    /// Emits one `add_edge_at_depth` per child and increments `node_count`
    /// once.  The `depth` parameter is the depth of `op` in the expression
    /// tree (0 = root).
    pub(crate) fn add_op_node_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        op: OpKind,
        child_ops: &[OpKind],
        depth: usize,
    ) {
        for &child_op in child_ops {
            self.add_edge_at_depth(emb, op, child_op, depth);
        }
        self.node_count += 1;
    }

    /// Add an Op node and all its edges to children (backward-compatible, depth = 1).
    ///
    /// Equivalent to `add_op_node_at_depth(emb, op, child_ops, 1)`.
    pub(crate) fn add_op_node(&mut self, emb: &OpEmbeddings, op: OpKind, child_ops: &[OpKind]) {
        self.add_op_node_at_depth(emb, op, child_ops, 1);
    }

    // ========== Incremental Removal (inverse of addition) ==========

    /// Remove a single edge with depth-aware VSA encoding — the exact
    /// inverse of [`Self::add_edge_at_depth`].
    ///
    /// Subtracts (instead of adds) the parent, child, and VSA binding
    /// contributions from each section. Decrements `edge_count` via
    /// saturating subtraction so underflow clamps to zero rather than
    /// wrapping.
    ///
    /// # Contract
    ///
    /// Callers must only remove edges that were previously added at the
    /// same `depth`. Removing an edge that was never added will corrupt
    /// the accumulator values (negative contributions) and is a logic
    /// error.
    #[inline]
    pub(crate) fn remove_edge_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: usize,
    ) {
        let p = emb.get(parent_op);
        let c = emb.get(child_op);
        let c_shifted = shift_by(c, depth);
        for i in 0..K {
            self.values[i] -= p[i]; // marginal parent
            self.values[K + i] -= c[i]; // marginal child
            self.values[2 * K + i] -= p[i] * c_shifted[i]; // VSA binding
        }
        self.edge_count = self.edge_count.saturating_sub(1);
    }

    /// Remove a single edge (backward-compatible, depth = 1) — the exact
    /// inverse of [`Self::add_edge`].
    ///
    /// Equivalent to `remove_edge_at_depth(emb, parent_op, child_op, 1)`.
    #[inline]
    pub(crate) fn remove_edge(&mut self, emb: &OpEmbeddings, parent_op: OpKind, child_op: OpKind) {
        self.remove_edge_at_depth(emb, parent_op, child_op, 1);
    }

    /// Remove an Op node and all its edges to children with depth-aware
    /// VSA — the exact inverse of [`Self::add_op_node_at_depth`].
    ///
    /// Calls `remove_edge_at_depth` for each child and decrements
    /// `node_count`.
    pub(crate) fn remove_op_node_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        op: OpKind,
        child_ops: &[OpKind],
        depth: usize,
    ) {
        for &child_op in child_ops {
            self.remove_edge_at_depth(emb, op, child_op, depth);
        }
        self.node_count = self.node_count.saturating_sub(1);
    }

    /// Remove an Op node and all its edges (backward-compatible, depth = 1)
    /// — the exact inverse of [`Self::add_op_node`].
    ///
    /// Equivalent to `remove_op_node_at_depth(emb, op, child_ops, 1)`.
    pub(crate) fn remove_op_node(&mut self, emb: &OpEmbeddings, op: OpKind, child_ops: &[OpKind]) {
        self.remove_op_node_at_depth(emb, op, child_ops, 1);
    }

    /// Remove a leaf node — the exact inverse of [`Self::add_leaf`].
    ///
    /// Decrements `node_count` only (leaves contribute no edges).
    pub(crate) fn remove_leaf(&mut self) {
        self.node_count = self.node_count.saturating_sub(1);
    }

    // ========== 2-hop Message Passing (1-round GNN) ==========

    /// Add a 2-hop (grandparent→parent→child) binding to the `[3K..4K]` section.
    ///
    /// Encodes 3-node path patterns like "Mul feeding Add feeding Sqrt" using
    /// the VSA triple product `E[grandparent] ⊙ shift₁(E[parent]) ⊙ shift²(E[child])`.
    /// The shift amounts break commutativity: `A→B→C` produces a different
    /// binding than any permutation of {A, B, C}.
    ///
    /// Does NOT modify the `[0..3K]` sections or `edge_count`/`node_count` —
    /// those are maintained by `add_edge*` / `add_op_node*`.
    #[inline]
    pub(crate) fn add_2hop_edge(
        &mut self,
        emb: &OpEmbeddings,
        grandparent_op: OpKind,
        parent_op: OpKind,
        child_op: OpKind,
    ) {
        let gp = emb.get(grandparent_op);
        let p = shift1(emb.get(parent_op));
        let c = shift_by(emb.get(child_op), 2);
        for i in 0..K {
            self.values[3 * K + i] += gp[i] * p[i] * c[i];
        }
    }

    /// Remove a 2-hop (grandparent→parent→child) binding — the exact inverse
    /// of [`Self::add_2hop_edge`].
    ///
    /// Subtracts the triple-product contribution from the `[3K..4K]` section.
    ///
    /// # Contract
    ///
    /// Callers must only remove 2-hop edges that were previously added with
    /// the same (grandparent, parent, child) triple. Removing a path that was
    /// never added will corrupt the accumulator values and is a logic error.
    #[inline]
    pub(crate) fn remove_2hop_edge(
        &mut self,
        emb: &OpEmbeddings,
        grandparent_op: OpKind,
        parent_op: OpKind,
        child_op: OpKind,
    ) {
        let gp = emb.get(grandparent_op);
        let p = shift1(emb.get(parent_op));
        let c = shift_by(emb.get(child_op), 2);
        for i in 0..K {
            self.values[3 * K + i] -= gp[i] * p[i] * c[i];
        }
    }

    /// Return a copy with each of the four sections independently L2-normalized.
    ///
    /// Raw sums grow proportionally to edge count, so a 200-edge graph has
    /// values ~20x larger than a 10-edge graph.  Normalizing makes the
    /// embedding scale-invariant: small rewrites on large graphs become visible
    /// instead of being swamped by magnitude.
    ///
    /// Each section is normalized independently because they represent different
    /// quantities with different natural scales:
    /// - `[0..K]`    marginal parent sums
    /// - `[K..2K]`   marginal child sums
    /// - `[2K..3K]`  1-hop VSA binding sums
    /// - `[3K..4K]`  2-hop VSA binding sums
    ///
    /// Scalar fields (`edge_count`, `node_count`, etc.) are copied as-is.
    ///
    /// A zero-norm section (no edges accumulated) is left as all-zeros rather
    /// than producing NaN/Inf.
    #[must_use]
    pub(crate) fn normalized(&self) -> Self {
        let mut out = self.clone();
        out.normalize_in_place();
        out
    }

    /// L2-normalize each of the four sections in place.
    ///
    /// See [`Self::normalized`] for rationale.
    pub(crate) fn normalize_in_place(&mut self) {
        l2_normalize_section(&mut self.values, 0, K);
        l2_normalize_section(&mut self.values, K, 2 * K);
        l2_normalize_section(&mut self.values, 2 * K, 3 * K);
        l2_normalize_section(&mut self.values, 3 * K, 4 * K);
    }
}

/// L2-normalize a contiguous slice `values[start..end]` in place.
///
/// If the section norm is zero (or negligibly small), it is left untouched
/// to avoid division by zero.
fn l2_normalize_section(values: &mut [f32], start: usize, end: usize) {
    let mut sum_sq: f32 = 0.0;
    for i in start..end {
        sum_sq += values[i] * values[i];
    }
    let norm = sqrtf(sum_sq);
    // Guard: skip normalization for zero/near-zero sections to avoid NaN/Inf.
    if norm < 1e-12 {
        return;
    }
    let inv_norm = 1.0 / norm;
    for i in start..end {
        values[i] *= inv_norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libm::sqrtf;

    // ========================================================================
    // GraphAccumulator normalization tests
    // ========================================================================

    #[test]
    fn graph_acc_normalize_unit_norm_per_section() {
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        // Build a non-trivial accumulator: several edges
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);
        gacc.add_edge(&emb, OpKind::Add, OpKind::Var);

        let normed = gacc.normalized();

        // Each section should have L2 norm ~1.0
        let section_norm = |start: usize, end: usize| -> f32 {
            let sum_sq: f32 = normed.values[start..end].iter().map(|v| v * v).sum();
            sqrtf(sum_sq)
        };
        let eps = 1e-5;
        assert!(
            (section_norm(0, K) - 1.0).abs() < eps,
            "parent section norm should be 1.0"
        );
        assert!(
            (section_norm(K, 2 * K) - 1.0).abs() < eps,
            "child section norm should be 1.0"
        );
        assert!(
            (section_norm(2 * K, 3 * K) - 1.0).abs() < eps,
            "1-hop VSA section norm should be 1.0"
        );
        // 2-hop section is zero (no 2-hop edges added) — normalization should leave it as zeros.
        let hop2_norm = section_norm(3 * K, 4 * K);
        assert!(
            hop2_norm < eps,
            "2-hop VSA section should be zero when no 2-hop edges added, got {hop2_norm}"
        );
    }

    #[test]
    fn graph_acc_normalize_scalars_preserved() {
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_leaf();
        gacc.add_leaf();
        gacc.node_budget = 100;
        gacc.epoch_budget = 50;

        let normed = gacc.normalized();
        assert_eq!(
            normed.edge_count, gacc.edge_count,
            "edge_count must be preserved"
        );
        assert_eq!(
            normed.node_count, gacc.node_count,
            "node_count must be preserved"
        );
        assert_eq!(
            normed.node_budget, gacc.node_budget,
            "node_budget must be preserved"
        );
        assert_eq!(
            normed.epoch_budget, gacc.epoch_budget,
            "epoch_budget must be preserved"
        );
    }

    #[test]
    fn graph_acc_normalize_zero_is_safe() {
        // A fresh (zero) accumulator should normalize without NaN/Inf.
        let gacc = GraphAccumulator::new();
        let normed = gacc.normalized();
        for (i, &v) in normed.values.iter().enumerate() {
            assert!(v.is_finite(), "values[{i}] must be finite, got {v}");
            assert_eq!(
                v, 0.0,
                "zero accumulator must stay zero after normalization"
            );
        }
    }

    #[test]
    fn graph_acc_normalize_in_place_matches_normalized() {
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);

        let copy_normed = gacc.normalized();

        let mut in_place = gacc.clone();
        in_place.normalize_in_place();

        for i in 0..GRAPH_ACC_DIM {
            assert!(
                (copy_normed.values[i] - in_place.values[i]).abs() < 1e-9,
                "values[{i}] mismatch: normalized()={} vs normalize_in_place()={}",
                copy_normed.values[i],
                in_place.values[i]
            );
        }
    }

    #[test]
    fn graph_acc_normalize_scale_invariance() {
        // Doubling all edges (adding each edge twice) should yield the same
        // normalized vector, proving scale invariance.
        let emb = OpEmbeddings::new_random(42);

        let mut small = GraphAccumulator::new();
        small.add_edge(&emb, OpKind::Add, OpKind::Mul);
        small.add_edge(&emb, OpKind::Mul, OpKind::Var);

        let mut large = GraphAccumulator::new();
        for _ in 0..10 {
            large.add_edge(&emb, OpKind::Add, OpKind::Mul);
            large.add_edge(&emb, OpKind::Mul, OpKind::Var);
        }

        let small_n = small.normalized();
        let large_n = large.normalized();

        for i in 0..GRAPH_ACC_DIM {
            assert!(
                (small_n.values[i] - large_n.values[i]).abs() < 1e-5,
                "normalized vectors should match regardless of scale: values[{i}] small={} large={}",
                small_n.values[i],
                large_n.values[i]
            );
        }
    }

    #[test]
    fn graph_acc_normalize_idempotent() {
        // Normalizing twice should produce the same result.
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);

        let once = gacc.normalized();
        let twice = once.normalized();

        for i in 0..GRAPH_ACC_DIM {
            assert!(
                (once.values[i] - twice.values[i]).abs() < 1e-6,
                "normalization must be idempotent: values[{i}] once={} twice={}",
                once.values[i],
                twice.values[i]
            );
        }
    }

    // ========================================================================
    // GraphAccumulator incremental remove tests
    // ========================================================================

    #[test]
    fn graph_acc_remove_leaf_saturates_at_zero() {
        let mut acc = GraphAccumulator::new();
        acc.remove_leaf();
        assert_eq!(acc.node_count, 0, "node_count must not underflow");
    }
}
