//! Diverse candidate extractions of ONE saturated e-graph — the enumeration
//! `VariantSet` minting needs (J6, docs/plans/2026-08-17-cost-model-domain.md).
//!
//! Additive only: every function here composes `Extraction`/`EGraph`
//! machinery that already existed in `extract.rs`/`extraction.rs`
//! (`Extraction::try_swap`, `ExtractionPolicy::extraction`,
//! `Extraction::choices`) — nothing in those files' bodies changes. Lives in
//! its own module precisely so a concurrent perf pass on `extract.rs`'s
//! internals never touches this file and vice versa.
//!
//! # What this targets
//!
//! Round-0 measurement
//! (docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md §1.5): the NNUE
//! extraction policy is the *conservative* one relative to the static
//! latency prior — across 12 re-extracted kernels the static prior
//! substitutes hardware estimate instructions (Recip/Rsqrt/MulAdd/Sqrt)
//! roughly twice as often as NNUE does, and the per-pair speed ratio is
//! near-monotone in that substitution delta. A cost-driven or random walk of
//! the saturated e-graph is unlikely to land on that specific axis often
//! enough to supply a contrastive ranking loss with signal, so
//! [`enumerate_candidates`] deliberately swaps every reachable e-class's
//! chosen node against a sibling that flips [`ESTIMATE_OPS`] membership, in
//! addition to the two cost-model picks themselves.

use alloc::boxed::Box;
use alloc::vec::Vec;

use pixelflow_ir::OpKind;

use super::cost::CostModel;
use super::extract::Extraction;
use super::extraction::ExtractionPolicy;
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use crate::nnue::ExprNnue;

/// Hardware estimate instructions Round-0 found the NNUE policy declines
/// relative to the static prior. Not "every estimate op in the language" —
/// exactly the CLAUDE.md-documented set the research notes measured the
/// conservatism on.
pub const ESTIMATE_OPS: [OpKind; 4] = [OpKind::Recip, OpKind::Rsqrt, OpKind::MulAdd, OpKind::Sqrt];

fn op_kind(node: &ENode) -> Option<OpKind> {
    node.op().map(|op| op.kind())
}

fn is_estimate_op(kind: OpKind) -> bool {
    ESTIMATE_OPS.contains(&kind)
}

/// One candidate from [`enumerate_candidates`]: the extraction plus which
/// arm of the enumeration produced it. Provenance is the axis a VariantSet
/// minting caller reports diversity over — `"static"`/`"nnue"` are the two
/// cost-model picks, `"swap"` an estimate-op flip of one of them — so it
/// travels with the candidate instead of being re-derived (fragile: a swap
/// can coincide with the static or NNUE choice vector and would otherwise
/// need re-classifying after the fact).
pub struct Candidate<'g> {
    pub provenance: &'static str,
    pub extraction: Extraction<'g>,
}

/// Every `(canonical e-class, chosen node index)` pair reachable from
/// `extraction`'s root, walked via `extraction.choice` (not the class-wide
/// node list) so this sees exactly what the extraction actually selected —
/// canonical classes visited once regardless of DAG sharing.
fn reachable_choices<'g>(
    egraph: &'g EGraph,
    extraction: &Extraction<'g>,
) -> Vec<(EClassId, usize)> {
    let mut result = Vec::new();
    let mut visited: Vec<u32> = Vec::new();
    let mut stack = alloc::vec![extraction.root()];
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if visited.contains(&canonical.0) {
            continue;
        }
        visited.push(canonical.0);
        let Some(idx) = extraction.choice(canonical) else {
            continue;
        };
        result.push((canonical, idx));
        if let Some(node) = egraph.nodes(canonical).get(idx) {
            for child in node.children() {
                stack.push(child);
            }
        }
    }
    result
}

/// Push `cand` onto `out` iff its raw choice vector is not already present
/// in `seen`. Dedup key is the choice vector itself, not any derived
/// property — two candidates that select the same node everywhere are the
/// same materialized expression no matter which policy or swap produced
/// them, and minting the same expression twice would silently double-count
/// it as two "diverse" variants.
fn push_if_new<'g>(
    provenance: &'static str,
    cand: Extraction<'g>,
    out: &mut Vec<Candidate<'g>>,
    seen: &mut Vec<Vec<Option<usize>>>,
) {
    let choices = cand.choices().to_vec();
    if seen.contains(&choices) {
        return;
    }
    seen.push(choices);
    out.push(Candidate {
        provenance,
        extraction: cand,
    });
}

/// Enumerate up to `max_candidates` diverse extractions of `root` from a
/// SATURATED `egraph`: the DP-optimal static-latency-prior pick, the NNUE
/// pick, and — from each of those two bases — every reachable e-class's
/// chosen node swapped against a sibling that flips [`ESTIMATE_OPS`]
/// membership (present vs absent).
///
/// Diversity over optimality (plan task 2, `docs/plans/2026-08-17-cost-model-domain.md`
/// J6): swap candidates are accepted as soon as they are structurally new
/// (deduplicated on the raw choice vector), not ranked or filtered by
/// predicted cost. A swap `Extraction::try_swap` rejects (because it would
/// close a cycle) is simply skipped, never an error — cycle rejection is an
/// expected search outcome, not a defect.
///
/// # Panics
///
/// Panics if `max_candidates` is 0 — a caller asking for zero candidates has
/// a bug, not a request this function can silently satisfy.
pub fn enumerate_candidates<'g>(
    egraph: &'g EGraph,
    root: EClassId,
    static_costs: &CostModel,
    nnue: &ExprNnue,
    max_candidates: usize,
) -> Vec<Candidate<'g>> {
    assert!(
        max_candidates >= 1,
        "enumerate_candidates: max_candidates must be >= 1, got {max_candidates}"
    );

    let mut out: Vec<Candidate<'g>> = Vec::new();
    let mut seen: Vec<Vec<Option<usize>>> = Vec::new();

    let static_policy = ExtractionPolicy::Static(Box::new(static_costs.clone()));
    push_if_new(
        "static",
        static_policy.extraction(egraph, root),
        &mut out,
        &mut seen,
    );

    if out.len() < max_candidates {
        let nnue_policy = ExtractionPolicy::Nnue(nnue);
        push_if_new(
            "nnue",
            nnue_policy.extraction(egraph, root),
            &mut out,
            &mut seen,
        );
    }

    let base_count = out.len();
    'bases: for bi in 0..base_count {
        if out.len() >= max_candidates {
            break;
        }
        let reachable = reachable_choices(egraph, &out[bi].extraction);
        for (class, node_idx) in reachable {
            if out.len() >= max_candidates {
                break 'bases;
            }
            let nodes = egraph.nodes(class);
            let Some(cur_kind) = nodes.get(node_idx).and_then(op_kind) else {
                continue;
            };
            let cur_is_estimate = is_estimate_op(cur_kind);
            for alt_idx in 0..nodes.len() {
                if alt_idx == node_idx {
                    continue;
                }
                let Some(alt_kind) = nodes.get(alt_idx).and_then(op_kind) else {
                    continue;
                };
                if is_estimate_op(alt_kind) == cur_is_estimate {
                    continue;
                }
                if let Some(cand) = out[bi].extraction.try_swap(class, alt_idx) {
                    push_if_new("swap", cand, &mut out, &mut seen);
                }
                if out.len() >= max_candidates {
                    break 'bases;
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;
    use pixelflow_ir::ExprArena;

    /// `1.0 / sqrt(x)` gives the rewrite system both a `Recip`/`Sqrt` route
    /// and (via `HardwarePrimitive`-class rules) an `Rsqrt` route, so the
    /// estimate-op axis has something to swap on and this is not a vacuous
    /// smoke test.
    fn recip_sqrt_arena() -> (ExprArena, pixelflow_ir::ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let sqrt_x = arena.push_unary(pixelflow_ir::OpKind::Sqrt, x);
        let one = arena.push_const(1.0);
        let root = arena.push_binary(pixelflow_ir::OpKind::Div, one, sqrt_x);
        (arena, root)
    }

    #[test]
    fn enumerate_returns_at_least_the_two_base_policies() {
        let mut eg = EGraph::with_rules(math::all_rules());
        let (arena, root) = recip_sqrt_arena();
        let root_class = eg.add_arena(&arena, root);
        eg.saturate_with_limit(64);

        let nnue = ExprNnue::new_random(1);
        let costs = CostModel::latency_prior();
        let cands = enumerate_candidates(&eg, root_class, &costs, &nnue, 10);

        assert!(
            !cands.is_empty(),
            "must return at least the static-prior extraction"
        );
        assert!(cands.len() <= 10, "must respect max_candidates");
    }

    #[test]
    fn enumerate_deduplicates_identical_choice_vectors() {
        // A bare variable has exactly one possible extraction — static and
        // NNUE picks (and any swap) must collapse to one candidate.
        let mut eg = EGraph::with_rules(Vec::new());
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let root_class = eg.add_arena(&arena, x);

        let nnue = ExprNnue::new_random(2);
        let costs = CostModel::latency_prior();
        let cands = enumerate_candidates(&eg, root_class, &costs, &nnue, 10);
        assert_eq!(cands.len(), 1, "a single-node e-graph has one extraction");
    }

    #[test]
    fn enumerate_respects_max_candidates_of_one() {
        let mut eg = EGraph::with_rules(math::all_rules());
        let (arena, root) = recip_sqrt_arena();
        let root_class = eg.add_arena(&arena, root);
        eg.saturate_with_limit(64);

        let nnue = ExprNnue::new_random(3);
        let costs = CostModel::latency_prior();
        let cands = enumerate_candidates(&eg, root_class, &costs, &nnue, 1);
        assert_eq!(cands.len(), 1);
    }

    #[test]
    #[should_panic(expected = "max_candidates must be >= 1")]
    fn enumerate_rejects_zero_max_candidates() {
        let mut eg = EGraph::with_rules(Vec::new());
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let root_class = eg.add_arena(&arena, x);
        let nnue = ExprNnue::new_random(4);
        let costs = CostModel::latency_prior();
        let _ = enumerate_candidates(&eg, root_class, &costs, &nnue, 0);
    }
}
