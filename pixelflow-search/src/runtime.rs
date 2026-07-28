//! E-graph optimization for runtime-built kernels.
//!
//! `kernel!`/`kernel_jit!` already run full saturation at macro-expansion
//! time: `pixelflow-compiler::optimize` builds an e-graph from the parsed
//! AST, saturates, and extracts before ever touching an
//! [`ExprArena`](pixelflow_ir::ExprArena). Anything stamped by those macros
//! reaches `pixelflow_ir::jit_cache` already optimized.
//!
//! `Kernel` values composed directly at runtime — `Kernel::over`, `.at()`,
//! `.select()`, arithmetic — never go through that macro, so their arenas hit
//! [`pixelflow_ir::jit_cache::compile_cached`]/`compile_collapse_cached` raw:
//! no CSE, no FMA fusion, no algebraic simplification. [`optimize_runtime_arena`]
//! is the same pipeline applied to an arena directly, for exactly that gap —
//! today's highest-volume instance is the font glyph bake
//! (`pixelflow-graphics`'s `Font::glyph_kernel_scaled`, cached per
//! `(codepoint, size, density)` bucket).
//!
//! `pixelflow-ir` itself must stay free of a `pixelflow-search` dependency
//! (the suckless constraint from
//! docs/plans/2026-07-20-kernel-unification.md), so this cannot live inside
//! `jit_cache`. Callers that want optimized runtime kernels — today,
//! `pixelflow-core`'s `Lattice::bake` — call this function before handing the
//! arena to `jit_cache`.

use crate::egraph::{
    EClassId, EGraph, ENode, all_rules, choices_to_arena, config_for_node_count,
    env_extraction_policy,
};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use std::collections::HashMap;

/// Optimize a runtime-built arena via bounded e-graph saturation, using the
/// same rule set and extraction policy (static latency-prior by default,
/// `PIXELFLOW_NNUE_WEIGHTS` opt-in) as the `kernel!`/`kernel_jit!` macros.
///
/// Returns `None`, unchanged, when the subgraph reachable from `root`
/// contains a construct the e-graph doesn't model:
///
/// - `Buffer`/`Gather`/`RawGather` (bound-memory reads) — no rewrite rule
///   reasons about memory, and the e-graph would just be dead weight around
///   an opaque leaf.
/// - `Nary` (the `Reduce` binder, from `Kernel::over`) — rewriting under a
///   binder is unsound without binder-aware rules (none exist yet); this is
///   a documented follow-up, not silently wrong output.
/// - `Param` — a `pixelflow-compiler` macro-parameter slot that should never
///   reach a runtime-built `Kernel` in the first place.
///
/// Callers compile the original arena unchanged in that case — `optimize_runtime_arena`
/// is strictly an optimization, never required for correctness.
#[must_use]
pub fn optimize_runtime_arena(arena: &ExprArena, root: ExprId) -> Option<(ExprArena, ExprId)> {
    let mut egraph = EGraph::with_rules(all_rules());
    let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
    let root_class = arena_to_egraph(arena, root, &mut egraph, &mut memo)?;

    let node_count = reachable_count(arena, root);
    let config = config_for_node_count(node_count);
    crate::egraph::saturate_with_full_budget(
        &mut egraph,
        config.max_iterations,
        config.max_classes,
        config.hard_timeout,
    );

    let policy = env_extraction_policy();
    let choices = policy.choices(&egraph, root_class);
    Some(choices_to_arena(&egraph, root_class, &choices))
}

/// Insert the subgraph reachable from `id` into `egraph`, memoized by
/// `ExprId` (on top of the e-graph's own hash-consing by node shape) so a
/// DAG-shared arena is walked once per node, not once per reference.
///
/// Returns `None` — aborting the whole conversion — the moment it meets an
/// op [`crate::egraph::ops::op_from_kind`] doesn't model, or a `Param`.
/// Iterative (explicit stack), matching [`choices_to_arena`]'s style in the
/// same crate: arena depths are unbounded in principle (Dwrt chain-rule
/// expansion, deep composition), so this must not blow the Rust stack.
fn arena_to_egraph(
    arena: &ExprArena,
    root: ExprId,
    egraph: &mut EGraph,
    memo: &mut HashMap<ExprId, EClassId>,
) -> Option<EClassId> {
    enum Task {
        Visit(ExprId),
        Complete(ExprId),
    }

    let mut task_stack = vec![Task::Visit(root)];
    let mut result_stack: Vec<EClassId> = Vec::new();

    while let Some(task) = task_stack.pop() {
        match task {
            Task::Visit(id) => {
                if let Some(&class) = memo.get(&id) {
                    result_stack.push(class);
                    continue;
                }
                match arena.node(id) {
                    &ExprNode::Var(idx) => {
                        let class = egraph.add(ENode::Var(idx));
                        memo.insert(id, class);
                        result_stack.push(class);
                    }
                    &ExprNode::Const(val) => {
                        let class = egraph.add(ENode::constant(val));
                        memo.insert(id, class);
                        result_stack.push(class);
                    }
                    ExprNode::Param(_) | ExprNode::Buffer(_) => return None,
                    &ExprNode::Unary(kind, a) => {
                        crate::egraph::ops::op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(a));
                    }
                    &ExprNode::Binary(kind, a, b) => {
                        crate::egraph::ops::op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(b));
                        task_stack.push(Task::Visit(a));
                    }
                    &ExprNode::Ternary(kind, a, b, c) => {
                        crate::egraph::ops::op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(c));
                        task_stack.push(Task::Visit(b));
                        task_stack.push(Task::Visit(a));
                    }
                    // Reduce (the only Nary op) is a binder: its "children"
                    // include the bound index and static extent alongside
                    // the body, which rewriting can't treat as ordinary
                    // operands without capturing the binder. Bail out.
                    ExprNode::Nary(..) => return None,
                }
            }
            Task::Complete(id) => {
                if let Some(&class) = memo.get(&id) {
                    result_stack.push(class);
                    continue;
                }
                let (kind, arity) = match arena.node(id) {
                    &ExprNode::Unary(kind, _) => (kind, 1),
                    &ExprNode::Binary(kind, _, _) => (kind, 2),
                    &ExprNode::Ternary(kind, _, _, _) => (kind, 3),
                    _ => unreachable!("Complete scheduled only for Unary/Binary/Ternary"),
                };
                let op = crate::egraph::ops::op_from_kind(kind)
                    .expect("op_from_kind already checked in Visit");
                let start = result_stack.len() - arity;
                let children: Vec<EClassId> = result_stack.drain(start..).collect();
                let class = egraph.add(ENode::Op { op, children });
                memo.insert(id, class);
                result_stack.push(class);
            }
        }
    }

    result_stack.pop()
}

/// Count nodes reachable from `root` — a rough size measure for
/// [`config_for_node_count`], mirroring what `pixelflow_ir::jit_cache`'s
/// canonical-key reachability walk already does for the same arena.
fn reachable_count(arena: &ExprArena, root: ExprId) -> usize {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut count = 0usize;
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        count += 1;
        stack.extend(arena.children(id));
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::BufferDecl;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval_scalar;

    /// Every optimization must preserve the arena's denoted value, over a
    /// spread of coordinates — the load-bearing property. Anything that ever
    /// broke this would silently mis-render, not fail loudly.
    fn assert_semantics_preserved(arena: &ExprArena, root: ExprId, optimized: &(ExprArena, ExprId)) {
        let (opt_arena, opt_root) = optimized;
        let coords: &[(f32, f32, f32, f32)] = &[
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0, 4.0),
            (-1.5, 0.5, 2.25, -3.0),
            (3.7, -4.1, 0.0, 1.0),
        ];
        for &(x, y, z, w) in coords {
            let want = eval_scalar(arena, root, &[x, y, z, w], &BindingTable::empty());
            let got = eval_scalar(opt_arena, *opt_root, &[x, y, z, w], &BindingTable::empty());
            assert!(
                (want - got).abs() < 1e-3 || (want.is_nan() && got.is_nan()),
                "optimize_runtime_arena changed semantics at ({x},{y},{z},{w}): {want} != {got}"
            );
        }
    }

    #[test]
    fn fma_fusion_applies_to_a_runtime_arena() {
        // a*b + c, built directly as an arena (no macro involved) — exactly
        // the shape Kernel::over/.at() composition produces at runtime.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let z = a.push_var(2);
        let mul = a.push_binary(OpKind::Mul, x, y);
        let root = a.push_binary(OpKind::Add, mul, z);

        let (opt_arena, opt_root) = optimize_runtime_arena(&a, root)
            .expect("pure arithmetic arena must optimize");

        assert_semantics_preserved(&a, root, &(opt_arena.clone(), opt_root));
        assert!(
            matches!(opt_arena.node(opt_root), ExprNode::Ternary(OpKind::MulAdd, ..)),
            "expected a*b+c fused to MulAdd, got {:?}",
            opt_arena.node(opt_root)
        );
    }

    #[test]
    fn shared_subexpressions_stay_shared_and_correct() {
        // sin(X)*sin(X) + sin(X): the repeated sin(X) subtree must convert
        // to the e-graph once (via the ExprId memo) and extract back
        // correctly regardless of how many times it's referenced.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sin, x);
        let sq = a.push_binary(OpKind::Mul, s, s);
        let root = a.push_binary(OpKind::Add, sq, s);

        let (opt_arena, opt_root) =
            optimize_runtime_arena(&a, root).expect("trig arena must optimize");
        assert_semantics_preserved(&a, root, &(opt_arena, opt_root));
    }

    #[test]
    fn dwrt_derivative_kernel_optimizes() {
        // The font-coverage shape: X - ((Y - y0) * k + x0), differentiated.
        // Dwrt is representable in the e-graph (ChainRule reduces it), so
        // this must NOT bail out.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let y0 = a.push_const(0.3);
        let k = a.push_const(0.7);
        let x0 = a.push_const(-0.2);
        let y_sub = a.push_binary(OpKind::Sub, y, y0);
        let scaled = a.push_binary(OpKind::Mul, y_sub, k);
        let line = a.push_binary(OpKind::Add, scaled, x0);
        let d = a.push_binary(OpKind::Sub, x, line);
        let var_x = a.push_const(0.0); // Dwrt's second child is the var index, wrt X (0)
        let dx = a.push_binary(OpKind::Dwrt, d, var_x);

        let (opt_arena, opt_root) =
            optimize_runtime_arena(&a, dx).expect("Dwrt-bearing arena must optimize");

        // eval_scalar refuses a raw Dwrt (the interpreter evaluates the
        // post-calculus program, same as the JIT) — lower both sides before
        // comparing, cross-checking the e-graph's ChainRule reduction
        // against the dedicated lower_dwrt pass.
        use pixelflow_ir::backend::emit::lowering::lower_dwrt_owned;
        let (want_arena, want_root) = lower_dwrt_owned(&a, dx).expect("lower original");
        let (got_arena, got_root) = lower_dwrt_owned(&opt_arena, opt_root).expect("lower optimized");
        assert_semantics_preserved(&want_arena, want_root, &(got_arena, got_root));
    }

    #[test]
    fn buffer_gather_bails_out() {
        // BilinearSampler-shaped: a Gather anywhere reachable from root means
        // "don't touch this" — return None rather than mis-optimize memory
        // ops the e-graph doesn't model.
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            width: 4,
            height: 4,
        });
        let x = a.push_var(0);
        let y = a.push_var(1);
        let g = a.push_gather(buf, x, y);
        let one = a.push_const(1.0);
        let root = a.push_binary(OpKind::Add, g, one);

        assert!(
            optimize_runtime_arena(&a, root).is_none(),
            "an arena reaching a Buffer/Gather must not be optimized"
        );
    }

    #[test]
    fn reduce_bails_out() {
        // Kernel::over-shaped: a Reduce (Nary) anywhere reachable means the
        // binder is present — bail rather than rewrite under it unsoundly.
        let mut a = ExprArena::new();
        let i = a.push_var(4); // reduction index slot
        let body = a.push_binary(OpKind::Mul, i, i);
        let root = a.push_reduce(OpKind::Add, 4, 4, body);

        assert!(
            optimize_runtime_arena(&a, root).is_none(),
            "an arena containing a Reduce must not be optimized"
        );
    }

    #[test]
    fn constant_folds_through_bounded_saturation() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let one = a.push_const(1.0);
        let zero = a.push_const(0.0);
        let plus_zero = a.push_binary(OpKind::Add, x, zero);
        let times_one = a.push_binary(OpKind::Mul, plus_zero, one);
        let root = times_one;

        let (opt_arena, opt_root) =
            optimize_runtime_arena(&a, root).expect("identity arena must optimize");
        assert_semantics_preserved(&a, root, &(opt_arena.clone(), opt_root));
        // x + 0.0, then * 1.0 should collapse to bare X.
        assert!(
            matches!(opt_arena.node(opt_root), ExprNode::Var(0)),
            "expected identities to collapse to bare X, got {:?}",
            opt_arena.node(opt_root)
        );
    }
}
