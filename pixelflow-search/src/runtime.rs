//! E-graph optimization for runtime-built kernels.
//!
//! `kernel!`/`kernel_jit!` already run full saturation at macro-expansion
//! time: `pixelflow-compiler::optimize` builds an e-graph from the parsed
//! AST, saturates, and extracts before ever touching an
//! [`ExprArena`](pixelflow_ir::ExprArena). Anything stamped by those macros
//! reaches `pixelflow_codegen::jit_cache` already optimized.
//!
//! `Kernel` values composed directly at runtime — `Kernel::over`, `.at()`,
//! `.select()`, arithmetic — never go through that macro, so their arenas hit
//! [`pixelflow_codegen::jit_cache::compile`] raw:
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
    EClassId, EGraph, ENode, Op, all_rules, choices_to_arena, config_for_node_count,
    env_extraction_policy,
};
use pixelflow_ir::LatticeShape;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, ExprArena, ExprId, ExprNode};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Optimize a runtime-built arena via bounded e-graph saturation, using the
/// same rule set and extraction policy (the static latency prior, through
/// `env_extraction_policy`) as the `kernel!`/`kernel_jit!` macros.
///
/// `Buffer`/`Gather` (bound-memory reads) are representable: they enter the
/// e-graph as opaque structure — no rewrite rule can name them, so their
/// gain is hash-consing CSE (splice-duplicated sampler subtrees collapse to
/// one node) plus ordinary rewriting of the coordinate arithmetic that feeds
/// them. Extraction redeclares each distinct `BufferIdentity` once in the
/// output arena.
///
/// Returns `None`, unchanged, when the subgraph reachable from `root`
/// contains a construct the e-graph doesn't model:
///
/// - `RawGather` — produced by lowering, after the e-graph's place in the
///   pipeline; reaching one here means the arena is already lowered.
/// - `Nary` other than `Reduce` (`Tuple`) — not modelled. `Reduce` itself
///   is unrolled first (`passes::expand_reduce`, the same unroll `legalize`
///   performs later): the arena the e-graph sees is binder-free, so factoring
///   across the unrolled terms is ordinary rewriting rather than rewriting
///   under a binder.
/// - `Param` — a `pixelflow-compiler` macro-parameter slot that should never
///   reach a runtime-built `Kernel` in the first place.
///
/// Callers compile the original arena unchanged in that case — `optimize_runtime_arena`
/// is strictly an optimization, never required for correctness.
///
/// `shape` is the extent of the lattice the kernel is compiled for. It is
/// part of the cache key and is consulted by no rewrite yet (stage 0′ of
/// `docs/plans/2026-09-01-loop-aware-codegen.md`), so that extent-weighted
/// extraction (stage 1) is a policy change here and a signature change
/// nowhere. Saturation does not depend on it; when stage 1 lands, this cache
/// should hold the saturated e-graph per structure and extract per extent.
///
/// Cached by the structural shape of the reachable subgraph (mirroring
/// `pixelflow_codegen::jit_cache`'s own canonical-key cache): a caller that bakes
/// the same `Kernel` across many frames — every glyph, on the common path
/// through `GlyphCache` — pays saturation once, not once per bake. Skipping
/// this cache would make `optimize_runtime_arena` slower than not optimizing
/// at all for any repeatedly-baked kernel, since the JIT compile it feeds is
/// itself cached downstream.
///
/// The cached value is `Arc`-wrapped for the same reason `jit_cache` hands
/// back `Arc<JitManifold>` rather than owned code: a hit must be an atomic
/// refcount bump, not a deep clone of the (potentially large — real glyph
/// arenas run to thousands of nodes once construction garbage is counted)
/// optimized `ExprArena`. Returning an owned tuple here would silently
/// reintroduce a per-call cost the cache exists to eliminate.
#[must_use]
pub fn optimize_runtime_arena(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Option<Arc<(ExprArena, ExprId)>> {
    static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Option<Arc<(ExprArena, ExprId)>>>>> =
        OnceLock::new();

    // Buffer-bearing arenas bypass the cache entirely: `BufferIdentity` is
    // process-unique and minted per construction, so two compiles never
    // share a key — every lookup would miss while every insert stayed
    // forever (the cache is static and unbounded). A terminal resizing all
    // day would leak one full optimized arena per recompile for zero hits.
    if !arena.buffers().is_empty() {
        return optimize_runtime_arena_uncached(arena, root).map(Arc::new);
    }

    let mut key = canonical_key(arena, root);
    key.extend_from_slice(&shape.key_bytes());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .expect("optimize_runtime_arena: lock poisoned")
        .get(&key)
    {
        return hit.clone();
    }

    let result = optimize_runtime_arena_uncached(arena, root).map(Arc::new);
    cache
        .lock()
        .expect("optimize_runtime_arena: lock poisoned")
        .entry(key)
        .or_insert(result)
        .clone()
}

fn optimize_runtime_arena_uncached(arena: &ExprArena, root: ExprId) -> Option<(ExprArena, ExprId)> {
    // Resolve `Dwrt` FIRST, with the same exact symbolic pass the compile
    // entries run — then the e-graph sees pure arithmetic. Order matters
    // enormously: differentiation manufactures constants (the winding
    // kernels' `d = X − f(Y)` gives `DX(d) = 1` and, for straight edges, a
    // constant `DY(d)` — making the whole gradient magnitude `√(DX²+DY²)` a
    // compile-time number), and `ConstantFold` can only cascade over
    // constants that exist by the time saturation runs. Lowering after the
    // e-graph (the compile entries' own fallback position) leaves those
    // folds permanently on the table because nothing folds post-extraction.
    //
    // A lowering error (a genuinely non-differentiable op) bails to `None`;
    // the arena then compiles unoptimized and the compile entry's own
    // `lower_dwrt` reports the same error loudly at the right layer.
    let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(arena, root).ok()?;
    // Then unroll every `Reduce`, in `legalize`'s order. The extents are
    // static, so the binder disappears into N terms sharing their
    // index-invariant subtrees (`unroll_reduce` factors `⊕_i (f(i)·c)` as
    // `c·⊕_i f(i)` by declining to duplicate `c`), and the e-graph sees pure
    // arithmetic it can CSE and fold across the terms.
    let (arena, root) = pixelflow_ir::passes::expand_reduce_owned(&arena, root);

    let mut egraph = EGraph::with_rules(all_rules());
    let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
    let root_class = arena_to_egraph(&arena, root, &mut egraph, &mut memo)?;

    let node_count = reachable_count(&arena, root);
    let config = config_for_node_count(node_count);
    crate::egraph::saturate_with_full_budget(
        &mut egraph,
        config.max_iterations,
        config.max_classes,
        config.hard_timeout,
    );

    let policy = env_extraction_policy();
    let extraction = policy.extraction(&egraph, root_class);
    let (extracted, extracted_root) = choices_to_arena(&extraction);

    // The extracted arena declares buffers in extraction-traversal order,
    // which need not match the input's — and slot order is ABI: the JIT
    // loads slot i's base pointer from the caller's context array at i*8,
    // and callers bind in the order the arena THEY BUILT declared. A
    // different extraction (a commuted equivalent under another cost model)
    // must not silently permute their pointers. Re-splicing onto a table
    // pre-declared in input order makes the invariant structural: splice
    // dedups buffers by identity onto the existing slots.
    if arena.buffers().is_empty() {
        return Some((extracted, extracted_root));
    }
    let mut ordered = ExprArena::new();
    for decl in arena.buffers() {
        let _slot = ordered.declare_buffer(*decl);
    }
    let root = ordered.splice(&extracted, extracted_root);
    debug_assert!(
        ordered
            .buffers()
            .iter()
            .zip(arena.buffers())
            .all(|(a, b)| a.id == b.id),
        "buffer slot order must survive optimization"
    );
    Some((ordered, root))
}

/// Canonical serialization of the subgraph reachable from `root`: nodes in
/// ascending original id order (the arena is append-only, so children always
/// precede parents), child references remapped to dense indices — the same
/// shape as `pixelflow_codegen::jit_cache`'s private `canonical_key`, reimplemented
/// here since that one isn't exported and this cache's correctness condition
/// is different (it needs a key for *every* reachable node kind, including
/// `Buffer`/`Nary`, to memoize the bail-out case too, not just the
/// e-graph-representable ones).
fn canonical_key(arena: &ExprArena, root: ExprId) -> Vec<u8> {
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }

    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let mut key: Vec<u8> = Vec::with_capacity(len * 8);

    let push_id = |key: &mut Vec<u8>, dense: &[u32], id: ExprId| {
        let d = dense[id.0 as usize];
        debug_assert_ne!(d, u32::MAX, "canonical_key: child densified before parent");
        key.extend_from_slice(&d.to_le_bytes());
    };

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let id = ExprId(idx as u32);
        match arena.node(id) {
            &ExprNode::Var(i) => {
                key.push(0);
                key.push(i);
            }
            &ExprNode::Const(v) => {
                key.push(1);
                key.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            &ExprNode::Param(i) => {
                key.push(2);
                key.push(i);
            }
            &ExprNode::Buffer(b) => {
                key.push(3);
                // Key by process-unique BufferIdentity, NOT the arena-local
                // slot index: two arenas can both call their own buffer "slot
                // 0" with equal extents while naming different memory, and a
                // slot-keyed cache would hand one of them the other's
                // optimized arena — whose redeclared identity binds the wrong
                // pixels. Identity is process-local, which is exactly the
                // lifetime of this in-process cache. It has no byte accessor,
                // so serialize its (injective) Debug form.
                let BufferDecl { id, width, height } = *arena.buffer_decl(b);
                key.extend_from_slice(alloc::format!("{id:?}").as_bytes());
                key.extend_from_slice(&width.to_le_bytes());
                key.extend_from_slice(&height.to_le_bytes());
            }
            &ExprNode::Unary(op, a) => {
                key.push(4);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, a);
            }
            &ExprNode::Binary(op, a, b) => {
                key.push(5);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, a);
                push_id(&mut key, &dense, b);
            }
            &ExprNode::Ternary(op, a, b, c) => {
                key.push(6);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, a);
                push_id(&mut key, &dense, b);
                push_id(&mut key, &dense, c);
            }
            &ExprNode::Nary(op, start, n) => {
                key.push(7);
                key.extend_from_slice(&op.marshal().to_bytes());
                key.extend_from_slice(&n.to_le_bytes());
                let (s, l) = (start as usize, n as usize);
                for &child in &arena.nary_children_raw()[s..s + l] {
                    push_id(&mut key, &dense, child);
                }
            }
        }
        dense[idx] = next;
        next += 1;
    }

    key
}

// ─────────────────────────── Runtime-only mask ops ───────────────────────────
//
// `&`/`|` on comparison masks are surface-language ops (every glyph winding
// kernel's Y-range gate is `(Y >= lo) & (Y < hi)`), so runtime-built arenas
// must be representable with them present. They are deliberately NOT in
// `egraph::ops::op_from_kind`: registering them globally hands them to the
// AOT macro tier too, whose e-graph runs at macro-expansion time — BEFORE
// composition — where resolving the `Dwrt` nodes the masks travel with is
// unsound (a leaf's `DX` is 1 only until an enclosing `.at()` warp scales
// it; the fonts' density-dependent AA ramp broke exactly this way when
// these ops were briefly global). The runtime tier optimizes the final
// composed arena at bake time, where the calculus has its full context, so
// mask ops are safe — and only meaningful — here.
//
// No rewrite rule targets them; they participate as opaque structure plus
// `ConstantFold`, whose bitwise-domain exemption already models their
// all-ones/zero masks.
struct MaskAnd;
impl Op for MaskAnd {
    fn kind(&self) -> OpKind {
        OpKind::BitAnd
    }
}
struct MaskOr;
impl Op for MaskOr {
    fn kind(&self) -> OpKind {
        OpKind::BitOr
    }
}

// ─────────────────────── Runtime-only integer-domain ops ─────────────────────
//
// The packed cell-grid kernel's spine: clamp → `TruncToInt` → `Shl` →
// or-fold builds a `u32` pixel per lane, so the production frame kernel is
// unrepresentable — and therefore compiles with NO CSE across its four
// channels — unless these enter the e-graph. Runtime-tier only, for the
// same reason as the mask ops above. Opaque to TEMPLATES: no rewrite rule can
// name them (nothing here or in `op_from_kind` hands them to a template), and
// their results are bit patterns the float rule set has no semantics for.
//
// Template-opacity is NOT fold-opacity, and the distinction is load-bearing:
// `ConstantFold::apply` destructures any `ENode::Op` and reads `op.kind()`
// (`math::algebra`) — it never consults `op_from_kind`. So every op registered
// here folds, and each one needs its own answer to "does this fold agree with
// what the backends emit?" `OpKind::fold_is_platform_specific` is where that
// answer lives; being unnameable by a template guards nothing.
//
// `Shl`/`Shr` do keep `Const` shift operands, because extraction emits `Const`
// leaves verbatim — so the emitter's immediate-only contract holds. The count's
// RANGE is a separate matter, enforced where the `Const` narrows to an
// immediate (`emit::shift_immediate`) rather than assumed here.
struct IntTrunc;
impl Op for IntTrunc {
    fn kind(&self) -> OpKind {
        OpKind::TruncToInt
    }
}
struct IntFromInt;
impl Op for IntFromInt {
    fn kind(&self) -> OpKind {
        OpKind::IntToFloat
    }
}
struct IntAdd;
impl Op for IntAdd {
    fn kind(&self) -> OpKind {
        OpKind::IAdd
    }
}
struct IntShl;
impl Op for IntShl {
    fn kind(&self) -> OpKind {
        OpKind::Shl
    }
}
struct IntShr;
impl Op for IntShr {
    fn kind(&self) -> OpKind {
        OpKind::Shr
    }
}

/// Whether the runtime tier can represent `kind` in its e-graph — i.e.,
/// whether an arena containing it still optimizes rather than bailing.
/// Test hook for the representability guards; the semantics live in
/// [`runtime_op_from_kind`].
#[must_use]
pub fn is_egraph_representable(kind: OpKind) -> bool {
    runtime_op_from_kind(kind).is_some()
}

/// [`crate::egraph::ops::op_from_kind`] extended with the runtime-only mask
/// ops above and the opaque `Gather` op (absent from the global lookup so no
/// rewrite template can name it — its participation is hash-consing CSE
/// only). Every conversion in this module resolves ops through this.
fn runtime_op_from_kind(kind: OpKind) -> Option<&'static dyn Op> {
    match kind {
        OpKind::BitAnd => Some(&MaskAnd),
        OpKind::BitOr => Some(&MaskOr),
        OpKind::TruncToInt => Some(&IntTrunc),
        OpKind::IntToFloat => Some(&IntFromInt),
        OpKind::IAdd => Some(&IntAdd),
        OpKind::Shl => Some(&IntShl),
        OpKind::Shr => Some(&IntShr),
        OpKind::Gather => Some(&crate::egraph::ops::Gather),
        other => crate::egraph::ops::op_from_kind(other),
    }
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
                    ExprNode::Param(_) => return None,
                    &ExprNode::Buffer(b) => {
                        let class = egraph.add(ENode::Buffer(*arena.buffer_decl(b)));
                        memo.insert(id, class);
                        result_stack.push(class);
                    }
                    &ExprNode::Unary(kind, a) => {
                        runtime_op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(a));
                    }
                    &ExprNode::Binary(kind, a, b) => {
                        runtime_op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(b));
                        task_stack.push(Task::Visit(a));
                    }
                    &ExprNode::Ternary(kind, a, b, c) => {
                        runtime_op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(c));
                        task_stack.push(Task::Visit(b));
                        task_stack.push(Task::Visit(a));
                    }
                    // `Reduce` was unrolled and `Dwrt` lowered before this
                    // walk; what remains (`Tuple`) is not modelled. Bail out.
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
                let op = runtime_op_from_kind(kind)
                    .expect("runtime_op_from_kind already checked in Visit");
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
/// [`config_for_node_count`], mirroring what `pixelflow_codegen::jit_cache`'s
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
    fn assert_semantics_preserved(
        arena: &ExprArena,
        root: ExprId,
        optimized: &(ExprArena, ExprId),
    ) {
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
    fn repeated_bake_of_the_same_kernel_hits_the_cache() {
        // The exact regression this cache exists to close: Lattice::bake
        // calls optimize_runtime_arena on EVERY bake of a Kernel, but real
        // callers (GlyphCache, and criterion benches that measure "the JIT
        // compile is cached, so iterations measure tabulation") bake the
        // *same* kernel repeatedly. Without caching, every one of those
        // calls re-runs full saturation from scratch — slower than not
        // optimizing at all, since the downstream JIT compile was already
        // cached and free.
        //
        // Build something big enough to land in the "classical" budget
        // (>50 nodes) with real rewriting work to do (redundant
        // sub-multiplications an FMA pass and commutativity/associativity
        // actually have to chew on), so a cold run takes measurably longer
        // than a hash lookup.
        fn build_arena() -> (ExprArena, ExprId) {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let mut acc = a.push_const(0.0);
            for k in 0..15 {
                let c = a.push_const(1.0 + k as f32 * 0.37);
                let xc = a.push_binary(OpKind::Mul, x, c);
                let yc = a.push_binary(OpKind::Mul, y, c);
                let term = a.push_binary(OpKind::Add, xc, yc);
                acc = a.push_binary(OpKind::Add, acc, term);
            }
            (a, acc)
        }

        let (a1, r1) = build_arena();
        let cold_start = std::time::Instant::now();
        let arc1 = optimize_runtime_arena(&a1, r1, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let opt1 = &arc1.0;
        let cold = cold_start.elapsed();

        // A freshly built, structurally identical (but not reused) arena:
        // proves the cache keys on shape, not on the first call's identity.
        let (a2, r2) = build_arena();
        let warm_start = std::time::Instant::now();
        let arc2 = optimize_runtime_arena(&a2, r2, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let opt2 = &arc2.0;
        let warm = warm_start.elapsed();

        assert_eq!(
            opt1.nodes_raw().len(),
            opt2.nodes_raw().len(),
            "cached and fresh optimization must agree on the result shape"
        );
        assert!(
            warm < cold / 2 || warm < std::time::Duration::from_micros(200),
            "expected the second call to hit the cache (warm {warm:?} vs cold {cold:?}) — \
             a regression here means optimize_runtime_arena is re-saturating every bake"
        );
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

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("pure arithmetic arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        assert_semantics_preserved(&a, root, &(opt_arena.clone(), opt_root));
        assert!(
            matches!(
                opt_arena.node(opt_root),
                ExprNode::Ternary(OpKind::MulAdd, ..)
            ),
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

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("trig arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
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

        let arc = optimize_runtime_arena(&a, dx, pixelflow_ir::LatticeShape::POINT)
            .expect("Dwrt-bearing arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        // eval_scalar refuses a raw Dwrt (the interpreter evaluates the
        // post-calculus program, same as the JIT) — lower both sides before
        // comparing, cross-checking the e-graph's ChainRule reduction
        // against the dedicated lower_dwrt pass.
        use pixelflow_ir::passes::lower_dwrt_owned;
        let (want_arena, want_root) = lower_dwrt_owned(&a, dx).expect("lower original");
        let (got_arena, got_root) =
            lower_dwrt_owned(&opt_arena, opt_root).expect("lower optimized");
        assert_semantics_preserved(&want_arena, want_root, &(got_arena, got_root));
    }

    /// Ids of every node reachable from `root`, discovery order.
    fn reachable_ids(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        let mut out = Vec::new();
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            out.push(id);
            stack.extend(arena.children(id));
        }
        out
    }

    fn count_gathers(arena: &ExprArena, root: ExprId) -> usize {
        reachable_ids(arena, root)
            .iter()
            .filter(|&&id| matches!(arena.node(id), ExprNode::Ternary(OpKind::Gather, ..)))
            .count()
    }

    /// Identities of buffers referenced by reachable `Buffer` leaves.
    fn reachable_buffer_identities(
        arena: &ExprArena,
        root: ExprId,
    ) -> std::collections::BTreeSet<pixelflow_ir::arena::BufferIdentity> {
        reachable_ids(arena, root)
            .iter()
            .filter_map(|&id| match arena.node(id) {
                &ExprNode::Buffer(b) => Some(arena.buffer_decl(b).id),
                _ => None,
            })
            .collect()
    }

    /// Bind slices to an arena by buffer *identity*, not slot order: the
    /// optimizer redeclares buffers in extraction-traversal order, so the
    /// optimized arena's slot numbering can differ from the input's.
    fn bind_by_identity<'a>(
        arena: &ExprArena,
        by_id: &[(pixelflow_ir::arena::BufferIdentity, &'a [f32])],
    ) -> BindingTable<'a> {
        let slices: Vec<&[f32]> = arena
            .buffers()
            .iter()
            .map(|d| {
                by_id
                    .iter()
                    .find(|(id, _)| *id == d.id)
                    .unwrap_or_else(|| panic!("no slice for buffer identity {:?}", d.id))
                    .1
            })
            .collect();
        BindingTable::bind(arena, &slices).expect("bind_by_identity")
    }

    /// Eval parity for buffer-bearing arenas, both sides bound by identity.
    fn assert_gather_semantics_preserved(
        arena: &ExprArena,
        root: ExprId,
        optimized: &(ExprArena, ExprId),
        by_id: &[(pixelflow_ir::arena::BufferIdentity, &[f32])],
    ) {
        let (opt_arena, opt_root) = optimized;
        let want_bind = bind_by_identity(arena, by_id);
        let got_bind = bind_by_identity(opt_arena, by_id);
        // Coordinates chosen off integer boundaries so Gather's floor cannot
        // flip cells on rounding differences introduced by rewrites.
        let coords: &[(f32, f32)] = &[(0.3, 0.4), (1.5, 0.6), (2.2, 1.7), (3.6, 2.4), (-1.2, 9.5)];
        for &(cx, cy) in coords {
            let want = eval_scalar(arena, root, &[cx, cy, 0.0, 0.0], &want_bind);
            let got = eval_scalar(opt_arena, *opt_root, &[cx, cy, 0.0, 0.0], &got_bind);
            assert!(
                (want - got).abs() < 1e-3,
                "gather optimization changed semantics at ({cx},{cy}): {want} != {got}"
            );
        }
    }

    #[test]
    fn gather_arena_round_trips_through_the_egraph() {
        // BilinearSampler-shaped: the e-graph must now carry the Gather as
        // opaque structure and hand back an arena that declares the same
        // buffer (by identity) and evaluates identically.
        let identity = pixelflow_ir::arena::BufferIdentity::mint();
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 3.0 + 1.0).collect();

        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: identity,
            width: 4,
            height: 4,
        });
        let x = a.push_var(0);
        let y = a.push_var(1);
        let g = a.push_gather(buf, x, y);
        let one = a.push_const(1.0);
        let root = a.push_binary(OpKind::Add, g, one);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("a Gather-bearing arena must optimize, not bail");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        assert_eq!(
            reachable_buffer_identities(&opt_arena, opt_root),
            reachable_buffer_identities(&a, root),
            "extraction must redeclare the same buffers, by identity"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(identity, data.as_slice())],
        );
    }

    #[test]
    fn duplicated_gathers_cse_into_one_node() {
        // The composition problem this change exists to solve: every use of a
        // sampler Kernel re-splices its fragment, so the SAME gather (same
        // buffer identity, same coordinate subtree) appears twice as two
        // disjoint copies. Hash-consing must collapse them to one node.
        let identity = pixelflow_ir::arena::BufferIdentity::mint();
        let data: Vec<f32> = (0..16).map(|i| (i * i) as f32).collect();

        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: identity,
            width: 4,
            height: 4,
        });
        // Two structurally identical copies, pushed separately — exactly what
        // splice produces.
        let mut push_dup = |a: &mut ExprArena| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let one = a.push_const(1.0);
            let xx = a.push_binary(OpKind::Add, x, one);
            a.push_gather(buf, xx, y)
        };
        let g1 = push_dup(&mut a);
        let g2 = push_dup(&mut a);
        // Mul (not Add) so the doubling rule can't restructure the root and
        // muddy the count assertions.
        let root = a.push_binary(OpKind::Mul, g1, g2);

        let before = reachable_count(&a, root);
        assert_eq!(
            count_gathers(&a, root),
            2,
            "input must contain the duplicate"
        );

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
        let after = reachable_count(&opt_arena, opt_root);

        assert!(
            after < before,
            "CSE must strictly shrink the arena (before={before}, after={after})"
        );
        assert_eq!(
            count_gathers(&opt_arena, opt_root),
            1,
            "the two identical gathers must share one node"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(identity, data.as_slice())],
        );
    }

    /// Slot order is the binding ABI: the JIT loads slot i's base pointer
    /// from the context array at i*8, and callers bind in the order the arena
    /// THEY built declared. Extraction traverses in its own order — here the
    /// root's first child reads the SECOND-declared buffer — so a rebuild
    /// that declared buffers in traversal order would silently swap the
    /// caller's two pointers and read the wrong memory.
    #[test]
    fn optimization_preserves_buffer_slot_order() {
        let mut a = ExprArena::new();
        let buf_a = a.declare_buffer(BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width: 4,
            height: 1,
        });
        let buf_b = a.declare_buffer(BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width: 8,
            height: 1,
        });
        let x = a.push_var(0);
        let y = a.push_var(1);
        let gb = a.push_gather(buf_b, x, y);
        let ga = a.push_gather(buf_a, x, y);
        let root = a.push_binary(OpKind::Add, gb, ga);

        let out = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("buffer kernel must optimize");
        let input: Vec<_> = a.buffers().iter().map(|d| d.id).collect();
        let output: Vec<_> = out.0.buffers().iter().map(|d| d.id).collect();
        assert_eq!(input, output, "slot order must survive optimization");
    }

    #[test]
    fn distinct_buffer_identities_never_merge() {
        // Equal extents and identical coordinates are a coincidence, not the
        // same memory: gathers of different identities must stay distinct.
        let id_a = pixelflow_ir::arena::BufferIdentity::mint();
        let id_b = pixelflow_ir::arena::BufferIdentity::mint();
        let data_a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..16).map(|i| 1000.0 - i as f32).collect();

        let mut a = ExprArena::new();
        let buf_a = a.declare_buffer(BufferDecl {
            id: id_a,
            width: 4,
            height: 4,
        });
        let buf_b = a.declare_buffer(BufferDecl {
            id: id_b,
            width: 4,
            height: 4,
        });
        let x = a.push_var(0);
        let y = a.push_var(1);
        let ga = a.push_gather(buf_a, x, y);
        let gb = a.push_gather(buf_b, x, y);
        let root = a.push_binary(OpKind::Sub, ga, gb);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        assert_eq!(
            count_gathers(&opt_arena, opt_root),
            2,
            "different identities with identical extents/coords must not merge"
        );
        assert_eq!(
            reachable_buffer_identities(&opt_arena, opt_root).len(),
            2,
            "both identities must survive extraction"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(id_a, data_a.as_slice()), (id_b, data_b.as_slice())],
        );
    }

    #[test]
    fn composed_cell_grid_kernel_shape_deduplicates() {
        // Faithful synthetic of the packed terminal kernel: cell-grid
        // coordinate arithmetic (cell index + intra-cell offset) feeding 5
        // gathers of one buffer (glyph atlas channels) and 4 of another
        // (color planes), with the coordinate subtree re-spliced VERBATIM for
        // every gather — exactly the duplication Kernel composition produces.
        let atlas_id = pixelflow_ir::arena::BufferIdentity::mint();
        let color_id = pixelflow_ir::arena::BufferIdentity::mint();
        let atlas: Vec<f32> = (0..(64 * 32)).map(|i| (i % 97) as f32).collect();
        let colors: Vec<f32> = (0..(64 * 32)).map(|i| (i % 251) as f32 * 0.5).collect();

        let mut a = ExprArena::new();
        let atlas_buf = a.declare_buffer(BufferDecl {
            id: atlas_id,
            width: 64,
            height: 32,
        });
        let color_buf = a.declare_buffer(BufferDecl {
            id: color_id,
            width: 64,
            height: 32,
        });

        // The shared coordinate arithmetic, duplicated per gather: cell
        // coords (floor(X/8), floor(Y/16)), intra-cell offsets, and an
        // atlas-space remap. The atlas cell stride (10, 18) differs from the
        // screen cell size (8, 16) so no algebraic rule can cancel the
        // arithmetic away — deduplication must come from hash-consing, as in
        // the real kernel.
        let mut push_coords = |a: &mut ExprArena| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let cw = a.push_const(8.0);
            let ch = a.push_const(16.0);
            let aw = a.push_const(10.0);
            let ah = a.push_const(18.0);
            let xc = a.push_binary(OpKind::Div, x, cw);
            let cx = a.push_unary(OpKind::Floor, xc);
            let yc = a.push_binary(OpKind::Div, y, ch);
            let cy = a.push_unary(OpKind::Floor, yc);
            let cxw = a.push_binary(OpKind::Mul, cx, cw);
            let fx = a.push_binary(OpKind::Sub, x, cxw);
            let cyh = a.push_binary(OpKind::Mul, cy, ch);
            let fy = a.push_binary(OpKind::Sub, y, cyh);
            let cxa = a.push_binary(OpKind::Mul, cx, aw);
            let cya = a.push_binary(OpKind::Mul, cy, ah);
            let sx = a.push_binary(OpKind::Add, cxa, fx);
            let sy = a.push_binary(OpKind::Add, cya, fy);
            (sx, sy)
        };

        let mut acc = a.push_const(0.0);
        for i in 0..9 {
            let (sx, sy) = push_coords(&mut a);
            let buf = if i < 5 { atlas_buf } else { color_buf };
            let g = a.push_gather(buf, sx, sy);
            let w = a.push_const(0.1 + i as f32 * 0.07);
            let wg = a.push_binary(OpKind::Mul, g, w);
            acc = a.push_binary(OpKind::Add, acc, wg);
        }
        let root = acc;

        let before = reachable_count(&a, root);
        assert_eq!(count_gathers(&a, root), 9);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
        let after = reachable_count(&opt_arena, opt_root);

        // Report shape for the record: 9 duplicated ~13-node coordinate
        // subtrees must collapse to (at most) one shared copy, and the 5+4
        // gathers to one per (buffer, coords) pair — here 2 total.
        assert!(
            after < before,
            "composed kernel must come back deduplicated (before={before}, after={after})"
        );
        assert_eq!(
            count_gathers(&opt_arena, opt_root),
            2,
            "5 atlas + 4 color gathers of identical coords must CSE to one each"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(atlas_id, atlas.as_slice()), (color_id, colors.as_slice())],
        );

        // Keep the measured counts visible in test output (`--nocapture`).
        println!("composed cell-grid shape: before={before} nodes, after={after} nodes");
    }

    #[test]
    fn reduce_is_unrolled_and_optimized() {
        // Kernel::over-shaped: Σ_{i<4} i² over the reduction index slot. The
        // binder is distributed before saturation, so what the e-graph sees
        // is 0·0 + 1·1 + 2·2 + 3·3 — ordinary arithmetic it can fold.
        //
        // What this pins is the binder's disappearance and the value, not the
        // folder's reach: saturation runs under a wall-clock budget (10ms for
        // an arena this small), so *how far* the fold cascades is a property
        // of the machine, not of the compiler. Asserting `Const(14.0)` here
        // passed locally and failed on a loaded CI runner, which is the
        // assertion being wrong rather than the code.
        let mut a = ExprArena::new();
        let i = a.push_var(4);
        let body = a.push_binary(OpKind::Mul, i, i);
        let root = a.push_reduce(OpKind::Add, 4, 4, body);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("a Reduce-bearing arena must optimize once distributed");
        let (opt, opt_root) = &*arc;
        assert!(
            !reaches_nary(opt, *opt_root),
            "the binder must be gone from the optimized arena"
        );
        assert_eq!(
            eval_scalar(opt, *opt_root, &[0.0; 4], &BindingTable::empty()),
            14.0,
            "Σ_{{i<4}} i² = 0 + 1 + 4 + 9"
        );

        // Σ_{i<3} X·i: the surviving terms depend on X, and the optimized
        // form agrees with the interpreter on the distributed original.
        let mut b = ExprArena::new();
        let x = b.push_var(0);
        let j = b.push_var(5);
        let body = b.push_binary(OpKind::Mul, x, j);
        let root = b.push_reduce(OpKind::Add, 5, 3, body);
        let (unrolled, unrolled_root) = pixelflow_ir::passes::expand_reduce_owned(&b, root);
        let arc = optimize_runtime_arena(&b, root, pixelflow_ir::LatticeShape::POINT)
            .expect("X-dependent Reduce must optimize");
        let (opt, opt_root) = &*arc;
        assert!(!reaches_nary(opt, *opt_root));
        for x in [0.0f32, 1.5, -2.25, 7.0] {
            let want = eval_scalar(
                &unrolled,
                unrolled_root,
                &[x, 0.0, 0.0, 0.0],
                &BindingTable::empty(),
            );
            let got = eval_scalar(opt, *opt_root, &[x, 0.0, 0.0, 0.0], &BindingTable::empty());
            assert_eq!(got, want, "Σ_{{i<3}} X·i at X={x}");
        }
    }

    /// Whether any `Nary` (the `Reduce` binder) is reachable from `root`.
    fn reaches_nary(arena: &ExprArena, root: ExprId) -> bool {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            if matches!(arena.node(id), ExprNode::Nary(..)) {
                return true;
            }
            stack.extend(arena.children(id));
        }
        false
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

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("identity arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
        assert_semantics_preserved(&a, root, &(opt_arena.clone(), opt_root));
        // x + 0.0, then * 1.0 should collapse to bare X.
        assert!(
            matches!(opt_arena.node(opt_root), ExprNode::Var(0)),
            "expected identities to collapse to bare X, got {:?}",
            opt_arena.node(opt_root)
        );
    }
}

/// Production saturation telemetry (docs/results/2026-09-01-production-saturation-telemetry.md).
///
/// [`optimize_runtime_arena_uncached`] computes a `SaturationResult` and
/// drops it (`runtime.rs:128-133`), so production cannot say whether a real
/// core-term kernel quiesces or is cut off by the iteration cap, the class
/// cap, or the wall-clock ceiling. This `#[ignore]`d test replays the same
/// three production calls — `EGraph::with_rules(all_rules())` (`:122`),
/// `config_for_node_count` + `saturate_with_full_budget` (`:126-133`),
/// `env_extraction_policy()` + `extraction` + `choices_to_arena` (`:135-137`)
/// — on arenas dumped from the production constructors (the packed cell grid
/// in `pixelflow-core`'s `cell_grid.rs`, glyphs from `Font::glyph_kernel_scaled`
/// in `pixelflow-graphics`), and keeps what production throws away. It lives
/// here, not beside the constructors, because `arena_to_egraph` and the
/// runtime-only mask/int ops it resolves are private to this module; moving
/// them out would be the public-API change this measurement must not make.
///
/// Nothing here changes production behavior: no signature, no visibility,
/// no code outside `#[cfg(test)]`.
#[cfg(test)]
mod production_telemetry {
    use super::*;
    use crate::egraph::{CostModel, ExtractionPolicy, SaturationConfig, extract_dag};
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
    use std::fmt::Write as _;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    const DIR_VAR: &str = "PIXELFLOW_TELEMETRY_DIR";
    const OUT_VAR: &str = "PIXELFLOW_TELEMETRY_OUT";
    const REF_MULT_VAR: &str = "PIXELFLOW_TELEMETRY_REF_MULT";
    const REF_CEILING_VAR: &str = "PIXELFLOW_TELEMETRY_REF_CEILING_S";
    /// Reference-run multiplier over the production tier's iteration cap
    /// (and, for the cap-lifted reference, its class cap), mirroring the
    /// Guide registration's "unguided-at-4B" comparison.
    const DEFAULT_REF_MULT: usize = 4;
    /// Reference-run wall-clock is a SAFETY CEILING, never a metric: if it
    /// binds the run panics rather than reporting a truncated reference.
    const DEFAULT_REF_CEILING_S: u64 = 600;

    fn env_required(var: &str) -> String {
        std::env::var(var).unwrap_or_else(|e| panic!("{var} must be set ({e})"))
    }

    /// Inverse of the dumpers' `dump_arena` (cell_grid.rs tests /
    /// pixelflow-graphics/tests/production_glyph_arena_dump.rs): replays
    /// reachable nodes in original id order through the public `push_*`
    /// API, which never hash-conses, so the rebuilt arena has exactly the
    /// dumped node multiset and `reachable_count` is preserved. Buffer
    /// identities are re-minted per distinct dumped ordinal, preserving the
    /// equality structure (two slots naming one buffer still share an
    /// identity) without depending on the dumping process's counter.
    fn load_arena(path: &Path) -> (String, ExprArena, ExprId) {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut lines = text.lines();
        assert_eq!(
            lines.next(),
            Some("# pixelflow arena dump v1"),
            "{}: bad header",
            path.display()
        );
        let mut name = None;
        let mut arena = ExprArena::new();
        let mut idents: Vec<BufferIdentity> = Vec::new();
        let mut root = None;
        let mut next_id: u32 = 0;
        let mut buf_count: u16 = 0;
        let op = |s: &str| -> OpKind {
            OpKind::all()
                .find(|k| format!("{k:?}") == s)
                .unwrap_or_else(|| panic!("{}: unknown OpKind {s:?}", path.display()))
        };
        let id = |s: &str| -> ExprId {
            ExprId(s.parse().unwrap_or_else(|e| panic!("{}: bad id {s:?}: {e}", path.display())))
        };
        for line in lines {
            let f: Vec<&str> = line.split_whitespace().collect();
            let pushed = match f.as_slice() {
                ["name", n] => {
                    name = Some((*n).to_string());
                    continue;
                }
                ["buf", ord, w, h] => {
                    let ord: usize = ord.parse().expect("buf ordinal");
                    while idents.len() <= ord {
                        idents.push(BufferIdentity::mint());
                    }
                    let slot = arena.declare_buffer(BufferDecl {
                        id: idents[ord],
                        width: w.parse().expect("buf width"),
                        height: h.parse().expect("buf height"),
                    });
                    assert_eq!(slot.0, buf_count, "{}: buffer slot order drifted", path.display());
                    buf_count += 1;
                    continue;
                }
                ["root", r] => {
                    root = Some(id(r));
                    continue;
                }
                ["V", i] => arena.push_var(i.parse().expect("var index")),
                ["C", bits] => arena.push_const(f32::from_bits(bits.parse().expect("const bits"))),
                ["B", slot] => arena.push_buffer(pixelflow_ir::arena::BufferId(
                    slot.parse().expect("buffer slot"),
                )),
                ["U", k, a] => arena.push_unary(op(k), id(a)),
                ["Bi", k, a, b] => arena.push_binary(op(k), id(a), id(b)),
                ["T", k, a, b, c] => arena.push_ternary(op(k), id(a), id(b), id(c)),
                other => panic!("{}: unparseable line {other:?}", path.display()),
            };
            assert_eq!(
                pushed,
                ExprId(next_id),
                "{}: replay drifted from dumped ids",
                path.display()
            );
            next_id += 1;
        }
        let name = name.unwrap_or_else(|| panic!("{}: no name line", path.display()));
        let root = root.unwrap_or_else(|| panic!("{}: no root line", path.display()));
        (name, arena, root)
    }

    /// Latency-prior cost of the arena the JIT would actually execute: the
    /// per-op table summed over every reachable operation once (DAG cost;
    /// leaves are free, as in `CostModel::node_op_cost`). This is the
    /// quality metric — NOT `ExtractedDAG::total_cost`, which adds a
    /// 1,000,000 `CYCLE_COST` penalty per cycle-breaking pick and so does
    /// not describe the emitted code.
    fn arena_cost(arena: &ExprArena, root: ExprId, costs: &CostModel) -> usize {
        let len = arena.nodes_raw().len();
        let mut seen = vec![false; len];
        let mut stack = vec![root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let kind = match arena.node(id) {
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Buffer(_) => None,
                ExprNode::Unary(k, _) | ExprNode::Binary(k, _, _) | ExprNode::Ternary(k, _, _, _) => Some(*k),
                other @ (ExprNode::Param(_) | ExprNode::Nary(..)) => {
                    panic!("extracted arena contains {other:?}")
                }
            };
            if let Some(k) = kind {
                assert_ne!(k, OpKind::Dwrt, "Dwrt survived extraction");
                total += costs.cost(k);
            }
            stack.extend(arena.children(id));
        }
        total
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Stop {
        Quiesced,
        IterationCap,
        ClassCap,
        Timeout,
    }

    struct Run {
        iterations: usize,
        total_unions: usize,
        classes_after: usize,
        applications: usize,
        journal_unions: usize,
        elapsed: Duration,
        cost: usize,
        dp_cost: usize,
        extracted_nodes: usize,
    }

    impl Run {
        /// The budget-independent trajectory signature: two runs of the same
        /// arena that were never cut differently agree on all of these.
        fn signature(&self) -> (usize, usize, usize, usize, usize) {
            (self.iterations, self.total_unions, self.classes_after, self.applications, self.cost)
        }
    }

    /// The production sequence of `optimize_runtime_arena_uncached`
    /// (`runtime.rs:106-137`) from the e-graph build onward, with the budget
    /// as parameters so the same function runs the production tier and both
    /// references. `lower_dwrt_owned` (`:120`) and `config_for_node_count`
    /// (`:126-127`) run once in the caller since they are budget-independent.
    fn run(arena: &ExprArena, root: ExprId, max_iterations: usize, max_classes: usize, timeout: Duration) -> Run {
        // runtime.rs:122-124
        let mut egraph = EGraph::with_rules(all_rules());
        let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
        let root_class = arena_to_egraph(arena, root, &mut egraph, &mut memo)
            .expect("production arena must be e-graph representable (no Param/Nary)");

        // runtime.rs:128-133 — the SaturationResult production discards.
        let started = Instant::now();
        let result = crate::egraph::saturate_with_full_budget(&mut egraph, max_iterations, max_classes, timeout);
        let elapsed = started.elapsed();

        // runtime.rs:135-137
        let policy = env_extraction_policy();
        assert!(
            matches!(policy, ExtractionPolicy::Static(_)),
            "PIXELFLOW_NNUE_WEIGHTS is set; this measures the default production policy — unset it"
        );
        let extraction = policy.extraction(&egraph, root_class);
        let (extracted, extracted_root) = choices_to_arena(&extraction);

        let costs = CostModel::latency_prior();
        // The Static arm's own DP (extraction.rs:59) re-run to read the
        // total it computes and discards; same inputs, same answer. Kept as
        // a raw column only — see `arena_cost` for why it is not the metric.
        let dp_cost = extract_dag(&egraph, root_class, &costs).total_cost;

        Run {
            iterations: result.iterations,
            total_unions: result.total_unions,
            classes_after: result.classes_after,
            applications: egraph.provenance().application_count(),
            journal_unions: egraph.provenance().union_count(),
            elapsed,
            cost: arena_cost(&extracted, extracted_root, &costs),
            dp_cost,
            extracted_nodes: reachable_count(&extracted, extracted_root),
        }
    }

    /// Why the production run stopped, decided against two unceilinged
    /// references rather than `SaturationResult::saturated` (which reads
    /// `iterations < max || total_unions == 0` and so calls a class-cap or
    /// timeout stop "saturated"). The class cap cannot be read off any
    /// count: `apply_rule_at_index_timed` refuses an application whenever
    /// `classes + estimated_new > max_classes` (graph.rs:937,964), so a
    /// cap-bound graph settles AT or under the cap and the next round
    /// commits zero unions exactly like a quiesced one. Only a run with the
    /// cap lifted can tell the two apart: if lifting the cap changes the
    /// trajectory, the cap was binding.
    ///
    /// `refr` = same class cap, `mult`× iterations, no time ceiling.
    /// `lifted` = `mult`× iterations AND `mult`× class cap, no time ceiling.
    fn classify(prod: &Run, refr: &Run, lifted: &Run, config: &SaturationConfig) -> Stop {
        let stopped_on_its_own_at_prod_round = refr.iterations == prod.iterations;
        if stopped_on_its_own_at_prod_round {
            let cap_bound = refr.signature() != lifted.signature();
            return if cap_bound { Stop::ClassCap } else { Stop::Quiesced };
        }
        if prod.iterations < config.max_iterations {
            return Stop::Timeout;
        }
        Stop::IterationCap
    }

    /// Stop reason of an unceilinged reference: the cap bound if the
    /// cap-lifted run diverged from it; otherwise it either used every
    /// iteration or quiesced.
    fn classify_reference(refr: &Run, lifted: &Run, max_iterations: usize) -> Stop {
        if refr.signature() != lifted.signature() {
            Stop::ClassCap
        } else if refr.iterations >= max_iterations {
            Stop::IterationCap
        } else {
            Stop::Quiesced
        }
    }

    fn classify_lifted(lifted: &Run, max_iterations: usize, max_classes: usize) -> &'static str {
        if lifted.iterations >= max_iterations {
            "IterationCap"
        } else if lifted.classes_after * 10 >= max_classes * 9 {
            // Cannot be proven without lifting again; flagged, not asserted.
            "NearLiftedCap"
        } else {
            "Quiesced"
        }
    }

    fn tier_name(config: &SaturationConfig) -> &'static str {
        match config.max_iterations {
            20 => "blitz",
            50 => "rapid",
            100 => "classical",
            other => panic!("unknown tier with max_iterations={other}"),
        }
    }

    fn loss_pct(cost: usize, baseline: usize) -> f64 {
        (cost as f64 - baseline as f64) / baseline as f64 * 100.0
    }

    fn median(v: &[f64]) -> f64 {
        assert!(!v.is_empty(), "median of nothing");
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let n = v.len();
        if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
    }

    fn percentile(v: &[f64], p: f64) -> f64 {
        assert!(!v.is_empty(), "percentile of nothing");
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let idx = ((v.len() - 1) as f64 * p).round() as usize;
        v[idx]
    }

    #[test]
    #[ignore = "measurement: PIXELFLOW_TELEMETRY_DIR=<dumps> PIXELFLOW_TELEMETRY_OUT=<tsv> cargo test -p pixelflow-search --release -- --ignored production_saturation_telemetry --nocapture"]
    fn production_saturation_telemetry() {
        let dir = PathBuf::from(env_required(DIR_VAR));
        let out_path = PathBuf::from(env_required(OUT_VAR));
        let mult: usize = std::env::var(REF_MULT_VAR)
            .map(|s| s.parse().expect("REF_MULT must be an integer"))
            .unwrap_or(DEFAULT_REF_MULT);
        let ceiling = Duration::from_secs(
            std::env::var(REF_CEILING_VAR)
                .map(|s| s.parse().expect("REF_CEILING_S must be an integer"))
                .unwrap_or(DEFAULT_REF_CEILING_S),
        );
        assert!(
            std::env::var("PIXELFLOW_NNUE_WEIGHTS").is_err(),
            "PIXELFLOW_NNUE_WEIGHTS is set; this measures the default production policy — unset it"
        );

        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "arena"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no *.arena files in {}", dir.display());

        let header = "name\tgroup\tnodes\ttier\tstop\tmachine_dependent\titers\tmax_iters\tclasses\tmax_classes\tapps\tunions\tjournal_unions\telapsed_ms\tcost\tdp_cost\text_nodes\
                      \tref_stop\tref_iters\tref_classes\tref_apps\tref_elapsed_ms\tref_cost\tloss_vs_ref_pct\
                      \tlifted_stop\tlifted_iters\tlifted_classes\tlifted_apps\tlifted_elapsed_ms\tlifted_cost\tloss_vs_lifted_pct\tanomaly";
        let mut tsv = String::new();
        writeln!(tsv, "{header}").expect("write");
        println!("{header}");

        struct Row {
            group: String,
            stop: Stop,
            ref_stop: Stop,
            loss_vs_ref: f64,
            loss_vs_lifted: f64,
            apps: usize,
            anomaly: Option<String>,
            fatal: bool,
        }
        let mut rows: Vec<Row> = Vec::new();

        for path in &files {
            let (name, raw_arena, raw_root) = load_arena(path);
            let group = name.split(':').next().expect("group prefix").to_string();

            // runtime.rs:120
            let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(&raw_arena, raw_root)
                .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
            // runtime.rs:126-127
            let node_count = reachable_count(&arena, root);
            let config = config_for_node_count(node_count);

            let prod = run(&arena, root, config.max_iterations, config.max_classes, config.hard_timeout);

            let ref_iters = config.max_iterations * mult;
            let lifted_classes = config.max_classes * mult;
            let refr = run(&arena, root, ref_iters, config.max_classes, ceiling);
            assert!(
                refr.elapsed < ceiling,
                "{name}: reference run hit its {ceiling:?} safety ceiling after {} iterations — \
                 the ceiling is a stop condition, not a metric; raise {REF_CEILING_VAR} and re-run",
                refr.iterations
            );
            let lifted = run(&arena, root, ref_iters, lifted_classes, ceiling);
            assert!(
                lifted.elapsed < ceiling,
                "{name}: cap-lifted run hit its {ceiling:?} safety ceiling after {} iterations — \
                 the ceiling is a stop condition, not a metric; raise {REF_CEILING_VAR} and re-run",
                lifted.iterations
            );

            let stop = classify(&prod, &refr, &lifted, &config);
            let ref_stop = classify_reference(&refr, &lifted, ref_iters);
            let lifted_stop = classify_lifted(&lifted, ref_iters, lifted_classes);
            let loss_vs_ref = loss_pct(prod.cost, refr.cost);
            let loss_vs_lifted = loss_pct(prod.cost, lifted.cost);

            let mut anomaly: Option<String> = None;
            let mut fatal = false;
            if matches!(stop, Stop::Quiesced | Stop::ClassCap)
                && (prod.total_unions, prod.classes_after, prod.applications, prod.cost)
                    != (refr.total_unions, refr.classes_after, refr.applications, refr.cost)
            {
                // Same round reached on their own, yet different work done:
                // either a mid-round deadline cut production's last round
                // (a Timeout wearing Quiesced's clothes) or the optimizer
                // is nondeterministic. Either way the row is not trustworthy.
                fatal = true;
                anomaly = Some(format!(
                    "reference stopped at production's round but trajectories differ: prod(unions={},classes={},apps={},cost={}) ref(unions={},classes={},apps={},cost={})",
                    prod.total_unions, prod.classes_after, prod.applications, prod.cost,
                    refr.total_unions, refr.classes_after, refr.applications, refr.cost
                ));
            } else if prod.cost < refr.cost || prod.cost < lifted.cost || refr.cost < lifted.cost {
                anomaly = Some(format!(
                    "more saturation extracted WORSE: cost prod={} ref={} lifted={}",
                    prod.cost, refr.cost, lifted.cost
                ));
            }

            let line = format!(
                "{name}\t{group}\t{node_count}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\
                 \t{:?}\t{}\t{}\t{}\t{:.1}\t{}\t{:.2}\
                 \t{}\t{}\t{}\t{}\t{:.1}\t{}\t{:.2}\t{}",
                tier_name(&config),
                stop,
                stop == Stop::Timeout,
                prod.iterations,
                config.max_iterations,
                prod.classes_after,
                config.max_classes,
                prod.applications,
                prod.total_unions,
                prod.journal_unions,
                prod.elapsed.as_secs_f64() * 1e3,
                prod.cost,
                prod.dp_cost,
                prod.extracted_nodes,
                ref_stop,
                refr.iterations,
                refr.classes_after,
                refr.applications,
                refr.elapsed.as_secs_f64() * 1e3,
                refr.cost,
                loss_vs_ref,
                lifted_stop,
                lifted.iterations,
                lifted.classes_after,
                lifted.applications,
                lifted.elapsed.as_secs_f64() * 1e3,
                lifted.cost,
                loss_vs_lifted,
                anomaly.as_deref().unwrap_or("-"),
            );
            println!("{line}");
            writeln!(tsv, "{line}").expect("write");
            rows.push(Row { group, stop, ref_stop, loss_vs_ref, loss_vs_lifted, apps: prod.applications, anomaly, fatal });
        }

        assert_eq!(rows.len(), files.len(), "every dumped kernel must produce a row");
        std::fs::write(&out_path, &tsv).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));

        let mut groups: Vec<String> = rows.iter().map(|r| r.group.clone()).collect();
        groups.sort();
        groups.dedup();
        groups.push("ALL".to_string());
        println!("\n== summary (ref = {mult}x iterations/same cap; lifted = {mult}x iterations/{mult}x cap; no time ceilings) ==");
        println!("group\tn\tquiesced\titer_cap\tclass_cap\ttimeout\tref_cap_bound\tmed_loss_ref%\tp90_loss_ref%\tmax_loss_ref%\tmed_loss_lifted%\tp90_loss_lifted%\tmax_loss_lifted%\tmed_apps\tmax_apps");
        for g in &groups {
            let sel: Vec<&Row> = rows.iter().filter(|r| g == "ALL" || &r.group == g).collect();
            let count = |s: Stop| sel.iter().filter(|r| r.stop == s).count();
            let lr: Vec<f64> = sel.iter().map(|r| r.loss_vs_ref).collect();
            let ll: Vec<f64> = sel.iter().map(|r| r.loss_vs_lifted).collect();
            let apps: Vec<f64> = sel.iter().map(|r| r.apps as f64).collect();
            println!(
                "{g}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.0}\t{:.0}",
                sel.len(),
                count(Stop::Quiesced),
                count(Stop::IterationCap),
                count(Stop::ClassCap),
                count(Stop::Timeout),
                sel.iter().filter(|r| r.ref_stop == Stop::ClassCap).count(),
                median(&lr),
                percentile(&lr, 0.9),
                percentile(&lr, 1.0),
                median(&ll),
                percentile(&ll, 0.9),
                percentile(&ll, 1.0),
                median(&apps),
                percentile(&apps, 1.0),
            );
        }
        let soft: Vec<&Row> = rows.iter().filter(|r| r.anomaly.is_some() && !r.fatal).collect();
        println!("\nnon-fatal anomalies (more saturation extracted worse): {}", soft.len());

        let fatal: Vec<String> = rows
            .iter()
            .filter(|r| r.fatal)
            .map(|r| format!("{}: {}", r.group, r.anomaly.as_deref().unwrap_or("?")))
            .collect();
        assert!(
            fatal.is_empty(),
            "table complete ({} rows, written to {}) but {} row(s) are not trustworthy:\n{}",
            rows.len(),
            out_path.display(),
            fatal.len(),
            fatal.join("\n")
        );
    }
}
