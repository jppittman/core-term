//! Global JIT compile cache: identical kernels compile once.
//!
//! Skipping codegen entirely avoids recompiling a kernel that was already
//! compiled. The macro-emitted builders can hit this frequently: every
//! call of an N-param builder with the same arguments (window resizes), and
//! every structurally identical glyph kernel in a bake sweep, produces a
//! byte-identical schedule.
//!
//! Keys are the [`LatticeShape`] the kernel is compiled for — its extents,
//! so a lattice of a different size is a different kernel and a window
//! resize recompiles, by decision — plus the **canonical form of the
//! reachable subgraph**: nodes in ascending id order with ids remapped
//! dense. Construction garbage (dead
//! nodes left behind by `substitute_params` / splicing rebuilds) does not
//! perturb the key, so logically identical kernels hit regardless of build
//! history. Keys are compared by full equality — a hash collision can cause
//! a wasted probe, never wrong code.
//!
//! Kernels that read bound memory (`Buffer` leaves) are compiled fresh: their
//! ABI differs (context-pointer calls) and their code bakes buffer slot
//! metadata. No kernel-macro surface produces them today.
//!
//! The cache is unbounded: entries are one executable-memory region each and
//! the population is the program's distinct kernel set, which is bounded by
//! construction (kernels are made at load/composition time, not per frame).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::vec::Vec;

use crate::CompiledKernel;
use crate::emit;
use crate::error::CompileError;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Arc<CompiledKernel>>>> = OnceLock::new();

/// Compile the kernel rooted at `root` to an executable [`CompiledKernel`] (2D collapse loop)
/// for a lattice of the given `shape`, sharing previously compiled code for
/// canonically identical kernels at the same extents.
///
/// The returned `Arc` is the shared handle — two constructions of the same
/// kernel at the same shape yield pointer-equal manifolds.
///
/// # Errors
///
/// Whatever [`emit::compile`] reports for the (optimized) arena.
pub fn compile(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Result<Arc<CompiledKernel>, CompileError> {
    // Optimize, then emit. This is not a step callers get to sequence: an
    // arena reaching a backend unoptimized is never what anyone wanted, and
    // when the choice was on offer, two of the three production call sites
    // took the wrong one and compiled the terminal's cell-grid kernels with
    // no CSE and no FMA fusion. It is inside the compile entry because that
    // is the only place it cannot be forgotten.
    //
    // It bails to the arena as given for constructs the e-graph does not
    // model (a `Tuple` root; `Reduce` is unrolled ahead of saturation and
    // does optimize); those still compile, just without the extra fusion.
    let emit_fn = |arena: &ExprArena, root: ExprId| {
        let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape);
        let (arena, root) = optimized
            .as_deref()
            .map(|(a, r)| (a, *r))
            .unwrap_or((arena, root));
        emit::compile(arena, root)
    };

    let Some(mut key) = canonical_key(arena, root) else {
        // Uncacheable (bound memory): compile fresh.
        let result = emit_fn(arena, root)?;
        return Ok(Arc::new(CompiledKernel::new(result.code, shape)));
    };

    // Keyed on the arena *as handed in*, before optimization, plus the shape.
    // Optimization is a deterministic function of those two, so equal inputs
    // yield equal output and a hit skips the saturation as well as the codegen.
    key.extend_from_slice(&shape.key_bytes());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().expect("jit_cache: lock poisoned").get(&key) {
        return Ok(hit.clone());
    }

    // Compile outside the lock so concurrent distinct-kernel constructions
    // don't serialize. A racing duplicate compile wastes work; the first
    // insertion wins so all callers share one region.
    let result = emit_fn(arena, root)?;
    let compiled = Arc::new(CompiledKernel::new(result.code, shape));
    let mut guard = cache.lock().expect("jit_cache: lock poisoned");
    Ok(guard.entry(key).or_insert(compiled).clone())
}

/// Number of distinct kernels interned so far (test/telemetry hook).
#[must_use]
pub fn entry_count() -> usize {
    CACHE
        .get()
        .map(|c| c.lock().expect("jit_cache: lock poisoned").len())
        .unwrap_or(0)
}

/// Canonical serialization of the subgraph reachable from `root`: nodes in
/// ascending original id order (the arena is append-only, so children always
/// precede parents), child references remapped to dense indices. `None` if the
/// subgraph reads bound memory.
fn canonical_key(arena: &ExprArena, root: ExprId) -> Option<Vec<u8>> {
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }

    // Dense remap in ascending id order.
    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let mut key: Vec<u8> = Vec::with_capacity(len * 8);

    let push_id = |key: &mut Vec<u8>, dense: &[u32], id: ExprId| {
        let d = dense[id.0 as usize];
        debug_assert_ne!(d, u32::MAX, "child densified before parent");
        key.extend_from_slice(&d.to_le_bytes());
    };

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        match arena.node(ExprId(idx as u32)) {
            ExprNode::Var(i) => {
                key.push(0);
                key.push(*i);
            }
            ExprNode::Const(v) => {
                key.push(1);
                key.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            ExprNode::Param(i) => {
                key.push(2);
                key.push(*i);
            }
            ExprNode::Buffer(_) => return None,
            ExprNode::Unary(op, a) => {
                key.push(3);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, *a);
            }
            ExprNode::Binary(op, a, b) => {
                key.push(4);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, *a);
                push_id(&mut key, &dense, *b);
            }
            ExprNode::Ternary(op, a, b, c) => {
                key.push(5);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, *a);
                push_id(&mut key, &dense, *b);
                push_id(&mut key, &dense, *c);
            }
            ExprNode::Nary(op, start, n) => {
                key.push(6);
                key.extend_from_slice(&op.marshal().to_bytes());
                key.extend_from_slice(&n.to_le_bytes());
                let (s, l) = (*start as usize, *n as usize);
                for child in &arena.nary_children_raw()[s..s + l] {
                    push_id(&mut key, &dense, *child);
                }
            }
        }
        dense[idx] = next;
        next += 1;
    }

    Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::kind::OpKind;

    const TEST_SHAPE: LatticeShape = LatticeShape::new([64, 64, 1, 1]);

    fn circle_arena(garbage: bool) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        if garbage {
            // Construction garbage: unreachable nodes must not perturb the key.
            let g = a.push_const(123.0);
            let _ = a.push_unary(OpKind::Sqrt, g);
        }
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let s = a.push_binary(OpKind::Add, x2, y2);
        let root = a.push_unary(OpKind::Sqrt, s);
        (a, root)
    }

    #[test]
    fn identical_kernels_share_code() {
        let (a1, r1) = circle_arena(false);
        let (a2, r2) = circle_arena(true);
        let m1 = compile(&a1, r1, TEST_SHAPE).expect("compile");
        let m2 = compile(&a2, r2, TEST_SHAPE).expect("compile");
        assert!(
            Arc::ptr_eq(&m1, &m2),
            "canonically identical kernels must share one compiled region"
        );
    }

    #[test]
    fn distinct_kernels_do_not_collide() {
        let (a1, r1) = circle_arena(false);
        let mut a2 = ExprArena::new();
        let x = a2.push_var(0);
        let y = a2.push_var(1);
        let r2 = a2.push_binary(OpKind::Sub, x, y);
        let m1 = compile(&a1, r1, TEST_SHAPE).expect("compile");
        let m2 = compile(&a2, r2, TEST_SHAPE).expect("compile");
        assert!(!Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn entry_count_is_monotonic_and_grows_on_new_kernels() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let k = a.push_const(424_242.0);
        let r = a.push_binary(OpKind::Mul, x, k);
        let before = entry_count();
        let _m1 = compile(&a, r, TEST_SHAPE).expect("compile");
        let after_one = entry_count();
        assert!(
            after_one > before,
            "compiling a new kernel must grow the cache"
        );

        let mut a2 = ExprArena::new();
        let y = a2.push_var(1);
        let k2 = a2.push_const(535_353.0);
        let r2 = a2.push_binary(OpKind::Mul, y, k2);
        let _m2 = compile(&a2, r2, TEST_SHAPE).expect("compile");
        let after_two = entry_count();
        assert!(
            after_two > after_one,
            "compiling another distinct kernel must grow the cache again"
        );
    }

    #[test]
    fn nary_reduce_children_affect_kernel_identity() {
        // Two back-to-back `Reduce` nodes in the same arena: the second one's
        // children start at a nonzero offset into the flat nary-children slab,
        // so an off-by-slicing bug in the second node's child range either
        // indexes out of bounds or silently drops its children from the key,
        // making two kernels that differ only in the second reduce collide.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let body1 = a.push_binary(OpKind::Add, x, x);
        let r1 = a.push_reduce(OpKind::Add, 4, 3, body1);
        let body2 = a.push_binary(OpKind::Mul, x, x);
        let r2 = a.push_reduce(OpKind::Mul, 5, 6, body2);
        let root = a.push_binary(OpKind::Add, r1, r2);

        let mut a2 = ExprArena::new();
        let x2 = a2.push_var(0);
        let body1b = a2.push_binary(OpKind::Add, x2, x2);
        let r1b = a2.push_reduce(OpKind::Add, 4, 3, body1b);
        let body2b = a2.push_binary(OpKind::Mul, x2, x2);
        let r2b = a2.push_reduce(OpKind::Mul, 5, 9, body2b); // extent differs only here
        let root2 = a2.push_binary(OpKind::Add, r1b, r2b);

        let m1 = compile(&a, root, TEST_SHAPE).expect("compile");
        let m2 = compile(&a2, root2, TEST_SHAPE).expect("compile");
        assert!(
            !Arc::ptr_eq(&m1, &m2),
            "a change confined to the second reduce's extent must not share a cache entry"
        );
    }

    #[test]
    fn operand_order_flip_is_a_distinct_kernel() {
        let (a1, r1) = circle_arena(false);

        let mut a3 = ExprArena::new();
        let x = a3.push_var(0);
        let y = a3.push_var(1);
        let x2 = a3.push_binary(OpKind::Mul, x, x);
        let y2 = a3.push_binary(OpKind::Mul, y, y);
        let s = a3.push_binary(OpKind::Add, y2, x2); // operand order flipped
        let r3 = a3.push_unary(OpKind::Sqrt, s);

        let m1 = compile(&a1, r1, TEST_SHAPE).expect("compile");
        let m3 = compile(&a3, r3, TEST_SHAPE).expect("compile");
        assert!(
            !Arc::ptr_eq(&m1, &m3),
            "flipping operand order must not share a cache entry"
        );
    }

    #[test]
    fn same_kernel_at_two_extents_is_two_entries() {
        let (a, r) = circle_arena(false);
        let frame = compile(&a, r, TEST_SHAPE).expect("compile");
        let again = compile(&a, r, TEST_SHAPE).expect("compile");
        let wider = compile(&a, r, LatticeShape::new([65, 64, 1, 1])).expect("compile");
        assert!(
            Arc::ptr_eq(&frame, &again),
            "the same kernel at the same extents must share one entry"
        );
        assert!(
            !Arc::ptr_eq(&frame, &wider),
            "one more column is a different lattice, hence a different kernel"
        );
        assert_eq!(frame.shape(), TEST_SHAPE);
        assert_eq!(wider.shape().extent(), [65, 64, 1, 1]);
    }
}
