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
//! ## The link step
//!
//! A kernel that reads bound memory or a uniform names it by *identity*, and
//! identities are minted per instance — so keyed by identity, two
//! compositions of the same shape would never share code. They share it
//! instead by **dense slot**: the same canonical traversal that produces the
//! key numbers each distinct buffer and uniform identity by first
//! occurrence, and that numbering is what the emitted code is compiled
//! against — the buffer's context entry, the uniform's offset in the block.
//! The traversal is a function of structure alone, so the same shape gets
//! the same numbering, and a thousand circles differing only in which
//! instances they read are one compile. What differs between them is the
//! [`Linked`] table handed back with the code: which identity each slot
//! binds, which the caller uses to build the context and the block.
//!
//! Buffer *extents* stay in the key. Two kernels over buffers of different
//! shapes are different code, since the shape is what the address arithmetic
//! was folded against.
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
use pixelflow_ir::arena::{BufferDecl, ExprArena, ExprId, ExprNode, UniformDecl};

static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Arc<CompiledKernel>>>> = OnceLock::new();

/// Compiled code plus the link: which identity each slot the code was
/// compiled against binds.
///
/// The code is shared through the cache with every kernel of the same shape;
/// the link is this kernel's own. A caller builds the context from
/// `buffers` in slot order, and the block from `uniforms` in offset order —
/// the block pointer goes in the context entry after the last buffer, and
/// only when `uniforms` is non-empty.
pub struct Linked {
    /// The shared code.
    pub kernel: Arc<CompiledKernel>,
    /// The buffer each context slot binds, in slot order.
    pub buffers: Vec<BufferDecl>,
    /// The uniform each block offset holds, in offset order.
    pub uniforms: Vec<UniformDecl>,
}

/// Compile the kernel rooted at `root` to an executable [`CompiledKernel`] (2D collapse loop)
/// for a lattice of the given `shape`, sharing previously compiled code for
/// canonically identical kernels at the same extents.
///
/// The returned `Arc` is the shared handle — two constructions of the same
/// kernel at the same shape yield pointer-equal manifolds — and the link
/// beside it says what this construction's slots bind.
///
/// # Errors
///
/// Whatever [`emit::compile`] reports for the (optimized) arena.
pub fn compile(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Result<Linked, CompileError> {
    let Canonical {
        mut key,
        buffers,
        uniforms,
    } = canonical(arena, root);

    // Optimize, link, then emit. This is not a step callers get to sequence:
    // an arena reaching a backend unoptimized is never what anyone wanted,
    // and when the choice was on offer, two of the three production call
    // sites took the wrong one and compiled the terminal's cell-grid kernels
    // with no CSE and no FMA fusion. It is inside the compile entry because
    // that is the only place it cannot be forgotten.
    //
    // It bails to the arena as given for constructs the e-graph does not
    // model (a `Tuple` root; `Reduce` is unrolled ahead of saturation and
    // does optimize); those still compile, just without the extra fusion.
    //
    // The relink after it renumbers the tables into the canonical order the
    // key was built from — extraction redeclares identities in its own
    // walk order — without touching a node, so the bytes of a kernel that
    // declares neither a buffer nor a uniform are exactly what they were.
    let emit_fn = |arena: &ExprArena, root: ExprId| {
        let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape);
        let (arena, root) = optimized
            .as_deref()
            .map(|(a, r)| (a, *r))
            .unwrap_or((arena, root));
        if buffers.is_empty() && uniforms.is_empty() {
            return emit::compile(arena, root);
        }
        let (linked, root) = arena.relink(root, &buffers, &uniforms);
        emit::compile(&linked, root)
    };

    // Keyed on the arena *as handed in*, before optimization, plus the shape.
    // Optimization is a deterministic function of those two, so equal inputs
    // yield equal output and a hit skips the saturation as well as the codegen.
    key.extend_from_slice(&shape.key_bytes());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().expect("jit_cache: lock poisoned").get(&key) {
        return Ok(Linked {
            kernel: hit.clone(),
            buffers,
            uniforms,
        });
    }

    // Compile outside the lock so concurrent distinct-kernel constructions
    // don't serialize. A racing duplicate compile wastes work; the first
    // insertion wins so all callers share one region.
    let result = emit_fn(arena, root)?;
    let compiled = Arc::new(CompiledKernel::new(result.code, shape));
    let mut guard = cache.lock().expect("jit_cache: lock poisoned");
    let kernel = guard.entry(key).or_insert(compiled).clone();
    Ok(Linked {
        kernel,
        buffers,
        uniforms,
    })
}

/// Number of distinct kernels interned so far (test/telemetry hook).
#[must_use]
pub fn entry_count() -> usize {
    CACHE
        .get()
        .map(|c| c.lock().expect("jit_cache: lock poisoned").len())
        .unwrap_or(0)
}

/// The canonical form of a reachable subgraph: its key, and the buffer and
/// uniform declarations in the dense order the key numbers them by.
struct Canonical {
    key: Vec<u8>,
    buffers: Vec<BufferDecl>,
    uniforms: Vec<UniformDecl>,
}

/// Canonical serialization of the subgraph reachable from `root`: nodes in
/// ascending original id order (the arena is append-only, so children always
/// precede parents), child references remapped to dense indices, and buffer
/// and uniform leaves remapped to dense slots by first occurrence in that
/// same order — never by identity, which is what lets two compositions of
/// one shape share code.
fn canonical(arena: &ExprArena, root: ExprId) -> Canonical {
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
    let mut buffers: Vec<BufferDecl> = Vec::new();
    let mut uniforms: Vec<UniformDecl> = Vec::new();

    let push_id = |key: &mut Vec<u8>, dense: &[u32], id: ExprId| {
        let d = dense[id.0 as usize];
        debug_assert_ne!(d, u32::MAX, "child densified before parent");
        key.extend_from_slice(&d.to_le_bytes());
    };
    /// The dense slot of `decl` in `table`, appending it on first sight.
    fn dense_slot<T: PartialEq + Copy>(table: &mut Vec<T>, decl: T) -> u16 {
        let slot = table.iter().position(|d| *d == decl).unwrap_or_else(|| {
            table.push(decl);
            table.len() - 1
        });
        u16::try_from(slot).expect("dense slot fits the table index width")
    }

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
            // Slot by first occurrence, extents in the key: the code folds
            // its address arithmetic against them.
            ExprNode::Buffer(b) => {
                let decl = *arena.buffer_decl(*b);
                key.push(7);
                key.extend_from_slice(&dense_slot(&mut buffers, decl).to_le_bytes());
                key.extend_from_slice(&decl.width.to_le_bytes());
                key.extend_from_slice(&decl.height.to_le_bytes());
            }
            // Offset by first occurrence; the default is the block's
            // business, not the code's.
            ExprNode::Uniform(u) => {
                let decl = *arena.uniform_decl(*u);
                key.push(8);
                key.extend_from_slice(&dense_slot(&mut uniforms, decl).to_le_bytes());
            }
        }
        dense[idx] = next;
        next += 1;
    }

    Canonical {
        key,
        buffers,
        uniforms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::arena::{BufferIdentity, UniformIdentity};
    use pixelflow_ir::kind::OpKind;

    const TEST_SHAPE: LatticeShape = LatticeShape::new([64, 64, 1, 1]);

    fn kernel_of(arena: &ExprArena, root: ExprId) -> Arc<CompiledKernel> {
        compile(arena, root, TEST_SHAPE).expect("compile").kernel
    }

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
        let m1 = kernel_of(&a1, r1);
        let m2 = kernel_of(&a2, r2);
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
        let m1 = kernel_of(&a1, r1);
        let m2 = kernel_of(&a2, r2);
        assert!(!Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn entry_count_is_monotonic_and_grows_on_new_kernels() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let k = a.push_const(424_242.0);
        let r = a.push_binary(OpKind::Mul, x, k);
        let before = entry_count();
        let _m1 = kernel_of(&a, r);
        let after_one = entry_count();
        assert!(
            after_one > before,
            "compiling a new kernel must grow the cache"
        );

        let mut a2 = ExprArena::new();
        let y = a2.push_var(1);
        let k2 = a2.push_const(535_353.0);
        let r2 = a2.push_binary(OpKind::Mul, y, k2);
        let _m2 = kernel_of(&a2, r2);
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

        let m1 = kernel_of(&a, root);
        let m2 = kernel_of(&a2, root2);
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

        let m1 = kernel_of(&a1, r1);
        let m3 = kernel_of(&a3, r3);
        assert!(
            !Arc::ptr_eq(&m1, &m3),
            "flipping operand order must not share a cache entry"
        );
    }

    #[test]
    fn same_kernel_at_two_extents_is_two_entries() {
        let (a, r) = circle_arena(false);
        let frame = kernel_of(&a, r);
        let again = kernel_of(&a, r);
        let wider = compile(&a, r, LatticeShape::new([65, 64, 1, 1]))
            .expect("compile")
            .kernel;
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

    // ─────────────────────── the link step ───────────────────────

    /// `(x − cx)·r + cy` over fresh uniform instances: one shape, many
    /// factors. `declared_first` flips the table order so the link, not the
    /// declaration order, is what the code is compiled against.
    fn circle_of(declared_first: bool) -> (ExprArena, ExprId, [UniformDecl; 3]) {
        let decl = |default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        };
        let (cx, cy, r) = (decl(0.0), decl(0.0), decl(1.0));
        let mut a = ExprArena::new();
        let (scx, scy, sr) = if declared_first {
            let scx = a.declare_uniform(cx);
            let scy = a.declare_uniform(cy);
            let sr = a.declare_uniform(r);
            (scx, scy, sr)
        } else {
            let sr = a.declare_uniform(r);
            let scy = a.declare_uniform(cy);
            let scx = a.declare_uniform(cx);
            (scx, scy, sr)
        };
        let x = a.push_var(0);
        let ucx = a.push_uniform(scx);
        let ur = a.push_uniform(sr);
        let ucy = a.push_uniform(scy);
        let d = a.push_binary(OpKind::Sub, x, ucx);
        let scaled = a.push_binary(OpKind::Mul, d, ur);
        let root = a.push_binary(OpKind::Add, scaled, ucy);
        (a, root, [cx, r, cy])
    }

    /// Every instance shares the one region and gets its own link. The
    /// cache's *count* is asserted in `tests/one_compile_per_shape.rs`, a
    /// binary of its own: the count is process-global, and the other tests
    /// in this one compile kernels concurrently.
    #[test]
    fn a_thousand_circles_share_one_region_with_a_thousand_links() {
        let mut first: Option<Arc<CompiledKernel>> = None;
        for i in 0..1000 {
            let (a, root, [cx, r, cy]) = circle_of(i % 2 == 0);
            let linked = compile(&a, root, TEST_SHAPE).expect("compile");
            // The link is this instance's, in first-occurrence order.
            assert_eq!(linked.uniforms, [cx, r, cy]);
            assert!(linked.buffers.is_empty());
            match &first {
                None => first = Some(linked.kernel),
                Some(k) => assert!(Arc::ptr_eq(k, &linked.kernel), "circle {i} recompiled"),
            }
        }
    }

    #[test]
    fn a_uniform_and_a_constant_are_different_kernels() {
        let (a, root, _) = circle_of(true);
        let mut folded = ExprArena::new();
        let x = folded.push_var(0);
        let cx = folded.push_const(0.0);
        let r = folded.push_const(1.0);
        let cy = folded.push_const(0.0);
        let d = folded.push_binary(OpKind::Sub, x, cx);
        let scaled = folded.push_binary(OpKind::Mul, d, r);
        let froot = folded.push_binary(OpKind::Add, scaled, cy);
        assert!(!Arc::ptr_eq(
            &kernel_of(&a, root),
            &kernel_of(&folded, froot)
        ));
    }

    fn gather_over(decl: BufferDecl) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(decl);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_gather(buf, x, y);
        (a, root)
    }

    #[test]
    fn bound_memory_of_one_shape_shares_code_and_of_another_does_not() {
        let decl = |width, height| BufferDecl {
            id: BufferIdentity::mint(),
            width,
            height,
        };
        let (a1, r1) = gather_over(decl(8, 4));
        let (a2, r2) = gather_over(decl(8, 4));
        let (a3, r3) = gather_over(decl(8, 5));
        let l1 = compile(&a1, r1, TEST_SHAPE).expect("compile");
        let l2 = compile(&a2, r2, TEST_SHAPE).expect("compile");
        let l3 = compile(&a3, r3, TEST_SHAPE).expect("compile");
        assert!(
            Arc::ptr_eq(&l1.kernel, &l2.kernel),
            "two atlases of one shape are one kernel — the link tells them apart"
        );
        assert_ne!(l1.buffers[0].id, l2.buffers[0].id);
        assert!(
            !Arc::ptr_eq(&l1.kernel, &l3.kernel),
            "a buffer of different extents is different code"
        );
    }

    /// The link is by first occurrence in the canonical traversal, and the
    /// relinked arena's slots follow it regardless of declaration order.
    #[test]
    fn slots_follow_first_occurrence_not_declaration_order() {
        let (a, root, [cx, r, cy]) = circle_of(false);
        assert_eq!(a.uniforms(), &[r, cy, cx], "declared in the other order");
        let Canonical { uniforms, .. } = canonical(&a, root);
        assert_eq!(uniforms, [cx, r, cy]);
        let (linked, lroot) = a.relink(root, &[], &uniforms);
        assert_eq!(linked.uniforms(), &[cx, r, cy]);
        assert_eq!(
            linked.nodes_raw().len(),
            a.nodes_raw().len(),
            "every node here is reachable, so relinking keeps them all"
        );
        assert_eq!(
            canonical(&linked, lroot).key,
            canonical(&a, root).key,
            "relinking changes no structure"
        );
        // Both declaration orders canonicalize to one key — the shape, not
        // the table, is what the code is a function of.
        let (b, broot, _) = circle_of(true);
        assert_eq!(canonical(&a, root).key, canonical(&b, broot).key);
    }

    /// A table naming an instance the graph never reads — what `Kernel::at`
    /// leaves behind for every axis the receiver does not read — must link.
    /// The link omits it; the relink drops it with the garbage. Forced down
    /// the optimizer's bail path (`RawGather` is post-lowering, so the
    /// e-graph declines and the input arena, table and all, is what gets
    /// linked), which is where an unreachable declaration used to panic.
    #[test]
    fn a_declared_but_unread_uniform_links_on_the_bail_path() {
        let decl = |default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        };
        let (phase, scale) = (decl(0.0), decl(2.0));
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 4,
            height: 1,
        });
        // Declared and read by a node nothing reaches — the `.at` shape.
        let dead = a.declare_uniform(phase);
        let _dead_read = a.push_uniform(dead);
        let live = a.declare_uniform(scale);
        let x = a.push_var(0);
        let b = a.push_buffer(buf);
        let g = a.push_binary(OpKind::RawGather, b, x);
        let s = a.push_uniform(live);
        let root = a.push_binary(OpKind::Mul, g, s);
        assert_eq!(a.uniforms().len(), 2, "the table names both");

        let linked =
            compile(&a, root, TEST_SHAPE).expect("a dead declaration must not refuse to link");
        assert_eq!(
            linked.uniforms,
            [scale],
            "the link holds what the graph reads"
        );
        assert_eq!(linked.buffers.len(), 1);
    }
}
