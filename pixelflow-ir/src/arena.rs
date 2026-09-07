//! Arena-allocated expression storage.
//!
//! [`ExprArena`] stores expression nodes in a flat `Vec<ExprNode>`, indexed by
//! [`ExprId`] (a 4-byte Copy handle). This eliminates per-node Arc overhead and
//! gives O(1) `len()` for node counting.
//!
//! The arena is append-only. [`ExprArena::clear`] truncates without deallocating,
//! ready for reuse.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::kernel::Scalar;
use crate::kind::OpKind;

/// Coordinate axes a lattice has, and so the coordinate `Var` indices: `X = 0`,
/// `Y = 1`.
///
/// There were four. Z and W had extent 1 in every production call — an axis
/// that never varies is not an axis — so they left the language and the
/// scalars they carried became [`UniformDecl`]s
/// (docs/plans/2026-09-06-lattice-is-the-index.md).
pub const COORD_AXES: usize = 2;

/// The `Var` indices Z and W had. Reserved, never reissued: a reduction
/// binder taking one of them would make an arena written before the change
/// read back as a different program.
pub const RETIRED_COORD_AXES: [u8; 2] = [2, 3];

// ───────────────────────────────────────── ExprId ─────────────────────────────

/// Index into an [`ExprArena`]. Copy, 4 bytes, no refcount.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ExprId(pub u32);

// ───────────────────────────────────────── Buffers ────────────────────────────

/// Slot index into an [`ExprArena`]'s buffer table. Copy, 2 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BufferId(pub u16);

/// Which block of memory a declaration refers to, independent of any arena.
///
/// [`BufferId`] is a slot index into *one* arena's table, so it cannot answer
/// "the same buffer?" across a merge — two fragments each call their own
/// buffer slot 0. Extents cannot answer it either: two atlases of equal size
/// are a coincidence, not a fact, and treating them as one would bind a single
/// pointer for both and silently read the wrong pixels.
///
/// So identity is provenance. You get one by minting it, and copy it into
/// every declaration that names that memory; nothing else can collide with it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BufferIdentity(u32);

impl BufferIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and aliasing two unrelated buffers.
    #[must_use]
    pub fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        Self(mint_identity(&NEXT, "BufferIdentity"))
    }
}

/// The one counter discipline behind every provenance identity.
///
/// `fetch_add` + assert was wrong: the add WRAPS before the assert fires, so
/// if that panic is ever caught — or merely unwinds a non-fatal worker thread
/// — the counter has already returned to 0 and the next mint hands out an
/// identity that is still live. Two unrelated buffers (or uniforms) would
/// then compare identical and merge into one splice/JIT slot. `fetch_update`
/// declining to store leaves the counter permanently exhausted instead.
fn mint_identity(counter: &core::sync::atomic::AtomicU32, what: &str) -> u32 {
    counter
        .fetch_update(
            core::sync::atomic::Ordering::Relaxed,
            core::sync::atomic::Ordering::Relaxed,
            |n| n.checked_add(1),
        )
        .unwrap_or_else(|_| panic!("{what}: counter exhausted"))
}

/// Declaration of a bound memory buffer: the static shape of a collapsed
/// lattice. The extents are part of the IR (like a `Const`) even though the
/// contents are bound later, at JIT-compile time. Static extents are what
/// allow the emitter to fold address arithmetic, drop provably in-bounds
/// clamps, and unroll reduction loops.
///
/// Layout is row-major with `stride == width`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferDecl {
    /// Which memory this names. Two declarations of the same buffer must carry
    /// the same identity — that is what lets a merge collapse them into one
    /// slot instead of binding the pointer twice.
    pub id: BufferIdentity,
    /// X extent (samples per row).
    pub width: u32,
    /// Y extent (number of rows).
    pub height: u32,
}

// ───────────────────────────────────────── Uniforms ───────────────────────────

/// Slot index into an [`ExprArena`]'s uniform table. Copy, 2 bytes. Not an
/// identity: two arenas each call their own first uniform slot 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformId(pub u16);

/// Which uniform a declaration refers to, independent of any arena.
///
/// A uniform is a scalar that is invariant across the lattice and supplied
/// at call time — the JIT tier's spelling of a builder's struct field. Its
/// identity is the *instance*: two instances of the same builder are two
/// factors of the kernel's parameter space, and one instance read from
/// twenty places is one factor. Neither a name nor a fragment-local index can
/// say that across a splice, so, exactly as for [`BufferIdentity`], identity
/// is provenance: minted once, copied into every declaration of the instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformIdentity(u32);

impl UniformIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and aliasing two unrelated uniforms.
    #[must_use]
    pub fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        Self(mint_identity(&NEXT, "UniformIdentity"))
    }
}

/// Declaration of a uniform: its identity plus the value the kernel holds
/// for it when nothing has been bound. The default is part of the IR so that
/// a bake without a block, and the scalar oracle, are total.
///
/// Compared and hashed by the default's *bit pattern*, so a declaration is a
/// plain key: `-0.0` and `0.0` are two defaults, and a NaN default is equal
/// to itself.
#[derive(Clone, Copy, Debug)]
pub struct UniformDecl {
    /// Which instance this names.
    pub id: UniformIdentity,
    /// The value bound when no block supplies one.
    pub default: f32,
}

impl PartialEq for UniformDecl {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.default.to_bits() == other.default.to_bits()
    }
}

impl Eq for UniformDecl {}

impl core::hash::Hash for UniformDecl {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.default.to_bits().hash(state);
    }
}

// ───────────────────────────────────────── ExprNode ───────────────────────────

/// A single expression node stored in the arena.
///
/// Layout is kept tight: the static assertion below guarantees <= 16 bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprNode {
    /// A bound variable: a lattice coordinate ([`COORD_AXES`] of them, X and
    /// Y), a reduction binder's index (`4..8`), or — in the macro front end
    /// only, before substitution — a parameter placeholder. Which one an
    /// index means is [`ExprArena::push_var`]'s documentation; the indices
    /// between the coordinates and the binders are not a hole to grow into,
    /// they are where Z and W used to be and nothing may claim them.
    Var(u8),
    Const(f32),
    Param(u8),
    /// Bound-memory leaf: references a [`BufferDecl`] in the arena's buffer
    /// table. Read through `Ternary(OpKind::Gather, buffer, x, y)`.
    Buffer(BufferId),
    /// Lattice-invariant scalar supplied per call: references a
    /// [`UniformDecl`] in the arena's uniform table. Never folded — its value
    /// is unknown until the call — and constant across the lattice, so it is
    /// loaded once per call rather than once per batch.
    Uniform(UniformId),
    Unary(OpKind, ExprId),
    Binary(OpKind, ExprId, ExprId),
    Ternary(OpKind, ExprId, ExprId, ExprId),
    /// N-ary node. Children live in `ExprArena::nary_children[start..start+len]`.
    Nary(OpKind, u32, u16),
}

const _: () = assert!(
    core::mem::size_of::<ExprNode>() <= 16,
    "ExprNode must fit in 16 bytes"
);

// ───────────────────────────────────── ExprChildren ──────────────────────────

/// Iterator over the child [`ExprId`]s of a node.
pub enum ExprChildren<'a> {
    Zero,
    One(ExprId),
    Two(ExprId, ExprId),
    Three(ExprId, ExprId, ExprId),
    Nary(&'a [ExprId]),
}

impl<'a> Iterator for ExprChildren<'a> {
    type Item = ExprId;

    fn next(&mut self) -> Option<ExprId> {
        match self {
            Self::Zero => None,
            Self::One(id) => {
                let id = *id;
                *self = Self::Zero;
                Some(id)
            }
            Self::Two(a, b) => {
                let a = *a;
                let b = *b;
                *self = Self::One(b);
                Some(a)
            }
            Self::Three(a, b, c) => {
                let a = *a;
                let b = *b;
                let c = *c;
                *self = Self::Two(b, c);
                Some(a)
            }
            Self::Nary(slice) => {
                if slice.is_empty() {
                    None
                } else {
                    let first = slice[0];
                    *self = Self::Nary(&slice[1..]);
                    Some(first)
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self {
            Self::Zero => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
            Self::Three(_, _, _) => 3,
            Self::Nary(s) => s.len(),
        };
        (n, Some(n))
    }
}

impl ExactSizeIterator for ExprChildren<'_> {}

// ───────────────────────────────────── ExprArena ─────────────────────────────

/// Arena-allocated expression storage. Append-only, O(1) drop.
#[derive(Clone)]
pub struct ExprArena {
    nodes: Vec<ExprNode>,
    nary_children: Vec<ExprId>,
    /// Buffer declarations, indexed by [`BufferId`]. The memory analogue of
    /// the symbol table: shapes are static IR, contents are bound at JIT time.
    buffers: Vec<BufferDecl>,
    /// Uniform declarations, indexed by [`UniformId`]: the scalar arguments
    /// of the kernel, each with its default. Values are bound per call.
    uniforms: Vec<UniformDecl>,
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprArena {
    /// Create an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            nary_children: Vec::new(),
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// Create an arena pre-allocated for `n` nodes.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n),
            nary_children: Vec::new(),
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// Truncate to zero nodes without deallocating backing storage.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nary_children.clear();
        self.buffers.clear();
        self.uniforms.clear();
    }

    /// Number of nodes in the arena. This is the O(1) node count.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the arena contains no nodes.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ───────────────────── push helpers ──────────────────────

    fn push_node(&mut self, node: ExprNode) -> ExprId {
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Push a `Var(i)` node.
    ///
    /// Only `0..COORD_AXES` are lattice coordinates. [`RETIRED_COORD_AXES`]
    /// are the Z and W axes, which no longer exist: an arena that names one
    /// is refused where it would become code
    /// ([`Kernel::from_parts`](crate::Kernel::from_parts), and the JIT cache),
    /// rather than here, because the same node is also a reduction binder's
    /// index and a rewrite rule's pattern metavariable, and those namespaces
    /// are dense from zero.
    pub fn push_var(&mut self, i: u8) -> ExprId {
        self.push_node(ExprNode::Var(i))
    }

    /// The retired coordinate axis this arena names, if any — the guard
    /// [`Kernel::from_parts`](crate::Kernel::from_parts) and the JIT cache
    /// apply before an arena can become a compiled kernel.
    ///
    /// A `Var(2)` reaching the emitter would read the third base coordinate,
    /// which a collapse passes as zero and no longer means anything: the
    /// pixels would be plausible and wrong. Refusing it is what makes "no
    /// emitted kernel reads the retired lanes" a fact rather than a habit.
    #[must_use]
    pub fn retired_axis(&self) -> Option<u8> {
        self.nodes.iter().find_map(|n| match n {
            ExprNode::Var(i) if RETIRED_COORD_AXES.contains(i) => Some(*i),
            _ => None,
        })
    }

    /// Push a `Const(v)` node.
    pub fn push_const(&mut self, v: f32) -> ExprId {
        self.push_node(ExprNode::Const(v))
    }

    /// Push a `Param(i)` node.
    pub fn push_param(&mut self, i: u8) -> ExprId {
        self.push_node(ExprNode::Param(i))
    }

    /// Declare a buffer slot, returning its [`BufferId`].
    ///
    /// # Panics
    ///
    /// Panics if the buffer table is full (`u16::MAX` slots).
    pub fn declare_buffer(&mut self, decl: BufferDecl) -> BufferId {
        assert!(
            self.buffers.len() < u16::MAX as usize,
            "declare_buffer: buffer table full ({} slots)",
            self.buffers.len()
        );
        let id = BufferId(self.buffers.len() as u16);
        self.buffers.push(decl);
        id
    }

    /// Push a `Buffer(id)` leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been declared via [`ExprArena::declare_buffer`].
    pub fn push_buffer(&mut self, id: BufferId) -> ExprId {
        assert!(
            (id.0 as usize) < self.buffers.len(),
            "push_buffer: BufferId({}) not declared (table has {} entries)",
            id.0,
            self.buffers.len()
        );
        self.push_node(ExprNode::Buffer(id))
    }

    /// Declare a uniform slot, returning its [`UniformId`].
    ///
    /// # Panics
    ///
    /// Panics if the uniform table is full (`u16::MAX` slots).
    pub fn declare_uniform(&mut self, decl: UniformDecl) -> UniformId {
        assert!(
            self.uniforms.len() < u16::MAX as usize,
            "declare_uniform: uniform table full ({} slots)",
            self.uniforms.len()
        );
        let id = UniformId(self.uniforms.len() as u16);
        self.uniforms.push(decl);
        id
    }

    /// The slot naming `decl`'s instance in this arena, declaring one if this
    /// is the first time that identity has been seen.
    ///
    /// Two declarations of one identity with different defaults cannot happen
    /// through the handle that minted it — the default travels with the
    /// identity in one `Copy` value — so it is a corrupt graph, and an
    /// assertion rather than a silent alias onto whichever arrived first.
    pub(crate) fn uniform_slot_for(&mut self, decl: UniformDecl) -> UniformId {
        match self.uniforms.iter().position(|d| d.id == decl.id) {
            Some(i) => {
                assert_eq!(
                    self.uniforms[i], decl,
                    "two declarations share a UniformIdentity but disagree on the default"
                );
                UniformId(i as u16)
            }
            None => self.declare_uniform(decl),
        }
    }

    /// Push a `Uniform(id)` leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been declared via [`ExprArena::declare_uniform`].
    pub fn push_uniform(&mut self, id: UniformId) -> ExprId {
        assert!(
            (id.0 as usize) < self.uniforms.len(),
            "push_uniform: UniformId({}) not declared (table has {} entries)",
            id.0,
            self.uniforms.len()
        );
        self.push_node(ExprNode::Uniform(id))
    }

    /// Get the declaration for a uniform slot.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    #[must_use]
    pub fn uniform_decl(&self, id: UniformId) -> &UniformDecl {
        &self.uniforms[id.0 as usize]
    }

    /// All declared uniforms, indexed by [`UniformId`].
    #[inline]
    #[must_use]
    pub fn uniforms(&self) -> &[UniformDecl] {
        &self.uniforms
    }

    /// Push a `Gather(buffer, x, y)` read of a declared buffer.
    ///
    /// Semantics: floor the indices, clamp to the declared extents, gather
    /// row-major. `DiscreteManifold::kernel` is exactly one of these.
    pub fn push_gather(&mut self, buffer: BufferId, x: ExprId, y: ExprId) -> ExprId {
        let buf = self.push_buffer(buffer);
        self.push_ternary(OpKind::Gather, buf, x, y)
    }

    /// Push a reduction `Nary(Reduce, [Const(combiner), Const(reduce_var),
    /// Const(extent), body])`.
    ///
    /// `combiner` is the monoid op folded with (`Add`/`Mul`/`Min`/`Max`);
    /// `reduce_var` is the index (4..8) that `body` folds over; `extent` is the
    /// trip count. Lowered to an unrolled accumulation by `expand_reduce`.
    ///
    /// # Panics
    ///
    /// Panics if `combiner` is not a monoid op or `reduce_var` is outside 4..8.
    pub fn push_reduce(
        &mut self,
        combiner: OpKind,
        reduce_var: u8,
        extent: u32,
        body: ExprId,
    ) -> ExprId {
        assert!(
            combiner.is_monoid(),
            "push_reduce: {combiner:?} is not a valid reduction combiner"
        );
        assert!(
            (4..8).contains(&reduce_var),
            "push_reduce: reduce_var {reduce_var} out of range (must be 4..8)"
        );
        let c = self.push_const(combiner.index() as f32);
        let v = self.push_const(reduce_var as f32);
        let n = self.push_const(extent as f32);
        self.push_nary(OpKind::Reduce, &[c, v, n, body])
    }

    /// Get the declaration for a buffer slot.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    #[must_use]
    pub fn buffer_decl(&self, id: BufferId) -> &BufferDecl {
        &self.buffers[id.0 as usize]
    }

    /// All declared buffers, indexed by [`BufferId`].
    #[inline]
    #[must_use]
    pub fn buffers(&self) -> &[BufferDecl] {
        &self.buffers
    }

    /// Push a unary operation node.
    pub fn push_unary(&mut self, op: OpKind, child: ExprId) -> ExprId {
        self.push_node(ExprNode::Unary(op, child))
    }

    /// Push a binary operation node.
    pub fn push_binary(&mut self, op: OpKind, a: ExprId, b: ExprId) -> ExprId {
        self.push_node(ExprNode::Binary(op, a, b))
    }

    /// Push a ternary operation node.
    pub fn push_ternary(&mut self, op: OpKind, a: ExprId, b: ExprId, c: ExprId) -> ExprId {
        self.push_node(ExprNode::Ternary(op, a, b, c))
    }

    /// Push an N-ary operation node. Children are copied into the internal slab.
    ///
    /// # Panics
    ///
    /// Panics if `children.len()` exceeds `u16::MAX`.
    pub fn push_nary(&mut self, op: OpKind, children: &[ExprId]) -> ExprId {
        assert!(
            children.len() <= u16::MAX as usize,
            "push_nary: {} children exceeds u16::MAX",
            children.len()
        );
        let start = self.nary_children.len() as u32;
        let len = children.len() as u16;
        self.nary_children.extend_from_slice(children);
        self.push_node(ExprNode::Nary(op, start, len))
    }

    // ───────────────────── raw access (serialization) ───────

    /// Raw slice of all nodes in the arena.
    #[inline]
    #[must_use]
    pub fn nodes_raw(&self) -> &[ExprNode] {
        &self.nodes
    }

    /// Raw slice of the nary-children slab.
    #[inline]
    #[must_use]
    pub fn nary_children_raw(&self) -> &[ExprId] {
        &self.nary_children
    }

    /// Reconstruct an arena from raw parts.
    ///
    /// # Safety contract (logical, not `unsafe`)
    ///
    /// The caller must ensure that every `ExprId` referenced by nodes in
    /// `nodes` is in-bounds, and that `Nary` start/len pairs index validly
    /// into `nary_children`. Violating this will cause panics on access,
    /// not UB.
    /// The reconstructed arena has empty buffer and uniform tables, so it
    /// cannot hold `Buffer` or `Uniform` nodes.
    #[must_use]
    pub fn from_raw(nodes: Vec<ExprNode>, nary_children: Vec<ExprId>) -> Self {
        Self {
            nodes,
            nary_children,
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    // ───────────────────── access ────────────────────────────

    /// Get the node at `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    #[must_use]
    pub fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.0 as usize]
    }

    /// Get the N-ary children slice for a `Nary(_, start, len)` node.
    ///
    /// # Panics
    ///
    /// Panics if `start + len` exceeds the internal nary_children buffer.
    #[inline]
    #[must_use]
    pub fn nary_children_slice(&self, start: u32, len: u16) -> &[ExprId] {
        let s = start as usize;
        let l = len as usize;
        &self.nary_children[s..s + l]
    }

    /// Get the [`OpKind`] of the node at `id`.
    ///
    /// Leaf nodes map to: `Var -> OpKind::Var`, `Const/Param -> OpKind::Const`,
    /// `Buffer -> OpKind::Buffer`, `Uniform -> OpKind::Uniform`.
    #[inline]
    #[must_use]
    pub fn kind(&self, id: ExprId) -> OpKind {
        match &self.nodes[id.0 as usize] {
            ExprNode::Var(_) => OpKind::Var,
            ExprNode::Const(_) | ExprNode::Param(_) => OpKind::Const,
            ExprNode::Buffer(_) => OpKind::Buffer,
            ExprNode::Uniform(_) => OpKind::Uniform,
            ExprNode::Unary(op, _) => *op,
            ExprNode::Binary(op, _, _) => *op,
            ExprNode::Ternary(op, _, _, _) => *op,
            ExprNode::Nary(op, _, _) => *op,
        }
    }

    /// Iterate over the child [`ExprId`]s of the node at `id`.
    #[inline]
    #[must_use]
    pub fn children(&self, id: ExprId) -> ExprChildren<'_> {
        match &self.nodes[id.0 as usize] {
            ExprNode::Var(_)
            | ExprNode::Const(_)
            | ExprNode::Param(_)
            | ExprNode::Buffer(_)
            | ExprNode::Uniform(_) => ExprChildren::Zero,
            ExprNode::Unary(_, a) => ExprChildren::One(*a),
            ExprNode::Binary(_, a, b) => ExprChildren::Two(*a, *b),
            ExprNode::Ternary(_, a, b, c) => ExprChildren::Three(*a, *b, *c),
            ExprNode::Nary(_, start, len) => {
                let s = *start as usize;
                let l = *len as usize;
                ExprChildren::Nary(&self.nary_children[s..s + l])
            }
        }
    }

    // ───────────────────── traversal ─────────────────────────

    /// Compute the depth of the subtree rooted at `root` (iterative).
    #[must_use]
    pub fn depth(&self, root: ExprId) -> usize {
        let mut stack: Vec<(ExprId, usize)> = Vec::new();
        stack.push((root, 1));
        let mut max_depth: usize = 0;

        while let Some((id, d)) = stack.pop() {
            match &self.nodes[id.0 as usize] {
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => {
                    max_depth = max_depth.max(d);
                }
                ExprNode::Unary(_, a) => {
                    stack.push((*a, d + 1));
                }
                ExprNode::Binary(_, a, b) => {
                    stack.push((*a, d + 1));
                    stack.push((*b, d + 1));
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push((*a, d + 1));
                    stack.push((*b, d + 1));
                    stack.push((*c, d + 1));
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    if l == 0 {
                        max_depth = max_depth.max(d);
                    } else {
                        for child in &self.nary_children[s..s + l] {
                            stack.push((*child, d + 1));
                        }
                    }
                }
            }
        }
        max_depth
    }

    /// Returns `true` if the subtree rooted at `root` contains at least one `Var` node.
    #[must_use]
    pub fn has_var(&self, root: ExprId) -> bool {
        let mut stack: Vec<ExprId> = Vec::new();
        stack.push(root);

        while let Some(id) = stack.pop() {
            match &self.nodes[id.0 as usize] {
                ExprNode::Var(_) => return true,
                ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => {}
                ExprNode::Unary(_, a) => stack.push(*a),
                ExprNode::Binary(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
            }
        }
        false
    }

    /// Returns `true` if the subtree contains degenerate subexpressions:
    /// NaN/Inf constants, `recip(0)`, `div(_, 0)`.
    #[must_use]
    pub fn has_degenerate(&self, root: ExprId) -> bool {
        let mut stack: Vec<ExprId> = vec![root];

        while let Some(id) = stack.pop() {
            match &self.nodes[id.0 as usize] {
                ExprNode::Const(v) if !v.is_finite() => return true,
                ExprNode::Unary(OpKind::Recip, a) => {
                    if matches!(self.nodes[a.0 as usize], ExprNode::Const(v) if v == 0.0) {
                        return true;
                    }
                    stack.push(*a);
                }
                ExprNode::Binary(OpKind::Div, a, b) => {
                    if matches!(self.nodes[b.0 as usize], ExprNode::Const(v) if v == 0.0) {
                        return true;
                    }
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => {}
                ExprNode::Unary(_, a) => stack.push(*a),
                ExprNode::Binary(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
            }
        }
        false
    }

    /// Count total nodes reachable from `root` (iterative).
    ///
    /// Note: if the DAG shares subtrees (same ExprId referenced multiple times),
    /// shared nodes are counted once per reference. This matches `Expr::node_count`
    /// behavior on Arc trees (where shared subtrees are traversed per reference).
    #[must_use]
    pub fn node_count_subtree(&self, root: ExprId) -> usize {
        let mut stack: Vec<ExprId> = Vec::new();
        stack.push(root);
        let mut count: usize = 0;

        while let Some(id) = stack.pop() {
            count += 1;
            match &self.nodes[id.0 as usize] {
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => {}
                ExprNode::Unary(_, a) => stack.push(*a),
                ExprNode::Binary(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
            }
        }
        count
    }

    /// Replace every `Param(i)` node with what `params[i]` says it is: a
    /// `Const` folded into the fragment, or a `Uniform` slot declared for the
    /// handle's identity (one slot per identity, however many placeholders
    /// name it).
    ///
    /// Returns the new root [`ExprId`] in the **same** arena. Old nodes become
    /// unreachable garbage — that is fine for an append-only arena.
    ///
    /// # Panics
    ///
    /// Panics if any `Param(i)` has `i >= params.len()`.
    pub fn substitute_params(&mut self, root: ExprId, params: &[Scalar]) -> ExprId {
        // Iterative post-order: map old ExprId -> new ExprId.
        // We use a Vec as a dense map since IDs are contiguous 0..n.
        enum Task {
            Descend(ExprId),
            Emit(ExprId),
        }

        // We'll build a mapping: old_id -> new_id.
        // Initialize with sentinel values.
        let old_len = self.nodes.len();
        let mut id_map: Vec<Option<ExprId>> = Vec::new();
        id_map.resize(old_len, None);

        let mut work: Vec<Task> = vec![Task::Descend(root)];

        while let Some(task) = work.pop() {
            match task {
                Task::Descend(id) => {
                    // If already mapped (shared subtree), skip.
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    match &self.nodes[id.0 as usize] {
                        ExprNode::Var(_)
                        | ExprNode::Const(_)
                        | ExprNode::Param(_)
                        | ExprNode::Buffer(_)
                        | ExprNode::Uniform(_) => {}
                        ExprNode::Unary(_, a) => {
                            work.push(Task::Descend(*a));
                        }
                        ExprNode::Binary(_, a, b) => {
                            work.push(Task::Descend(*b));
                            work.push(Task::Descend(*a));
                        }
                        ExprNode::Ternary(_, a, b, c) => {
                            work.push(Task::Descend(*c));
                            work.push(Task::Descend(*b));
                            work.push(Task::Descend(*a));
                        }
                        ExprNode::Nary(_, start, len) => {
                            let s = *start as usize;
                            let l = *len as usize;
                            for child in self.nary_children[s..s + l].iter().rev() {
                                work.push(Task::Descend(*child));
                            }
                        }
                    }
                }
                Task::Emit(id) => {
                    // Skip if already emitted (can happen with shared subtrees).
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    let new_id = match self.nodes[id.0 as usize].clone() {
                        ExprNode::Param(i) => {
                            let idx = i as usize;
                            assert!(
                                idx < params.len(),
                                "substitute_params: param index {} out of range (have {} params)",
                                idx,
                                params.len()
                            );
                            match params[idx] {
                                Scalar::Const(v) => self.push_const(v),
                                Scalar::Uniform(u) => {
                                    let slot = self.uniform_slot_for(u.decl());
                                    self.push_uniform(slot)
                                }
                            }
                        }
                        ExprNode::Var(i) => self.push_var(i),
                        ExprNode::Const(v) => self.push_const(v),
                        // Buffer and uniform ids stay valid: the tables live
                        // in this arena.
                        ExprNode::Buffer(b) => self.push_node(ExprNode::Buffer(b)),
                        ExprNode::Uniform(u) => self.push_node(ExprNode::Uniform(u)),
                        ExprNode::Unary(op, a) => {
                            let na = id_map[a.0 as usize]
                                .expect("substitute_params: child not yet mapped for Unary");
                            self.push_unary(op, na)
                        }
                        ExprNode::Binary(op, a, b) => {
                            let na = id_map[a.0 as usize]
                                .expect("substitute_params: child a not yet mapped for Binary");
                            let nb = id_map[b.0 as usize]
                                .expect("substitute_params: child b not yet mapped for Binary");
                            self.push_binary(op, na, nb)
                        }
                        ExprNode::Ternary(op, a, b, c) => {
                            let na = id_map[a.0 as usize]
                                .expect("substitute_params: child a not yet mapped for Ternary");
                            let nb = id_map[b.0 as usize]
                                .expect("substitute_params: child b not yet mapped for Ternary");
                            let nc = id_map[c.0 as usize]
                                .expect("substitute_params: child c not yet mapped for Ternary");
                            self.push_ternary(op, na, nb, nc)
                        }
                        ExprNode::Nary(op, start, len) => {
                            let s = start as usize;
                            let l = len as usize;
                            let child_ids: Vec<ExprId> = self.nary_children[s..s + l]
                                .iter()
                                .map(|old_child| {
                                    id_map[old_child.0 as usize]
                                        .expect("substitute_params: nary child not yet mapped")
                                })
                                .collect();
                            self.push_nary(op, &child_ids)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("substitute_params: root was never mapped")
    }

    // ───────────────────── composition (P4: arena splicing) ─────────────────

    /// Copy the fragment reachable from `root` in `other` into this arena,
    /// returning the fragment's new root here. Shared subexpressions are
    /// copied once, so a DAG stays a DAG. This is the substrate of kernel
    /// composition: the spliced fragment reads this arena's coordinate
    /// variables directly (an identity contramap); warp it afterwards with
    /// [`ExprArena::substitute_vars_with`].
    ///
    /// Buffers the fragment reads are merged into this arena's table by
    /// [`BufferIdentity`] and its `Buffer` leaves remapped, so a sampler
    /// composes like anything else and reading the same memory from twenty
    /// places still binds one pointer. Uniforms merge the same way, by
    /// [`UniformIdentity`]: one instance read from twenty places is one slot,
    /// and two instances of one builder stay two.
    pub fn splice(&mut self, other: &ExprArena, root: ExprId) -> ExprId {
        let mut id_map: Vec<Option<ExprId>> = vec![None; other.nodes.len()];
        // Fragment-local BufferId -> this arena's slot, filled lazily.
        let mut buf_map: Vec<Option<BufferId>> = vec![None; other.buffers.len()];
        let mut uni_map: Vec<Option<UniformId>> = vec![None; other.uniforms.len()];

        enum Task {
            Descend(ExprId),
            Emit(ExprId),
        }
        let mut work: Vec<Task> = vec![Task::Descend(root)];

        while let Some(task) = work.pop() {
            match task {
                Task::Descend(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    let children: Vec<ExprId> = other.children(id).collect();
                    for child in children.into_iter().rev() {
                        work.push(Task::Descend(child));
                    }
                }
                Task::Emit(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    let m = |old: ExprId| {
                        id_map[old.0 as usize].expect("splice: child copied before parent")
                    };
                    let new_id = match other.nodes[id.0 as usize].clone() {
                        ExprNode::Var(i) => self.push_var(i),
                        ExprNode::Const(v) => self.push_const(v),
                        ExprNode::Param(i) => self.push_param(i),
                        ExprNode::Buffer(b) => {
                            let slot = match buf_map[b.0 as usize] {
                                Some(slot) => slot,
                                None => {
                                    let decl = other.buffers[b.0 as usize];
                                    let slot =
                                        match self.buffers.iter().position(|d| d.id == decl.id) {
                                            Some(i) => {
                                                assert_eq!(
                                                    self.buffers[i], decl,
                                                    "splice: two declarations share a \
                                                 BufferIdentity but disagree on extents"
                                                );
                                                BufferId(i as u16)
                                            }
                                            None => self.declare_buffer(decl),
                                        };
                                    buf_map[b.0 as usize] = Some(slot);
                                    slot
                                }
                            };
                            self.push_buffer(slot)
                        }
                        ExprNode::Uniform(u) => {
                            let slot = match uni_map[u.0 as usize] {
                                Some(slot) => slot,
                                None => {
                                    let slot = self.uniform_slot_for(other.uniforms[u.0 as usize]);
                                    uni_map[u.0 as usize] = Some(slot);
                                    slot
                                }
                            };
                            self.push_uniform(slot)
                        }
                        ExprNode::Unary(op, a) => {
                            let a = m(a);
                            self.push_unary(op, a)
                        }
                        ExprNode::Binary(op, a, b) => {
                            let (a, b) = (m(a), m(b));
                            self.push_binary(op, a, b)
                        }
                        ExprNode::Ternary(op, a, b, c) => {
                            let (a, b, c) = (m(a), m(b), m(c));
                            self.push_ternary(op, a, b, c)
                        }
                        ExprNode::Nary(op, start, len) => {
                            let (s, l) = (start as usize, len as usize);
                            let mapped: Vec<ExprId> = other.nary_children[s..s + l]
                                .iter()
                                .map(|c| m(*c))
                                .collect();
                            self.push_nary(op, &mapped)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("splice: root was never copied")
    }

    /// Rebuild the subgraph at `root`, replacing every `Var(i)` for which
    /// `subs` has an entry with the given (already existing) node — the
    /// generic contramap: a coordinate warp substitutes `Var(0..4)` with
    /// coordinate expressions, which is what `Kernel::at` is built from.
    ///
    /// Entries must reference nodes already in this arena (e.g. from
    /// [`ExprArena::splice`]). Unlisted variables are preserved. Returns the
    /// new root in the same arena; old nodes become unreachable garbage, as
    /// with [`ExprArena::substitute_params`].
    pub fn substitute_vars_with(&mut self, root: ExprId, subs: &[(u8, ExprId)]) -> ExprId {
        let lookup = |i: u8| subs.iter().find(|(v, _)| *v == i).map(|(_, id)| *id);

        let old_len = self.nodes.len();
        let mut id_map: Vec<Option<ExprId>> = vec![None; old_len];

        enum Task {
            Descend(ExprId),
            Emit(ExprId),
        }
        let mut work: Vec<Task> = vec![Task::Descend(root)];

        while let Some(task) = work.pop() {
            match task {
                Task::Descend(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    let children: Vec<ExprId> = self.children(id).collect();
                    for child in children.into_iter().rev() {
                        work.push(Task::Descend(child));
                    }
                }
                Task::Emit(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    let m = |old: ExprId| {
                        id_map[old.0 as usize]
                            .expect("substitute_vars_with: child rebuilt before parent")
                    };
                    let new_id = match self.nodes[id.0 as usize].clone() {
                        ExprNode::Var(i) => match lookup(i) {
                            Some(replacement) => replacement,
                            None => self.push_var(i),
                        },
                        ExprNode::Const(v) => self.push_const(v),
                        ExprNode::Param(i) => self.push_param(i),
                        ExprNode::Buffer(b) => self.push_node(ExprNode::Buffer(b)),
                        ExprNode::Uniform(u) => self.push_node(ExprNode::Uniform(u)),
                        ExprNode::Unary(op, a) => {
                            let a = m(a);
                            self.push_unary(op, a)
                        }
                        ExprNode::Binary(op, a, b) => {
                            let (a, b) = (m(a), m(b));
                            self.push_binary(op, a, b)
                        }
                        ExprNode::Ternary(op, a, b, c) => {
                            let (a, b, c) = (m(a), m(b), m(c));
                            self.push_ternary(op, a, b, c)
                        }
                        ExprNode::Nary(op, start, len) => {
                            let (s, l) = (start as usize, len as usize);
                            let child_ids: Vec<ExprId> = self.nary_children[s..s + l].to_vec();
                            let mapped: Vec<ExprId> = child_ids.into_iter().map(m).collect();
                            self.push_nary(op, &mapped)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("substitute_vars_with: root was never rebuilt")
    }

    // ───────────────────── linking ───────────────────────────

    /// The subgraph reachable from `root`, with its buffer and uniform tables
    /// replaced by the given orders — the link step: slot `i` of the result
    /// names `buffers[i]` / `uniforms[i]` and every reachable leaf is
    /// remapped to its slot there. Reachable nodes keep their relative order
    /// (ascending id, so still topological), which is the order a schedule
    /// is built in; construction garbage is dropped, which no schedule ever
    /// saw. So nothing downstream of the tables — the schedule, the
    /// registers, the bytes — can move.
    ///
    /// A declaration this arena holds but no reachable node reads is left
    /// behind with the garbage: `Kernel::at` splices all four coordinate
    /// fragments whether or not the receiver reads that axis, so a table
    /// routinely names an instance the graph does not, and the link — which
    /// is computed over the reachable subgraph — rightly omits it. The orders
    /// may likewise declare identities nothing here reads; those slots exist
    /// in the result unread.
    ///
    /// # Panics
    ///
    /// Panics if a *reachable* leaf's declaration has no entry in the given
    /// order, or disagrees with it (extents, default).
    #[must_use]
    pub fn relink(
        &self,
        root: ExprId,
        buffers: &[BufferDecl],
        uniforms: &[UniformDecl],
    ) -> (ExprArena, ExprId) {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut reachable[id.0 as usize], true) {
                continue;
            }
            stack.extend(self.children(id));
        }

        let buffer_slot = |b: BufferId| -> BufferId {
            let decl = self.buffers[b.0 as usize];
            let i = buffers
                .iter()
                .position(|d| d.id == decl.id)
                .unwrap_or_else(|| panic!("relink: reachable {decl:?} is not in the link"));
            assert_eq!(buffers[i], decl, "relink: buffer declaration disagrees");
            BufferId(i as u16)
        };
        let uniform_slot = |u: UniformId| -> UniformId {
            let decl = self.uniforms[u.0 as usize];
            let i = uniforms
                .iter()
                .position(|d| d.id == decl.id)
                .unwrap_or_else(|| panic!("relink: reachable {decl:?} is not in the link"));
            assert_eq!(uniforms[i], decl, "relink: uniform declaration disagrees");
            UniformId(i as u16)
        };

        let mut out = ExprArena {
            nodes: Vec::with_capacity(self.nodes.len()),
            nary_children: Vec::new(),
            buffers: buffers.to_vec(),
            uniforms: uniforms.to_vec(),
        };
        let mut dense: Vec<Option<ExprId>> = vec![None; self.nodes.len()];
        for (idx, node) in self.nodes.iter().enumerate() {
            if !reachable[idx] {
                continue;
            }
            let m =
                |old: ExprId| dense[old.0 as usize].expect("relink: child densified before parent");
            let new_id = match node {
                ExprNode::Var(i) => out.push_var(*i),
                ExprNode::Const(v) => out.push_const(*v),
                ExprNode::Param(i) => out.push_param(*i),
                ExprNode::Buffer(b) => out.push_buffer(buffer_slot(*b)),
                ExprNode::Uniform(u) => out.push_uniform(uniform_slot(*u)),
                ExprNode::Unary(op, a) => out.push_unary(*op, m(*a)),
                ExprNode::Binary(op, a, b) => out.push_binary(*op, m(*a), m(*b)),
                ExprNode::Ternary(op, a, b, c) => out.push_ternary(*op, m(*a), m(*b), m(*c)),
                ExprNode::Nary(op, start, len) => {
                    let (s, l) = (*start as usize, *len as usize);
                    let mapped: Vec<ExprId> =
                        self.nary_children[s..s + l].iter().map(|c| m(*c)).collect();
                    out.push_nary(*op, &mapped)
                }
            };
            dense[idx] = Some(new_id);
        }
        let new_root = dense[root.0 as usize].expect("relink: root is reachable from itself");
        (out, new_root)
    }

    // ───────────────────── display ───────────────────────────

    /// Format the subtree rooted at `root` as an S-expression, matching the
    /// [`Expr`] display format.
    pub(crate) fn fmt_expr(&self, root: ExprId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Task {
            Visit(ExprId),
            WriteStr(&'static str),
        }

        let mut stack: Vec<Task> = vec![Task::Visit(root)];

        while let Some(task) = stack.pop() {
            match task {
                Task::WriteStr(s) => f.write_str(s)?,
                Task::Visit(id) => match &self.nodes[id.0 as usize] {
                    ExprNode::Var(i) => write!(f, "Var({})", i)?,
                    ExprNode::Const(v) => write!(f, "Const({})", v)?,
                    ExprNode::Param(i) => write!(f, "Param({})", i)?,
                    ExprNode::Buffer(b) => write!(f, "Buffer({})", b.0)?,
                    ExprNode::Uniform(u) => write!(f, "Uniform({})", u.0)?,
                    ExprNode::Unary(op, a) => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*a));
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                    ExprNode::Binary(op, a, b) => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*b));
                        stack.push(Task::WriteStr(", "));
                        stack.push(Task::Visit(*a));
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                    ExprNode::Ternary(op, a, b, c) => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*c));
                        stack.push(Task::WriteStr(", "));
                        stack.push(Task::Visit(*b));
                        stack.push(Task::WriteStr(", "));
                        stack.push(Task::Visit(*a));
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                    ExprNode::Nary(op, start, len) => {
                        let s = *start as usize;
                        let l = *len as usize;
                        stack.push(Task::WriteStr(")"));
                        for (i, child) in self.nary_children[s..s + l].iter().enumerate().rev() {
                            stack.push(Task::Visit(*child));
                            if i > 0 {
                                stack.push(Task::WriteStr(", "));
                            }
                        }
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                },
            }
        }
        Ok(())
    }

    /// Return a [`Display`]-able wrapper for the subtree rooted at `root`.
    #[must_use]
    pub fn display(&self, root: ExprId) -> DisplayExpr<'_> {
        DisplayExpr { arena: self, root }
    }

    /// Compare two subtrees for structural equality without allocating `Expr` trees.
    ///
    /// `self[a]` is compared against `other[b]` node-by-node in lockstep using an
    /// iterative work stack. Both subtrees may live in the same arena (pass `self`
    /// for both `self` and `other`) or in different arenas.
    ///
    /// Constant nodes are compared by exact bit equality (same behaviour as
    /// [`Expr`]'s `PartialEq`). Var and Param indices are compared by value.
    #[must_use]
    pub fn subtree_eq(&self, a: ExprId, other: &ExprArena, b: ExprId) -> bool {
        // Stack of (self-id, other-id) pairs still to be compared.
        let mut stack: Vec<(ExprId, ExprId)> = Vec::with_capacity(16);
        stack.push((a, b));

        while let Some((s_id, o_id)) = stack.pop() {
            let s_node = &self.nodes[s_id.0 as usize];
            let o_node = &other.nodes[o_id.0 as usize];

            match (s_node, o_node) {
                (ExprNode::Var(si), ExprNode::Var(oi)) => {
                    if si != oi {
                        return false;
                    }
                }
                (ExprNode::Const(sv), ExprNode::Const(ov)) => {
                    // Bit-exact comparison matches Expr's PartialEq behaviour.
                    if sv.to_bits() != ov.to_bits() {
                        return false;
                    }
                }
                (ExprNode::Param(si), ExprNode::Param(oi)) => {
                    if si != oi {
                        return false;
                    }
                }
                // Buffer slots compare by id AND declared shape, so the
                // comparison is meaningful across arenas with different tables.
                (ExprNode::Buffer(sb), ExprNode::Buffer(ob)) => {
                    if sb != ob || self.buffers[sb.0 as usize] != other.buffers[ob.0 as usize] {
                        return false;
                    }
                }
                // Uniform slots likewise: by slot AND declaration.
                (ExprNode::Uniform(su), ExprNode::Uniform(ou)) => {
                    if su != ou || self.uniforms[su.0 as usize] != other.uniforms[ou.0 as usize] {
                        return false;
                    }
                }
                (ExprNode::Unary(s_op, s_a), ExprNode::Unary(o_op, o_a)) => {
                    if s_op != o_op {
                        return false;
                    }
                    stack.push((*s_a, *o_a));
                }
                (ExprNode::Binary(s_op, s_a, s_b), ExprNode::Binary(o_op, o_a, o_b)) => {
                    if s_op != o_op {
                        return false;
                    }
                    stack.push((*s_a, *o_a));
                    stack.push((*s_b, *o_b));
                }
                (
                    ExprNode::Ternary(s_op, s_a, s_b, s_c),
                    ExprNode::Ternary(o_op, o_a, o_b, o_c),
                ) => {
                    if s_op != o_op {
                        return false;
                    }
                    stack.push((*s_a, *o_a));
                    stack.push((*s_b, *o_b));
                    stack.push((*s_c, *o_c));
                }
                (ExprNode::Nary(s_op, s_start, s_len), ExprNode::Nary(o_op, o_start, o_len)) => {
                    if s_op != o_op || s_len != o_len {
                        return false;
                    }
                    let ss = *s_start as usize;
                    let os = *o_start as usize;
                    let len = *s_len as usize;
                    for i in 0..len {
                        stack.push((self.nary_children[ss + i], other.nary_children[os + i]));
                    }
                }
                // Different node variants — structurally unequal.
                _ => return false,
            }
        }

        true
    }
}

// ───────────────────────────────────── DisplayExpr ───────────────────────────

/// Wrapper that implements [`fmt::Display`] for an arena subtree.
pub struct DisplayExpr<'a> {
    arena: &'a ExprArena,
    root: ExprId,
}

impl fmt::Display for DisplayExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.arena.fmt_expr(self.root, f)
    }
}

// ───────────────────────────────────── Tests ─────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    // 1. test_push_and_access
    #[test]
    fn push_and_access() {
        let mut arena = ExprArena::new();

        let v = arena.push_var(0);
        assert_eq!(arena.kind(v), OpKind::Var);
        assert_eq!(arena.children(v).count(), 0);

        let c = arena.push_const(core::f32::consts::PI);
        assert_eq!(arena.kind(c), OpKind::Const);
        assert_eq!(arena.children(c).count(), 0);

        let p = arena.push_param(1);
        assert_eq!(arena.kind(p), OpKind::Const); // Param maps to Const kind
        assert_eq!(arena.children(p).count(), 0);

        let u = arena.push_unary(OpKind::Neg, v);
        assert_eq!(arena.kind(u), OpKind::Neg);
        let u_children: Vec<ExprId> = arena.children(u).collect();
        assert_eq!(u_children, vec![v]);

        let b = arena.push_binary(OpKind::Add, v, c);
        assert_eq!(arena.kind(b), OpKind::Add);
        let b_children: Vec<ExprId> = arena.children(b).collect();
        assert_eq!(b_children, vec![v, c]);

        let t = arena.push_ternary(OpKind::MulAdd, v, c, p);
        assert_eq!(arena.kind(t), OpKind::MulAdd);
        let t_children: Vec<ExprId> = arena.children(t).collect();
        assert_eq!(t_children, vec![v, c, p]);
    }

    // 2. test_node_count
    #[test]
    fn node_count() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let c1 = arena.push_const(1.0);
        let root = arena.push_binary(OpKind::Add, v0, c1);
        assert_eq!(arena.len(), 3);
        assert_eq!(arena.node_count_subtree(root), 3);

        let mut arena2 = ExprArena::new();
        let a0 = arena2.push_var(0);
        let a1 = arena2.push_var(1);
        let add = arena2.push_binary(OpKind::Add, a0, a1);
        let c2 = arena2.push_const(2.0);
        let root2 = arena2.push_binary(OpKind::Mul, add, c2);
        assert_eq!(arena2.len(), 5);
        assert_eq!(arena2.node_count_subtree(root2), 5);
    }

    // 3. test_depth
    #[test]
    fn verify_depth() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let v1 = arena.push_var(1);
        let c3 = arena.push_const(3.0);
        let mul = arena.push_binary(OpKind::Mul, v1, c3);
        let root = arena.push_binary(OpKind::Add, v0, mul);
        assert_eq!(arena.depth(root), 3);
    }

    // 4. test_has_var
    #[test]
    fn verify_has_var() {
        let mut arena1 = ExprArena::new();
        let v0 = arena1.push_var(0);
        let c1 = arena1.push_const(1.0);
        let root1 = arena1.push_binary(OpKind::Add, v0, c1);
        assert!(arena1.has_var(root1));

        let mut arena2 = ExprArena::new();
        let c1 = arena2.push_const(1.0);
        let c2 = arena2.push_const(2.0);
        let root2 = arena2.push_binary(OpKind::Add, c1, c2);
        assert!(!arena2.has_var(root2));
    }

    // 5. test_has_degenerate
    #[test]
    fn verify_has_degenerate() {
        let mut arena1 = ExprArena::new();
        let root1 = arena1.push_const(f32::NAN);
        assert!(arena1.has_degenerate(root1));

        let mut arena2 = ExprArena::new();
        let root2 = arena2.push_const(f32::INFINITY);
        assert!(arena2.has_degenerate(root2));

        let mut arena3 = ExprArena::new();
        let v0 = arena3.push_var(0);
        let c0 = arena3.push_const(0.0);
        let root3 = arena3.push_binary(OpKind::Div, v0, c0);
        assert!(arena3.has_degenerate(root3));

        let mut arena4 = ExprArena::new();
        let c0 = arena4.push_const(0.0);
        let root4 = arena4.push_unary(OpKind::Recip, c0);
        assert!(arena4.has_degenerate(root4));

        let mut arena5 = ExprArena::new();
        let v0 = arena5.push_var(0);
        let c1 = arena5.push_const(1.0);
        let root5 = arena5.push_binary(OpKind::Add, v0, c1);
        assert!(!arena5.has_degenerate(root5));
    }

    // 7. test_clear_preserves_capacity
    #[test]
    fn clear_preserves_capacity() {
        let mut arena = ExprArena::with_capacity(64);
        let _v = arena.push_var(0);
        let _c = arena.push_const(1.0);
        assert_eq!(arena.len(), 2);

        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());

        // Push again — should work fine, capacity preserved.
        let v2 = arena.push_var(1);
        assert_eq!(v2, ExprId(0));
        assert_eq!(arena.len(), 1);
    }

    // 7. test_substitute_params
    #[test]
    fn verify_substitute_params() {
        let mut arena = ExprArena::new();
        let p0 = arena.push_param(0);
        let p1 = arena.push_param(1);
        let root = arena.push_binary(OpKind::Add, p0, p1);

        let new_root = arena.substitute_params(root, &[Scalar::Const(10.0), Scalar::Const(20.0)]);

        match arena.node(new_root) {
            ExprNode::Binary(OpKind::Add, a, b) => {
                assert!(matches!(arena.node(*a), ExprNode::Const(v) if (*v - 10.0).abs() < 1e-6));
                assert!(matches!(arena.node(*b), ExprNode::Const(v) if (*v - 20.0).abs() < 1e-6));
            }
            other => panic!("expected Binary(Add, ...), got {:?}", other),
        }
    }

    #[test]
    fn push_gather_should_create_node_when_valid() {
        let mut arena = ExprArena::new();
        let buf = arena.declare_buffer(BufferDecl {
            id: crate::arena::BufferIdentity::mint(),
            width: 16,
            height: 8,
        });
        assert_eq!(arena.buffers().len(), 1);
        assert_eq!(arena.buffer_decl(buf).width, 16);

        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let gather = arena.push_gather(buf, x, y);

        // Gather is a ternary whose first child is the Buffer leaf.
        assert_eq!(arena.kind(gather), OpKind::Gather);
        let children: Vec<ExprId> = arena.children(gather).collect();
        assert_eq!(children.len(), 3);
        assert!(matches!(arena.node(children[0]), ExprNode::Buffer(b) if *b == buf));
        assert_eq!(arena.kind(children[0]), OpKind::Buffer);
        assert_eq!(arena.children(children[0]).count(), 0); // Buffer is a leaf

        assert_eq!(format!("{}", arena.display(children[0])), "Buffer(0)");
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn push_buffer_should_panic_when_undeclared() {
        let mut arena = ExprArena::new();
        let _ = arena.push_buffer(BufferId(0));
    }

    #[test]
    #[should_panic(expected = "is not a valid reduction combiner")]
    fn push_reduce_should_panic_when_the_combiner_is_not_a_monoid_op() {
        let mut arena = ExprArena::new();
        let body = arena.push_var(4);
        let _ = arena.push_reduce(OpKind::Sin, 4, 4, body);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn push_reduce_should_panic_when_the_reduce_var_is_outside_4_to_8() {
        let mut arena = ExprArena::new();
        let body = arena.push_var(0);
        let _ = arena.push_reduce(OpKind::Add, 3, 4, body);
    }

    // 8. test_nary
    #[test]
    fn nary() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let v1 = arena.push_var(1);
        let c = arena.push_const(42.0);

        let tup = arena.push_nary(OpKind::Tuple, &[v0, v1, c]);
        assert_eq!(arena.kind(tup), OpKind::Tuple);

        let children: Vec<ExprId> = arena.children(tup).collect();
        assert_eq!(children, vec![v0, v1, c]);
        assert_eq!(arena.children(tup).len(), 3);
    }

    // 9. test_display
    #[test]
    fn verify_display() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let v1 = arena.push_var(1);
        let c2 = arena.push_const(2.0);
        let root = arena.push_ternary(OpKind::MulAdd, v0, v1, c2);
        // `display` matches the canonical `Expr` S-expression format.
        assert_eq!(
            format!("{}", arena.display(root)),
            "mul_add(Var(0), Var(1), Const(2))"
        );
    }

    #[test]
    fn size_of_expr_node() {
        // Compile-time assertion exists above, but also verify at runtime.
        assert!(
            core::mem::size_of::<ExprNode>() <= 16,
            "ExprNode is {} bytes, expected <= 16",
            core::mem::size_of::<ExprNode>()
        );
    }

    #[test]
    fn expr_children_exact_size() {
        let mut arena = ExprArena::new();
        let v = arena.push_var(0);
        let c = arena.push_const(1.0);
        let bin = arena.push_binary(OpKind::Add, v, c);

        assert_eq!(arena.children(v).len(), 0);
        assert_eq!(arena.children(bin).len(), 2);
    }
}

#[cfg(test)]
mod composition_tests {
    use super::*;
    use crate::binding::BindingTable;
    use crate::eval::eval_scalar;
    use crate::kind::OpKind;

    fn eval(a: &ExprArena, root: ExprId, vars: &[f32; 2]) -> f32 {
        eval_scalar(a, root, vars, &BindingTable::empty())
    }

    #[test]
    fn splice_copies_reachable_fragment_only() {
        let mut donor = ExprArena::new();
        let x = donor.push_var(0);
        let _dead = donor.push_const(99.0);
        let y = donor.push_var(1);
        let frag = donor.push_binary(OpKind::Mul, x, y);

        let mut host = ExprArena::new();
        let hx = host.push_var(0);
        let spliced = host.splice(&donor, frag);
        let root = host.push_binary(OpKind::Add, hx, spliced);

        // x + x*y, and the donor's dead node did not come along.
        assert_eq!(eval(&host, root, &[3.0, 4.0]), 3.0 + 12.0);
        assert!(
            !host
                .nodes_raw()
                .iter()
                .any(|n| matches!(n, ExprNode::Const(v) if *v == 99.0)),
            "unreachable donor node was copied"
        );
    }

    #[test]
    fn splice_preserves_dag_sharing() {
        // Donor: s = x*y used twice — must stay shared after splicing.
        let mut donor = ExprArena::new();
        let x = donor.push_var(0);
        let y = donor.push_var(1);
        let s = donor.push_binary(OpKind::Mul, x, y);
        let frag = donor.push_binary(OpKind::Add, s, s);

        let mut host = ExprArena::new();
        let before = host.nodes_raw().len();
        let spliced = host.splice(&donor, frag);
        // x, y, s, add = 4 nodes — not 6 (s duplicated).
        assert_eq!(host.nodes_raw().len() - before, 4);
        assert_eq!(eval(&host, spliced, &[3.0, 2.0]), 12.0);
    }

    #[test]
    fn splice_merges_buffer_tables_keeping_reads_distinct() {
        // Two donors that each call their own buffer slot 0. Merging must give
        // two slots, and each gather must still read the buffer it was written
        // against — remapping the id, not just renumbering it.
        let mut donor_a = ExprArena::new();
        let pa = donor_a.declare_buffer(BufferDecl {
            id: crate::arena::BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let (ax, ay) = (donor_a.push_var(0), donor_a.push_var(1));
        let a_root = donor_a.push_gather(pa, ax, ay);

        let mut donor_b = ExprArena::new();
        let qb = donor_b.declare_buffer(BufferDecl {
            id: crate::arena::BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let (bx, by) = (donor_b.push_var(0), donor_b.push_var(1));
        let b_root = donor_b.push_gather(qb, bx, by);

        let mut host = ExprArena::new();
        let sa = host.splice(&donor_a, a_root);
        let sb = host.splice(&donor_b, b_root);
        let root = host.push_binary(OpKind::Add, sa, sb);
        assert_eq!(host.buffers().len(), 2, "two donors, two slots");

        let (p, q) = ([10.0f32, 20.0], [3.0f32, 4.0]);
        let binding = BindingTable::bind(&host, &[&p[..], &q[..]]).expect("bind");
        assert_eq!(eval_scalar(&host, root, &[0.0; 2], &binding), 13.0);
        assert_eq!(eval_scalar(&host, root, &[1.0, 0.0], &binding), 24.0);
    }

    #[test]
    fn splice_gives_one_slot_to_a_buffer_read_twice() {
        // Within a single splice, one buffer stays one slot however many times
        // the fragment gathers from it. (Across separate splices it does not —
        // that needs an identity outliving the arena-local BufferId.)
        let mut donor = ExprArena::new();
        let buf = donor.declare_buffer(BufferDecl {
            id: crate::arena::BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let (x, y) = (donor.push_var(0), donor.push_var(1));
        let one = donor.push_const(1.0);
        let x1 = donor.push_binary(OpKind::Add, x, one);
        let g0 = donor.push_gather(buf, x, y);
        let g1 = donor.push_gather(buf, x1, y);
        let frag = donor.push_binary(OpKind::Add, g0, g1);

        let mut host = ExprArena::new();
        let root = host.splice(&donor, frag);
        assert_eq!(host.buffers().len(), 1, "one buffer read twice, one slot");

        let p = [5.0f32, 7.0];
        let binding = BindingTable::bind(&host, &[&p[..]]).expect("bind");
        assert_eq!(eval_scalar(&host, root, &[0.0; 2], &binding), 12.0);
    }

    #[test]
    fn splicing_one_buffer_from_two_places_binds_it_once() {
        // Separate splices, same memory: identity merges them. This is what
        // lets a sampler be read from all over a kernel and still cost one
        // binding — without it each use would want its own pointer.
        let mut donor = ExprArena::new();
        let buf = donor.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let (x, y) = (donor.push_var(0), donor.push_var(1));
        let frag = donor.push_gather(buf, x, y);

        let mut host = ExprArena::new();
        let first = host.splice(&donor, frag);
        let second = host.splice(&donor, frag);
        let root = host.push_binary(OpKind::Add, first, second);
        assert_eq!(host.buffers().len(), 1, "one buffer, one slot");

        let p = [5.0f32, 7.0];
        let binding = BindingTable::bind(&host, &[&p[..]]).expect("bind");
        assert_eq!(eval_scalar(&host, root, &[0.0; 2], &binding), 10.0);
    }

    #[test]
    fn substitute_vars_with_replaces_slot_with_fragment() {
        // Body template: sqrt(Var(8)) + X, where Var(8) is a manifold slot.
        let mut a = ExprArena::new();
        let slot = a.push_var(8);
        let sq = a.push_unary(OpKind::Sqrt, slot);
        let x = a.push_var(0);
        let root = a.push_binary(OpKind::Add, sq, x);

        // Fragment: x*x + y*y.
        let fx = a.push_var(0);
        let fy = a.push_var(1);
        let fx2 = a.push_binary(OpKind::Mul, fx, fx);
        let fy2 = a.push_binary(OpKind::Mul, fy, fy);
        let frag = a.push_binary(OpKind::Add, fx2, fy2);

        let root = a.substitute_vars_with(root, &[(8, frag)]);
        // sqrt(x²+y²) + x at (3,4) = 5 + 3.
        assert_eq!(eval(&a, root, &[3.0, 4.0]), 8.0);
    }

    #[test]
    fn substitute_vars_with_as_coordinate_warp() {
        // Warp: evaluate x*y at (x+1, 2y) — the contramap use of the same API.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let body = a.push_binary(OpKind::Mul, x, y);

        let one = a.push_const(1.0);
        let two = a.push_const(2.0);
        let wx = a.push_binary(OpKind::Add, x, one);
        let wy = a.push_binary(OpKind::Mul, y, two);

        let warped = a.substitute_vars_with(body, &[(0, wx), (1, wy)]);
        // (x+1) * 2y at (3, 4) = 4 * 8 = 32.
        assert_eq!(eval(&a, warped, &[3.0, 4.0]), 32.0);

        // The warp expressions' own Var(0)/Var(1) still read raw coordinates.
        assert_eq!(eval(&a, warped, &[0.0, 1.0]), 2.0);
    }

    #[test]
    fn spliced_fragment_differentiates_in_host() {
        // The composition story end-to-end at the arena level: splice a
        // distance fragment under a Dwrt and lower — d/dx √(x²+y²) = x/r.
        use crate::passes::lower_dwrt_owned;

        let mut donor = ExprArena::new();
        let x = donor.push_var(0);
        let y = donor.push_var(1);
        let x2 = donor.push_binary(OpKind::Mul, x, x);
        let y2 = donor.push_binary(OpKind::Mul, y, y);
        let sum = donor.push_binary(OpKind::Add, x2, y2);
        let dist = donor.push_unary(OpKind::Sqrt, sum);

        let mut host = ExprArena::new();
        let frag = host.splice(&donor, dist);
        let v0 = host.push_const(0.0);
        let root = host.push_binary(OpKind::Dwrt, frag, v0);

        let (out, out_root) = lower_dwrt_owned(&host, root).expect("lower_dwrt");
        let got = eval(&out, out_root, &[3.0, 4.0]);
        assert!((got - 0.6).abs() < 1e-4, "d/dx dist at (3,4): got {got}");
    }

    // ───────────────────────── uniforms ─────────────────────────

    /// A fragment reading one uniform, as `Uniform::kernel` would build it.
    fn uniform_fragment(decl: UniformDecl) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let slot = a.declare_uniform(decl);
        let root = a.push_uniform(slot);
        (a, root)
    }

    fn uniform_decl(default: f32) -> UniformDecl {
        UniformDecl {
            id: UniformIdentity::mint(),
            default,
        }
    }

    #[test]
    fn splice_merges_one_identity_into_one_slot_and_keeps_two_apart() {
        let same = uniform_decl(1.0);
        let other = uniform_decl(1.0); // equal default, distinct instance
        let (da, ra) = uniform_fragment(same);
        let (db, rb) = uniform_fragment(same);
        let (dc, rc) = uniform_fragment(other);

        let mut host = ExprArena::new();
        let a = host.splice(&da, ra);
        let b = host.splice(&db, rb);
        let c = host.splice(&dc, rc);
        let ab = host.push_binary(OpKind::Add, a, b);
        let root = host.push_binary(OpKind::Add, ab, c);

        assert_eq!(
            host.uniforms().len(),
            2,
            "one slot per identity, not per read"
        );
        assert_eq!(host.uniforms()[0], same);
        assert_eq!(host.uniforms()[1], other);
        assert!(matches!(host.node(a), ExprNode::Uniform(UniformId(0))));
        assert!(matches!(host.node(b), ExprNode::Uniform(UniformId(0))));
        assert!(matches!(host.node(c), ExprNode::Uniform(UniformId(1))));

        // Defaults when nothing is bound; the block, slot by slot, when it is.
        assert_eq!(eval(&host, root, &[0.0; 2]), 3.0);
        let bound = BindingTable::empty()
            .bind_uniforms(&host, &[(other.id, 7.0), (same.id, 5.0)])
            .expect("both are declared");
        assert_eq!(eval_scalar(&host, root, &[0.0; 2], &bound), 17.0);
        // An identity the arena does not declare is refused by name.
        let stranger = uniform_decl(0.0).id;
        assert_eq!(
            BindingTable::empty()
                .bind_uniforms(&host, &[(stranger, 1.0)])
                .err(),
            Some(crate::binding::BindError::Uniform(stranger))
        );
    }

    #[test]
    fn a_fragment_spliced_twice_reads_one_slot() {
        // `k.add(&k)`: the receiver's arena already holds the slot, and the
        // spliced copy must find it rather than declare a second.
        let decl = uniform_decl(2.0);
        let (donor, r) = uniform_fragment(decl);
        let mut host = donor.clone();
        let again = host.splice(&donor, r);
        let root = host.push_binary(OpKind::Mul, r, again);
        assert_eq!(host.uniforms().len(), 1);
        assert_eq!(eval(&host, root, &[0.0; 2]), 4.0);
    }

    #[test]
    #[should_panic(expected = "disagree on the default")]
    fn one_identity_with_two_defaults_is_refused() {
        let id = UniformIdentity::mint();
        let (donor, r) = uniform_fragment(UniformDecl { id, default: 1.0 });
        let mut host = ExprArena::new();
        let _ = host.declare_uniform(UniformDecl { id, default: 2.0 });
        let _ = host.splice(&donor, r);
    }

    #[test]
    fn substitute_params_declares_a_slot_for_a_uniform_and_folds_a_const() {
        use crate::kernel::Uniform;
        let mut arena = ExprArena::new();
        let p0 = arena.push_param(0);
        let p1 = arena.push_param(1);
        let p0_again = arena.push_param(0);
        let sum = arena.push_binary(OpKind::Add, p0, p1);
        let root = arena.push_binary(OpKind::Add, sum, p0_again);

        let u = Uniform::new(0.5);
        let root = arena.substitute_params(root, &[Scalar::Uniform(u), Scalar::Const(3.0)]);

        assert_eq!(arena.uniforms(), &[u.decl()]);
        assert_eq!(
            format!("{}", arena.display(root)),
            "add(add(Uniform(0), Const(3)), Uniform(0))"
        );
        assert_eq!(eval(&arena, root, &[0.0; 2]), 4.0);
    }

    #[test]
    fn f32_arguments_substitute_to_the_same_arena_as_before() {
        // The fold path is byte-for-byte what it was: `f32` keeps its meaning.
        let build = || {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let p = a.push_param(0);
            let root = a.push_binary(OpKind::Mul, x, p);
            (a, root)
        };
        let (mut folded, root) = build();
        let folded_root = folded.substitute_params(root, &[Scalar::from(2.5)]);
        let (mut by_hand, root) = build();
        let x = by_hand.push_var(0);
        let c = by_hand.push_const(2.5);
        let hand_root = by_hand.push_binary(OpKind::Mul, x, c);
        let _ = root;
        assert_eq!(folded.nodes_raw(), by_hand.nodes_raw());
        assert_eq!(folded_root, hand_root);
        assert!(folded.uniforms().is_empty());
    }

    #[test]
    fn subtree_eq_distinguishes_uniform_declarations() {
        let (a, ra) = uniform_fragment(uniform_decl(1.0));
        let (b, rb) = uniform_fragment(uniform_decl(1.0));
        assert!(a.subtree_eq(ra, &a, ra));
        assert!(!a.subtree_eq(ra, &b, rb), "same slot, different instance");
    }
}
