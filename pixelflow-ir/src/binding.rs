//! Binding bound-memory buffers to their declared slots for execution.
//!
//! An [`ExprArena`] declares buffers by *shape* ([`BufferDecl`]) via a
//! [`BufferId`]. Before a kernel that contains `Gather` nodes can run, each
//! slot must be bound to actual contents. This module provides the binding
//! used by the reference interpreter ([`crate::eval`]); the JIT path will
//! grow its own owned/`Arc` binding in M2 (see `KERNELS_AND_LATTICES.md`).
//!
//! Bindings here *borrow* their contents: a [`BindingTable`] is valid for the
//! duration of one evaluation, not the lifetime of a compiled kernel.

use crate::arena::{BufferId, ExprArena, UniformId};
use alloc::vec::Vec;

/// Why binding a buffer table failed. Binding fails loud rather than reading
/// out of bounds — consistent with the workspace's no-silent-failure rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindError {
    /// The number of supplied slices does not match the arena's buffer count.
    Count {
        /// Buffers the arena declares.
        declared: usize,
        /// Slices supplied.
        supplied: usize,
    },
    /// A slice length does not match its declared `width * height`.
    Length {
        /// The offending slot.
        slot: u16,
        /// Length the declaration requires.
        expected: usize,
        /// Length supplied.
        actual: usize,
    },
}

impl core::fmt::Display for BindError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BindError::Count { declared, supplied } => write!(
                f,
                "binding count mismatch: arena declares {declared} buffer(s), {supplied} supplied"
            ),
            BindError::Length {
                slot,
                expected,
                actual,
            } => write!(
                f,
                "buffer slot {slot}: declared length {expected}, bound slice has {actual}"
            ),
        }
    }
}

/// Borrowed contents for every buffer an [`ExprArena`] declares, indexed by
/// [`BufferId`]. Row-major, `stride == width`, matching `BufferDecl`.
///
/// Also the values of the arena's uniforms, indexed by [`UniformId`] — the
/// oracle's block. A table bound without one evaluates every uniform at its
/// declared default, which is what a bake without a block does.
#[derive(Clone, Debug)]
pub struct BindingTable<'a> {
    slots: Vec<&'a [f32]>,
    uniforms: &'a [f32],
}

impl<'a> BindingTable<'a> {
    /// Supply a value for each of the arena's uniform slots, in
    /// [`UniformId`] order. Slots past the end of `values` keep their
    /// declared default.
    #[must_use]
    pub fn with_uniforms(mut self, values: &'a [f32]) -> Self {
        self.uniforms = values;
        self
    }

    /// The value bound to uniform slot `id`, or `None` to mean its default.
    #[inline]
    #[must_use]
    pub fn uniform(&self, id: UniformId) -> Option<f32> {
        self.uniforms.get(id.0 as usize).copied()
    }

    /// Bind `slices` to the arena's buffer slots, in [`BufferId`] order.
    ///
    /// Validates that the count and every length match the declarations, so a
    /// later `Gather` can index without bounds surprises.
    ///
    /// # Errors
    ///
    /// Returns [`BindError`] if the count or any length disagrees with the
    /// arena's [`BufferDecl`]s.
    pub fn bind(arena: &ExprArena, slices: &[&'a [f32]]) -> Result<Self, BindError> {
        let decls = arena.buffers();
        if decls.len() != slices.len() {
            return Err(BindError::Count {
                declared: decls.len(),
                supplied: slices.len(),
            });
        }
        for (i, (decl, slice)) in decls.iter().zip(slices.iter()).enumerate() {
            let expected = decl.width as usize * decl.height as usize;
            if slice.len() != expected {
                return Err(BindError::Length {
                    slot: i as u16,
                    expected,
                    actual: slice.len(),
                });
            }
        }
        Ok(Self {
            slots: slices.to_vec(),
            uniforms: &[],
        })
    }

    /// An empty binding table, for arenas that declare no buffers.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            slots: Vec::new(),
            uniforms: &[],
        }
    }

    /// The contents bound to `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range for this table.
    #[inline]
    #[must_use]
    pub fn slot(&self, id: BufferId) -> &'a [f32] {
        self.slots[id.0 as usize]
    }
}
