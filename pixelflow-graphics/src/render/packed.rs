//! # A frame is one packed-pixel kernel over the frame lattice
//!
//! Four channel kernels in `[0, 1]` — red, green, blue, alpha, which is all a
//! colour output ever is — packed to bytes and OR-folded into a `u32` pixel
//! **inside the kernel**, then compiled once against the frame's lattice. One
//! collapse call bakes a band of finished pixels: no per-batch FFI boundary,
//! no virtual dispatch, no per-pixel scalar pack, and the collapse kernel's
//! two-level LICM prologues (per call, per row) active.
//!
//! The layer below ([`pixelflow_core::PlaneProgram`]) knows lattices, buffers
//! and bit patterns and nothing about colour; this is where byte lanes and
//! pixel formats begin.

use std::sync::Arc;

use pixelflow_core::{PlaneFrame, PlaneProgram, PlaneRegion};
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
use pixelflow_ir::Kernel;

/// The four channel kernels packed to one `u32` pixel, each channel packed to
/// a byte exactly as `Pixel::from_rgba` does — `(x·255).clamp(0, 255)` then
/// truncate toward zero — shifted to its byte lane and OR-folded at the root.
///
/// `shifts[c]` is the bit position of channel `c` in `(r, g, b, a)` order.
/// Both pixel orders are little-endian byte arrays wrapping a `u32`, so byte
/// `i` sits at bit `8·i`: `Rgba8` is `[r, g, b, a]` → `[0, 8, 16, 24]`, and
/// `Bgra8` is `[b, g, r, a]` → `[16, 8, 0, 24]`.
///
/// Truncation (not rounding) is deliberate twice over: it is bit-exact with
/// the scalar pack's `as u8`, and `cvttps2dq`/`fcvtzs` both truncate toward
/// zero — no per-target tie divergence to inherit.
pub(crate) fn packed_kernel(channels: &[Kernel; 4], shifts: [u32; 4]) -> Kernel {
    let k = Kernel::constant;
    let byte = |c: usize| {
        channels[c]
            .mul(&k(255.0))
            .clamp(&k(0.0), &k(255.0))
            .trunc_to_int()
            .shl(shifts[c])
    };
    // Fold in the bit domain, then take the single named exit: the packed
    // pattern IS the kernel's output.
    byte(0).or(&byte(1)).or(&byte(2)).or(&byte(3)).into_kernel()
}

/// A colour output compiled at a frame's lattice: ONE program whose root is a
/// packed `u32` pixel, so the collapse loop stores finished pixels directly.
///
/// Compile once per (channels, format, frame extent); bind whatever memory the
/// channels read per frame ([`PackedProgram::bind`]).
pub struct PackedProgram {
    /// The byte lanes this kernel packs into. A frame stores raw words, so
    /// whoever consumes them must confirm their format agrees.
    shifts: [u32; 4],
    program: PlaneProgram,
}

impl PackedProgram {
    /// Compose and JIT-compile the packed program for four channel kernels
    /// over a `extent[0] × extent[1]` pixel frame.
    ///
    /// # Panics
    ///
    /// Panics on shifts that are not a permutation of the four byte lanes
    /// (channels would overlap), on a degenerate extent, or for anything
    /// [`PlaneProgram::compile`] refuses — too many buffer slots, a buffer
    /// past exact `f32` indexing, a `Field` width that does not match the
    /// JIT's, or a compile failure.
    #[must_use]
    pub fn compile(channels: &[Kernel; 4], shifts: [u32; 4], extent: [u32; 2]) -> Self {
        {
            let mut lanes = shifts;
            lanes.sort_unstable();
            assert_eq!(
                lanes,
                [0, 8, 16, 24],
                "PackedProgram::compile: shifts {shifts:?} are not a \
                 permutation of the byte lanes; channels would overlap"
            );
        }
        Self {
            shifts,
            program: PlaneProgram::compile(&packed_kernel(channels, shifts), extent),
        }
    }

    /// The byte lanes this program's kernel packs into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.shifts
    }

    /// The pixel extents this program was compiled for.
    #[must_use]
    pub fn extent(&self) -> [u32; 2] {
        self.program.extent()
    }

    /// The memory the composed kernel declared, in slot order — how a caller
    /// that minted the identities checks that its reads merged the way it
    /// intended.
    #[must_use]
    pub fn buffers(&self) -> &[BufferDecl] {
        self.program.buffers()
    }

    /// The compiled kernel's emitted bytes, for the profiling harness in
    /// `cell_grid`'s tests. `PlaneProgram::code_bytes` is the public way in.
    #[cfg(test)]
    pub(crate) fn code_bytes(&self) -> &[u8] {
        self.program.code_bytes()
    }

    /// Bind one frame's memory, by buffer identity — see
    /// [`PlaneProgram::bind`]. A program whose channels read nothing (a
    /// procedural shader) binds the empty slice.
    ///
    /// # Panics
    ///
    /// Panics if a declared slot has no buffer bound to it, or if a bound
    /// buffer's length is not the one its declaration promised.
    #[must_use]
    pub fn bind(&self, buffers: &[(BufferIdentity, Arc<Vec<f32>>)]) -> PackedFrame {
        PackedFrame {
            shifts: self.shifts,
            frame: self.program.bind(buffers),
        }
    }
}

/// One frame of a packed program: the compiled kernel plus the memory it
/// reads. Cheap to clone.
#[derive(Clone)]
pub struct PackedFrame {
    shifts: [u32; 4],
    frame: PlaneFrame,
}

impl PackedFrame {
    /// The byte lanes these words are packed into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.shifts
    }

    /// The padded row stride (in pixels) a bake writes: `width` rounded up to
    /// whole SIMD batches.
    #[must_use]
    pub fn padded_width(width: usize) -> usize {
        PlaneFrame::padded_width(width)
    }

    /// Bake packed pixels over the pixel rows `y0 .. y0 + rows` at
    /// pixel-center coordinates, into a plane of
    /// [`PackedFrame::padded_width`]`(width)`-stride rows.
    ///
    /// The kernel's root is int-domain: each lane already holds a packed
    /// pixel's bit pattern, and the collapse store is a raw vector store, so
    /// what reaches memory is exactly what the OR-fold built.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero or `out` is shorter than
    /// `rows * padded_width(width)`.
    pub fn bake_packed_rows(&self, region: PlaneRegion, out: &mut [u32]) {
        self.frame.bake_int_rows(region, out);
    }
}
