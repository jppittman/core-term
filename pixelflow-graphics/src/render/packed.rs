//! # A colour output is a colour tree compiled at the frame's shape
//!
//! Four channel kernels in `[0, 1]` — red, green, blue, alpha, which is all a
//! colour output ever is — packed to bytes and OR-folded into a `u32` pixel
//! **inside the kernel**, then compiled at the frame's lattice shape. That is
//! a compiled manifold with four channels, and collapsing a band of its rows
//! writes finished pixels: no per-batch FFI boundary, no virtual dispatch, no
//! per-pixel scalar pack, and the collapse kernel's two-level LICM prologues
//! (per call, per row) active.
//!
//! A colour is an [`Rgba`] tree rather than an array, so a *choice* between
//! colours reaches here as one node and leaves as one `Select` on the packed
//! words ([`pixelflow_ir::Bits::select`] — the blend the hardware already
//! does). That is the difference between a scene the emitter can short-circuit
//! and one it cannot: with four selects sharing a mask, everything either arm
//! computes is shared from each select's point of view, and none of it can be
//! skipped.
//!
//! The layer below ([`pixelflow_core::PlaneProgram`]) is the same object with
//! one channel and no pack — it knows lattices, buffers and bit patterns and
//! nothing about colour. This is where byte lanes and pixel formats begin.

use std::sync::Arc;

use pixelflow_core::{PlaneFrame, PlaneProgram, PlaneRegion};
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
use pixelflow_ir::{Bits, Kernel};

use crate::scene3d::Rgba;

/// A colour packed to one `u32` pixel: each leaf's four channels packed to a
/// byte exactly as `Pixel::from_rgba` does — `(x·255).clamp(0, 255)` then
/// truncate toward zero — shifted to its byte lane and OR-folded, and each
/// choice between colours blended as one `Select` on those words.
///
/// `shifts[c]` is the bit position of channel `c` in `(r, g, b, a)` order.
/// Both pixel orders are little-endian byte arrays wrapping a `u32`, so byte
/// `i` sits at bit `8·i`: `Rgba8` is `[r, g, b, a]` → `[0, 8, 16, 24]`, and
/// `Bgra8` is `[b, g, r, a]` → `[16, 8, 0, 24]`.
///
/// Truncation (not rounding) is deliberate twice over: it is bit-exact with
/// the scalar pack's `as u8`, and `cvttps2dq`/`fcvtzs` both truncate toward
/// zero — no per-target tie divergence to inherit.
///
/// Selecting words and selecting channels give the same bits, since `Select`
/// is a lanewise bitwise blend and the pack is lanewise: whatever the
/// not-taken arm packed is masked away whole. That equality is why this
/// changes no pixel and why the goldens do not move
/// (`selecting_packed_words_is_selecting_the_channels`).
pub(crate) fn packed_kernel(color: &Rgba, shifts: [u32; 4]) -> Kernel {
    let k = Kernel::constant;
    color
        .fold(
            &|channels: &[Kernel; 4]| {
                let byte = |c: usize| {
                    channels[c]
                        .mul(&k(255.0))
                        .clamp(&k(0.0), &k(255.0))
                        .trunc_to_int()
                        .shl(shifts[c])
                };
                byte(0).or(&byte(1)).or(&byte(2)).or(&byte(3))
            },
            &|mask, if_true: Bits, if_false: Bits| Bits::select(mask, &if_true, &if_false),
        )
        // The single named exit from the bit domain: the packed pattern IS the
        // kernel's output.
        .into_kernel()
}

/// A colour output compiled at a frame's lattice shape: a compiled manifold
/// with four channels, whose root is a packed `u32` pixel, so the collapse
/// loop stores finished pixels directly.
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
    /// Compose and JIT-compile the packed program for a colour over an
    /// `extent[0] × extent[1]` pixel frame.
    ///
    /// Four channel kernels are a colour with no choice in it
    /// (`Rgba::from(&channels)`), which is what a procedural shader hands over.
    ///
    /// # Panics
    ///
    /// Panics on shifts that are not a permutation of the four byte lanes
    /// (channels would overlap), on a degenerate extent, or for anything
    /// [`PlaneProgram::compile`] refuses — too many buffer slots, a buffer
    /// past exact `f32` indexing, a `Field` width that does not match the
    /// JIT's, or a compile failure.
    #[must_use]
    pub fn compile(color: &Rgba, shifts: [u32; 4], extent: [u32; 2]) -> Self {
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
            program: PlaneProgram::compile(&packed_kernel(color, shifts), extent),
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

    /// The compiled kernel's emitted bytes — what the scene will actually
    /// run, for a harness that measures or disassembles it
    /// ([`pixelflow_core::PlaneProgram::code_bytes`] is the same hook one
    /// layer down). Inspection only: nothing about rendering goes through it.
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
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
            slice: [0.0, 0.0],
        }
    }
}

/// One frame of a packed program: the compiled manifold, the memory it reads,
/// and the plane of the kernel's four-coordinate domain this frame samples.
/// Cheap to clone.
#[derive(Clone)]
pub struct PackedFrame {
    shifts: [u32; 4],
    frame: PlaneFrame,
    /// The `Z` and `W` every pixel of this frame carries — see
    /// [`PackedFrame::on_slice`].
    slice: [f32; 2],
}

impl PackedFrame {
    /// The byte lanes these words are packed into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.shifts
    }

    /// The same compiled program sampled on the plane at `(z, w)` instead of
    /// the origin plane.
    ///
    /// This is how an animated scene stays compiled: the channel kernels read
    /// time as `Kernel::w()`, the program is compiled once, and each frame
    /// binds its timestamp here. Baking the time into the kernel as a
    /// constant would recompile the scene every frame instead.
    #[must_use]
    pub fn on_slice(mut self, z: f32, w: f32) -> Self {
        self.slice = [z, w];
        self
    }

    /// The plane this frame samples: `(z, w)`.
    #[must_use]
    pub fn slice(&self) -> (f32, f32) {
        (self.slice[0], self.slice[1])
    }

    /// Collapse the pixel rows `y0 .. y0 + rows` at pixel-center coordinates,
    /// straight into `out` — a plane of `u32` words whose rows are `stride`
    /// pixels apart, which for a frame is its own row width.
    ///
    /// The kernel's root is int-domain: each lane already holds a packed
    /// pixel's bit pattern, and the collapse store is a raw vector store, so
    /// what reaches memory is exactly what the OR-fold built.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, or
    /// `out` cannot hold the band.
    pub fn collapse_rows(&self, region: PlaneRegion, out: &mut [u32], stride: usize) {
        self.frame
            .collapse_int_rows(region.on_slice(self.slice[0], self.slice[1]), out, stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::Lattice;

    const RGBA: [u32; 4] = [0, 8, 16, 24];
    const BGRA: [u32; 4] = [16, 8, 0, 24];

    /// The packed words a kernel produces over a `w × 1` strip. The root is
    /// int-domain, so the lane's bit pattern IS the pixel.
    fn words(kernel: &Kernel, w: usize) -> Vec<u32> {
        Lattice::frame(w, 1, 0.0)
            .bake(kernel)
            .buffer()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// The four channels of a colour that has no choice in it.
    fn leaf(color: &Rgba) -> [Kernel; 4] {
        color.fold(&|channels| channels.clone(), &|_, _, _| {
            panic!("this colour is a choice, not a leaf")
        })
    }

    /// **The equality S3b rests on.** Choosing between two packed words is
    /// choosing between each of their channels: `Select` is a lanewise
    /// bitwise blend and the pack is lanewise, so the not-taken arm's bytes
    /// are masked away whole. Bit-exact, under both byte orders — which is
    /// why one select over a whole colour draws the same picture as four, and
    /// why no golden moves when the colour becomes a tree.
    #[test]
    fn selecting_packed_words_is_selecting_the_channels() {
        let k = Kernel::constant;
        let ramp = |scale: f32| Kernel::x().mul(&k(scale / 8.0));
        let a = Rgba::new(ramp(1.0), k(0.25), k(0.5), k(1.0));
        let b = Rgba::new(k(0.75), ramp(0.5), k(0.125), k(1.0));
        // A mask that is neither uniform nor aligned to a SIMD batch, so the
        // two forms are compared where the blend actually blends.
        let mask = Kernel::x().lt(&k(3.0));

        let (ca, cb) = (leaf(&a), leaf(&b));
        let per_channel = Rgba::new(
            mask.select(&ca[0], &cb[0]),
            mask.select(&ca[1], &cb[1]),
            mask.select(&ca[2], &cb[2]),
            mask.select(&ca[3], &cb[3]),
        );
        let one_select = a.select(&mask, &b);

        for shifts in [RGBA, BGRA] {
            assert_eq!(
                words(&packed_kernel(&one_select, shifts), 8),
                words(&packed_kernel(&per_channel, shifts), 8),
                "one select on the words disagrees with four on the channels, \
                 shifts {shifts:?}"
            );
        }
    }

    /// The two byte orders are the same pixels in a different arrangement, so
    /// a test that passed by packing everything into one lane would not
    /// notice. This one does.
    #[test]
    fn the_two_byte_orders_disagree_about_where_red_goes() {
        let k = Kernel::constant;
        let red = Rgba::new(k(1.0), k(0.0), k(0.0), k(1.0));
        assert_eq!(words(&packed_kernel(&red, RGBA), 1)[0], 0xff00_00ff);
        assert_eq!(
            words(&packed_kernel(&red, BGRA), 1)[0],
            0xff00_0000 | 0xff_0000
        );
    }
}
