//! # A kernel compiled at a frame's lattice, baked in bands of rows
//!
//! One [`Kernel`], one frame extent, one compiled collapse program: the loop
//! over batches and rows lives inside the emitted code, with the two-level
//! LICM prologues (per call, per row) active, and one call bakes a whole band.
//! Whatever memory the kernel declared is bound by identity once per frame and
//! stays bound for every band baked from it.
//!
//! This is the shape every frame path in the tree already had — the cell
//! grid's four channel programs and its packed sibling were two copies of it
//! — with nothing above it: no colour, no channels, no pixel format. Those
//! live a layer up, in `pixelflow-graphics`, which composes a packed-pixel
//! kernel and hands it here.
//!
//! ## Output planes
//!
//! A band is written as `rows` rows at [`PlaneFrame::padded_width`] stride —
//! the sampled width rounded up to whole SIMD batches. The padding lanes hold
//! whatever the kernel computed past the right edge and are simply not read
//! back. The store is a raw vector store, type-blind bit movement, so a kernel
//! whose root is int-domain (a packed pixel, a mask) bakes through
//! [`PlaneFrame::bake_int_rows`] into a `u32` plane exactly: no float
//! operation touches the value between the root and memory.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::Field;
use pixelflow_codegen::JitManifold;
use pixelflow_ir::Kernel;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};

/// Buffer slots a [`PlaneFrame`] can bind without allocating: binding builds
/// its base-pointer array on the stack, so the bound is what makes that array
/// a fixed size. A kernel declaring more is refused at compile time rather
/// than silently overflowing it.
pub const MAX_BOUND_BUFFERS: usize = 4;

/// A horizontal band of rows to bake: `width` samples across, rows
/// `y0 .. y0 + rows`.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRegion {
    /// Samples per row.
    pub width: usize,
    /// First row.
    pub y0: usize,
    /// Number of rows.
    pub rows: usize,
}

/// One kernel compiled against one frame extent.
///
/// Compile once per shape; bind memory per frame ([`PlaneProgram::bind`]).
/// The program's size and compile time are independent of the frame's.
pub struct PlaneProgram {
    jit: Arc<JitManifold>,
    /// The lattice the kernel was specialized to. Every region baked through
    /// it lies within these extents.
    extent: [u32; 2],
    /// The memory the kernel declared, in the slot order its ABI binds.
    /// Shared so a [`PlaneFrame`] stays cheap to clone.
    slots: Arc<[BufferDecl]>,
}

impl PlaneProgram {
    /// JIT-compile `kernel` in collapse mode over a `extent[0] × extent[1]`
    /// lattice.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate extent, when this build's `Field` width does not
    /// match the JIT's emitted width, when the kernel declares more buffers
    /// than [`MAX_BOUND_BUFFERS`] or one too large to index exactly in `f32`,
    /// or if compilation fails.
    #[must_use]
    pub fn compile(kernel: &Kernel, extent: [u32; 2]) -> Self {
        assert!(
            extent[0] > 0 && extent[1] > 0,
            "PlaneProgram::compile: degenerate extent {extent:?}"
        );
        assert_eq!(
            core::mem::size_of::<Field>(),
            pixelflow_codegen::JIT_VECTOR_BYTES,
            "PlaneProgram::compile: Field width does not match the JIT's emitted width"
        );
        let (arena, root) = kernel.parts();
        let slots: Vec<BufferDecl> = arena.buffers().to_vec();
        assert!(
            slots.len() <= MAX_BOUND_BUFFERS,
            "PlaneProgram::compile: kernel needs {} buffer slots, over the \
             {MAX_BOUND_BUFFERS} a frame can bind without allocating",
            slots.len()
        );
        for decl in &slots {
            assert!(
                buffer_len(decl) <= EXACT_F32_INDEX,
                "PlaneProgram::compile: buffer of {} elements exceeds the \
                 exactly f32-indexable range (2^24); gathers would alias \
                 adjacent samples",
                buffer_len(decl)
            );
        }
        // Bound-memory arenas are uncacheable (the code bakes buffer slot
        // metadata); the cache recognizes that and compiles fresh.
        let jit = pixelflow_codegen::jit_cache::compile(
            arena,
            root,
            pixelflow_ir::LatticeShape::new([extent[0], extent[1], 1, 1]),
        )
        .expect("PlaneProgram: kernel failed to compile");
        Self {
            jit,
            extent,
            slots: slots.into(),
        }
    }

    /// The lattice extents this program was compiled for.
    #[must_use]
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// The memory this program's kernel declared, in slot order. A caller that
    /// minted the identities can say which slot is which without inferring it
    /// from extents.
    #[must_use]
    pub fn buffers(&self) -> &[BufferDecl] {
        &self.slots
    }

    /// The compiled kernel's emitted bytes (research/profiling harness).
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        self.jit.code_bytes()
    }

    /// Bind one frame's memory: each declared slot takes the buffer carrying
    /// its identity. `buffers` may be given in any order and may carry entries
    /// this kernel does not read. Buffers are `Arc`s so a frame in flight
    /// keeps its data alive while the caller prepares the next one.
    ///
    /// # Panics
    ///
    /// Panics if a declared slot has no buffer bound to it, or if a buffer's
    /// length is not the `width × height` its declaration promised — the
    /// gathers address the declared shape, so a shorter buffer would be read
    /// past its end through an entirely safe API.
    #[must_use]
    pub fn bind(&self, buffers: &[(BufferIdentity, Arc<Vec<f32>>)]) -> PlaneFrame {
        let mut bound: [Option<Arc<Vec<f32>>>; MAX_BOUND_BUFFERS] = Default::default();
        for (slot, decl) in bound.iter_mut().zip(self.slots.iter()) {
            let data = buffers
                .iter()
                .find(|(id, _)| *id == decl.id)
                .map(|(_, data)| data)
                .unwrap_or_else(|| panic!("PlaneProgram::bind: nothing bound to slot {decl:?}"));
            assert_eq!(
                data.len(),
                buffer_len(decl),
                "PlaneProgram::bind: buffer of {} floats bound to slot {decl:?}",
                data.len()
            );
            *slot = Some(Arc::clone(data));
        }
        PlaneFrame {
            jit: Arc::clone(&self.jit),
            extent: self.extent,
            bound,
        }
    }
}

/// Gathers compute a row-major linear index in `f32`, which is exact only
/// below 2^24 — beyond that adjacent samples alias.
const EXACT_F32_INDEX: usize = 1 << 24;

/// Elements a declaration promises.
///
/// # Panics
///
/// Panics if the product overflows `usize`. A wrapped length would be SMALL,
/// so `bind` would accept a correspondingly small buffer while the compiled
/// kernel still declared the true row width — and the gathers would read
/// billions of elements past the end through an entirely safe API.
fn buffer_len(decl: &BufferDecl) -> usize {
    (decl.width as usize)
        .checked_mul(decl.height as usize)
        .expect("PlaneProgram: declared buffer length overflows usize")
}

/// One frame: the compiled program plus the memory it reads. Cheap to clone
/// (one `Arc` for the code, one per bound buffer).
#[derive(Clone)]
pub struct PlaneFrame {
    jit: Arc<JitManifold>,
    extent: [u32; 2],
    /// Bound memory in slot order; entries past the declared slots stay
    /// `None` and are never addressed, because the kernel only reads slots it
    /// declared.
    bound: [Option<Arc<Vec<f32>>>; MAX_BOUND_BUFFERS],
}

impl PlaneFrame {
    /// The padded row stride (in samples) a bake writes: `width` rounded up to
    /// whole SIMD batches. Planes are laid out at this stride; the padding
    /// lanes hold whatever the kernel computed past the right edge and are
    /// simply not read back.
    #[must_use]
    pub fn padded_width(width: usize) -> usize {
        let lanes = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
        width.div_ceil(lanes).max(1) * lanes
    }

    /// The lattice extents the kernel was compiled for.
    #[must_use]
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Bake the region at sample-center coordinates (`x + ½`, `y + ½`) into a
    /// plane of [`PlaneFrame::padded_width`]`(width)`-stride rows.
    ///
    /// ONE call into the collapse kernel per invocation.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero or `out` is shorter than
    /// `rows * padded_width(width)`.
    pub fn bake_rows(&self, region: PlaneRegion, out: &mut [f32]) {
        let rows = self.check_region("bake_rows", region, out.len());
        // SAFETY: see `bake`. `out` is an `f32` plane, which is what the
        // collapse ABI writes.
        unsafe { self.bake(region, out.as_mut_ptr(), rows) }
    }

    /// [`PlaneFrame::bake_rows`] for a kernel whose root is int-domain: each
    /// lane already holds a bit pattern (a packed pixel, a mask), and the
    /// collapse store is a raw vector store — type-blind bit movement — so
    /// writing through the ABI's `*mut f32` into a `u32` plane is exact.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero or `out` is shorter than
    /// `rows * padded_width(width)`.
    pub fn bake_int_rows(&self, region: PlaneRegion, out: &mut [u32]) {
        let rows = self.check_region("bake_int_rows", region, out.len());
        // SAFETY: see `bake`. `u32` and `f32` share size and alignment, and
        // the store moves the root's bit pattern without interpreting it.
        unsafe { self.bake(region, out.as_mut_ptr().cast::<f32>(), rows) }
    }

    /// Shared guard: returns the number of rows to bake, or zero for an empty
    /// band.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, the region leaves the compiled
    /// extents, or `out_len` cannot hold the band.
    fn check_region(&self, what: &str, region: PlaneRegion, out_len: usize) -> usize {
        let PlaneRegion { width, y0, rows } = region;
        assert!(width > 0, "{what}: zero width");
        // The kernel was compiled for `extent`; a region outside it would run
        // the collapse loop past the lattice it was specialized to.
        //
        // `debug_assert`, matching `JitManifold::call_collapse`'s own check of
        // the same promise: today's emitted code takes its loop bounds from
        // the tile at run time, so a wider region is merely a stale cache key,
        // not wrong samples. It becomes load-bearing when the emitted code
        // specializes on the extents, and is promoted with that change rather
        // than ahead of it — a release panic for a promise nothing yet relies
        // on is a new way for a terminal to die.
        let (fw, fh) = (self.extent[0] as usize, self.extent[1] as usize);
        debug_assert!(
            width <= fw && y0.saturating_add(rows) <= fh,
            "{what}: region {width}×{rows} at row {y0} lies outside the \
             {fw}×{fh} lattice this program was compiled for"
        );
        let stride = Self::padded_width(width);
        // Checked: `rows * stride` wraps in release for a caller-supplied
        // region large enough, and a wrapped product would let an undersized
        // `out` pass this guard while the collapse call below still received
        // the real (enormous) row count and wrote past the slice. The
        // documented panic must fire before any unsafe call, not after.
        let needed = rows
            .checked_mul(stride)
            .expect("bake: rows * stride overflows usize");
        assert!(
            out_len >= needed,
            "{what}: plane of {out_len} elements cannot hold {rows} rows at stride {stride}"
        );
        rows
    }

    /// # Safety
    ///
    /// `out` must be writable for `rows * padded_width(region.width)` 4-byte
    /// elements — which [`PlaneFrame::check_region`] asserted for the slice it
    /// came from.
    unsafe fn bake(&self, region: PlaneRegion, out: *mut f32, rows: usize) {
        if rows == 0 {
            return;
        }
        let stride = Self::padded_width(region.width);
        let groups = stride / (pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>());
        // One base pointer per declared slot, in slot order. Stack-allocated
        // against the MAX_BOUND_BUFFERS bound `compile` checked, so baking a
        // band allocates nothing; trailing entries stay null and are never
        // read because the kernel only addresses slots it declared.
        let mut ctx = [core::ptr::null::<f32>(); MAX_BOUND_BUFFERS];
        for (dst, src) in ctx.iter_mut().zip(self.bound.iter()) {
            if let Some(data) = src {
                *dst = data.as_ptr();
            }
        }
        // Sample centers: the rasterizer convention (x + ½, y + ½).
        let x0 = Field::sequential(0.5);
        let y = Field::from(region.y0 as f32 + 0.5);
        let zero = Field::from(0.0);
        // SAFETY: `compile` checked size_of::<Field>() == JIT_VECTOR_BYTES and
        // that every declared slot fits `ctx`; `bind` bound a buffer of the
        // declared length to each of them and this frame holds those `Arc`s
        // alive for the duration of the call; the caller's guard proved `out`
        // holds `rows` full rows of `groups` batches with no scalar tail
        // (row_skip = 0 — the stride IS whole batches).
        unsafe {
            self.jit.call_collapse(
                ctx.as_ptr(),
                pixelflow_codegen::TileSlice::contiguous(out, groups, rows),
                pixelflow_codegen::Point4::new(x0, y, zero, zero),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Bake a kernel of the coordinates themselves and read the plane back:
    /// the region's absolute rows, the sample-center convention, and the
    /// padded stride, in one.
    #[test]
    fn a_band_bakes_its_absolute_rows_at_sample_centers() {
        let kernel = Kernel::x().mul(&Kernel::constant(100.0)).add(&Kernel::y());
        let program = PlaneProgram::compile(&kernel, [8, 8]);
        let frame = program.bind(&[]);
        let stride = PlaneFrame::padded_width(5);
        let mut out = vec![0.0f32; 3 * stride];
        frame.bake_rows(
            PlaneRegion {
                width: 5,
                y0: 4,
                rows: 3,
            },
            &mut out,
        );
        for row in 0..3 {
            for col in 0..5 {
                let expected = (col as f32 + 0.5) * 100.0 + (4 + row) as f32 + 0.5;
                assert!(
                    (out[row * stride + col] - expected).abs() < 1e-3,
                    "row {row} col {col}: {} != {expected}",
                    out[row * stride + col]
                );
            }
        }
    }

    /// An int-domain root reaches memory as the bit pattern the kernel built,
    /// with no float operation in between.
    #[test]
    fn an_int_domain_root_bakes_into_a_u32_plane_bit_exactly() {
        let kernel = Kernel::x()
            .trunc_to_int()
            .shl(8)
            .or(&Kernel::y().trunc_to_int())
            .into_kernel();
        let program = PlaneProgram::compile(&kernel, [4, 4]);
        let frame = program.bind(&[]);
        let stride = PlaneFrame::padded_width(4);
        let mut out = vec![0u32; 2 * stride];
        frame.bake_int_rows(
            PlaneRegion {
                width: 4,
                y0: 0,
                rows: 2,
            },
            &mut out,
        );
        for row in 0..2u32 {
            for col in 0..4u32 {
                assert_eq!(
                    out[row as usize * stride + col as usize],
                    (col << 8) | row,
                    "row {row} col {col}"
                );
            }
        }
    }

    #[test]
    fn padded_width_rounds_up_to_whole_simd_batches() {
        let lanes = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
        assert_eq!(PlaneFrame::padded_width(1), lanes);
        assert_eq!(PlaneFrame::padded_width(lanes), lanes);
        assert_eq!(PlaneFrame::padded_width(lanes + 1), lanes * 2);
    }

    #[test]
    #[should_panic(expected = "nothing bound to slot")]
    fn a_declared_slot_with_no_buffer_is_refused() {
        let buffer = BufferIdentity::mint();
        let kernel = crate::lattice::DiscreteManifold::kernel_for(buffer, 4, 4).at(
            &Kernel::x(),
            &Kernel::y(),
            &Kernel::z(),
            &Kernel::w(),
        );
        let program = PlaneProgram::compile(&kernel, [4, 4]);
        let _refused = program.bind(&[]);
    }

    #[test]
    #[should_panic(expected = "floats bound to slot")]
    fn a_buffer_of_the_wrong_length_is_refused() {
        let buffer = BufferIdentity::mint();
        let kernel = crate::lattice::DiscreteManifold::kernel_for(buffer, 4, 4).at(
            &Kernel::x(),
            &Kernel::y(),
            &Kernel::z(),
            &Kernel::w(),
        );
        let program = PlaneProgram::compile(&kernel, [4, 4]);
        let _refused = program.bind(&[(buffer, Arc::new(vec![0.0f32; 15]))]);
    }
}
