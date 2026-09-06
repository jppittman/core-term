//! # A kernel compiled at a 2D lattice shape, collapsed in bands of rows
//!
//! The middle object of `kernel ──compile(shape)──▶ manifold
//! ──collapse(lattice)──▶ buffer`, for a plane. A [`Kernel`] is the
//! description; compiling it at an extent gives a **compiled manifold**
//! ([`PlaneProgram`], plus its bound memory as [`PlaneFrame`]); collapsing
//! that over a band of rows gives numbers. The loop over batches and rows
//! lives inside the emitted code, with the two-level LICM prologues (per
//! call, per row) active, so one collapse call covers a whole band, and
//! whatever memory the kernel declared is bound by identity once and stays
//! bound for every band collapsed from it.
//!
//! This is the shape every frame path in the tree already had — the cell
//! grid's four channel programs and its packed sibling were two copies of it
//! — with nothing above it: no colour, no channels, no pixel format. Those
//! live a layer up, in `pixelflow-graphics`, which composes a packed-pixel
//! kernel and compiles it here.
//!
//! ## Output planes
//!
//! A band is written straight into the caller's plane, whose rows are however
//! many elements apart the caller says: the collapse ABI stores whole SIMD
//! batches and steps the output pointer by the leftover bytes between rows,
//! so a destination at any stride is filled in place — no staging plane, no
//! per-row copy. The one thing a batch store cannot do is a row's final
//! *partial* batch when the stride has no room for its overhang; that batch
//! alone goes through a one-batch scratch, and it is the only place in this
//! module where the SIMD width is visible at all.
//!
//! The store is a raw vector store, type-blind bit movement, so a kernel
//! whose root is int-domain (a packed pixel, a mask) collapses through
//! [`PlaneFrame::collapse_int_rows`] into a `u32` plane exactly: no float
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

/// A horizontal band of the lattice to collapse: `width` samples across, rows
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

/// One kernel compiled at one 2D lattice shape: a compiled manifold, before
/// its memory is bound.
///
/// Compile once per shape; bind memory per frame ([`PlaneProgram::bind`]).
/// The program's size and compile time are independent of the lattice's.
///
/// **This is the compiled manifold that can bind memory**, and the only one.
/// A kernel composed over a buffer —
/// [`DiscreteManifold::kernel_for`](crate::DiscreteManifold::kernel_for)'s
/// gather, [`BilinearSampler::kernel_for`](crate::BilinearSampler::kernel_for)'s
/// 4-tap blend, or anything built on them — declares buffer slots, and
/// [`Lattice::bake`](crate::Lattice::bake) refuses those: it binds nothing, so
/// the gathers would load their base pointers out of a null context. Compile
/// the same kernel here, bind its buffers by identity, and collapse it into
/// `f32` rows.
///
/// The three colour-shaped things in the tree are all this object wearing
/// different numbers of channels: `Lattice::bake` is the one-channel,
/// buffer-free instance (`collapse(compile(k))` fused, and can stop being a
/// separate name); a field over bound memory is the one-channel instance with
/// its slots filled; and `pixelflow-graphics`'s packed program is four channel
/// kernels compiled through here with an integer pack at the root. Not three
/// paths — one, sampled three ways.
pub struct PlaneProgram {
    jit: Arc<JitManifold>,
    /// The lattice shape the kernel was specialized to. Every region
    /// collapsed through it lies within these extents.
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
        // Bound-memory arenas are uncacheable (the code embeds buffer slot
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
    /// The lattice extents the kernel was compiled for.
    #[must_use]
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Collapse the region at sample-center coordinates (`x + ½`, `y + ½`)
    /// into `out`, whose rows are `stride` elements apart and whose first
    /// `region.width` elements each row are the samples.
    ///
    /// The destination is written in place — the collapse loop's own stores
    /// land in it — so `stride` is whatever the caller's plane already is: a
    /// frame's packed row width, a padded scratch, a sub-rectangle of
    /// something larger.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, or
    /// `out` cannot hold the band.
    pub fn collapse_rows(&self, region: PlaneRegion, out: &mut [f32], stride: usize) {
        let band = self.plan("collapse_rows", region, stride, out.len());
        // SAFETY: see `collapse`. `out` is an `f32` plane, which is what the
        // collapse ABI writes, and `plan` proved it holds the band.
        unsafe { self.collapse(region, out.as_mut_ptr(), band) }
    }

    /// [`PlaneFrame::collapse_rows`] for a kernel whose root is int-domain:
    /// each lane already holds a bit pattern (a packed pixel, a mask), and the
    /// collapse store is a raw vector store — type-blind bit movement — so
    /// writing through the ABI's `*mut f32` into a `u32` plane is exact.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, or
    /// `out` cannot hold the band.
    pub fn collapse_int_rows(&self, region: PlaneRegion, out: &mut [u32], stride: usize) {
        let band = self.plan("collapse_int_rows", region, stride, out.len());
        // SAFETY: see `collapse`. `u32` and `f32` share size and alignment,
        // and the store moves the root's bit pattern without interpreting it.
        unsafe { self.collapse(region, out.as_mut_ptr().cast::<f32>(), band) }
    }

    /// How a band lands in a destination of a given stride, and the guard that
    /// the destination can hold it.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, the
    /// region leaves the compiled extents, or `out_len` cannot hold the band.
    fn plan(&self, what: &str, region: PlaneRegion, stride: usize, out_len: usize) -> BandPlan {
        let PlaneRegion { width, y0, rows } = region;
        assert!(width > 0, "{what}: zero width");
        assert!(
            stride >= width,
            "{what}: stride {stride} is narrower than the {width} samples a row holds"
        );
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
        let plan = BandPlan::new(width, rows, stride);
        // Checked: the span wraps in release for a caller-supplied region
        // large enough, and a wrapped product would let an undersized `out`
        // pass this guard while the collapse call below still received the
        // real (enormous) row count and wrote past the slice. The documented
        // panic must fire before any unsafe call, not after.
        let needed = plan.span().expect("collapse: band span overflows usize");
        assert!(
            out_len >= needed,
            "{what}: plane of {out_len} elements cannot hold {rows} rows at stride {stride}"
        );
        plan
    }

    /// # Safety
    ///
    /// `out` must be writable for `band.span()` 4-byte elements — which
    /// [`PlaneFrame::plan`] asserted for the slice it came from.
    unsafe fn collapse(&self, region: PlaneRegion, out: *mut f32, band: BandPlan) {
        if band.rows == 0 {
            return;
        }
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
        let zero = Field::from(0.0);
        // Sample centers: the rasterizer convention (x + ½, y + ½).
        if band.groups > 0 {
            // SAFETY: `compile` checked size_of::<Field>() == JIT_VECTOR_BYTES
            // and that every declared slot fits `ctx`; `bind` bound a buffer of
            // the declared length to each of them and this frame holds those
            // `Arc`s alive for the duration of the call; the caller's guard
            // proved `out` holds `rows` rows of `groups` whole batches at
            // `stride`, which is what `row_skip_bytes` steps between.
            unsafe {
                self.jit.call_collapse(
                    ctx.as_ptr(),
                    pixelflow_codegen::TileSlice::new(
                        out,
                        band.groups,
                        band.rows,
                        band.row_skip_bytes(),
                    ),
                    pixelflow_codegen::Point4::new(
                        Field::sequential(0.5),
                        Field::from(region.y0 as f32 + 0.5),
                        zero,
                        zero,
                    ),
                );
            }
        }
        let Some(tail) = band.tail() else { return };
        // The row's last, partial batch. A whole-batch store there would run
        // past the row into the next one, so it lands in a scratch batch and
        // only the samples that belong to the row are copied back — one extra
        // call per row, and only when the stride left no room for the overhang.
        let mut scratch = [0.0f32; BATCH_LANES];
        for row in 0..band.rows {
            // SAFETY: as above; `scratch` is exactly the one batch
            // `TileSlice::single` writes.
            unsafe {
                self.jit.call_collapse(
                    ctx.as_ptr(),
                    pixelflow_codegen::TileSlice::single(scratch.as_mut_ptr()),
                    pixelflow_codegen::Point4::new(
                        Field::sequential(band.whole_lanes() as f32 + 0.5),
                        Field::from((region.y0 + row) as f32 + 0.5),
                        zero,
                        zero,
                    ),
                );
            }
            // SAFETY: `plan` proved `out` holds `row * stride + width`
            // elements, and this writes the last `tail` of them.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    scratch.as_ptr(),
                    out.add(row * band.stride + band.whole_lanes()),
                    tail,
                );
            }
        }
    }
}

/// Lanes in one SIMD batch of the emitted code. The only place in this module
/// the width is visible, and it is here for one reason: a row's final partial
/// batch needs somewhere to land that is not the next row.
const BATCH_LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

/// How one band of rows is written into a destination plane: whole SIMD
/// batches stored straight through the collapse loop, and whatever partial
/// batch the stride left no room for.
#[derive(Clone, Copy, Debug)]
struct BandPlan {
    /// Samples per row the caller asked for.
    width: usize,
    rows: usize,
    /// Elements between the starts of two destination rows.
    stride: usize,
    /// Whole batches the collapse loop stores per row.
    groups: usize,
}

impl BandPlan {
    /// A final partial batch overhangs `width`, which is harmless exactly when
    /// the destination row is wide enough to absorb it — the padding lanes hold
    /// whatever the kernel computed past the right edge and are not read back.
    /// Otherwise it is left to the scratch batch, and the loop stores only the
    /// batches that fit.
    fn new(width: usize, rows: usize, stride: usize) -> Self {
        let batches = width.div_ceil(BATCH_LANES);
        let groups = if stride >= batches * BATCH_LANES {
            batches
        } else {
            width / BATCH_LANES
        };
        Self {
            width,
            rows,
            stride,
            groups,
        }
    }

    /// Samples of each row the collapse loop's whole batches cover.
    fn whole_lanes(self) -> usize {
        self.groups * BATCH_LANES
    }

    /// Bytes the collapse loop steps the output pointer by between rows.
    fn row_skip_bytes(self) -> usize {
        (self.stride - self.whole_lanes()) * core::mem::size_of::<f32>()
    }

    /// Samples per row left for the scratch batch, or `None` when the whole
    /// batches already covered the row.
    fn tail(self) -> Option<usize> {
        self.width
            .checked_sub(self.whole_lanes())
            .filter(|t| *t > 0)
    }

    /// Elements the destination must hold: every row but the last at full
    /// stride, then whichever of the batch overhang and the sampled width
    /// reaches further.
    fn span(self) -> Option<usize> {
        let Some(before_last) = self.rows.checked_sub(1) else {
            return Some(0);
        };
        before_last
            .checked_mul(self.stride)?
            .checked_add(self.whole_lanes().max(self.width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// `x * 100 + y` — every sample names the coordinate it was taken at, so a
    /// misplaced row or column is visible in the value rather than only in a
    /// difference.
    fn coordinate_kernel() -> Kernel {
        Kernel::x().mul(&Kernel::constant(100.0)).add(&Kernel::y())
    }

    fn expected(col: usize, row: usize) -> f32 {
        (col as f32 + 0.5) * 100.0 + row as f32 + 0.5
    }

    /// Collapse a kernel of the coordinates themselves and read the plane back:
    /// the region's absolute rows, the sample-center convention, and a
    /// destination whose rows are exactly as wide as the samples, in one.
    #[test]
    fn a_band_collapses_its_absolute_rows_at_sample_centers() {
        let program = PlaneProgram::compile(&coordinate_kernel(), [8, 8]);
        let frame = program.bind(&[]);
        let (width, y0, rows) = (5, 4, 3);
        let mut out = vec![0.0f32; rows * width];
        frame.collapse_rows(PlaneRegion { width, y0, rows }, &mut out, width);
        for row in 0..rows {
            for col in 0..width {
                let want = expected(col, y0 + row);
                assert!(
                    (out[row * width + col] - want).abs() < 1e-3,
                    "row {row} col {col}: {} != {want}",
                    out[row * width + col]
                );
            }
        }
    }

    /// The destination's rows are `stride` apart because the caller said so,
    /// not because the batch width worked out that way: every row's samples
    /// name their own coordinates, so a band placed at the wrong pitch reads
    /// back the wrong values. This is the case the collapse ABI's
    /// `row_skip_bytes` exists for, and the packed frame path is built on it.
    ///
    /// A spare row past the band stays pristine: the collapse fills the rows
    /// it was given and no more. (Within a row, everything past `width` is
    /// scratch — a final partial batch's lanes land there — which is why only
    /// the samples themselves are read back.)
    #[test]
    fn a_band_lands_at_the_stride_the_caller_asked_for() {
        let program = PlaneProgram::compile(&coordinate_kernel(), [16, 8]);
        let frame = program.bind(&[]);
        let (width, stride, rows) = (9, BATCH_LANES * 4 + 1, 4);
        const UNTOUCHED: f32 = -1.0;
        let mut out = vec![UNTOUCHED; (rows + 1) * stride];
        frame.collapse_rows(PlaneRegion { width, y0: 0, rows }, &mut out, stride);
        for row in 0..rows {
            for col in 0..width {
                let want = expected(col, row);
                assert!(
                    (out[row * stride + col] - want).abs() < 1e-3,
                    "row {row} col {col}: {} != {want}",
                    out[row * stride + col]
                );
            }
        }
        assert!(
            out[rows * stride..].iter().all(|&x| x == UNTOUCHED),
            "the collapse wrote past the {rows} rows it was given"
        );
    }

    /// Whatever the stride, the samples are the same: a row's final partial
    /// batch reaches memory through the scratch batch with the values it would
    /// have had from a whole-batch store into a padded row.
    #[test]
    fn a_partial_last_batch_collapses_the_same_samples_at_any_stride() {
        let program = PlaneProgram::compile(&coordinate_kernel(), [32, 8]);
        let frame = program.bind(&[]);
        let (width, rows) = (BATCH_LANES * 2 - 1, 3);
        let padded = BATCH_LANES * 2;
        let mut packed = vec![0.0f32; rows * width];
        let mut spread = vec![0.0f32; rows * padded];
        let region = PlaneRegion { width, y0: 1, rows };
        frame.collapse_rows(region, &mut packed, width);
        frame.collapse_rows(region, &mut spread, padded);
        for row in 0..rows {
            let a = &packed[row * width..(row + 1) * width];
            let b = &spread[row * padded..row * padded + width];
            assert_eq!(a, b, "row {row}: the stride changed the samples");
        }
    }

    /// An int-domain root reaches memory as the bit pattern the kernel built,
    /// with no float operation in between — including through the scratch
    /// batch, which is why the width here is not a whole batch.
    #[test]
    fn an_int_domain_root_collapses_into_a_u32_plane_bit_exactly() {
        let kernel = Kernel::x()
            .trunc_to_int()
            .shl(8)
            .or(&Kernel::y().trunc_to_int())
            .into_kernel();
        let width = BATCH_LANES + 1;
        let program = PlaneProgram::compile(&kernel, [width as u32, 4]);
        let frame = program.bind(&[]);
        let mut out = vec![0u32; 2 * width];
        frame.collapse_int_rows(
            PlaneRegion {
                width,
                y0: 0,
                rows: 2,
            },
            &mut out,
            width,
        );
        for row in 0..2u32 {
            for col in 0..width as u32 {
                assert_eq!(
                    out[row as usize * width + col as usize],
                    (col << 8) | row,
                    "row {row} col {col}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "narrower than")]
    fn a_stride_below_the_sampled_width_is_refused() {
        let program = PlaneProgram::compile(&coordinate_kernel(), [8, 8]);
        let frame = program.bind(&[]);
        let mut out = vec![0.0f32; 32];
        frame.collapse_rows(
            PlaneRegion {
                width: 5,
                y0: 0,
                rows: 2,
            },
            &mut out,
            4,
        );
    }

    /// The same object with one channel and no pack: a kernel whose root is a
    /// gather over a declared buffer, compiled at the buffer's shape, bound by
    /// identity, and collapsed into one `f32` field. `Lattice::bake` cannot do
    /// this — it binds nothing and refuses a kernel that declares slots — so
    /// this is how a buffer-backed sampler (a glyph's coverage, a texture)
    /// reaches numbers.
    #[test]
    fn a_kernel_over_bound_memory_collapses_its_samples() {
        let buffer = BufferIdentity::mint();
        let (bw, bh) = (4u32, 3u32);
        let data: Vec<f32> = (0..bw * bh).map(|i| i as f32 * 0.25).collect();
        let kernel = crate::lattice::DiscreteManifold::kernel_for(buffer, bw, bh).at(
            &Kernel::x(),
            &Kernel::y(),
            &Kernel::z(),
            &Kernel::w(),
        );
        let program = PlaneProgram::compile(&kernel, [bw, bh]);
        assert_eq!(
            program.buffers().len(),
            1,
            "the sampler's slot must survive to the compiled program"
        );
        let frame = program.bind(&[(buffer, Arc::new(data.clone()))]);
        let (width, rows) = (bw as usize, bh as usize);
        let mut out = vec![0.0f32; rows * width];
        frame.collapse_rows(PlaneRegion { width, y0: 0, rows }, &mut out, width);
        assert_eq!(out, data, "the collapse did not read the bound buffer");
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
