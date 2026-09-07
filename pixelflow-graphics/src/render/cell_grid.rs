//! # The cell grid as a packed program
//!
//! [`pixelflow_core::CellGridShape`] denotes four channel kernels over two
//! bound buffers — cell data and a coverage atlas — and knows nothing about
//! bytes or pixel formats. This is the instance that turns them into a frame:
//! the four channels packed to one `u32` pixel in IR ([`PackedManifold`]) for a
//! given byte order, with the two buffers bound per frame.
//!
//! What the program is compiled against is the shape — extents, and nothing
//! else. The grid's *metric* (cell size, sample density, tile size, display
//! scale) is a [`CellGridMetrics`] written into a [`UniformBlock`], so a
//! zoom, a font-size change or a DPI change reuses the compiled kernel; see
//! the core module's docs. The frame indices the grid does not cover are an
//! [`IndexRange`]'s complement, painted rather than computed.
//!
//! The terminal's text grid is the production caller; sprite grids and
//! tile-based scenes are the same primitive.

use std::sync::Arc;

use pixelflow_core::{
    paint_outside, CellGridBuffers, CellGridKernels, CellGridMetrics, CellGridShape, CellGridSlots,
    IndexRange, PlaneRegion, UniformBlock,
};

use crate::render::packed::{pack_scalar, PackedFrame, PackedManifold};
use crate::scene3d::Rgba;

/// A cell grid compiled for one shape and one pixel byte order.
///
/// Compile once per shape ([`CellGridPackedManifold::compile`]) — only an
/// extent recompiles anything — lay out a metric with
/// [`CellGridPackedManifold::params`], and stamp per-frame data into it with
/// [`CellGridPackedManifold::frame`].
pub struct CellGridPackedManifold {
    shape: CellGridShape,
    buffers: CellGridBuffers,
    slots: CellGridSlots,
    /// The word painted over the frame indices outside the grid, packed for
    /// this program's shifts. Not in the kernel: a constant on a range is
    /// data, and a program is not recompiled to change its background.
    border: u32,
    packed: PackedManifold,
}

/// One metric, laid out for one packed program: the argument values its
/// kernel reads, and the index range the grid covers at that metric.
///
/// Making one is where a resize's arithmetic happens; a frame that reuses it
/// costs nothing but refcounts, which is what keeps the render path free of
/// per-frame allocation.
pub struct CellGridPackedParams {
    block: UniformBlock,
    grid: [u32; 2],
}

impl CellGridPackedParams {
    /// The frame indices the grid covers at this metric.
    #[must_use]
    pub fn grid_range(&self) -> IndexRange {
        IndexRange::new(0, 0, self.grid[0] as usize, self.grid[1] as usize)
    }
}

impl CellGridPackedManifold {
    /// Compose and JIT-compile the packed program for `shape`, with
    /// `default_bg` (linear RGBA) painted outside the grid and `shifts[c]`
    /// giving the bit position of channel `c` in `(r, g, b, a)` order —
    /// `[0, 8, 16, 24]` for `Rgba8`, `[16, 8, 0, 24]` for `Bgra8`.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate shape, on shifts that are not a permutation
    /// of the four byte lanes, if the composed kernel's buffer reads do not
    /// merge to exactly one cell slot and one atlas slot, or if compilation
    /// fails.
    #[must_use]
    pub fn compile(shape: CellGridShape, default_bg: [f32; 4], shifts: [u32; 4]) -> Self {
        let CellGridKernels {
            channels,
            buffers,
            slots,
        } = shape.channel_kernels();
        let packed = PackedManifold::compile(
            &Rgba::from(&channels),
            shifts,
            [shape.frame_w, shape.frame_h],
        );
        // Identity-merge across the four channels' splices is load-bearing:
        // all cell reads and atlas taps must land in the same two slots a
        // frame binds. A third slot means splice stopped merging — refuse.
        let decls = packed.buffers();
        assert_eq!(
            decls.iter().filter(|d| d.id == buffers.cells).count(),
            1,
            "packed cell-grid kernel did not merge its cell reads to one slot"
        );
        assert_eq!(
            decls.iter().filter(|d| d.id == buffers.atlas).count(),
            1,
            "packed cell-grid kernel did not merge its atlas taps to one slot"
        );
        Self {
            shape,
            buffers,
            slots,
            border: pack_scalar(default_bg, shifts),
            packed,
        }
    }

    /// The shape this program was compiled for.
    #[must_use]
    pub fn shape(&self) -> &CellGridShape {
        &self.shape
    }

    /// The byte lanes this program's kernel packs into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.packed.shifts()
    }

    /// The compiled kernel's emitted bytes, for the profiling harness in
    /// `cell_grid`'s tests. `Manifold::code_bytes` is the public way in.
    #[cfg(test)]
    pub(crate) fn code_bytes(&self) -> &[u8] {
        self.packed.code_bytes()
    }

    /// Lay `metrics` out for this program: the block its kernel reads, plus
    /// the grid's index range at that metric. **This is the whole of a resize
    /// that keeps the shape** — no arena, no saturation, no compile.
    ///
    /// # Panics
    ///
    /// Panics on a non-positive or non-finite metric.
    #[must_use]
    pub fn params(&self, metrics: &CellGridMetrics) -> CellGridPackedParams {
        let mut block = self.packed.block();
        self.slots
            .write(&mut block, metrics)
            .expect("the slots were minted with this program's kernels");
        CellGridPackedParams {
            block,
            grid: self.shape.grid_extent(metrics),
        }
    }

    /// Bind one frame's data. `cells` is `CELL_STRIDE` `f32`s per cell,
    /// row-major; `atlas` is the coverage atlas, row-major texels. Both are
    /// `Arc`s so a frame in flight keeps its data alive while the caller
    /// prepares the next one.
    ///
    /// `params` is required rather than defaulted, and goes through
    /// [`PackedManifold::bind_with`]: the plain `bind` **refuses** a program
    /// that has arguments, precisely so a frame cannot silently render at the
    /// slots' defaults — a one-point-cell grid is a picture, and a plausible
    /// wrong picture is the worst kind.
    ///
    /// # Panics
    ///
    /// Panics if either buffer's length does not match the shape, or if
    /// `params` was laid out for a different program.
    #[must_use]
    pub fn frame(
        &self,
        params: &CellGridPackedParams,
        cells: Arc<Vec<f32>>,
        atlas: Arc<Vec<f32>>,
    ) -> CellGridPackedFrame {
        CellGridPackedFrame {
            frame: self.packed.bind_with(
                &[(self.buffers.cells, cells), (self.buffers.atlas, atlas)],
                &params.block,
            ),
            grid: params.grid,
            border: self.border,
        }
    }
}

/// One frame of a packed cell grid: the compiled program with its memory and
/// metric bound, the index range the grid covers, and the word outside it.
/// Cheap to clone.
///
/// **The two-summand union, specialized.** The grid's kernel answers for
/// [`Self::grid_range`] and the constant border answers for the rest; the two
/// are disjoint by construction, so nothing arbitrates between them and no
/// mask is multiplied through a program every pixel of the frame runs.
#[derive(Clone)]
pub struct CellGridPackedFrame {
    frame: PackedFrame,
    /// The grid's device-pixel extent, not the range itself: a grid is
    /// anchored at the frame origin, so two of a range's four numbers are
    /// structurally zero — and this value travels inside a `Scene`, on the
    /// stack, once per frame.
    grid: [u32; 2],
    border: u32,
}

impl CellGridPackedFrame {
    /// The byte lanes these words are packed into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.frame.shifts()
    }

    /// The frame indices the grid covers.
    #[must_use]
    pub fn grid_range(&self) -> IndexRange {
        IndexRange::new(0, 0, self.grid[0] as usize, self.grid[1] as usize)
    }

    /// Collapse the pixel index band `band` into `out`, a plane of `u32`
    /// words whose rows are `stride` pixels apart and whose first element is
    /// the band's first index.
    ///
    /// The band is an [`IndexRange`] and not a coordinate: *where* a frame's
    /// samples are taken is the frame's business (pixel centers), and what a
    /// caller chooses is *which* samples. The grid's program is collapsed
    /// over `band ∩ grid_range()`; the rest of the band is painted with the
    /// border word.
    ///
    /// # Panics
    ///
    /// Panics if the band starts at a column other than 0, its width is zero,
    /// `stride` is less than it, or `out` cannot hold the band.
    pub fn collapse_rows(&self, band: IndexRange, out: &mut [u32], stride: usize) {
        assert_eq!(
            band.x0(),
            0,
            "CellGridPackedFrame::collapse_rows: a frame band starts at column 0"
        );
        let claim = self.grid_range().intersect(&band);
        if !claim.is_empty() {
            self.frame.collapse_subrect(
                PlaneRegion::rows(claim.width(), claim.y0(), claim.rows()),
                out,
                stride,
            );
        }
        paint_outside(out, stride, band, claim, self.border);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::color::Rgba8;
    use crate::render::packed::packed_kernel;
    use crate::render::Pixel;
    use pixelflow_core::{CellGridFrame, CellGridProgram};
    use pixelflow_ir::arena::BufferIdentity;
    use pixelflow_ir::{ExprArena, ExprId};

    /// The 2x1 grid's shape and metric: 4x4-point cells over a 2-tile atlas
    /// of 4x4-content tiles with 1-texel aprons (12x6 texels), in a 12x6
    /// frame.
    const TINY_SHAPE: CellGridShape = CellGridShape {
        cols: 2,
        rows: 1,
        atlas_width: 12,
        atlas_height: 6,
        frame_w: 12,
        frame_h: 6,
    };
    const TINY_METRICS: CellGridMetrics = CellGridMetrics {
        cell_w: 4.0,
        cell_h: 4.0,
        density: 1.0,
        tile_w: 4,
        tile_h: 4,
        scale: 1.0,
    };

    /// Two 4x4-content tiles with 1-texel aprons: tile 0 solid coverage 1,
    /// tile 1 the coverage `fill` field.
    fn two_tile_atlas(fill: f32) -> Vec<f32> {
        let (aw, slot) = (12usize, 6usize);
        let mut atlas = vec![0.0f32; aw * 6];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0;
                atlas[(1 + r) * aw + slot + 1 + c] = fill;
            }
        }
        atlas
    }

    /// A 2×1 grid over a 2-tile atlas: tile A solid coverage 1, tile B a
    /// half-coverage field. Colors chosen so every channel distinguishes
    /// fg, bg, and the default background.
    fn tiny_scene() -> (CellGridProgram, Arc<Vec<f32>>, Arc<Vec<f32>>) {
        let program = CellGridProgram::compile(TINY_SHAPE, [0.9, 0.8, 0.7, 0.6]);
        // Cell 0 → tile 0, red-on-black; cell 1 → tile 1, white-on-blue.
        let cells = vec![
            1.0, 1.0, /* uv */ 1.0, 0.0, 0.0, 1.0, /* fg */ 0.0, 0.0, 0.0,
            1.0, /* bg */
            7.0, 1.0, /* uv */ 1.0, 1.0, 1.0, 1.0, /* fg */ 0.0, 0.0, 1.0,
            1.0, /* bg */
        ];
        (program, Arc::new(cells), Arc::new(two_tile_atlas(0.5)))
    }

    /// The whole `w x h` frame as one index band.
    fn whole(w: usize, h: usize) -> IndexRange {
        IndexRange::new(0, 0, w, h)
    }

    /// Collapse one channel over pixel centers into a dense w×h plane —
    /// the four-channel oracle the pack is checked against.
    fn plane(frame: &CellGridFrame, channel: usize, w: usize, h: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; h * w];
        frame.collapse_channel_rows(channel, whole(w, h), &mut dense, w);
        dense
    }

    fn reachable_nodes(arena: &ExprArena, root: ExprId) -> usize {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            count += 1;
            stack.extend(arena.children(id));
        }
        count
    }

    /// Research harness: dump the packed kernel's machine code and hot-loop
    /// it in one process, so a sampling profiler's addresses correlate to
    /// the dumped bytes exactly. `PIXELFLOW_CODE_DUMP=<path>` receives the
    /// raw bytes; the base pointer prints to stdout.
    #[test]
    #[ignore = "manual profiling harness"]
    fn profile_dump_packed_kernel() {
        let shape = CellGridShape {
            cols: 213,
            rows: 66,
            atlas_width: 64,
            atlas_height: 32,
            frame_w: 2560,
            frame_h: 1584,
        };
        let metrics = CellGridMetrics {
            cell_w: 12.0,
            cell_h: 24.0,
            density: 1.0,
            tile_w: 12,
            tile_h: 24,
            scale: 1.0,
        };
        let program = CellGridPackedManifold::compile(shape, [0.1, 0.1, 0.1, 1.0], [0, 8, 16, 24]);
        let code = program.code_bytes();
        println!(
            "CODE_BASE=0x{:x} CODE_LEN={}",
            code.as_ptr() as usize,
            code.len()
        );
        if let Ok(path) = std::env::var("PIXELFLOW_CODE_DUMP") {
            std::fs::write(&path, code).expect("dump write failed");
        }
        let mut atlas = vec![0.0f32; shape.atlas_len()];
        for (i, t) in atlas.iter_mut().enumerate() {
            *t = ((i * 7) % 11) as f32 / 10.0;
        }
        let mut cells = vec![0.0f32; shape.cells_len()];
        for (i, c) in cells.iter_mut().enumerate() {
            *c = ((i * 13) % 17) as f32 / 16.0;
        }
        let frame = program.frame(&program.params(&metrics), Arc::new(cells), Arc::new(atlas));
        let (w, h) = (2560usize, 1584usize);
        let mut out = vec![0u32; w * h];
        for _ in 0..150 {
            frame.collapse_rows(whole(w, h), &mut out, w);
            std::hint::black_box(&out);
        }
    }

    /// `Rgba8`'s byte lanes: little-endian `[r, g, b, a]`.
    const RGBA_SHIFTS: [u32; 4] = [0, 8, 16, 24];
    /// `Bgra8`'s byte lanes: little-endian `[b, g, r, a]`.
    const BGRA_SHIFTS: [u32; 4] = [16, 8, 0, 24];

    /// The border word a partially-covering scene paints is the production
    /// pixel conversion, not a restatement of it: `pack_scalar` — the one
    /// definition `CellGridPackedManifold::compile` uses — must agree with
    /// `Pixel::from_rgba` composed with `Pixel::to_u32` for the format whose
    /// shifts it was given.
    #[test]
    fn border_word_is_the_pixel_conversion() {
        for rgba in [
            [0.0, 0.0, 0.0, 1.0],
            [0.9, 0.8, 0.7, 0.6],
            [1.0, 1.0, 1.0, 1.0],
            [0.1, 0.25, 0.5, 0.75],
        ] {
            let [r, g, b, a] = rgba;
            assert_eq!(
                pack_scalar(rgba, RGBA_SHIFTS),
                Rgba8::from_rgba(r, g, b, a).to_u32(),
                "border word for {rgba:?}"
            );
        }
    }

    /// Collapse packed pixels over pixel centers into a dense w×h plane.
    fn packed_plane(frame: &CellGridPackedFrame, w: usize, h: usize) -> Vec<u32> {
        let mut dense = vec![0u32; h * w];
        frame.collapse_rows(whole(w, h), &mut dense, w);
        dense
    }

    #[test]
    fn packed_bake_is_bit_exact_with_channel_bakes_under_both_byte_orders() {
        // 12×6 over tiny_scene's 8×4 grid: the right and bottom margins are
        // off-grid, so default_bg flows through the pack too. THE invariant:
        // for every pixel, the packed collapse equals the four channel ones
        // composed with the scalar pack — exact u32 equality, no epsilon.
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(&program.params(&TINY_METRICS), cells.clone(), atlas.clone());
        let (w, h) = (12, 6);
        let planes: [Vec<f32>; 4] = core::array::from_fn(|c| plane(&frame, c, w, h));
        for shifts in [RGBA_SHIFTS, BGRA_SHIFTS] {
            let packed =
                CellGridPackedManifold::compile(*program.shape(), [0.9, 0.8, 0.7, 0.6], shifts);
            let got = packed_plane(
                &packed.frame(&packed.params(&TINY_METRICS), cells.clone(), atlas.clone()),
                w,
                h,
            );
            for i in 0..w * h {
                let expected = pack_scalar(
                    [planes[0][i], planes[1][i], planes[2][i], planes[3][i]],
                    shifts,
                );
                assert_eq!(
                    got[i], expected,
                    "pixel {i} under shifts {shifts:?}: {:#010x} != {:#010x}",
                    got[i], expected
                );
            }
        }
    }

    #[test]
    fn packed_bake_parity_holds_on_an_asymmetric_grid() {
        // The 1×2 grid from the row-offset test (cols ≠ rows, per-row
        // colors), sampled 6×10 so the right and bottom margins exercise a
        // non-gray default background through every channel's shift.
        let shape = CellGridShape {
            cols: 1,
            rows: 2,
            frame_w: 6,
            frame_h: 10,
            ..TINY_SHAPE
        };
        let default_bg = [0.25, 0.5, 0.75, 1.0];
        let cells = Arc::new(vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ]);
        let atlas = Arc::new(two_tile_atlas(0.5));
        let (w, h) = (6, 10);
        let program = CellGridProgram::compile(shape, default_bg);
        let frame = program.frame(&program.params(&TINY_METRICS), cells.clone(), atlas.clone());
        let planes: [Vec<f32>; 4] = core::array::from_fn(|c| plane(&frame, c, w, h));
        let packed = CellGridPackedManifold::compile(shape, default_bg, RGBA_SHIFTS);
        let got = packed_plane(
            &packed.frame(&packed.params(&TINY_METRICS), cells, atlas),
            w,
            h,
        );
        for i in 0..w * h {
            let expected = pack_scalar(
                [planes[0][i], planes[1][i], planes[2][i], planes[3][i]],
                RGBA_SHIFTS,
            );
            assert_eq!(
                got[i], expected,
                "pixel {i}: {:#010x} != {:#010x}",
                got[i], expected
            );
        }
    }

    /// **A resize that keeps the shape is a parameter write.** One packed
    /// program, re-parameterized to a second metric, draws exactly what a
    /// separately constructed program built for that metric draws — and the
    /// two programs are the same compiled code.
    #[test]
    fn a_new_metric_reuses_the_compiled_program() {
        let shape = CellGridShape {
            cols: 2,
            rows: 2,
            frame_w: 20,
            frame_h: 20,
            ..TINY_SHAPE
        };
        let small = TINY_METRICS;
        let large = CellGridMetrics {
            cell_w: 8.0,
            cell_h: 8.0,
            ..TINY_METRICS
        };
        let cells = Arc::new(
            (0..4)
                .flat_map(|i| {
                    let f = i as f32;
                    [1.0, 1.0, f * 0.25, 0.5, 0.75, 1.0, 0.1, 0.2, 0.3, 1.0]
                })
                .collect::<Vec<f32>>(),
        );
        let atlas = Arc::new(two_tile_atlas(0.5));
        let bg = [0.9, 0.8, 0.7, 0.6];

        let a = CellGridPackedManifold::compile(shape, bg, RGBA_SHIFTS);
        let b = CellGridPackedManifold::compile(shape, bg, RGBA_SHIFTS);
        assert!(
            core::ptr::eq(a.code_bytes().as_ptr(), b.code_bytes().as_ptr()),
            "one shape, one compiled kernel — the JIT cache is keyed on the \
             extents and the leaf slots, and a metric is neither"
        );

        let at_small = packed_plane(
            &a.frame(&a.params(&small), cells.clone(), atlas.clone()),
            20,
            20,
        );
        let from_write = packed_plane(
            &a.frame(&a.params(&large), cells.clone(), atlas.clone()),
            20,
            20,
        );
        let fresh = packed_plane(&b.frame(&b.params(&large), cells, atlas), 20, 20);
        assert_eq!(
            from_write, fresh,
            "the metric written into the block is not the metric the kernel read"
        );
        assert_ne!(
            at_small, from_write,
            "the two metrics must differ, or the equality above proves nothing"
        );
        // And the border moved with the metric: 2 cells of 4 points cover 8
        // pixels of the 20-wide frame, 2 of 8 cover 16.
        assert_eq!(a.params(&small).grid_range(), IndexRange::new(0, 0, 8, 8));
        assert_eq!(a.params(&large).grid_range(), IndexRange::new(0, 0, 16, 16));
    }

    /// The packed kernel splices four channel fragments (each with its own
    /// cell reads and atlas taps), and identity-merge must still land them
    /// all in exactly two slots — one per distinct buffer.
    #[test]
    fn packed_kernel_reads_land_in_exactly_two_slots() {
        let shape = CellGridShape {
            frame_w: 8,
            frame_h: 4,
            ..TINY_SHAPE
        };
        let CellGridKernels {
            channels, buffers, ..
        } = shape.channel_kernels();
        let kernel = packed_kernel(&Rgba::from(&channels), RGBA_SHIFTS);
        let slots = kernel.parts().0.buffers();
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.cells).count(),
            1,
            "all four channels' cell reads, one buffer, one slot"
        );
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.atlas).count(),
            1,
            "all four channels' atlas taps, one slot"
        );
        assert_eq!(slots.len(), 2, "exactly {{Cells, Atlas}}, nothing else");
    }

    /// Pins the packed program's size, in the spirit of
    /// `reads_of_one_buffer_share_a_slot_but_not_nodes`: 667 = 4 × 157 (each
    /// `or` re-splices a whole channel fragment) + 4 × 9 (each channel's pack
    /// chain: const 255, mul, const 0, max, const 255, min, trunc, const
    /// shift, shl) + 3 ors. Node sharing is the e-graph's call once it can
    /// see `Gather`; this number falling is the sign that landed.
    #[test]
    fn packed_kernel_node_count_is_the_channel_kernels_plus_the_pack() {
        let shape = CellGridShape {
            frame_w: 8,
            frame_h: 4,
            ..TINY_SHAPE
        };
        let kernel = packed_kernel(&Rgba::from(&shape.channel_kernels().channels), RGBA_SHIFTS);
        let (arena, root) = kernel.parts();
        assert_eq!(
            reachable_nodes(arena, root),
            667,
            "composed packed node count (4 channels of 157, plus the pack)"
        );
    }

    #[test]
    fn packed_row_offset_band_matches_the_full_bake() {
        // Mirrors `baking_a_row_offset_band_samples_the_correct_absolute_row`
        // for the packed path: a collapse starting at y0 = 4 must reproduce
        // rows 4..8 of the full one exactly (u32 equality — these are packed
        // pixels, not floats).
        let shape = CellGridShape {
            cols: 1,
            rows: 2,
            frame_w: 4,
            frame_h: 8,
            ..TINY_SHAPE
        };
        let program = CellGridPackedManifold::compile(shape, [0.0; 4], RGBA_SHIFTS);
        let cells = Arc::new(vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ]);
        let frame = program.frame(
            &program.params(&TINY_METRICS),
            cells,
            Arc::new(two_tile_atlas(0.5)),
        );

        let full = packed_plane(&frame, 4, 8);

        let mut offset = vec![0u32; 4 * 4];
        frame.collapse_rows(IndexRange::new(0, 4, 4, 4), &mut offset, 4);
        assert_eq!(
            offset,
            full[4 * 4..],
            "the offset collapse disagrees with the full one"
        );
    }

    #[test]
    #[should_panic(expected = "permutation of the byte lanes")]
    fn overlapping_pack_shifts_are_refused() {
        let shape = CellGridShape {
            cols: 1,
            rows: 1,
            frame_w: 4,
            frame_h: 4,
            ..TINY_SHAPE
        };
        // R and B both at bit 0: the OR would blend two channels' bytes.
        let _refused = CellGridPackedManifold::compile(shape, [0.0; 4], [0, 8, 0, 24]);
    }

    /// **The metric cannot arrive by default.** The packed program declares
    /// arguments, and `PackedManifold::bind` refuses a program that has any —
    /// so the only way to a frame is `bind_with` and a block, which is what
    /// `CellGridPackedManifold::frame` requires a `CellGridPackedParams` for.
    /// Without that refusal, a caller who forgot the metric would get a grid
    /// of one-point cells: a picture, and a plausible one.
    #[test]
    #[should_panic(expected = "argument")]
    fn a_frame_cannot_be_bound_without_its_metric() {
        let shape = CellGridShape {
            frame_w: 8,
            frame_h: 4,
            ..TINY_SHAPE
        };
        let CellGridKernels { channels, .. } = shape.channel_kernels();
        let packed = PackedManifold::compile(&Rgba::from(&channels), RGBA_SHIFTS, [8, 4]);
        assert_eq!(packed.uniforms().len(), 8, "the metric is eight slots");
        let _refused = packed.bind(&[]);
    }

    /// Production saturation telemetry, stage 1 of 2
    /// (docs/results/2026-09-01-production-saturation-telemetry.md): write
    /// the packed cell-grid arena core-term actually compiles, for the
    /// geometries it actually compiles it at, so the measurer in
    /// `pixelflow-search/src/runtime.rs` (`production_telemetry`) can replay
    /// `optimize_runtime_arena`'s calls on it and keep the `SaturationResult`
    /// production discards. Lives here because `packed_kernel` is private to
    /// this crate, and stays private itself.
    ///
    /// The shape is core-term's, not a fixture's: `TerminalApp::build_scene`
    /// builds a `CellGridShape` from the snapshot's grid dimensions and the
    /// device-pixel frame those imply (config defaults `cell_width_px: 10`,
    /// `cell_height_px: 16`, `config.rs`, times the display density), plus the
    /// atlas extents of `GlyphAtlas::new(cell_height, density,
    /// ATLAS_CAPACITY = 128)`, whose arithmetic (`atlas.rs:90-98`, PAD = 1) is
    /// restated here and cross-checked against a real `GlyphAtlas` by the
    /// glyph dumper. Shifts are `PlatformColorCube::PACKED_SHIFTS` on macOS
    /// (`RgbaColorCube`, `[0, 8, 16, 24]`). Density 1.0 is the startup
    /// compile, 2.0 the Retina recompile after `WindowCreated`; 120x40 is one
    /// resize.
    ///
    /// **The metric no longer reaches the arena at all** — cell size, sample
    /// density and tile size are uniform slots, and the default background is
    /// a border fill — so the density below survives only in the atlas and
    /// frame *extents* it implies, which are still what the program is
    /// compiled against.
    #[test]
    #[ignore = "telemetry dumper: PIXELFLOW_TELEMETRY_DIR=<dir> cargo test -p pixelflow-graphics --release -- --ignored dump_production_cell_grid_arenas"]
    fn dump_production_cell_grid_arenas() {
        let dir = std::path::PathBuf::from(
            std::env::var("PIXELFLOW_TELEMETRY_DIR").expect("PIXELFLOW_TELEMETRY_DIR must be set"),
        );
        std::fs::create_dir_all(&dir).expect("create dump dir");
        const CELL_W_PT: f32 = 10.0;
        const CELL_H_PT: f32 = 16.0;
        const ATLAS_SLOTS_PER_ROW: u32 = 12; // ceil(sqrt(128))
        const ATLAS_SLOT_ROWS: u32 = 11; // ceil(128 / 12)
        const ATLAS_PAD: u32 = 1;
        for (label, cols, rows, density) in [
            ("80x24_d1", 80u32, 24u32, 1.0f32),
            ("80x24_d2", 80, 24, 2.0),
            ("120x40_d2", 120, 40, 2.0),
        ] {
            let tile_px = (CELL_H_PT * density).round().max(1.0) as u32;
            let slot_px = tile_px + 2 * ATLAS_PAD;
            let cell_w = CELL_W_PT * density;
            let cell_h = CELL_H_PT * density;
            let shape = CellGridShape {
                cols,
                rows,
                atlas_width: ATLAS_SLOTS_PER_ROW * slot_px,
                atlas_height: ATLAS_SLOT_ROWS * slot_px,
                frame_w: (cols as f32 * cell_w).round() as u32,
                frame_h: (rows as f32 * cell_h).round() as u32,
            };
            let kernel = packed_kernel(&Rgba::from(&shape.channel_kernels().channels), RGBA_SHIFTS);
            let (arena, root) = kernel.parts();
            let name = format!("cellgrid:{label}");
            let path = dir.join(format!("cellgrid_{label}.arena"));
            dump_arena(arena, root, &name, &path);
            println!(
                "{name}: {} reachable nodes -> {}",
                reachable_nodes(arena, root),
                path.display()
            );
        }
    }

    /// Text dump of the subgraph reachable from `root`: nodes in ascending
    /// original id order (children precede parents), ids remapped dense,
    /// constants as bit patterns, buffer identities as dense ordinals. The
    /// loader in `pixelflow-search/src/runtime.rs` is the inverse. Duplicated
    /// verbatim in `pixelflow-graphics/tests/production_glyph_arena_dump.rs`
    /// rather than shared, because a unit test and an integration test share no
    /// code, and `pixelflow-ir` must not grow a test-only serializer.
    fn dump_arena(
        arena: &ExprArena,
        root: pixelflow_ir::ExprId,
        name: &str,
        path: &std::path::Path,
    ) {
        use core::fmt::Write as _;
        use pixelflow_ir::arena::ExprNode;
        let len = arena.nodes_raw().len();
        let mut reachable = vec![false; len];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut reachable[id.0 as usize], true) {
                continue;
            }
            stack.extend(arena.children(id));
        }
        let mut out = std::string::String::new();
        writeln!(out, "# pixelflow arena dump v1").expect("fmt");
        writeln!(out, "name {name}").expect("fmt");
        let mut idents: Vec<BufferIdentity> = Vec::new();
        for decl in arena.buffers() {
            let ord = match idents.iter().position(|i| *i == decl.id) {
                Some(p) => p,
                None => {
                    idents.push(decl.id);
                    idents.len() - 1
                }
            };
            writeln!(out, "buf {ord} {} {}", decl.width, decl.height).expect("fmt");
        }
        let mut dense: Vec<u32> = vec![u32::MAX; len];
        let mut next = 0u32;
        let d = |dense: &[u32], id: pixelflow_ir::ExprId| -> u32 {
            let v = dense[id.0 as usize];
            assert_ne!(v, u32::MAX, "child dumped before parent");
            v
        };
        for idx in 0..len {
            if !reachable[idx] {
                continue;
            }
            let id = pixelflow_ir::ExprId(idx as u32);
            match arena.node(id) {
                ExprNode::Var(i) => writeln!(out, "V {i}"),
                ExprNode::Const(v) => writeln!(out, "C {}", v.to_bits()),
                ExprNode::Buffer(b) => writeln!(out, "B {}", b.0),
                ExprNode::Uniform(u) => writeln!(out, "Un {}", u.0),
                ExprNode::Unary(k, a) => writeln!(out, "U {k:?} {}", d(&dense, *a)),
                ExprNode::Binary(k, a, b) => {
                    writeln!(out, "Bi {k:?} {} {}", d(&dense, *a), d(&dense, *b))
                }
                ExprNode::Ternary(k, a, b, c) => writeln!(
                    out,
                    "T {k:?} {} {} {}",
                    d(&dense, *a),
                    d(&dense, *b),
                    d(&dense, *c)
                ),
                other @ (ExprNode::Param(_) | ExprNode::Nary(..)) => {
                    panic!("{name}: production arena contains {other:?}, which optimize_runtime_arena bails on")
                }
            }
            .expect("fmt");
            dense[idx] = next;
            next += 1;
        }
        writeln!(out, "root {}", d(&dense, root)).expect("fmt");
        std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}
