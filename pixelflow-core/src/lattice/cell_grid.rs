//! # CellGrid: a JIT-compiled tile-grid sampler
//!
//! A grid of uniformly sized cells, each referencing a tile in a coverage
//! atlas and carrying two RGBA colors; every pixel blends the cell's colors
//! by the atlas coverage under the tile's bilinear filter. This is the
//! tilemap primitive: sprite grids, tile-based scenes, and text grids are
//! all instances (a terminal is a text grid whose atlas holds glyphs — that
//! mapping lives with the caller; this module only ever sees floats).
//!
//! The whole scene is ONE compiled program per color channel — cell-index
//! arithmetic, per-cell data gathers, a 4-tap atlas read, and the color
//! blend — over two bound buffers (the per-cell data and the atlas). The
//! per-frame update rewrites the cell buffer; a geometry change (grid
//! dimensions, cell size, atlas extents) recompiles the program — a few
//! milliseconds through the JIT, dominated by the e-graph optimizer that
//! the compile path always runs, and paid only on resize. This replaces the only
//! alternative the combinator layer offered — a per-frame tree of boxed
//! `Select`s sized by the runtime grid — with a program whose size is
//! independent of the grid's.
//!
//! ## Cell data layout
//!
//! Row-major, [`CELL_STRIDE`] `f32`s per cell:
//!
//! ```text
//! [u0, v0, fg_r, fg_g, fg_b, fg_a, bg_r, bg_g, bg_b, bg_a]
//! ```
//!
//! `(u0, v0)` is the cell's tile origin in atlas texels (the caller points
//! it at the first *content* texel — any padding inset is the caller's).
//! Colors are linear [0, 1] floats.
//!
//! ## Coordinate convention
//!
//! Queries are in point space, like [`CachedGlyph`-style samplers]: cell
//! `(c, r)` spans `[c·cell_w, (c+1)·cell_w) × [r·cell_h, (r+1)·cell_h)`
//! points; within a cell, the atlas is addressed at
//! `(u0 + lx·density − ½, v0 + ly·density − ½)` texels, where `(lx, ly)`
//! is the point-space offset into the cell — the same texel-center
//! embedding the glyph cache uses, so a density-matched sample grid reads
//! tiles losslessly. Queries outside the grid return the default
//! background color. The bilinear taps reach one texel beyond the sampled
//! region on each side, so tiles need a one-texel apron in the atlas
//! (zero coverage for "nothing outside the tile").

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::manifold::{BoundManifold, Manifold, PlaneRegion};
use super::{BilinearSampler, DiscreteManifold};
use pixelflow_ir::Kernel;
use pixelflow_ir::arena::BufferIdentity;

/// `f32`s per cell in the cell-data buffer.
pub const CELL_STRIDE: usize = 10;

/// The compiled-against shape of a [`CellGridProgram`]: grid dimensions,
/// cell extents, atlas extents in texels, and the atlas sample density.
/// Changing any field means recompiling.
///
/// ## Coordinate units
///
/// Cell extents and `density` are in the CALLER'S sampling space — the
/// coordinates the compiled kernels will be evaluated at. The program
/// applies no unit conversion of its own, so whoever drives the bake
/// decides what a coordinate means and must express every field in that
/// one space. In particular, `pixelflow-runtime` renders cell-grid scenes
/// on the frame buffer's DEVICE-PIXEL grid without any DPI contramap: a
/// scene destined for the runtime bakes its display scale in here
/// (extents × scale, density in texels per device pixel), as core-term
/// does.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CellGridGeometry {
    /// Grid width in cells.
    pub cols: u32,
    /// Grid height in cells.
    pub rows: u32,
    /// Cell width, in sampling-space coordinate units (see the type docs).
    pub cell_w: f32,
    /// Cell height, in sampling-space coordinate units.
    pub cell_h: f32,
    /// Atlas texels per sampling-space coordinate unit.
    pub density: f32,
    /// Atlas width in texels.
    pub atlas_width: u32,
    /// Atlas height in texels.
    pub atlas_height: u32,
    /// Tile content width in texels. Samples past it clamp into the
    /// zero apron between slots instead of reaching a neighbor's content
    /// (a cell wider than its tile shows background there, exactly as the
    /// per-glyph sampler's out-of-extent zero did).
    pub tile_w: u32,
    /// Tile content height in texels (see `tile_w`).
    pub tile_h: u32,
    /// Width in pixels of the frame this program renders. Every
    /// [`PlaneRegion`] baked through it lies within `frame_w × frame_h`;
    /// pixels past the grid's own edge (a frame is rarely a whole number of
    /// cells) show the default background. Part of the geometry because the
    /// compiled kernels are specialized to the frame's lattice: a resize is
    /// a recompile.
    pub frame_w: u32,
    /// Height in pixels of the frame this program renders (see `frame_w`).
    pub frame_h: u32,
}

impl CellGridGeometry {
    /// Cell-data buffer length this geometry implies.
    ///
    /// # Panics
    ///
    /// Panics if the product overflows `usize`. A wrapped length would be
    /// SMALL, so `frame` would accept a correspondingly small buffer while
    /// the compiled kernel still declared the true row width — and the
    /// gathers would read billions of elements past the end through an
    /// entirely safe API.
    #[must_use]
    pub fn cells_len(&self) -> usize {
        (self.cols as usize)
            .checked_mul(self.rows as usize)
            .and_then(|cells| cells.checked_mul(CELL_STRIDE))
            .expect("CellGridGeometry::cells_len overflows usize")
    }

    /// Atlas buffer length this geometry implies.
    ///
    /// # Panics
    ///
    /// Panics if the product overflows `usize` — see [`Self::cells_len`].
    #[must_use]
    pub fn atlas_len(&self) -> usize {
        (self.atlas_width as usize)
            .checked_mul(self.atlas_height as usize)
            .expect("CellGridGeometry::atlas_len overflows usize")
    }

    /// Width of the cell-data buffer as the kernel declares it, in `f32`s.
    ///
    /// # Panics
    ///
    /// Panics if it overflows `u32`. `channel_kernel` computed this as a
    /// wrapping `u32` product, so a wrapped value silently declared a buffer
    /// of a completely different shape than the one bound to it.
    #[must_use]
    pub(crate) fn cells_row_width(&self) -> u32 {
        self.cols
            .checked_mul(CELL_STRIDE as u32)
            .expect("CellGridGeometry: cell-buffer row width overflows u32")
    }

    /// The four channel kernels this geometry denotes — everything from pixel
    /// coordinate to blended channel value — over freshly minted identities
    /// for the two buffers they read.
    ///
    /// This is the whole of the cell grid as a *program*: what a consumer does
    /// with the four channels — bake them into planes ([`CellGridProgram`]) or
    /// compose them into one packed-pixel kernel (`pixelflow-graphics`, which
    /// is where byte lanes and pixel formats live) — is the consumer's.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate geometry: zero cells, non-positive cell extent
    /// or density, or an empty atlas.
    #[must_use]
    pub fn channel_kernels(&self, default_bg: [f32; 4]) -> CellGridKernels {
        assert_compilable(self);
        let buffers = CellGridBuffers::mint();
        CellGridKernels {
            channels: core::array::from_fn(|c| channel_kernel(self, buffers, c, default_bg[c])),
            buffers,
        }
    }
}

/// One channel's program: everything from pixel coordinate to blended channel
/// value, composed from the two samplers rather than assembled node by node.
///
/// Keyed on extents, not data. A sampler's fragment carries only its buffer
/// declaration, so the whole program composes from geometry alone and the
/// per-frame buffers bind later.
fn channel_kernel(
    geom: &CellGridGeometry,
    bufs: CellGridBuffers,
    channel: usize,
    default_bg: f32,
) -> Kernel {
    let k = Kernel::constant;
    let (x, y) = (Kernel::x(), Kernel::y());

    // Which cell, and where inside it.
    let col = x.mul(&k(1.0 / geom.cell_w)).floor();
    let row = y.mul(&k(1.0 / geom.cell_h)).floor();
    let lx = x.sub(&col.mul(&k(geom.cell_w)));
    let ly = y.sub(&row.mul(&k(geom.cell_h)));

    // Per-cell data: gathers clamp to the buffer edge, so out-of-grid queries
    // read the border cell — harmless, since the final select replaces them
    // with the default background anyway.
    let cells = DiscreteManifold::kernel_for(bufs.cells, geom.cells_row_width(), geom.rows);
    let cx = col.mul(&k(CELL_STRIDE as f32));
    let field = |offset: usize| {
        let idx = if offset == 0 {
            cx.clone()
        } else {
            cx.add(&k(offset as f32))
        };
        cells.at(&idx, &row)
    };
    let u0 = field(0);
    let v0 = field(1);
    let fg = field(2 + channel);
    let bg = field(6 + channel);

    // Atlas coordinates: point offset into the cell, scaled to texels, shifted
    // to texel centres. Clamping to half a texel past the tile content keeps
    // the taps on this slot's apron and (at most) the neighbor's — both zero —
    // so a cell larger than its tile fades to background rather than showing
    // fragments of the adjacent tile.
    let au = u0.add(&lx.mul(&k(geom.density)).min(&k(geom.tile_w as f32 + 0.5)));
    let av = v0.add(&ly.mul(&k(geom.density)).min(&k(geom.tile_h as f32 + 0.5)));
    let atlas = BilinearSampler::kernel_for(bufs.atlas, geom.atlas_width, geom.atlas_height);
    let cov = atlas.at(&au.sub(&k(0.5)), &av.sub(&k(0.5)));

    // blended = bg + cov·(fg − bg), and the default background off-grid.
    let blended = bg.add(&cov.mul(&fg.sub(&bg)));
    let in_grid = x
        .ge(&k(0.0))
        .and(&x.lt(&k(geom.cols as f32 * geom.cell_w)))
        .and(&y.ge(&k(0.0)))
        .and(&y.lt(&k(geom.rows as f32 * geom.cell_h)));
    in_grid.select(&blended, &k(default_bg))
}

/// The two blocks of memory a cell-grid program reads, identified so that
/// composing many reads of one buffer still merges to one slot, and so binding
/// can say which slot is which without inferring it from extents.
#[derive(Clone, Copy, Debug)]
pub struct CellGridBuffers {
    /// The per-cell data buffer: [`CELL_STRIDE`] `f32`s per cell, row-major.
    pub cells: BufferIdentity,
    /// The coverage atlas, row-major texels.
    pub atlas: BufferIdentity,
}

impl CellGridBuffers {
    /// Two identities distinct from every other in this process.
    #[must_use]
    pub fn mint() -> Self {
        Self {
            cells: BufferIdentity::mint(),
            atlas: BufferIdentity::mint(),
        }
    }
}

/// A geometry's four channel kernels and the buffers they read
/// ([`CellGridGeometry::channel_kernels`]).
pub struct CellGridKernels {
    /// One kernel per channel, in `(r, g, b, a)` order.
    pub channels: [Kernel; 4],
    /// The identities those kernels declared, for binding data to them.
    pub buffers: CellGridBuffers,
}

/// The geometry preconditions every cell-grid compile shares.
///
/// # Panics
///
/// Panics on a degenerate geometry: zero cells, non-positive cell extent or
/// density, or an empty atlas. Buffers too large to index exactly in `f32`
/// are refused by [`Manifold::compile`], which sees every declared
/// buffer's real extents rather than this geometry's idea of them.
fn assert_compilable(geom: &CellGridGeometry) {
    assert!(
        geom.cols > 0 && geom.rows > 0 && geom.atlas_width > 0 && geom.atlas_height > 0,
        "cell-grid compile: degenerate geometry {geom:?}"
    );
    assert!(
        geom.cell_w > 0.0 && geom.cell_h > 0.0 && geom.density > 0.0,
        "cell-grid compile: non-positive cell extent or density {geom:?}"
    );
    // Both products must be computable before anything downstream trusts
    // them: cells_len/cells_row_width panic on overflow rather than wrapping,
    // and a wrapped length paired with an unwrapped row width is how a safe
    // API ends up gathering past the end of the bound buffer.
    // (Both accessors panic on overflow; calling them here is the check.)
    assert!(
        geom.cells_len() > 0 && geom.cells_row_width() > 0,
        "cell-grid compile: empty cell buffer for {geom:?}"
    );
}

/// The four compiled channel kernels for one grid/atlas geometry, each baking
/// into its own `f32` plane.
///
/// The production frame path is the *packed* program a layer up
/// (`pixelflow-graphics`), which folds these four channels into one kernel
/// whose root is a finished pixel; this one is the per-channel form — the
/// parity oracle that pack is checked against, and the way to read one
/// channel as a field.
///
/// Compile once per geometry ([`CellGridProgram::compile`]); stamp per-frame
/// data into it with [`CellGridProgram::frame`]. A resize is: build the new
/// geometry, compile, done — the program's size and compile time are
/// independent of the grid's dimensions.
pub struct CellGridProgram {
    geom: CellGridGeometry,
    buffers: CellGridBuffers,
    channels: [Manifold; 4],
}

impl CellGridProgram {
    /// JIT-compile the four channel programs for `geom`, with `default_bg`
    /// (linear RGBA) painted outside the grid.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate geometry (zero cells, non-positive cell extent
    /// or density, empty atlas), when this build's `Field` width does not
    /// match the JIT's emitted width, or if compilation fails.
    #[must_use]
    pub fn compile(geom: CellGridGeometry, default_bg: [f32; 4]) -> Self {
        let CellGridKernels { channels, buffers } = geom.channel_kernels(default_bg);
        let extent = [geom.frame_w, geom.frame_h];
        Self {
            geom,
            buffers,
            channels: channels.map(|kernel| Manifold::compile(&kernel, extent)),
        }
    }

    /// The geometry this program was compiled for.
    #[must_use]
    pub fn geometry(&self) -> &CellGridGeometry {
        &self.geom
    }

    /// Bind one frame's data. `cells` is [`CELL_STRIDE`] `f32`s per cell,
    /// row-major; `atlas` is the coverage atlas, row-major texels. Both are
    /// `Arc`s so a frame in flight keeps its data alive while the caller
    /// prepares the next one.
    ///
    /// # Panics
    ///
    /// Panics if either buffer's length does not match the geometry.
    #[must_use]
    pub fn frame(&self, cells: Arc<Vec<f32>>, atlas: Arc<Vec<f32>>) -> CellGridFrame {
        let bound = [(self.buffers.cells, cells), (self.buffers.atlas, atlas)];
        CellGridFrame {
            channels: core::array::from_fn(|c| self.channels[c].bind(&bound)),
        }
    }
}

/// One frame of a cell grid: the compiled channels plus the data they read.
/// Cheap to clone.
#[derive(Clone)]
pub struct CellGridFrame {
    channels: [BoundManifold; 4],
}

impl CellGridFrame {
    /// Collapse one color channel (0 = R, 1 = G, 2 = B, 3 = A) over the pixel
    /// rows `y0 .. y0 + rows` at pixel-center coordinates, into a plane whose
    /// rows are `stride` samples apart — see [`BoundManifold::collapse_rows`].
    ///
    /// # Panics
    ///
    /// Panics if `channel >= 4`, the region's width is zero, `stride` is less
    /// than it, or `out` cannot hold the band.
    pub fn collapse_channel_rows(
        &self,
        channel: usize,
        region: PlaneRegion,
        out: &mut [f32],
        stride: usize,
    ) {
        self.channels[channel].collapse_rows(region, out, stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use pixelflow_ir::ExprArena;

    /// A 2×1 grid over a 2-tile atlas: tile A solid coverage 1, tile B a
    /// half-coverage field. Colors chosen so every channel distinguishes
    /// fg, bg, and the default background.
    fn tiny_scene() -> (CellGridProgram, Arc<Vec<f32>>, Arc<Vec<f32>>) {
        // Atlas: two 4×4-content tiles with 1-texel aprons, laid out side by
        // side → 12×6 texels.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 12,
            frame_h: 6,
        };
        let program = CellGridProgram::compile(geom, [0.9, 0.8, 0.7, 0.6]);
        // Cell 0 → tile 0, red-on-black; cell 1 → tile 1, white-on-blue.
        let cells = vec![
            1.0, 1.0, /* uv */ 1.0, 0.0, 0.0, 1.0, /* fg */ 0.0, 0.0, 0.0,
            1.0, /* bg */
            7.0, 1.0, /* uv */ 1.0, 1.0, 1.0, 1.0, /* fg */ 0.0, 0.0, 1.0,
            1.0, /* bg */
        ];
        (program, Arc::new(cells), Arc::new(atlas))
    }

    /// Splicing merges buffer declarations by identity, so the nine reads in
    /// a channel program (four cell fields — `bg` twice — plus the atlas)
    /// occupy exactly two slots: one per distinct buffer.
    ///
    /// Nodes are a different story, and this pins the gap. 146 against the 77
    /// the hand-written arena built, because every *use* of a Kernel value
    /// re-splices its whole fragment: `row` is used by four cell reads, `cx`
    /// by four, `bg` by two. Identity fixed the slots; it cannot fix this.
    /// Node sharing is an inlining decision, and it belongs to the e-graph —
    /// which cannot see this arena at all while `Gather` is unrepresentable
    /// there. This number falling is the sign that landed.
    #[test]
    fn reads_of_one_buffer_share_a_slot_but_not_nodes() {
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let CellGridKernels { channels, buffers } = geom.channel_kernels([0.0; 4]);
        let (arena, root) = channels[0].parts();

        let slots = arena.buffers();
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.cells).count(),
            1,
            "five cell reads, one buffer, one slot"
        );
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.atlas).count(),
            1,
            "one slot for the atlas"
        );
        assert_eq!(slots.len(), 2, "exactly {{Cells, Atlas}}, nothing else");
        assert_eq!(
            reachable_nodes(arena, root),
            146,
            "composed node count (hand-built was 77)"
        );
    }

    fn reachable_nodes(arena: &ExprArena, root: pixelflow_ir::ExprId) -> usize {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            count += 1;
            stack.extend(arena.children(id));
        }
        count
    }

    /// Row-major index into an 8-wide plane.
    fn px(row: usize, col: usize) -> usize {
        row * 8 + col
    }

    /// Bake one channel over pixel centers and unpad to a dense w×h plane.
    fn plane(frame: &CellGridFrame, channel: usize, w: usize, h: usize) -> alloc::vec::Vec<f32> {
        let mut dense = vec![0.0f32; h * w];
        frame.collapse_channel_rows(channel, PlaneRegion::rows(w, 0, h), &mut dense, w);
        dense
    }

    #[test]
    fn interior_texels_blend_fg_over_bg_by_coverage() {
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        let r = plane(&frame, 0, 8, 4);
        let b = plane(&frame, 2, 8, 4);
        // Cell 0 interior: coverage 1 → pure fg (red).
        assert!((r[px(1, 1)] - 1.0).abs() < 1e-5, "cell0 R = {}", r[9]);
        assert!(b[px(1, 1)].abs() < 1e-5, "cell0 B = {}", b[9]);
        // Cell 1 interior: coverage 0.5 → half fg, half bg.
        // R: 0.5·1 + 0.5·0 = 0.5;  B: 0.5·1 + 0.5·1 = 1.0.
        assert!((r[px(1, 5)] - 0.5).abs() < 1e-5, "cell1 R = {}", r[13]);
        assert!((b[px(1, 5)] - 1.0).abs() < 1e-5, "cell1 B = {}", b[13]);
    }

    #[test]
    fn outside_the_grid_is_the_default_background() {
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        // Sample a lattice wider and taller than the 8×4 grid.
        let g = plane(&frame, 1, 12, 6);
        // (10.5, 0.5) is right of the grid; (0.5, 5.5) below it.
        assert!((g[10] - 0.8).abs() < 1e-5, "right of grid G = {}", g[10]);
        assert!(
            (g[5 * 12] - 0.8).abs() < 1e-5,
            "below grid G = {}",
            g[5 * 12]
        );
        // Alpha channel outside: 0.6.
        let a = plane(&frame, 3, 12, 6);
        assert!((a[11] - 0.6).abs() < 1e-5, "outside A = {}", a[11]);
    }

    #[test]
    fn aligned_pixel_centers_read_tiles_losslessly() {
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        let r = plane(&frame, 0, 8, 4);
        // Density-matched pixel centers land exactly on texel centers, so
        // the bilinear read degenerates to lookup: the LAST pixel column of
        // cell 0 still reads pure coverage 1 (its own last content texel),
        // and the FIRST pixel column of cell 1 reads its tile's 0.5.
        assert!((r[px(1, 3)] - 1.0).abs() < 1e-5, "cell0 edge R = {}", r[11]);
        assert!((r[px(1, 4)] - 0.5).abs() < 1e-5, "cell1 edge R = {}", r[12]);
    }

    #[test]
    fn misaligned_samples_fade_through_the_apron_not_into_neighbors() {
        // Density 0.8 over a 5-point cell: pixel centers land BETWEEN texel
        // centers, so the bilinear filter engages. The last pixel of the
        // cell (x = 4.5 → tile-space 3.6) taps the last content texel
        // (weight 0.9) and the zero APRON (weight 0.1) — never the next
        // slot's content, which sits two texels further.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 1.0; // tile 1: ALSO solid
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 5.0,
            cell_h: 5.0,
            density: 0.8,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 5,
            frame_h: 5,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 5, 5);
        // Row 2 (tile-space y = 2.0·... = interior): x = 4.5 → 0.9 content,
        // 0.1 apron. If the tap bled into tile 1's solid content instead,
        // this would read 1.0.
        assert!(
            (r[2 * 5 + 4] - 0.9).abs() < 1e-5,
            "fractional edge sample R = {}",
            r[2 * 5 + 4]
        );
    }

    #[test]
    fn cells_wider_than_their_tile_fade_to_background() {
        // 1x1 grid, 8-point-wide cell over a 4-texel tile: the right half
        // of the cell has no tile content. The clamp must land those taps
        // in the zero apron (background), never in a neighboring slot.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 1.0; // tile 1: ALSO solid
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 8.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 8, 4);
        // Left half: tile content, fg. Right half: past the tile — if the
        // clamp failed, taps would reach tile 1 (also solid) and read 1.0.
        assert!(
            (r[px(1, 1)] - 1.0).abs() < 1e-5,
            "tile content R = {}",
            r[px(1, 1)]
        );
        for col in 5..8 {
            assert!(
                r[px(1, col)].abs() < 1e-5,
                "cell past its tile must be background at col {col}, got {}",
                r[px(1, col)]
            );
        }
    }

    #[test]
    #[should_panic(expected = "exactly")]
    fn oversized_atlas_is_refused() {
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 1 << 13,
            atlas_height: 1 << 12, // 2^25 texels: past exact f32 indexing
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 4,
        };
        let _refused = CellGridProgram::compile(geom, [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "floats bound to slot")]
    fn mismatched_cell_buffer_is_refused() {
        let (program, _, atlas) = tiny_scene();
        // Binding satisfies unused_must_use; the call panics first.
        let _refused = program.frame(Arc::new(alloc::vec![0.0; 3]), atlas);
    }

    #[test]
    fn cells_len_and_atlas_len_match_declared_geometry() {
        let geom = CellGridGeometry {
            cols: 3,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 12,
            frame_h: 8,
        };
        assert_eq!(geom.cells_len(), 3 * 2 * CELL_STRIDE);
        assert_eq!(geom.atlas_len(), 12 * 6);
    }

    #[test]
    fn green_channel_reads_its_own_bg_field_not_a_neighboring_one() {
        // In `tiny_scene`, cell 1 has fg_g = 1.0, bg_g = 0.0, coverage 0.5,
        // so the correct blend is 0.5. Every neighboring field at cell 1
        // (fg_a, bg_r, bg_b) is 1.0 or 0.0 in a way that a wrong bg-field
        // offset would still coincidentally read 0.5-ish through cell 0's
        // R/B channels the other tests already check — G is the one channel
        // whose neighbors in the field layout (fg_a at offset 5, bg_r at
        // offset 6) both read 1.0/0.0 in a way that distinguishes a bad
        // offset from the correct one.
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        let g = plane(&frame, 1, 8, 4);
        assert!(
            (g[px(1, 5)] - 0.5).abs() < 1e-5,
            "cell1 G = {}",
            g[px(1, 5)]
        );
    }

    #[test]
    fn cells_taller_than_their_tile_fade_to_background() {
        // Mirrors `cells_wider_than_their_tile_fade_to_background` in the
        // row direction: a 1x1 grid with a cell taller than its tile. The
        // bottom half has no tile content; the vertical apron clamp must
        // land those taps in the zero apron, never a neighboring slot.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 1.0; // tile 1: ALSO solid
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 4.0,
            cell_h: 8.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 8,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 4, 8);
        // Top half: tile content, fg. Bottom half: past the tile — if the
        // clamp failed, taps would reach tile 1 (also solid) and read 1.0.
        assert!((r[5] - 1.0).abs() < 1e-5, "tile content R = {}", r[5]);
        for row in 5..8 {
            assert!(
                r[row * 4 + 1].abs() < 1e-5,
                "cell past its tile must be background at row {row}, got {}",
                r[row * 4 + 1]
            );
        }
    }

    #[test]
    fn grid_extent_is_cell_count_times_cell_size_not_their_sum() {
        // A 3x3 grid of 2x2 cells: grid_w = grid_h = 6.0. If the boundary
        // check ever computed cols + cell_w (5.0) instead of cols * cell_w,
        // the last row/column of real cells (spanning [4, 6)) would be
        // wrongly treated as outside the grid.
        let (aw, ah) = (6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // solid tile
            }
        }
        let geom = CellGridGeometry {
            cols: 3,
            rows: 3,
            cell_w: 2.0,
            cell_h: 2.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 8,
        };
        let program = CellGridProgram::compile(geom, [0.4; 4]);
        let block: [f32; CELL_STRIDE] = [
            1.0, 1.0, /* fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let mut cells = vec![];
        for _ in 0..9 {
            cells.extend_from_slice(&block);
        }
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 8, 8);
        // (5.5, 5.5) is interior to the last (col 2, row 2) cell — must
        // read pure fg (1.0), not the default background (0.4) a
        // cols+cell_w-sized boundary would wrongly clip it to.
        assert!(
            (r[5 * 8 + 5] - 1.0).abs() < 1e-5,
            "last cell R = {}",
            r[5 * 8 + 5]
        );
    }

    #[test]
    fn baking_a_row_offset_region_samples_the_correct_absolute_row() {
        // A 1x2 grid (one column, two rows) with visibly different content
        // per row. Baking a region whose `PlaneRegion::y0` starts mid-grid
        // must read the ABSOLUTE row that `y0` implies. A constant error in
        // the pixel-center offset shifts every sampled row by the same
        // amount, so a self-consistency check (comparing this bake against
        // a full bake sliced at the same relative offset) can't see it —
        // this asserts a concrete expected value instead.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 8,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));

        let mut dense = vec![0.0f32; 4 * 4];
        frame.collapse_channel_rows(0, PlaneRegion::rows(4, 4, 4), &mut dense, 4);
        // y0 = 4 → the first baked row samples absolute y = 4.5, inside row
        // 1's cell (coverage 0.5, R: bg 1.0 -> fg 0.0 blends to 0.5). An
        // off-by-one-pixel-center bug would instead sample y = 3.5, still
        // inside row 0's cell (solid fg, R = 1.0).
        assert!(
            (dense[1] - 0.5).abs() < 1e-5,
            "row-offset bake read the wrong absolute row: R = {}",
            dense[1]
        );
    }
}
