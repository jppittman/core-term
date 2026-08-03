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
//! dimensions, cell size, atlas extents) recompiles the four channel
//! kernels, which is microseconds through the JIT. This replaces the only
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

use super::JitVec;
use crate::Field;
use pixelflow_codegen::JitManifold;
use pixelflow_ir::arena::BufferDecl;
use pixelflow_ir::{ExprArena, ExprId, OpKind};

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
}

impl CellGridGeometry {
    /// Cell-data buffer length this geometry implies.
    #[must_use]
    pub fn cells_len(&self) -> usize {
        self.cols as usize * self.rows as usize * CELL_STRIDE
    }

    /// Atlas buffer length this geometry implies.
    #[must_use]
    pub fn atlas_len(&self) -> usize {
        self.atlas_width as usize * self.atlas_height as usize
    }
}

/// One channel's program: everything from pixel coordinate to blended
/// channel value, as a single arena over the two buffers.
fn channel_arena(geom: &CellGridGeometry, channel: usize, default_bg: f32) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    // Buffer 0: per-cell data; buffer 1: the atlas. `CellGridChannel::eval`
    // binds the context pointers in this order.
    let cells = a.declare_buffer(BufferDecl {
        width: geom.cols * CELL_STRIDE as u32,
        height: geom.rows,
    });
    let atlas = a.declare_buffer(BufferDecl {
        width: geom.atlas_width,
        height: geom.atlas_height,
    });

    let x = a.push_var(0);
    let y = a.push_var(1);
    let one = a.push_const(1.0);
    let half = a.push_const(0.5);

    // Which cell, and where inside it.
    let inv_cw = a.push_const(1.0 / geom.cell_w);
    let inv_ch = a.push_const(1.0 / geom.cell_h);
    let xc = a.push_binary(OpKind::Mul, x, inv_cw);
    let yc = a.push_binary(OpKind::Mul, y, inv_ch);
    let col = a.push_unary(OpKind::Floor, xc);
    let row = a.push_unary(OpKind::Floor, yc);
    let cw = a.push_const(geom.cell_w);
    let ch = a.push_const(geom.cell_h);
    let col_px = a.push_binary(OpKind::Mul, col, cw);
    let row_px = a.push_binary(OpKind::Mul, row, ch);
    let lx = a.push_binary(OpKind::Sub, x, col_px);
    let ly = a.push_binary(OpKind::Sub, y, row_px);

    // Per-cell data: gathers clamp to the buffer edge, so out-of-grid
    // queries read the border cell — harmless, since the final select
    // replaces them with the default background anyway.
    let stride = a.push_const(CELL_STRIDE as f32);
    let cx = a.push_binary(OpKind::Mul, col, stride);
    let field = |a: &mut ExprArena, base: ExprId, offset: usize| {
        let off = a.push_const(offset as f32);
        a.push_binary(OpKind::Add, base, off)
    };
    let u_idx = cx; // offset 0
    let v_idx = field(&mut a, cx, 1);
    let fg_idx = field(&mut a, cx, 2 + channel);
    let bg_idx = field(&mut a, cx, 6 + channel);
    let u0 = a.push_gather(cells, u_idx, row);
    let v0 = a.push_gather(cells, v_idx, row);
    let fg = a.push_gather(cells, fg_idx, row);
    let bg = a.push_gather(cells, bg_idx, row);

    // Atlas coordinates: point offset into the cell, scaled to texels,
    // shifted to texel centers.
    let density = a.push_const(geom.density);
    let lxd_raw = a.push_binary(OpKind::Mul, lx, density);
    let lyd_raw = a.push_binary(OpKind::Mul, ly, density);
    // Clamp to half a texel past the tile content: the taps then land on
    // this slot's apron and (at most) the neighbor's apron — both zero —
    // so a cell larger than its tile fades to background rather than
    // displaying fragments of the adjacent tile.
    let tile_w_edge = a.push_const(geom.tile_w as f32 + 0.5);
    let tile_h_edge = a.push_const(geom.tile_h as f32 + 0.5);
    let lxd = a.push_binary(OpKind::Min, lxd_raw, tile_w_edge);
    let lyd = a.push_binary(OpKind::Min, lyd_raw, tile_h_edge);
    let au = a.push_binary(OpKind::Add, u0, lxd);
    let av = a.push_binary(OpKind::Add, v0, lyd);
    let ax = a.push_binary(OpKind::Sub, au, half);
    let ay = a.push_binary(OpKind::Sub, av, half);

    // 4-tap bilinear read of the atlas (the same blend `bilinear_arena`
    // builds for a whole-buffer sampler, at computed tile coordinates).
    let ax0 = a.push_unary(OpKind::Floor, ax);
    let ay0 = a.push_unary(OpKind::Floor, ay);
    let ax1 = a.push_binary(OpKind::Add, ax0, one);
    let ay1 = a.push_binary(OpKind::Add, ay0, one);
    let fx = a.push_binary(OpKind::Sub, ax, ax0);
    let fy = a.push_binary(OpKind::Sub, ay, ay0);
    let gx = a.push_binary(OpKind::Sub, one, fx);
    let gy = a.push_binary(OpKind::Sub, one, fy);
    let c00 = a.push_gather(atlas, ax0, ay0);
    let c10 = a.push_gather(atlas, ax1, ay0);
    let c01 = a.push_gather(atlas, ax0, ay1);
    let c11 = a.push_gather(atlas, ax1, ay1);
    let w00 = a.push_binary(OpKind::Mul, gx, gy);
    let w10 = a.push_binary(OpKind::Mul, fx, gy);
    let w01 = a.push_binary(OpKind::Mul, gx, fy);
    let w11 = a.push_binary(OpKind::Mul, fx, fy);
    let t00 = a.push_binary(OpKind::Mul, c00, w00);
    let t10 = a.push_binary(OpKind::Mul, c10, w10);
    let t01 = a.push_binary(OpKind::Mul, c01, w01);
    let t11 = a.push_binary(OpKind::Mul, c11, w11);
    let s0 = a.push_binary(OpKind::Add, t00, t10);
    let s1 = a.push_binary(OpKind::Add, s0, t01);
    let cov = a.push_binary(OpKind::Add, s1, t11);

    // blended = bg + cov·(fg − bg).
    let diff = a.push_binary(OpKind::Sub, fg, bg);
    let scaled = a.push_binary(OpKind::Mul, cov, diff);
    let blended = a.push_binary(OpKind::Add, bg, scaled);

    // Outside the grid there is no cell data: the default background.
    let zero = a.push_const(0.0);
    let grid_w = a.push_const(geom.cols as f32 * geom.cell_w);
    let grid_h = a.push_const(geom.rows as f32 * geom.cell_h);
    let x_lo = a.push_binary(OpKind::Ge, x, zero);
    let x_hi = a.push_binary(OpKind::Lt, x, grid_w);
    let y_lo = a.push_binary(OpKind::Ge, y, zero);
    let y_hi = a.push_binary(OpKind::Lt, y, grid_h);
    let x_in = a.push_binary(OpKind::BitAnd, x_lo, x_hi);
    let y_in = a.push_binary(OpKind::BitAnd, y_lo, y_hi);
    let in_grid = a.push_binary(OpKind::BitAnd, x_in, y_in);
    let bg_default = a.push_const(default_bg);
    let root = a.push_ternary(OpKind::Select, in_grid, blended, bg_default);
    (a, root)
}

/// The four compiled channel kernels for one grid/atlas geometry.
///
/// Compile once per geometry ([`CellGridProgram::compile`]); stamp per-frame
/// data into it with [`CellGridProgram::frame`]. A resize is: build the new
/// geometry, compile, done — the program's size and compile time are
/// independent of the grid's dimensions.
pub struct CellGridProgram {
    geom: CellGridGeometry,
    jits: [Arc<JitManifold>; 4],
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
        assert!(
            geom.cols > 0 && geom.rows > 0 && geom.atlas_width > 0 && geom.atlas_height > 0,
            "CellGridProgram::compile: degenerate geometry {geom:?}"
        );
        assert!(
            geom.cell_w > 0.0 && geom.cell_h > 0.0 && geom.density > 0.0,
            "CellGridProgram::compile: non-positive cell extent or density {geom:?}"
        );
        assert_eq!(
            core::mem::size_of::<Field>(),
            pixelflow_codegen::JIT_VECTOR_BYTES,
            "CellGridProgram::compile: Field width does not match the JIT's emitted width"
        );
        // Gather computes its row-major linear index in f32, which is exact
        // only below 2^24 — beyond that adjacent texels alias. Refuse
        // loudly instead of rendering corrupted gathers.
        const EXACT_F32_INDEX: usize = 1 << 24;
        assert!(
            geom.atlas_len() <= EXACT_F32_INDEX,
            "CellGridProgram::compile: atlas of {} texels exceeds the exactly \
             f32-indexable range (2^24); gathers would alias adjacent texels",
            geom.atlas_len()
        );
        assert!(
            geom.cells_len() <= EXACT_F32_INDEX,
            "CellGridProgram::compile: cell buffer of {} floats exceeds the \
             exactly f32-indexable range (2^24)",
            geom.cells_len()
        );
        let jits = core::array::from_fn(|c| {
            let (arena, root) = channel_arena(&geom, c, default_bg[c]);
            // Collapse-mode (internal 2D loop): one call bakes a whole run
            // of rows for a channel. Bound-memory arenas are uncacheable
            // (the code bakes buffer slot metadata); the cache recognizes
            // that and compiles fresh.
            pixelflow_codegen::jit_cache::compile_collapse_cached(&arena, root)
                .expect("cell-grid channel failed to compile")
        });
        Self { geom, jits }
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
        assert_eq!(
            cells.len(),
            self.geom.cells_len(),
            "cell buffer length does not match geometry {:?}",
            self.geom
        );
        assert_eq!(
            atlas.len(),
            self.geom.atlas_len(),
            "atlas buffer length does not match geometry {:?}",
            self.geom
        );
        CellGridFrame {
            jits: self.jits.clone(),
            cells,
            atlas,
        }
    }
}

/// One frame of a cell grid: the compiled channels plus the data they read.
/// Cheap to clone (four `Arc`s and two `Arc`s).
#[derive(Clone)]
pub struct CellGridFrame {
    jits: [Arc<JitManifold>; 4],
    cells: Arc<Vec<f32>>,
    atlas: Arc<Vec<f32>>,
}

/// A horizontal band of pixel rows to bake: `width` pixels across, rows
/// `y0 .. y0 + rows`.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRegion {
    /// Pixels per row.
    pub width: usize,
    /// First pixel row.
    pub y0: usize,
    /// Number of rows.
    pub rows: usize,
}

impl CellGridFrame {
    /// The padded row stride (in `f32`s) that [`CellGridFrame::bake_channel_rows`]
    /// writes: `width` rounded up to whole SIMD batches. Planes are laid out
    /// at this stride; the padding lanes hold whatever the kernel computed
    /// past the right edge (out-of-grid pixels — the default background) and
    /// are simply not read back.
    #[must_use]
    pub fn padded_width(width: usize) -> usize {
        let lanes = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
        width.div_ceil(lanes).max(1) * lanes
    }

    /// Bake one color channel (0 = R, 1 = G, 2 = B, 3 = A) over the pixel
    /// rows `y0 .. y0 + rows` at pixel-center coordinates, into a plane of
    /// [`CellGridFrame::padded_width`]`(width)`-stride rows.
    ///
    /// ONE call into the channel's 2D collapse kernel per invocation: the
    /// loop over batches and rows lives inside the JIT, with the two-level
    /// LICM prologues (per-call and per-row) active. This is the production
    /// render path — a frame is four of these per stripe, then a pack.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= 4`, the region's width is zero, or `out` is
    /// shorter than `rows * padded_width(width)`.
    pub fn bake_channel_rows(&self, channel: usize, region: PlaneRegion, out: &mut [f32]) {
        let PlaneRegion { width, y0, rows } = region;
        assert!(width > 0, "bake_channel_rows: zero width");
        let stride = Self::padded_width(width);
        assert!(
            out.len() >= rows * stride,
            "bake_channel_rows: plane of {} f32s cannot hold {rows} rows at stride {stride}",
            out.len()
        );
        if rows == 0 {
            return;
        }
        let groups = stride / (pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>());
        let ctx = [self.cells.as_ptr(), self.atlas.as_ptr()];
        // Pixel centers: the rasterizer convention (x + ½, y + ½).
        let x0 = Field::sequential(0.5);
        let y = Field::from(y0 as f32 + 0.5);
        let zero = Field::from(0.0);
        // SAFETY: `compile` checked size_of::<Field>() == JIT_VECTOR_BYTES;
        // the kernel was compiled from an arena declaring exactly the two
        // buffers whose lengths `CellGridProgram::frame` asserted against
        // the same geometry; `ctx` binds their live base pointers (in
        // declaration order) for the duration of the call; and `out` holds
        // `rows` full rows of `groups` batches (asserted above) with no
        // scalar tail (row_skip = 0 — the stride IS whole batches).
        unsafe {
            self.jits[channel].call_collapse(
                ctx.as_ptr(),
                out.as_mut_ptr(),
                groups,
                rows,
                0,
                core::mem::transmute::<Field, JitVec>(x0),
                core::mem::transmute::<Field, JitVec>(y),
                core::mem::transmute::<Field, JitVec>(zero),
                core::mem::transmute::<Field, JitVec>(zero),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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

    /// Row-major index into an 8-wide plane.
    fn px(row: usize, col: usize) -> usize {
        row * 8 + col
    }

    /// Bake one channel over pixel centers and unpad to a dense w×h plane.
    fn plane(frame: &CellGridFrame, channel: usize, w: usize, h: usize) -> alloc::vec::Vec<f32> {
        let stride = CellGridFrame::padded_width(w);
        let mut padded = vec![0.0f32; h * stride];
        frame.bake_channel_rows(
            channel,
            PlaneRegion {
                width: w,
                y0: 0,
                rows: h,
            },
            &mut padded,
        );
        let mut dense = alloc::vec::Vec::with_capacity(w * h);
        for row in 0..h {
            dense.extend_from_slice(&padded[row * stride..row * stride + w]);
        }
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
        };
        let _refused = CellGridProgram::compile(geom, [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "cell buffer length")]
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
        };
        assert_eq!(geom.cells_len(), 3 * 2 * CELL_STRIDE);
        assert_eq!(geom.atlas_len(), 12 * 6);
    }

    #[test]
    fn padded_width_rounds_up_to_whole_simd_batches() {
        let lanes = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
        assert_eq!(CellGridFrame::padded_width(1), lanes);
        assert_eq!(CellGridFrame::padded_width(lanes), lanes);
        assert_eq!(CellGridFrame::padded_width(lanes + 1), lanes * 2);
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
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));

        let stride = CellGridFrame::padded_width(4);
        let mut padded = vec![0.0f32; 4 * stride];
        frame.bake_channel_rows(
            0,
            PlaneRegion {
                width: 4,
                y0: 4,
                rows: 4,
            },
            &mut padded,
        );
        // y0 = 4 → the first baked row samples absolute y = 4.5, inside row
        // 1's cell (coverage 0.5, R: bg 1.0 -> fg 0.0 blends to 0.5). An
        // off-by-one-pixel-center bug would instead sample y = 3.5, still
        // inside row 0's cell (solid fg, R = 1.0).
        assert!(
            (padded[1] - 0.5).abs() < 1e-5,
            "row-offset bake read the wrong absolute row: R = {}",
            padded[1]
        );
    }
}
