//! # Scene: what the rasterizer renders
//!
//! The rasterizer's input is either an arbitrary color manifold — rendered
//! by dense per-batch evaluation through the `Manifold` trait — or a JIT
//! cell-grid frame ([`pixelflow_core::CellGridFrame`]), whose four channel
//! planes are baked by ONE internal-loop collapse call per channel per
//! stripe and then packed to pixels. The cell-grid lane is the production
//! frame path: no per-batch FFI boundary, no virtual dispatch, and the
//! collapse kernel's two-level LICM prologues (per-call, per-row) active.

use crate::render::frame::Frame;
use crate::render::Pixel;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pixelflow_core::{CellGridFrame, PlaneRegion};
use pixelflow_core::{Discrete, Manifold};
use std::sync::Arc;

/// A renderable scene. See the module docs for the two lanes.
#[derive(Clone)]
pub enum Scene {
    /// Dense per-batch evaluation of an arbitrary color manifold.
    Surface(Arc<dyn Manifold<Output = Discrete> + Send + Sync>),
    /// A JIT cell-grid frame: channel planes baked by the 2D collapse
    /// kernel, packed to pixels per stripe.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    CellGrid(CellGridFrame),
}

impl From<Arc<dyn Manifold<Output = Discrete> + Send + Sync>> for Scene {
    fn from(manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>) -> Self {
        Self::Surface(manifold)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl From<CellGridFrame> for Scene {
    fn from(frame: CellGridFrame) -> Self {
        Self::CellGrid(frame)
    }
}

impl core::fmt::Debug for Scene {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Surface(_) => f.debug_tuple("Scene::Surface").finish(),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::CellGrid(_) => f.debug_tuple("Scene::CellGrid").finish(),
        }
    }
}

impl Scene {
    /// Render this scene into `frame` with up to `num_threads` workers.
    ///
    /// Surfaces go through the work-stealing per-batch rasterizer; cell
    /// grids bake channel planes stripe-parallel through the collapse
    /// kernels and pack via [`Pixel::from_rgba`].
    pub fn render<P: Pixel + Send>(&self, frame: &mut Frame<P>, num_threads: usize) {
        match self {
            Self::Surface(manifold) => {
                crate::render::rasterizer::rasterize(manifold, frame, num_threads);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::CellGrid(grid) => render_cell_grid(grid, frame, num_threads),
        }
    }
}

/// Bake and pack a cell-grid frame, stripe-parallel.
///
/// Each worker owns a contiguous run of rows: it bakes the four channel
/// planes for its stripe (one collapse call per channel — the pixel loop
/// lives inside the JIT) and packs them row by row.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn render_cell_grid<P: Pixel + Send>(
    grid: &CellGridFrame,
    frame: &mut Frame<P>,
    num_threads: usize,
) {
    let (width, height) = (frame.width, frame.height);
    if width == 0 || height == 0 {
        return;
    }
    let workers = num_threads.max(1).min(height);
    let rows_per = height.div_ceil(workers);
    let mut bands: Vec<(usize, &mut [P])> = Vec::with_capacity(workers);
    {
        let mut rest: &mut [P] = &mut frame.data[..width * height];
        let mut y = 0usize;
        while y < height {
            let rows = rows_per.min(height - y);
            let (band, tail) = rest.split_at_mut(rows * width);
            bands.push((y, band));
            rest = tail;
            y += rows;
        }
    }

    std::thread::scope(|scope| {
        for (y0, band) in bands {
            scope.spawn(move || bake_and_pack_stripe(grid, width, y0, band));
        }
    });
}

/// Scratch budget per worker for the four channel planes. Bounds the
/// bake-and-pack working set to cache-friendly chunks instead of sizing
/// temporaries to the whole stripe (a 4K frame would otherwise stage
/// ~127 MiB of planes per render).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const PLANE_SCRATCH_BYTES: usize = 1 << 20;

/// Bake the four channel planes for rows `y0..y0 + band.len()/width` and
/// pack them into `band`, in bounded-height chunks reusing one scratch
/// allocation.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bake_and_pack_stripe<P: Pixel>(grid: &CellGridFrame, width: usize, y0: usize, band: &mut [P]) {
    let stride = CellGridFrame::padded_width(width);
    let chunk_rows = PLANE_SCRATCH_BYTES / (4 * stride * core::mem::size_of::<f32>());
    bake_and_pack_chunked(grid, width, y0, band, chunk_rows);
}

/// [`bake_and_pack_stripe`] with an explicit chunk height (rows of plane
/// scratch per pass) — split out so tests can force chunk boundaries that
/// the production budget would only hit on very large frames.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bake_and_pack_chunked<P: Pixel>(
    grid: &CellGridFrame,
    width: usize,
    y0: usize,
    band: &mut [P],
    chunk_rows: usize,
) {
    let rows = band.len() / width;
    let stride = CellGridFrame::padded_width(width);
    let chunk_rows = chunk_rows.clamp(1, rows.max(1));
    let mut planes = vec![0.0f32; 4 * chunk_rows * stride];

    let mut done = 0usize;
    while done < rows {
        let n = chunk_rows.min(rows - done);
        {
            let (r, rest) = planes.split_at_mut(chunk_rows * stride);
            let (g, rest) = rest.split_at_mut(chunk_rows * stride);
            let (b, a) = rest.split_at_mut(chunk_rows * stride);
            let region = PlaneRegion {
                width,
                y0: y0 + done,
                rows: n,
            };
            grid.bake_channel_rows(0, region, r);
            grid.bake_channel_rows(1, region, g);
            grid.bake_channel_rows(2, region, b);
            grid.bake_channel_rows(3, region, a);
        }
        let plane =
            |c: usize| &planes[c * chunk_rows * stride..c * chunk_rows * stride + n * stride];
        let (r, g, b, a) = (plane(0), plane(1), plane(2), plane(3));
        for row in 0..n {
            let p = row * stride;
            let o = (done + row) * width;
            for i in 0..width {
                band[o + i] = P::from_rgba(r[p + i], g[p + i], b[p + i], a[p + i]);
            }
        }
        done += n;
    }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::render::color::Rgba8;
    use pixelflow_core::{CellGridGeometry, CellGridProgram};

    /// A 2×2 grid of solid/half tiles; oracle is the scalar blend math.
    fn scene() -> Scene {
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
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
        };
        let program = CellGridProgram::compile(geom, [0.25, 0.25, 0.25, 1.0]);
        #[rustfmt::skip]
        let cells = vec![
            // (0,0): solid tile, red on black
            1.0, 1.0,  1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 1.0,
            // (1,0): half tile, white on blue
            7.0, 1.0,  1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 1.0, 1.0,
            // (0,1): half tile, green on black
            7.0, 1.0,  0.0, 1.0, 0.0, 1.0,  0.0, 0.0, 0.0, 1.0,
            // (1,1): solid tile, black on white
            1.0, 1.0,  0.0, 0.0, 0.0, 1.0,  1.0, 1.0, 1.0, 1.0,
        ];
        Scene::CellGrid(program.frame(Arc::new(cells), Arc::new(atlas)))
    }

    #[test]
    fn cell_grid_scene_packs_blended_channels() {
        let scene = scene();
        // 10×10 frame: the 8×8 grid plus a margin of default background.
        let mut frame = Frame::<Rgba8>::new(10, 10);
        scene.render(&mut frame, 2);

        let px = |x: usize, y: usize| frame.data[y * 10 + x];
        // Cell (0,0) interior: coverage 1 → pure red.
        assert_eq!((px(1, 1).r(), px(1, 1).g(), px(1, 1).b()), (255, 0, 0));
        // Cell (1,0) interior: coverage ½ of white over blue → (128, 128, 255)ish.
        let p = px(5, 1);
        assert!(p.r() >= 126 && p.r() <= 129, "half blend r = {}", p.r());
        assert_eq!(p.b(), 255);
        // Cell (1,1) interior: black over white at coverage 1 → black.
        assert_eq!(px(5, 5).r(), 0);
        // Outside the grid: the default background (0.25 → ~64).
        let m = px(9, 9);
        assert!(m.r() >= 62 && m.r() <= 66, "margin r = {}", m.r());
    }

    #[test]
    fn chunked_bake_matches_whole_stripe() {
        // Force chunk boundaries (3-row chunks over an 8-row band) and
        // require pixel identity with the unchunked bake.
        let Scene::CellGrid(grid) = scene() else {
            unreachable!()
        };
        let (w, h) = (9usize, 8usize);
        let mut whole = vec![Rgba8::from_u32(0); w * h];
        let mut chunked = vec![Rgba8::from_u32(0); w * h];
        super::bake_and_pack_chunked(&grid, w, 0, &mut whole, h);
        super::bake_and_pack_chunked(&grid, w, 0, &mut chunked, 3);
        assert_eq!(whole, chunked, "chunk boundaries must not change pixels");
    }

    #[test]
    fn stripe_split_matches_single_threaded() {
        let scene = scene();
        let mut one = Frame::<Rgba8>::new(9, 8);
        let mut many = Frame::<Rgba8>::new(9, 8);
        scene.render(&mut one, 1);
        scene.render(&mut many, 4);
        assert_eq!(one.data, many.data, "thread count must not change pixels");
    }
}
