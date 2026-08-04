//! # Scene: what the rasterizer renders
//!
//! The rasterizer's input is either an arbitrary color manifold — rendered
//! by dense per-batch evaluation through the `Manifold` trait — or a JIT
//! packed cell-grid frame ([`pixelflow_core::CellGridPackedFrame`]): ONE
//! collapse call per stripe whose kernel computes all four channels and
//! packs them to `u32` pixels in-register, so what reaches memory is
//! finished pixels. The cell-grid lane is the production frame path: no
//! per-batch FFI boundary, no virtual dispatch, no per-pixel scalar pack,
//! and the collapse kernel's two-level LICM prologues (per-call, per-row)
//! active. Byte order comes from the platform's ColorCube
//! ([`compile_platform_cell_grid`]), never from application code.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::render::color::PlatformColorCube;
use crate::render::frame::Frame;
use crate::render::Pixel;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pixelflow_core::{
    CellGridFrame, CellGridGeometry, CellGridPackedFrame, CellGridPackedProgram, PlaneRegion,
};
use pixelflow_core::{Discrete, Manifold};
use std::sync::Arc;

/// A renderable scene. See the module docs for the two lanes.
#[derive(Clone)]
pub enum Scene {
    /// Dense per-batch evaluation of an arbitrary color manifold.
    Surface(Arc<dyn Manifold<Output = Discrete> + Send + Sync>),
    /// A JIT packed cell-grid frame: the 2D collapse kernel writes
    /// finished `u32` pixels; stripes copy rows into the frame.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    CellGrid(CellGridPackedFrame),
}

impl From<Arc<dyn Manifold<Output = Discrete> + Send + Sync>> for Scene {
    fn from(manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>) -> Self {
        Self::Surface(manifold)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl From<CellGridPackedFrame> for Scene {
    fn from(frame: CellGridPackedFrame) -> Self {
        Self::CellGrid(frame)
    }
}

/// Compile the packed cell-grid program with THIS platform's pixel byte
/// order, taken from [`PlatformColorCube`] — the one home byte order has.
/// Applications pass geometry and colors; they never see a shift.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_platform_cell_grid(
    geom: CellGridGeometry,
    default_bg: [f32; 4],
) -> CellGridPackedProgram {
    CellGridPackedProgram::compile(geom, default_bg, PlatformColorCube::PACKED_SHIFTS)
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
    /// grids bake finished packed pixels stripe-parallel through the one
    /// collapse kernel and copy rows into the frame.
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

/// Bake a packed cell-grid frame, stripe-parallel.
///
/// Each worker owns a contiguous run of rows: one collapse call bakes its
/// stripe's finished pixels (the pixel loop and the pack both live inside
/// the JIT), and each row is copied `width`-wide into the frame.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn render_cell_grid<P: Pixel + Send>(
    grid: &CellGridPackedFrame,
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
            scope.spawn(move || bake_packed_stripe(grid, width, y0, band));
        }
    });
}

/// Staging budget per worker: the kernel writes whole batches on a padded
/// stride, and frame rows are width-packed, so rows stage here and are
/// copied `width`-wide. A quarter of the old four-plane scratch would
/// already hold the same rows; keeping the budget quadruples the rows per
/// collapse call instead.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const STAGING_SCRATCH_BYTES: usize = 1 << 20;

/// Bake rows `y0..y0 + band.len()/width` and copy them into `band`, in
/// bounded-height chunks reusing one staging allocation.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bake_packed_stripe<P: Pixel>(
    grid: &CellGridPackedFrame,
    width: usize,
    y0: usize,
    band: &mut [P],
) {
    let stride = CellGridFrame::padded_width(width);
    let chunk_rows = STAGING_SCRATCH_BYTES / (stride * core::mem::size_of::<u32>());
    bake_packed_chunked(grid, width, y0, band, chunk_rows);
}

/// [`bake_packed_stripe`] with an explicit chunk height — split out so tests
/// can force chunk boundaries the production budget would only hit on very
/// large frames.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bake_packed_chunked<P: Pixel>(
    grid: &CellGridPackedFrame,
    width: usize,
    y0: usize,
    band: &mut [P],
    chunk_rows: usize,
) {
    let rows = band.len() / width;
    let stride = CellGridFrame::padded_width(width);
    let chunk_rows = chunk_rows.clamp(1, rows.max(1));
    let mut staging = vec![0u32; chunk_rows * stride];

    let mut done = 0usize;
    while done < rows {
        let n = chunk_rows.min(rows - done);
        grid.bake_packed_rows(
            PlaneRegion {
                width,
                y0: y0 + done,
                rows: n,
            },
            &mut staging,
        );
        for row in 0..n {
            let src = &staging[row * stride..row * stride + width];
            let dst = &mut band[(done + row) * width..(done + row + 1) * width];
            for (d, s) in dst.iter_mut().zip(src) {
                *d = P::from_u32(*s);
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
    use crate::render::color::RgbaColorCube;
    use pixelflow_core::CellGridPackedProgram;

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
        let program = CellGridPackedProgram::compile(
            geom,
            [0.25, 0.25, 0.25, 1.0],
            RgbaColorCube::PACKED_SHIFTS,
        );
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
        //
        // `bake_packed_chunked` is production logic (`bake_packed_stripe`
        // calls it on the real render path too), not a test-only seam — but
        // its `chunk_rows` argument is derived internally from
        // `STAGING_SCRATCH_BYTES`, which only forces a chunk boundary on
        // frames far larger than a unit test should render. Calling it
        // directly with a small `chunk_rows` is the only way to exercise
        // that boundary deterministically; going through `Scene::render`
        // could only ever hit the always-one-chunk case. Flexibility-clause
        // exception to docs/STYLE.md's public-API testing rule, same as the
        // 2026-07-24 pass's call for actor-scheduler's timing-internal
        // backoff arithmetic.
        let Scene::CellGrid(grid) = scene() else {
            unreachable!()
        };
        let (w, h) = (9usize, 8usize);
        let mut whole = vec![Rgba8::from_u32(0); w * h];
        let mut chunked = vec![Rgba8::from_u32(0); w * h];
        super::bake_packed_chunked(&grid, w, 0, &mut whole, h);
        super::bake_packed_chunked(&grid, w, 0, &mut chunked, 3);
        assert_eq!(whole, chunked, "chunk boundaries must not change pixels");
    }

    #[test]
    fn platform_wrapper_needs_no_shifts_from_the_caller() {
        let Scene::CellGrid(reference) = scene() else {
            unreachable!()
        };
        // Same geometry/data through the platform wrapper: it must compile
        // and render; pixel values are platform byte order, so only shape
        // and the alpha-lane content are asserted here (Rgba8 and Bgra8
        // agree on the alpha byte).
        let _ = reference;
        let (aw, ah) = (12usize, 6usize);
        let atlas = vec![0.0f32; aw * ah];
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
        let program = super::compile_platform_cell_grid(geom, [0.0, 0.0, 0.0, 1.0]);
        // Zero coverage everywhere, so every in-grid pixel shows its cell's
        // BACKGROUND — which must be opaque for the alpha assertion to hold.
        let mut cells = vec![0.0f32; geom.cells_len()];
        for cell in cells.chunks_mut(pixelflow_core::CELL_STRIDE) {
            cell[9] = 1.0; // bg_a
        }
        let frame_data = program.frame(Arc::new(cells), Arc::new(atlas));
        let mut frame = Frame::<Rgba8>::new(8, 8);
        Scene::CellGrid(frame_data).render(&mut frame, 1);
        assert!(
            frame.data.iter().all(|p| p.0 >> 24 == 255),
            "alpha lane must be opaque regardless of platform byte order"
        );
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

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod frame_bench {
    //! Steady-state frame-bake comparison: the retired four-plane-then-pack
    //! shape against the packed kernel. Ignored by default — run manually:
    //! `cargo test --release -p pixelflow-graphics --lib -- --ignored
    //! bench_packed --nocapture`. The four-channel program still exists in
    //! pixelflow-core as the parity oracle, which is what lets the retired
    //! shape be reconstructed here without keeping dead render code.
    use super::*;
    use crate::render::color::{Rgba8, RgbaColorCube};
    use pixelflow_core::{CellGridGeometry, CellGridPackedProgram, CellGridProgram};

    fn realistic() -> (CellGridGeometry, Vec<f32>, Vec<f32>) {
        // A 2560x1584 frame of 12x24 cells: 213x66 — a full-screen terminal.
        let geom = CellGridGeometry {
            cols: 213,
            rows: 66,
            cell_w: 12.0,
            cell_h: 24.0,
            density: 1.0,
            atlas_width: 64,
            atlas_height: 32,
            tile_w: 12,
            tile_h: 24,
        };
        let mut atlas = vec![0.0f32; geom.atlas_len()];
        for (i, t) in atlas.iter_mut().enumerate() {
            *t = ((i * 7) % 11) as f32 / 10.0;
        }
        let mut cells = vec![0.0f32; geom.cells_len()];
        for (i, c) in cells.iter_mut().enumerate() {
            *c = ((i * 13) % 17) as f32 / 16.0;
        }
        (geom, cells, atlas)
    }

    fn median_ns(mut samples: Vec<u128>) -> u128 {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    #[test]
    #[ignore = "manual benchmark; run in --release with --nocapture"]
    fn bench_packed_vs_four_plane() {
        let (geom, cells, atlas) = realistic();
        let (w, h) = (2560usize, 1584usize);
        let stride = CellGridFrame::padded_width(w);
        let cells = Arc::new(cells);
        let atlas = Arc::new(atlas);

        let four = CellGridProgram::compile(geom, [0.1, 0.1, 0.1, 1.0])
            .frame(cells.clone(), atlas.clone());
        let packed = CellGridPackedProgram::compile(
            geom,
            [0.1, 0.1, 0.1, 1.0],
            RgbaColorCube::PACKED_SHIFTS,
        )
        .frame(cells, atlas);

        let mut band = vec![Rgba8::from_u32(0); w * h];
        let chunk_rows = STAGING_SCRATCH_BYTES / (stride * core::mem::size_of::<u32>());

        // The retired shape: four plane bakes plus a per-pixel from_rgba.
        let old_pass = |band: &mut [Rgba8]| {
            let mut planes = vec![0.0f32; 4 * chunk_rows * stride];
            let mut done = 0usize;
            while done < h {
                let n = chunk_rows.min(h - done);
                {
                    let (r, rest) = planes.split_at_mut(chunk_rows * stride);
                    let (g, rest) = rest.split_at_mut(chunk_rows * stride);
                    let (b, a) = rest.split_at_mut(chunk_rows * stride);
                    let region = PlaneRegion {
                        width: w,
                        y0: done,
                        rows: n,
                    };
                    four.bake_channel_rows(0, region, r);
                    four.bake_channel_rows(1, region, g);
                    four.bake_channel_rows(2, region, b);
                    four.bake_channel_rows(3, region, a);
                }
                let plane = |c: usize| &planes[c * chunk_rows * stride..];
                for row in 0..n {
                    let p = row * stride;
                    let o = (done + row) * w;
                    for i in 0..w {
                        band[o + i] = Rgba8::from_rgba(
                            plane(0)[p + i],
                            plane(1)[p + i],
                            plane(2)[p + i],
                            plane(3)[p + i],
                        );
                    }
                }
                done += n;
            }
        };
        let new_pass = |band: &mut [Rgba8]| bake_packed_chunked(&packed, w, 0, band, chunk_rows);

        const WARM: usize = 3;
        const RUNS: usize = 15;
        for _ in 0..WARM {
            old_pass(&mut band);
            new_pass(&mut band);
        }
        let mut old_ns = Vec::with_capacity(RUNS);
        let mut new_ns = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = std::time::Instant::now();
            old_pass(&mut band);
            old_ns.push(t.elapsed().as_nanos());
            let t = std::time::Instant::now();
            new_pass(&mut band);
            new_ns.push(t.elapsed().as_nanos());
        }
        let (o, n) = (median_ns(old_ns), median_ns(new_ns));
        let px = (w * h) as u128;
        println!("frame {w}x{h} ({px} px), single-threaded, median of {RUNS}:");
        println!(
            "  four-plane + per-pixel pack: {o} ns/frame ({:.3} ns/px)",
            o as f64 / px as f64
        );
        println!(
            "  packed kernel + row copy:    {n} ns/frame ({:.3} ns/px)",
            n as f64 / px as f64
        );
        println!("  speedup: {:.2}x", o as f64 / n as f64);
    }
}
