//! # Scene: what gets rendered into a frame
//!
//! A scene is a [`PackedFrame`] — four channel kernels compiled into ONE
//! program over the frame's lattice, with the pixel pack inside it — rendered
//! one collapse call per stripe, stripes across threads. No per-batch FFI
//! boundary, no virtual dispatch, no per-pixel scalar pack, and the collapse
//! kernel's two-level LICM prologues (per call, per row) active. Byte order
//! comes from the pixel format the frame is stored in
//! ([`compile_packed_for`], [`compile_platform_packed`]), never from
//! application code.
//!
//! The cell grid ([`compile_cell_grid_for`]) is one instance of that: a
//! geometry that denotes four channel kernels over a cell buffer and a
//! coverage atlas. A procedural shader is another: four kernels and nothing
//! bound.
//!
//! [`Scene::Surface`] is the one lane left over, and it is on its way out —
//! see its own docs.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::render::cell_grid::CellGridPackedProgram;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::render::color::{PlatformColorCube, PlatformPixel};
use crate::render::frame::Frame;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::render::packed::{PackedFrame, PackedProgram};
use crate::render::Pixel;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pixelflow_core::{CellGridGeometry, Kernel, PlaneRegion};
use pixelflow_core::{Discrete, Manifold};
use std::sync::Arc;

/// A renderable scene. See the module docs for the two lanes.
#[derive(Clone)]
pub enum Scene {
    /// Dense per-batch evaluation of an arbitrary color manifold: one
    /// `Manifold::eval` call per SIMD batch, across a `dyn` boundary.
    ///
    /// **The single per-batch remnant, kept for exactly one reason.** The
    /// plan of record (`docs/plans/2026-09-06-kernel-with-a-lattice.md`)
    /// deletes per-batch evaluation as an API; what still needs it is
    /// `scene3d`'s ray marching, which cannot lower while the IR has no
    /// iteration binder. This variant, `rasterize` and `execute_stripe` stay
    /// until that decision lands (add the binder, or retire the demos), and
    /// **no new consumer is permitted** — a scene that can be written as four
    /// channel kernels is a [`Scene::Packed`].
    Surface(Arc<dyn Manifold<Output = Discrete> + Send + Sync>),
    /// A packed-pixel program over the frame lattice: the 2D collapse kernel
    /// writes finished `u32` pixels; stripes copy rows into the frame.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    Packed(PackedFrame),
}

impl From<Arc<dyn Manifold<Output = Discrete> + Send + Sync>> for Scene {
    fn from(manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>) -> Self {
        Self::Surface(manifold)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl From<PackedFrame> for Scene {
    fn from(frame: PackedFrame) -> Self {
        Self::Packed(frame)
    }
}

/// Compile a colour output for the pixel format `P`: four channel kernels in
/// `[0, 1]` — red, green, blue, alpha, which is all a colour output ever is —
/// over a `frame[0] × frame[1]` pixel lattice.
///
/// This is the constructor a shader author uses. Byte order comes from `P`
/// itself, so the format the kernel packs for is the format the frame stores,
/// by construction; the author never sees a shift.
///
/// The channel kernels are sampled at **device pixel centers** of the frame
/// they are compiled for. A shader authored in some other space says so in
/// the language, by precomposing the embedding — `channel.at(&(X * s), &(Y *
/// s), &Z, &W)` — before compiling here; there is no separate scale for
/// anything downstream to keep in step.
///
/// # Panics
///
/// Panics for a `P` with no packed RGBA form ([`Pixel::packed_shifts`]
/// returns `None`) — a grayscale or exotic format has no byte lanes to pack
/// into, and silently packing RGBA into it would be garbage.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_packed_for<P: Pixel>(channels: &[Kernel; 4], frame: [u32; 2]) -> PackedProgram {
    PackedProgram::compile(channels, packed_shifts_of::<P>("compile_packed_for"), frame)
}

/// [`compile_packed_for`] with THIS platform's pixel byte order.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_platform_packed(channels: &[Kernel; 4], frame: [u32; 2]) -> PackedProgram {
    compile_packed_for::<PlatformPixel>(channels, frame)
}

/// The byte lanes `P` packs into.
///
/// # Panics
///
/// Panics for a `P` with no packed RGBA form.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn packed_shifts_of<P: Pixel>(what: &str) -> [u32; 4] {
    P::packed_shifts().unwrap_or_else(|| {
        panic!(
            "{what}: {} has no packed RGBA form; render it through the \
             surface lane instead",
            core::any::type_name::<P>()
        )
    })
}

/// Compile the packed cell-grid program for the pixel format `P`, taking
/// byte order from `P` itself — so the format the kernel packs for is the
/// format the frame stores, by construction. Applications pass geometry and
/// colors; they never see a shift.
///
/// Use [`compile_platform_cell_grid`] when the frame is the platform's own
/// format, which is the production path.
///
/// # Panics
///
/// Panics for a `P` with no packed RGBA form ([`Pixel::packed_shifts`]
/// returns `None`) — a grayscale or exotic format must render through the
/// surface lane, and silently packing RGBA into it would be garbage.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_cell_grid_for<P: Pixel>(
    geom: CellGridGeometry,
    default_bg: [f32; 4],
) -> CellGridPackedProgram {
    CellGridPackedProgram::compile(
        geom,
        default_bg,
        packed_shifts_of::<P>("compile_cell_grid_for"),
    )
}

/// Compile the packed cell-grid program with THIS platform's pixel byte
/// order — [`PlatformColorCube`] and [`PlatformPixel`] are the same choice,
/// pinned to each other below.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_platform_cell_grid(
    geom: CellGridGeometry,
    default_bg: [f32; 4],
) -> CellGridPackedProgram {
    debug_assert_eq!(
        PlatformPixel::packed_shifts(),
        Some(PlatformColorCube::PACKED_SHIFTS),
        "the platform pixel and platform color cube must agree on byte order"
    );
    compile_cell_grid_for::<PlatformPixel>(geom, default_bg)
}

impl core::fmt::Debug for Scene {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Surface(_) => f.debug_tuple("Scene::Surface").finish(),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::Packed(_) => f.debug_tuple("Scene::Packed").finish(),
        }
    }
}

impl Scene {
    /// Render this scene into `frame` with up to `num_threads` workers.
    ///
    /// Packed scenes bake finished pixels stripe-parallel through their one
    /// collapse kernel and copy rows into the frame; surfaces go through the
    /// work-stealing per-batch rasterizer.
    pub fn render<P: Pixel + Send>(&self, frame: &mut Frame<P>, num_threads: usize) {
        match self {
            Self::Surface(manifold) => {
                crate::render::rasterizer::rasterize(manifold, frame, num_threads);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::Packed(packed) => render_packed(packed, frame, num_threads),
        }
    }
}

/// Bake a packed frame, stripe-parallel.
///
/// Each worker owns a contiguous run of rows: one collapse call bakes its
/// stripe's finished pixels (the pixel loop and the pack both live inside
/// the JIT), and each row is copied `width`-wide into the frame.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn render_packed<P: Pixel + Send>(packed: &PackedFrame, frame: &mut Frame<P>, num_threads: usize) {
    let (width, height) = (frame.width, frame.height);
    if width == 0 || height == 0 {
        return;
    }
    // The kernel baked a packed word for the shifts it was compiled with;
    // storing that word as a format with different byte order would swap R
    // and B in every pixel. The old per-pixel `from_rgba` path converted, so
    // any `P` worked; this path moves bits, so the formats must match.
    assert_eq!(
        P::packed_shifts(),
        Some(packed.shifts()),
        "packed frame format does not match the shifts its kernel packed \
         for — compile it with `compile_packed_for::<P>`"
    );
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
            scope.spawn(move || bake_packed_stripe(packed, width, y0, band));
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
fn bake_packed_stripe<P: Pixel>(packed: &PackedFrame, width: usize, y0: usize, band: &mut [P]) {
    let stride = PackedFrame::padded_width(width);
    let chunk_rows = STAGING_SCRATCH_BYTES / (stride * core::mem::size_of::<u32>());
    bake_packed_chunked(packed, width, y0, band, chunk_rows);
}

/// [`bake_packed_stripe`] with an explicit chunk height — split out so tests
/// can force chunk boundaries the production budget would only hit on very
/// large frames.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bake_packed_chunked<P: Pixel>(
    packed: &PackedFrame,
    width: usize,
    y0: usize,
    band: &mut [P],
    chunk_rows: usize,
) {
    let rows = band.len() / width;
    let stride = PackedFrame::padded_width(width);
    let chunk_rows = chunk_rows.clamp(1, rows.max(1));
    let mut staging = vec![0u32; chunk_rows * stride];

    let mut done = 0usize;
    while done < rows {
        let n = chunk_rows.min(rows - done);
        packed.bake_packed_rows(
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
    use crate::render::cell_grid::CellGridPackedProgram;
    use crate::render::color::Rgba8;
    use crate::render::color::RgbaColorCube;

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
            frame_w: 10,
            frame_h: 10,
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
        Scene::Packed(program.frame(Arc::new(cells), Arc::new(atlas)))
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
        let Scene::Packed(grid) = scene() else {
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
            frame_w: 8,
            frame_h: 8,
        };
        let program = super::compile_platform_cell_grid(geom, [0.0, 0.0, 0.0, 1.0]);
        // Zero coverage everywhere, so every in-grid pixel shows its cell's
        // BACKGROUND — which must be opaque for the alpha assertion to hold.
        let mut cells = vec![0.0f32; geom.cells_len()];
        for cell in cells.chunks_mut(pixelflow_core::CELL_STRIDE) {
            cell[9] = 1.0; // bg_a
        }
        let frame_data = program.frame(Arc::new(cells), Arc::new(atlas));
        // The platform wrapper packs for PlatformPixel, so the frame must BE
        // PlatformPixel — pairing it with a hardcoded format is the mistake
        // the render-time format guard exists to catch, and did catch here
        // (this test asserted Rgba8 and failed on Linux, where the platform
        // is Bgra8). Alpha is byte 3 in both orders, so the assertion below
        // is the one claim that holds whichever platform runs it.
        let mut frame = Frame::<crate::render::color::PlatformPixel>::new(8, 8);
        Scene::Packed(frame_data).render(&mut frame, 1);
        assert!(
            frame.data.iter().all(|p| p.0 >> 24 == 255),
            "alpha lane must be opaque regardless of platform byte order"
        );
    }

    /// The shader-author path: four channel kernels and a frame size, with
    /// nothing bound. No geometry, no buffers, no `Manifold`.
    #[test]
    fn four_channel_kernels_and_a_frame_size_are_a_scene() {
        let k = Kernel::constant;
        // r ramps across the frame, g down it, b and a constant.
        let channels = [
            Kernel::x().mul(&k(1.0 / 8.0)),
            Kernel::y().mul(&k(1.0 / 8.0)),
            k(0.0),
            k(1.0),
        ];
        let program = compile_packed_for::<Rgba8>(&channels, [8, 8]);
        let scene = Scene::Packed(program.bind(&[]));
        let mut frame = Frame::<Rgba8>::new(8, 8);
        scene.render(&mut frame, 2);

        for y in 0..8usize {
            for x in 0..8usize {
                let p = frame.data[y * 8 + x];
                let expect = |v: f32| ((v * 255.0).clamp(0.0, 255.0)) as u8;
                assert_eq!(
                    (p.r(), p.g(), p.b(), p.a()),
                    (
                        expect((x as f32 + 0.5) / 8.0),
                        expect((y as f32 + 0.5) / 8.0),
                        0,
                        255
                    ),
                    "pixel ({x}, {y})"
                );
            }
        }
    }

    /// A shader authored in point space targets a denser device frame by
    /// PRECOMPOSITION — `Kernel::at` on the channel kernels, in the language
    /// — not by a wrapper somewhere downstream. The scale is spent at compile
    /// time, so nothing per-frame has to know about it.
    #[test]
    fn a_point_space_shader_reaches_a_denser_frame_by_precomposition() {
        let k = Kernel::constant;
        let red = Kernel::x().mul(&k(1.0 / 16.0));
        let channels = |r: Kernel| [r, k(0.0), k(0.0), k(1.0)];

        // Authored over a 16-point-wide surface...
        let points = compile_packed_for::<Rgba8>(&channels(red.clone()), [16, 16]);
        let mut point_frame = Frame::<Rgba8>::new(16, 16);
        Scene::Packed(points.bind(&[])).render(&mut point_frame, 1);

        // ...and sampled on a 2x device grid: x_point = x_device / 2.
        let half = |v: &Kernel| v.mul(&k(0.5));
        let device = compile_packed_for::<Rgba8>(
            &channels(red.at(
                &half(&Kernel::x()),
                &half(&Kernel::y()),
                &Kernel::z(),
                &Kernel::w(),
            )),
            [32, 32],
        );
        let mut device_frame = Frame::<Rgba8>::new(32, 32);
        Scene::Packed(device.bind(&[])).render(&mut device_frame, 1);

        for x in 0..32usize {
            let expected = (((x as f32 + 0.5) * 0.5 / 16.0) * 255.0) as u8;
            assert_eq!(
                device_frame.data[x].r(),
                expected,
                "device column {x} does not sample point space"
            );
        }
        // The two agree where they sample the same point: device pixel 2i+…
        // straddles point pixel i, so the point frame's value is between the
        // device frame's two.
        for i in 0..16usize {
            let (lo, hi) = (
                device_frame.data[2 * i].r(),
                device_frame.data[2 * i + 1].r(),
            );
            let mid = point_frame.data[i].r();
            assert!(
                lo <= mid && mid <= hi,
                "point column {i} ({mid}) is not between device columns ({lo}, {hi})"
            );
        }
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
    use crate::render::cell_grid::CellGridPackedProgram;
    use crate::render::color::{Rgba8, RgbaColorCube};
    use pixelflow_core::{CellGridGeometry, CellGridProgram};

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
            frame_w: 2560,
            frame_h: 1584,
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

    /// Hot-loop-only target for profilers: nothing but the packed path,
    /// long enough to sample. `cargo test --release ... --ignored
    /// profile_packed` under samply/xctrace.
    #[test]
    #[ignore = "manual profiling target"]
    fn profile_packed_hot_loop() {
        let (geom, cells, atlas) = realistic();
        let (w, h) = (2560usize, 1584usize);
        let stride = PackedFrame::padded_width(w);
        let packed = CellGridPackedProgram::compile(
            geom,
            [0.1, 0.1, 0.1, 1.0],
            RgbaColorCube::PACKED_SHIFTS,
        )
        .frame(Arc::new(cells), Arc::new(atlas));
        let chunk_rows = STAGING_SCRATCH_BYTES / (stride * core::mem::size_of::<u32>());
        let mut band = vec![Rgba8::from_u32(0); w * h];
        for _ in 0..150 {
            bake_packed_chunked(&packed, w, 0, &mut band, chunk_rows);
            std::hint::black_box(&band);
        }
    }

    #[test]
    #[ignore = "manual benchmark; run in --release with --nocapture"]
    fn bench_packed_vs_four_plane() {
        let (geom, cells, atlas) = realistic();
        let (w, h) = (2560usize, 1584usize);
        let stride = PackedFrame::padded_width(w);
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

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod pixel_format_tests {
    use super::*;
    use crate::render::color::{Bgra8, Rgba8};

    /// The kernel packs for one byte order and the frame stores raw words,
    /// so a mismatched frame format would swap R and B in every pixel. The
    /// old per-pixel `from_rgba` path converted; this one must refuse.
    #[test]
    #[should_panic(expected = "does not match the shifts its kernel packed for")]
    fn mismatched_frame_format_is_refused() {
        let (aw, ah) = (12usize, 6usize);
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
            frame_w: 8,
            frame_h: 8,
        };
        // Compiled for Rgba8's lanes...
        let program = compile_cell_grid_for::<Rgba8>(geom, [0.0, 0.0, 0.0, 1.0]);
        let frame_data = program.frame(
            Arc::new(vec![0.0f32; geom.cells_len()]),
            Arc::new(vec![0.0f32; aw * ah]),
        );
        // ...rendered into a Bgra8 frame.
        let mut frame = Frame::<Bgra8>::new(8, 8);
        Scene::Packed(frame_data).render(&mut frame, 1);
    }

    /// Both packed formats work when the kernel is compiled for them.
    #[test]
    fn each_packed_format_renders_through_its_own_shifts() {
        let (aw, ah) = (12usize, 6usize);
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
            frame_w: 8,
            frame_h: 8,
        };
        // Opaque red background everywhere: coverage is zero (empty atlas),
        // so every in-grid pixel is its cell's background.
        let mut cells = vec![0.0f32; geom.cells_len()];
        for cell in cells.chunks_mut(pixelflow_core::CELL_STRIDE) {
            cell[6] = 1.0; // bg_r
            cell[9] = 1.0; // bg_a
        }
        let atlas = Arc::new(vec![0.0f32; aw * ah]);
        let cells = Arc::new(cells);

        let rgba = compile_cell_grid_for::<Rgba8>(geom, [1.0, 0.0, 0.0, 1.0])
            .frame(cells.clone(), atlas.clone());
        let mut rf = Frame::<Rgba8>::new(8, 8);
        Scene::Packed(rgba).render(&mut rf, 1);

        let bgra = compile_cell_grid_for::<Bgra8>(geom, [1.0, 0.0, 0.0, 1.0]).frame(cells, atlas);
        let mut bf = Frame::<Bgra8>::new(8, 8);
        Scene::Packed(bgra).render(&mut bf, 1);

        // Same color through both formats: the raw words differ (byte order),
        // the decoded channels agree.
        assert_ne!(rf.data[0].0, bf.data[0].0, "byte orders must differ");
        assert_eq!((rf.data[0].r(), rf.data[0].b()), (255, 0));
        assert_eq!((bf.data[0].r(), bf.data[0].b()), (255, 0));
    }
}
