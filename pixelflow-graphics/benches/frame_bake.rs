//! Packed cell-grid frame baking: the retired four-plane-then-pack shape
//! against the packed kernel, plus a sustained hot-loop target for external
//! sampling profilers (samply/xctrace).
//!
//! Postsubmit-only (docs/POSTSUBMIT.md): only ever invoked via
//! `cargo bench --workspace --benches`, never presubmit. Moved out of
//! `pixelflow-graphics/src/render/scene.rs`'s `#[cfg(test)] mod frame_bench`
//! (previously `#[ignore]`d `#[test]`s) so "never runs during a regular test
//! pass" is structural rather than resting on the attribute not getting
//! dropped — see `work_stealing.rs` for the same move applied to
//! `tests/scene3d_test.rs::work_stealing_benchmark`.
//!
//! `bake_packed_chunked` and `STAGING_SCRATCH_BYTES` below mirror private
//! helpers of the same name in `pixelflow-graphics::render::scene` — a bench
//! target compiles as an external crate and can only see `pub` items, and
//! those two are deliberately not public (minimal API). Both are thin: a
//! staging-buffer chunking loop over `pixelflow_core`'s public
//! `CellGridPackedFrame::bake_packed_rows`, so reproducing them here measures
//! the same kernel work without widening the crate's surface.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pixelflow_core::{
    CellGridFrame, CellGridGeometry, CellGridPackedFrame, CellGridPackedProgram, CellGridProgram,
    PlaneRegion,
};
use pixelflow_graphics::render::color::{Rgba8, RgbaColorCube};
use pixelflow_graphics::render::Pixel;
use std::sync::Arc;

/// A 2560x1584 frame of 12x24 cells: 213x66 — a full-screen terminal.
fn realistic() -> (CellGridGeometry, Vec<f32>, Vec<f32>) {
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

/// Staging budget per chunk, mirroring `render::scene`'s private
/// `STAGING_SCRATCH_BYTES`.
const STAGING_SCRATCH_BYTES: usize = 1 << 20;

/// Mirrors `render::scene`'s private `bake_packed_chunked`: stage
/// `chunk_rows` at a time into one reused buffer via the public
/// `bake_packed_rows`, then copy each row out.
fn bake_packed_chunked(
    grid: &CellGridPackedFrame,
    width: usize,
    y0: usize,
    band: &mut [Rgba8],
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
                *d = Rgba8::from_u32(*s);
            }
        }
        done += n;
    }
}

/// Hot-loop-only target for external sampling profilers: nothing but the
/// packed path, long enough to sample.
fn bench_packed_hot_loop(c: &mut Criterion) {
    let (geom, cells, atlas) = realistic();
    let (w, h) = (2560usize, 1584usize);
    let stride = CellGridFrame::padded_width(w);
    let packed =
        CellGridPackedProgram::compile(geom, [0.1, 0.1, 0.1, 1.0], RgbaColorCube::PACKED_SHIFTS)
            .frame(Arc::new(cells), Arc::new(atlas));
    let chunk_rows = STAGING_SCRATCH_BYTES / (stride * core::mem::size_of::<u32>());
    let mut band = vec![Rgba8::from_u32(0); w * h];

    c.bench_function("packed_kernel_hot_loop", |b| {
        b.iter(|| {
            bake_packed_chunked(&packed, w, 0, &mut band, chunk_rows);
            black_box(&band);
        });
    });
}

/// Steady-state frame-bake comparison: the retired four-plane-then-pack
/// shape against the packed kernel. The four-channel program still exists in
/// pixelflow-core as the parity oracle, which is what lets the retired shape
/// be reconstructed here without keeping dead render code.
fn bench_packed_vs_four_plane(c: &mut Criterion) {
    let (geom, cells, atlas) = realistic();
    let (w, h) = (2560usize, 1584usize);
    let stride = CellGridFrame::padded_width(w);
    let cells = Arc::new(cells);
    let atlas = Arc::new(atlas);

    let four =
        CellGridProgram::compile(geom, [0.1, 0.1, 0.1, 1.0]).frame(cells.clone(), atlas.clone());
    let packed =
        CellGridPackedProgram::compile(geom, [0.1, 0.1, 0.1, 1.0], RgbaColorCube::PACKED_SHIFTS)
            .frame(cells, atlas);

    let chunk_rows = STAGING_SCRATCH_BYTES / (stride * core::mem::size_of::<u32>());
    let mut band = vec![Rgba8::from_u32(0); w * h];

    let mut group = c.benchmark_group("packed_vs_four_plane_2560x1584");
    // One full-frame bake per sample is already substantial work.
    group.sample_size(10);

    group.bench_function("four_plane_per_pixel_pack", |b| {
        b.iter(|| {
            let mut planes = vec![0.0f32; 4 * chunk_rows * stride];
            let mut done = 0usize;
            while done < h {
                let n = chunk_rows.min(h - done);
                {
                    let (r, rest) = planes.split_at_mut(chunk_rows * stride);
                    let (g, rest) = rest.split_at_mut(chunk_rows * stride);
                    let (blue, a) = rest.split_at_mut(chunk_rows * stride);
                    let region = PlaneRegion {
                        width: w,
                        y0: done,
                        rows: n,
                    };
                    four.bake_channel_rows(0, region, r);
                    four.bake_channel_rows(1, region, g);
                    four.bake_channel_rows(2, region, blue);
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
            black_box(&band);
        });
    });

    group.bench_function("packed_kernel_row_copy", |b| {
        b.iter(|| {
            bake_packed_chunked(&packed, w, 0, &mut band, chunk_rows);
            black_box(&band);
        });
    });

    group.finish();
}

criterion_group!(
    frame_bake_benches,
    bench_packed_hot_loop,
    bench_packed_vs_four_plane
);
criterion_main!(frame_bake_benches);
