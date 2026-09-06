//! Collapsing a cell-grid frame: the retired four-plane-then-pack shape
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
//! Nothing private is mirrored here any more. The packed lane collapses
//! straight into the destination through the public
//! `PackedFrame::collapse_rows`, which is exactly what `render::scene`'s
//! stripe loop calls — one collapse call per band, no staging plane and no
//! row copy — so the bench measures the shipped path rather than a
//! reconstruction of it.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pixelflow_core::{CellGridGeometry, CellGridProgram, PlaneRegion};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::scene::compile_cell_grid_for;
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

/// Hot-loop-only target for external sampling profilers: nothing but the
/// packed path, long enough to sample.
fn bench_packed_hot_loop(c: &mut Criterion) {
    let (geom, cells, atlas) = realistic();
    let (w, h) = (2560usize, 1584usize);
    let packed = compile_cell_grid_for::<Rgba8>(geom, [0.1, 0.1, 0.1, 1.0])
        .frame(Arc::new(cells), Arc::new(atlas));
    let region = PlaneRegion::rows(w, 0, h);
    let mut band = vec![0u32; w * h];

    c.bench_function("packed_kernel_hot_loop", |b| {
        b.iter(|| {
            packed.collapse_rows(region, &mut band, w);
            black_box(&band);
        });
    });
}

/// Steady-state whole-frame comparison: the retired four-plane-then-pack
/// shape against the packed kernel. The four-channel program still exists in
/// pixelflow-core as the parity oracle, which is what lets the retired shape
/// be reconstructed here without keeping dead render code.
fn bench_packed_vs_four_plane(c: &mut Criterion) {
    let (geom, cells, atlas) = realistic();
    let (w, h) = (2560usize, 1584usize);
    let cells = Arc::new(cells);
    let atlas = Arc::new(atlas);

    let four =
        CellGridProgram::compile(geom, [0.1, 0.1, 0.1, 1.0]).frame(cells.clone(), atlas.clone());
    let packed = compile_cell_grid_for::<Rgba8>(geom, [0.1, 0.1, 0.1, 1.0]).frame(cells, atlas);

    // The retired shape staged a megabyte of scratch per plane; keep that
    // band height so what is being priced is the shape, not a new tuning.
    let chunk_rows = (1 << 20) / (w * core::mem::size_of::<f32>());
    let mut band = vec![0u32; w * h];

    let mut group = c.benchmark_group("packed_vs_four_plane_2560x1584");
    // One full frame per sample is already substantial work.
    group.sample_size(10);

    group.bench_function("four_plane_per_pixel_pack", |b| {
        b.iter(|| {
            let mut planes = vec![0.0f32; 4 * chunk_rows * w];
            let mut done = 0usize;
            while done < h {
                let n = chunk_rows.min(h - done);
                {
                    let (r, rest) = planes.split_at_mut(chunk_rows * w);
                    let (g, rest) = rest.split_at_mut(chunk_rows * w);
                    let (blue, a) = rest.split_at_mut(chunk_rows * w);
                    let region = PlaneRegion::rows(w, done, n);
                    four.collapse_channel_rows(0, region, r, w);
                    four.collapse_channel_rows(1, region, g, w);
                    four.collapse_channel_rows(2, region, blue, w);
                    four.collapse_channel_rows(3, region, a, w);
                }
                let plane = |c: usize| &planes[c * chunk_rows * w..];
                for row in 0..n {
                    let p = row * w;
                    let o = (done + row) * w;
                    for i in 0..w {
                        band[o + i] = Rgba8::from_rgba(
                            plane(0)[p + i],
                            plane(1)[p + i],
                            plane(2)[p + i],
                            plane(3)[p + i],
                        )
                        .to_u32();
                    }
                }
                done += n;
            }
            black_box(&band);
        });
    });

    group.bench_function("packed_kernel_direct_write", |b| {
        let region = PlaneRegion::rows(w, 0, h);
        b.iter(|| {
            packed.collapse_rows(region, &mut band, w);
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
