//! Throughput benchmark for the core rasterizer loop.
//!
//! Measures performance of `execute_stripe` on a trivial manifold (solid color)
//! to isolate the overhead of the rasterizer loop itself (SIMD dispatch, memory writes).
//!
//! Run with: `cargo bench -p pixelflow-graphics --bench rasterizer_throughput`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pixelflow_graphics::render::color::{Color, NamedColor, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

fn bench_rasterizer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterizer_throughput");

    // Test at 1080p resolution
    let width = 1920;
    let height = 1080;
    let pixels = (width * height) as u64;

    group.throughput(Throughput::Elements(pixels));

    // Simple solid color manifold - practically zero evaluation cost
    // This forces the benchmark to measure the rasterizer loop overhead
    let color = Color::Named(NamedColor::Red); // Should be a constant manifold

    // Reuse frame buffer to avoid allocation in loop
    let mut frame = Frame::<Rgba8>::new(width, height);

    group.bench_function("solid_color_1080p", |b| {
        b.iter(|| {
            // Single-threaded rasterization
            rasterize(black_box(&color), black_box(&mut frame), 1);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_rasterizer_throughput);
criterion_main!(benches);
