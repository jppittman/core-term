use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pixelflow_core::{Discrete, Field, ManifoldExt, RgbaComponents};
use pixelflow_graphics::render::{rasterize, Frame, RenderOptions};
use pixelflow_graphics::{Color, NamedColor};
use pixelflow_graphics::render::color::Rgba8;

fn bench_pack_rgba_fields(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_packing");

    let r = Field::from(0.5);
    let g = Field::from(0.5);
    let b = Field::from(0.5);
    let a = Field::from(1.0);

    group.bench_function("pack_rgba_fields", |bencher| {
        bencher.iter(|| {
            black_box(Discrete::pack(RgbaComponents {
                r,
                g,
                b,
                a,
            }))
        })
    });

    group.finish();
}

fn bench_glyph_evaluation(c: &mut Criterion) {
    // This benchmark measures the end-to-end rasterization of a glyph-like shape.
    // It is intended to catch regressions in the Manifold evaluation pipeline.
    let mut group = c.benchmark_group("glyph_evaluation");

    // A simple signed distance field representing a circle (glyph approximation)
    struct CircleSdf;
    impl pixelflow_core::Manifold for CircleSdf {
        type Output = Discrete;
        fn eval_raw(
            &self,
            x: Field,
            y: Field,
            _z: Field,
            _w: Field,
        ) -> Discrete {
            // (x^2 + y^2).sqrt() - 100.0
            // Center at (128, 128), radius 100
            let dx = x - Field::from(128.0);
            let dy = y - Field::from(128.0);
            let dist = (dx * dx + dy * dy).sqrt().constant() - Field::from(100.0);

            // Antialiased edge
            // alpha = smoothstep(-0.5, 0.5, -dist)
            let alpha = (Field::from(0.5) - dist).max(Field::from(0.0)).min(Field::from(1.0));

            // White color
            let gray = alpha; // Premultiplied alpha if we were doing blending, but here just opacity

            Discrete::pack(RgbaComponents {
                r: gray,
                g: gray,
                b: gray,
                a: Field::from(1.0),
            })
        }
    }

    let manifold = CircleSdf;
    let mut frame: Frame<Rgba8> = Frame::new(256, 256);

    group.bench_function("rasterize_circle_256x256_st", |bencher| {
        bencher.iter(|| {
            // Single-threaded rasterization
            rasterize(black_box(&manifold), black_box(&mut frame), 1);
        })
    });

    // Also benchmark parallel rasterization to ensure overhead is low
    group.bench_function("rasterize_circle_256x256_mt_4", |bencher| {
        bencher.iter(|| {
            rasterize(black_box(&manifold), black_box(&mut frame), 4);
        })
    });

    group.finish();
}

fn bench_complex_manifold(c: &mut Criterion) {
    // Benchmarks a deeper expression tree
    let mut group = c.benchmark_group("complex_manifold");

    use pixelflow_core::{ManifoldExt, X, Y};

    // A pattern of repeating circles: sin(x/10) * cos(y/10) > 0
    let pattern = (X / 10.0).sin() * (Y / 10.0).cos();
    // Color logic: if pattern > 0 then Red else Blue
    let color_manifold = pattern.gt(0.0).select(
        Color::Named(NamedColor::Red),
        Color::Named(NamedColor::Blue),
    );

    let mut frame: Frame<Rgba8> = Frame::new(512, 512);

    group.bench_function("rasterize_pattern_512x512_mt_8", |bencher| {
        bencher.iter(|| {
            rasterize(black_box(&color_manifold), black_box(&mut frame), 8);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pack_rgba_fields,
    bench_glyph_evaluation,
    bench_complex_manifold
);
criterion_main!(benches);
