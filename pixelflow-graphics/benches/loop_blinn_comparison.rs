use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::{
    fonts::ttf::{loop_blinn_quad, Curve},
    render::{rasterizer::{rasterize, RenderOptions}, frame::Frame},
    Font, Grayscale, Rgba8,
};

// Font containing glyphs with complex curves
const FONT_DATA: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

fn bench_raw_quad_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_quad_evaluation");
    group.sample_size(200);

    // Create a test quadratic curve
    let control_points: [[f32; 2]; 3] = [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]];

    let standard_quad = Curve::<3>(control_points);
    let lb_quad = loop_blinn_quad(control_points);

    // Test evaluation over a grid
    let size = 64;
    group.throughput(Throughput::Elements((size * size) as u64));

    // Benchmark standard quad
    group.bench_function("standard_quad_raw", |b| {
        b.iter(|| {
            for y in 0..size {
                for x in 0..size {
                    let xf = Field::from(x as f32 / size as f32);
                    let yf = Field::from(y as f32 / size as f32);
                    let result = standard_quad.eval_raw(xf, yf, Field::from(0.0), Field::from(0.0));
                    black_box(result);
                }
            }
        })
    });

    // Benchmark Loop-Blinn quad
    group.bench_function("loop_blinn_quad_raw", |b| {
        b.iter(|| {
            for y in 0..size {
                for x in 0..size {
                    let xf = Field::from(x as f32 / size as f32);
                    let yf = Field::from(y as f32 / size as f32);
                    let result = lb_quad.eval_raw(xf, yf, Field::from(0.0), Field::from(0.0));
                    black_box(result);
                }
            }
        })
    });

    group.finish();
}

fn bench_geometry_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometry_comparison");
    group.sample_size(100);

    let pf_font = Font::parse(FONT_DATA).unwrap();

    // Test characters with varying curve complexity
    let test_cases = [
        ('g', "simple_curves"),
        ('@', "complex_curves"),
        ('S', "s_curves"),
        ('&', "ampersand"),
    ];

    for (char_code, desc) in test_cases.iter() {
        let size_px = 64u32;
        let total_pixels = size_px * size_px;
        group.throughput(Throughput::Elements(total_pixels as u64));

        let glyph = pf_font.glyph_scaled(*char_code, size_px as f32).unwrap();

        let mut frame = Frame::<Rgba8>::new(size_px, size_px);

        let bench_name = format!("standard/{}", desc);
        group.bench_function(&bench_name, |b| {
            b.iter(|| {
                let colored = Grayscale(glyph.clone());
                rasterize(&colored, &mut frame, RenderOptions { num_threads: 1 });
                black_box(&frame);
            })
        });
    }

    group.finish();
}

fn bench_full_glyph_rendering(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_glyph_rendering");
    group.sample_size(50);

    let pf_font = Font::parse(FONT_DATA).unwrap();

    // Benchmark at different sizes to see if performance scales differently
    let sizes = [32u32, 64u32, 128u32];

    for size_px in sizes.iter() {
        let total_pixels = size_px * size_px;
        group.throughput(Throughput::Elements(total_pixels as u64));

        let glyph = pf_font.glyph_scaled('@', *size_px as f32).unwrap();
        let mut frame = Frame::<Rgba8>::new(*size_px, *size_px);

        let bench_name = format!("standard_{}px", size_px);
        group.bench_function(&bench_name, |b| {
            b.iter(|| {
                let colored = Grayscale(glyph.clone());
                rasterize(&colored, &mut frame, RenderOptions { num_threads: 1 });
                black_box(&frame);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_raw_quad_evaluation,
    bench_geometry_comparison,
    bench_full_glyph_rendering
);
criterion_main!(benches);
