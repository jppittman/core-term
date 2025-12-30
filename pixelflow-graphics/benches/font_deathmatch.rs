use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use freetype::Library;
use pixelflow_graphics::{
    render::rasterizer::{
        execute, render_parallel, render_parallel_pooled, RenderOptions, TensorShape, ThreadPool,
    },
    Font, Lift, Rgba8,
};
use std::sync::Arc;

// ============================================================================
// Font Data
// ============================================================================

const FONT_DATA: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_rasterization_deathmatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("font_rasterization_deathmatch");

    // Common parameters
    let char_code = 'g';
    let size_px: u32 = 64;
    let total_pixels = size_px * size_px;

    group.throughput(Throughput::Elements(total_pixels as u64));

    // ------------------------------------------------------------------------
    // Freetype Setup
    // ------------------------------------------------------------------------
    let ft_lib = Library::init().unwrap();
    let ft_face = ft_lib.new_memory_face(FONT_DATA.to_vec(), 0).unwrap();
    ft_face.set_pixel_sizes(size_px, size_px).unwrap();

    group.bench_function(BenchmarkId::new("freetype", "64px_g"), |b| {
        b.iter(|| {
            // Load and render the glyph
            ft_face
                .load_char(char_code as usize, freetype::face::LoadFlag::RENDER)
                .unwrap();
            let glyph = ft_face.glyph();
            let bitmap = glyph.bitmap();
            black_box(bitmap.buffer());
        })
    });

    // ------------------------------------------------------------------------
    // PixelFlow Setup
    // ------------------------------------------------------------------------
    let pf_font = Font::parse(FONT_DATA).unwrap();

    // Pre-allocate buffer for PixelFlow (simulating reuse, Freetype also reuses its internal slot buffer)
    let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); (size_px * size_px) as usize];
    let shape = TensorShape::new(size_px as usize, size_px as usize);

    group.bench_function(BenchmarkId::new("pixelflow", "64px_g"), |b| {
        b.iter(|| {
            let glyph = pf_font
                .glyph_scaled(black_box(char_code), size_px as f32)
                .unwrap();
            let colored = Lift(glyph);
            execute(&colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    // ------------------------------------------------------------------------
    // PixelFlow Setup (Cached Glyph)
    // ------------------------------------------------------------------------
    let cached_glyph = pf_font.glyph_scaled(char_code, size_px as f32).unwrap();
    let cached_colored = Lift(cached_glyph);

    group.bench_function(BenchmarkId::new("pixelflow_cached", "64px_g"), |b| {
        b.iter(|| {
            execute(&cached_colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    // ------------------------------------------------------------------------
    // PixelFlow Setup (Pooled Parallel)
    // ------------------------------------------------------------------------
    // Create pool once (outside measurement loop in real app, here we include init or put outside?)
    // Criterion iter loops many times. We want the pool to persist across iterations.
    // So we must create it outside the b.iter closure.

    let pool_2 = ThreadPool::new(2);
    let pool_4 = ThreadPool::new(4);

    group.bench_function(BenchmarkId::new("pixelflow_pooled_2", "64px_g"), |b| {
        b.iter(|| {
            render_parallel_pooled(&pool_2, &cached_colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    group.bench_function(BenchmarkId::new("pixelflow_pooled_4", "64px_g"), |b| {
        b.iter(|| {
            render_parallel_pooled(&pool_4, &cached_colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    group.finish();
}

fn bench_heavy_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("heavy_workload");

    // 512x512 = 262,144 pixels.
    let size_px = 512;
    let total_pixels = size_px * size_px;
    group.throughput(Throughput::Elements(total_pixels as u64));

    let pf_font = Font::parse(FONT_DATA).unwrap();
    // Render a large '@' which has complex geometry
    let glyph = pf_font.glyph_scaled('@', size_px as f32).unwrap();
    let colored = Lift(glyph);

    let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); (size_px * size_px) as usize];
    let shape = TensorShape::new(size_px as usize, size_px as usize);

    // Serial
    group.bench_function(BenchmarkId::new("pixelflow_serial", "512px_@"), |b| {
        b.iter(|| {
            execute(&colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    // Parallel 2 (Pooled)
    let pool_2 = ThreadPool::new(2);
    group.bench_function(BenchmarkId::new("pixelflow_pooled_2", "512px_@"), |b| {
        b.iter(|| {
            render_parallel_pooled(&pool_2, &colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    // Parallel 4 (Pooled)
    let pool_4 = ThreadPool::new(4);
    group.bench_function(BenchmarkId::new("pixelflow_pooled_4", "512px_@"), |b| {
        b.iter(|| {
            render_parallel_pooled(&pool_4, &colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    // Parallel 8 (Pooled)
    let pool_8 = ThreadPool::new(8);
    group.bench_function(BenchmarkId::new("pixelflow_pooled_8", "512px_@"), |b| {
        b.iter(|| {
            render_parallel_pooled(&pool_8, &colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    group.finish();
}

fn bench_cold_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_start");

    let char_code = 'g';
    let size_px: u32 = 64;

    // ------------------------------------------------------------------------
    // Freetype Cold (Face Creation + Render)
    // ------------------------------------------------------------------------
    let ft_lib = Library::init().unwrap();

    group.bench_function(BenchmarkId::new("freetype", "parse_and_render"), |b| {
        b.iter(|| {
            // Freetype requires keeping the buffer alive or copying it.
            // The Rust wrapper's new_memory_face takes Rc<Vec<u8>> or similar usually,
            // but here we might need to be careful.
            // Using to_vec() mimics loading bytes from a file.
            let face = ft_lib.new_memory_face(FONT_DATA.to_vec(), 0).unwrap();
            face.set_pixel_sizes(size_px, size_px).unwrap();
            face.load_char(char_code as usize, freetype::face::LoadFlag::RENDER)
                .unwrap();
            let glyph = face.glyph();
            let bitmap = glyph.bitmap();
            black_box(bitmap.buffer());
        })
    });

    // ------------------------------------------------------------------------
    // PixelFlow Cold (Font Parse + Compile + Render)
    // ------------------------------------------------------------------------
    let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); (size_px * size_px) as usize];
    let shape = TensorShape::new(size_px as usize, size_px as usize);

    group.bench_function(BenchmarkId::new("pixelflow", "parse_and_render"), |b| {
        b.iter(|| {
            // Font::parse is zero-copy (borrowed)
            let font = Font::parse(black_box(FONT_DATA)).unwrap();
            let glyph = font
                .glyph_scaled(black_box(char_code), size_px as f32)
                .unwrap();
            let colored = Lift(glyph);
            execute(&colored, &mut buffer, shape);
            black_box(&buffer);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rasterization_deathmatch,
    bench_cold_start,
    bench_heavy_workload
);
criterion_main!(benches);
