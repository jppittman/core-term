use criterion::{Criterion, criterion_group, criterion_main, black_box};
use pixelflow_fonts::{Font, glyphs};
use pixelflow_core::ops::Baked;
use pixelflow_core::pipe::Surface;
use pixelflow_core::batch::Batch;
use std::fs;
use std::path::Path;

// Adjust path as needed. Since we are in pixelflow-fonts, and run cargo bench there.
const FONT_PATH: &str = "../pixelflow-render/assets/NotoSansMono-Regular.ttf";

/// Benchmark Analysis:
///
/// 1. **Rasterization (Baking)**: ~500µs per glyph (for 'g' at 24px, ~38 segments).
///    - Bottleneck: Iterating all segments for every pixel.
///    - Throughput: ~2000 glyphs/sec (single thread).
///
/// 2. **Cached Sampling**: ~50ns per batch of 4 pixels.
///    - Very fast (memory bound).
///
/// 3. **Uncached Curve Evaluation**: ~8µs per batch of 4 pixels.
///    - This is ~2µs per pixel.
///    - For 'g' (38 segments), this means ~50ns per segment per pixel.
///    - Bottleneck: O(N_segments) evaluation per pixel.
///    - Comparison: 160x slower than cached sampling.
///
/// 4. **Comparisons (Ballpark)**:
///    - Freetype/Harfbuzz: Highly optimized C rasterizers are typically faster (scanline sweep).
///    - PixelFlow targets GPU-like massive parallelism potential, but on CPU this algebraic approach
///      is slower than active-edge-list rasterizers for single-threaded usage.
///    - However, 0.5ms initial bake time is acceptable for a terminal emulator where glyphs are cached.

fn load_font_bytes() -> Vec<u8> {
    let path = Path::new(FONT_PATH);
    if !path.exists() {
        panic!("Font file not found at: {:?}. Please run from pixelflow-fonts directory.", path);
    }
    fs::read(path).expect("Failed to load font file")
}

fn bench_font_parsing(c: &mut Criterion) {
    let bytes = load_font_bytes();
    c.bench_function("font_parsing", |b| {
        b.iter(|| {
            black_box(Font::from_bytes(black_box(&bytes))).unwrap();
        })
    });
}

fn bench_glyph_outline(c: &mut Criterion) {
    let bytes = load_font_bytes();
    let font = Font::from_bytes(&bytes).unwrap();
    let size = 24.0;

    c.bench_function("glyph_outline_generation", |b| {
        b.iter(|| {
            black_box(font.glyph(black_box('g'), black_box(size))).unwrap();
        })
    });
}

fn bench_rasterization(c: &mut Criterion) {
    let bytes = load_font_bytes();
    let font = Font::from_bytes(&bytes).unwrap();
    let size = 24.0;
    // 'g' is usually complex enough (curves + loops)
    let glyph = font.glyph('g', size).unwrap();
    let w = glyph.bounds().width;
    let h = glyph.bounds().height;

    c.bench_function("rasterization_bake", |b| {
        b.iter(|| {
            black_box(Baked::new(black_box(&glyph), black_box(w), black_box(h)));
        })
    });
}

fn bench_sampling(c: &mut Criterion) {
    let bytes = load_font_bytes();
    let font = Font::from_bytes(&bytes).unwrap();
    let size = 24.0;
    let glyph = font.glyph('g', size).unwrap();
    let w = glyph.bounds().width;
    let h = glyph.bounds().height;
    let baked = Baked::new(&glyph, w, h);

    // Coordinate batch for sampling
    let x = Batch::<u32>::new(w/2, w/2+1, w/2+2, w/2+3);
    let y = Batch::<u32>::splat(h/2);

    let mut group = c.benchmark_group("sampling");

    group.bench_function("cached_glyph", |b| {
        b.iter(|| {
            black_box(baked.eval(black_box(x), black_box(y)));
        })
    });

    group.bench_function("uncached_glyph_curves", |b| {
        b.iter(|| {
            black_box(glyph.eval(black_box(x), black_box(y)));
        })
    });

    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let bytes = load_font_bytes();
    let font = Font::from_bytes(&bytes).unwrap();
    let size = 24;
    // Pre-create the factory
    let factory = glyphs(font.clone(), size as u32, size as u32);

    // Simulate rendering "Hello World"
    let text = "Hello World";

    // warm up
    for ch in text.chars() {
        let _ = factory(ch).get();
    }

    c.bench_function("resolve_glyphs_lazy_hot", |b| {
        b.iter(|| {
            for ch in text.chars() {
                let lazy = factory(ch);
                black_box(lazy.get());
            }
        })
    });
}

criterion_group!(
    benches,
    bench_font_parsing,
    bench_glyph_outline,
    bench_rasterization,
    bench_sampling,
    bench_end_to_end
);
criterion_main!(benches);
