//! Comprehensive benchmarks for pixelflow-graphics
//!
//! Tests font parsing, glyph rendering, rasterization, and color operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet2;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldExt, PARALLELISM};
use pixelflow_graphics::{
    render::rasterizer::{rasterize, RenderOptions, TensorShape},
    CachedGlyph, CachedText, Color, ColorCube, Font, GlyphCache, Grayscale, NamedColor, Rgba8,
};

// ============================================================================
// Font Data
// ============================================================================

const FONT_DATA: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

// ============================================================================
// Font Parsing Benchmarks
// ============================================================================

fn bench_font_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("font_parsing");

    group.bench_function("parse_font", |bencher| {
        bencher.iter(|| black_box(Font::parse(black_box(FONT_DATA)).unwrap()))
    });

    group.finish();
}

fn bench_cmap_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmap_lookup");

    let font = Font::parse(FONT_DATA).unwrap();

    // Test ASCII characters
    group.bench_function("lookup_ascii_A", |bencher| {
        bencher.iter(|| black_box(font.glyph(black_box('A'))))
    });

    group.bench_function("lookup_ascii_lowercase", |bencher| {
        bencher.iter(|| {
            for c in 'a'..='z' {
                black_box(font.glyph(c));
            }
        })
    });

    group.bench_function("lookup_digits", |bencher| {
        bencher.iter(|| {
            for c in '0'..='9' {
                black_box(font.glyph(c));
            }
        })
    });

    // Unicode
    group.bench_function("lookup_unicode_emoji", |bencher| {
        bencher.iter(|| black_box(font.glyph(black_box('★'))))
    });

    group.finish();
}

fn bench_glyph_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("glyph_compilation");

    let font = Font::parse(FONT_DATA).unwrap();

    group.bench_function("compile_simple_glyph_A", |bencher| {
        bencher.iter(|| black_box(font.glyph(black_box('A'))))
    });

    group.bench_function("compile_complex_glyph_@", |bencher| {
        bencher.iter(|| black_box(font.glyph(black_box('@'))))
    });

    group.bench_function("compile_and_scale_A", |bencher| {
        bencher.iter(|| black_box(font.glyph_scaled(black_box('A'), 24.0)))
    });

    group.bench_function("compile_full_alphabet", |bencher| {
        bencher.iter(|| {
            for c in 'A'..='Z' {
                black_box(font.glyph(c));
            }
        })
    });

    group.finish();
}

fn bench_advance_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("advance_width");

    let font = Font::parse(FONT_DATA).unwrap();

    group.bench_function("advance_single", |bencher| {
        bencher.iter(|| black_box(font.advance(black_box('M'))))
    });

    group.bench_function("advance_scaled_single", |bencher| {
        bencher.iter(|| black_box(font.advance_scaled(black_box('M'), 16.0)))
    });

    group.bench_function("advance_string", |bencher| {
        let text = "Hello, World!";
        bencher.iter(|| {
            let mut total = 0.0f32;
            for c in text.chars() {
                total += font.advance_scaled(c, 16.0).unwrap_or(0.0);
            }
            black_box(total)
        })
    });

    group.finish();
}

// ============================================================================
// Glyph Evaluation Benchmarks
// ============================================================================

fn bench_glyph_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("glyph_evaluation");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let font = Font::parse(FONT_DATA).unwrap();
    let glyph_a = font.glyph_scaled('A', 64.0).unwrap();
    let glyph_at = font.glyph_scaled('@', 64.0).unwrap();

    let x = Field::sequential(32.0);
    let y = Field::from(32.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("eval_glyph_A_single_point", |bencher| {
        bencher.iter(|| black_box(glyph_a.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("eval_glyph_@_single_point", |bencher| {
        bencher.iter(|| black_box(glyph_at.eval_raw(black_box(x), black_box(y), z, w)))
    });

    // Add AA benchmark using Jet2
    group.bench_function("eval_glyph_A_AA_single_point", |bencher| {
        use pixelflow_core::jet::Jet2;
        let xj = Jet2::x(x);
        let yj = Jet2::y(y);
        let zj = Jet2::constant(z);
        let wj = Jet2::constant(w);
        bencher.iter(|| black_box(glyph_a.eval_raw(black_box(xj), black_box(yj), zj, wj)))
    });

    // Evaluate across a small grid
    group.bench_function("eval_glyph_A_8x8_grid", |bencher| {
        bencher.iter(|| {
            for row in 0..8 {
                for col_batch in 0..(8 / PARALLELISM).max(1) {
                    let fx = (col_batch * PARALLELISM) as f32;
                    let fy = row as f32;
                    black_box(glyph_a.eval_raw(Field::sequential(fx), Field::from(fy), z, w));
                }
            }
        })
    });

    group.finish();
}

// ============================================================================
// Rasterization Benchmarks
// ============================================================================

fn bench_rasterize_solid_color(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterize_solid");

    for size in [16, 64, 256, 512].iter() {
        let total_pixels = size * size;
        group.throughput(Throughput::Elements(total_pixels as u64));

        group.bench_with_input(
            BenchmarkId::new("solid_red", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                let color = NamedColor::Red;
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&color, &mut buffer, shape, RenderOptions::default());
                    black_box(&buffer);
                })
            },
        );
    }

    group.finish();
}

fn bench_rasterize_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterize_gradient");

    for size in [64, 256, 512].iter() {
        let total_pixels = size * size;
        group.throughput(Throughput::Elements(total_pixels as u64));

        group.bench_with_input(
            BenchmarkId::new("horizontal_gradient", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                use pixelflow_core::X;
                let gradient = At {
                    inner: ColorCube,
                    x: X / (size as f32),
                    y: 0.5f32,
                    z: 0.5f32,
                    w: 1.0f32,
                };
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&gradient, &mut buffer, shape, RenderOptions::default());
                    black_box(&buffer);
                })
            },
        );
    }

    group.finish();
}

fn bench_rasterize_circle(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterize_circle");

    for size in [64, 256, 512].iter() {
        let total_pixels = size * size;
        group.throughput(Throughput::Elements(total_pixels as u64));

        group.bench_with_input(
            BenchmarkId::new("circle_sdf", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                use pixelflow_core::{ManifoldExt, X, Y};
                let cx = size as f32 / 2.0;
                let cy = size as f32 / 2.0;
                let r = size as f32 / 3.0;

                // Circle SDF: inside = white, outside = black
                let dx = X - cx;
                let dy = Y - cy;
                let inside = (dx * dx + dy * dy).lt(r * r);
                let circle = Grayscale(inside.select(1.0f32, 0.0f32));

                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&circle, &mut buffer, shape, RenderOptions::default());
                    black_box(&buffer);
                })
            },
        );
    }

    group.finish();
}

fn bench_rasterize_glyph(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterize_glyph");

    let font = Font::parse(FONT_DATA).unwrap();

    for size in [32, 64, 128].iter() {
        let total_pixels = size * size;
        group.throughput(Throughput::Elements(total_pixels as u64));

        group.bench_with_input(
            BenchmarkId::new("glyph_A", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                let glyph = font.glyph_scaled('A', size as f32).unwrap();
                let colored = Grayscale(glyph);
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&colored, &mut buffer, shape, RenderOptions::default());
                    black_box(&buffer);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("glyph_@", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                let glyph = font.glyph_scaled('@', size as f32).unwrap();
                let colored = Grayscale(glyph);
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&colored, &mut buffer, shape, RenderOptions::default());
                    rasterize(&colored, &mut buffer, shape, RenderOptions::default());
                    black_box(&buffer);
                })
            },
        );
    }

    group.finish();
}

fn bench_rasterize_glyph_aa(c: &mut Criterion) {
    use pixelflow_graphics::render::aa::aa_coverage;

    let mut group = c.benchmark_group("rasterize_glyph_aa");

    let font = Font::parse(FONT_DATA).unwrap();

    for size in [32, 64, 128].iter() {
        let total_pixels = size * size;
        group.throughput(Throughput::Elements(total_pixels as u64));

        // Anti-aliased glyph using Jet2 via aa_coverage wrapper
        group.bench_with_input(
            BenchmarkId::new("glyph_A_aa", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                let glyph = font.glyph_scaled('A', size as f32).unwrap();
                let aa_glyph = aa_coverage(glyph);
                let colored = Grayscale(aa_glyph);
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&colored, &mut buffer, shape, RenderOptions::default());
                    rasterize(&colored, &mut buffer, shape, RenderOptions::default());
                    black_box(&buffer);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("glyph_@_aa", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                let glyph = font.glyph_scaled('@', size as f32).unwrap();
                let aa_glyph = aa_coverage(glyph);
                let colored = Grayscale(aa_glyph);
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&colored, &mut buffer, shape, RenderOptions::default());
                    black_box(&buffer);
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Color Benchmarks
// ============================================================================

fn bench_color_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_conversion");

    group.bench_function("named_to_rgb", |bencher| {
        bencher.iter(|| black_box(NamedColor::BrightCyan.to_rgb()))
    });

    group.bench_function("color_to_rgba8", |bencher| {
        let color = Color::Rgb(128, 64, 255);
        bencher.iter(|| black_box(black_box(color).to_rgba8()))
    });

    group.bench_function("indexed_color_cube", |bencher| {
        let color = Color::Indexed(150); // In the 6x6x6 cube
        bencher.iter(|| black_box(black_box(color).to_rgba8()))
    });

    group.bench_function("indexed_grayscale", |bencher| {
        let color = Color::Indexed(240); // Grayscale ramp
        bencher.iter(|| black_box(black_box(color).to_rgba8()))
    });

    group.finish();
}

fn bench_discrete_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("discrete_pack");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    group.bench_function("pack_rgba_fields", |bencher| {
        let r = Field::from(0.5);
        let g = Field::from(0.25);
        let b = Field::from(0.75);
        let a = Field::from(1.0);

        bencher.iter(|| {
            black_box(Discrete::pack(
                black_box(r),
                black_box(g),
                black_box(b),
                black_box(a),
            ))
        })
    });

    group.bench_function("pack_sequential_fields", |bencher| {
        let r = (Field::sequential(0.0) / Field::from(PARALLELISM as f32)).constant();
        let g = Field::from(0.5);
        let b = Field::from(0.5);
        let a = Field::from(1.0);

        bencher.iter(|| black_box(Discrete::pack(r, g, b, a)))
    });

    group.finish();
}

fn bench_color_manifold(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_manifold");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(50.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("eval_named_color", |bencher| {
        let color = NamedColor::Green;
        bencher.iter(|| black_box(color.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("eval_color_cube_constant", |bencher| {
        let color = At {
            inner: ColorCube,
            x: 1.0f32,
            y: 0.5f32,
            z: 0.0f32,
            w: 1.0f32,
        };
        bencher.iter(|| black_box(color.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("eval_color_cube_gradient", |bencher| {
        use pixelflow_core::X;
        let color = At {
            inner: ColorCube,
            x: X / 100.0f32,
            y: 0.5f32,
            z: 0.5f32,
            w: 1.0f32,
        };
        bencher.iter(|| black_box(color.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("eval_grayscale", |bencher| {
        let gray = Grayscale(0.75f32);
        bencher.iter(|| black_box(gray.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

// ============================================================================
// Shape Benchmarks
// ============================================================================

fn bench_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("shapes");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(0.5);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("circle_unit", |bencher| {
        use pixelflow_graphics::shapes::{circle, EMPTY, SOLID};
        let c = circle(SOLID, EMPTY);
        bencher.iter(|| black_box(c.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("square_unit", |bencher| {
        use pixelflow_graphics::shapes::{square, EMPTY, SOLID};
        let s = square(SOLID, EMPTY);
        bencher.iter(|| black_box(s.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("half_plane_x", |bencher| {
        use pixelflow_graphics::shapes::{half_plane_x, EMPTY, SOLID};
        let h = half_plane_x(SOLID, EMPTY);
        bencher.iter(|| black_box(h.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("circle_inside_test", |bencher| {
        // Circle inside test using manifold composition
        use pixelflow_core::{ManifoldExt, X, Y};
        let m = (X * X + Y * Y).lt(1.0f32).select(1.0f32, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

// ============================================================================
// Font Caching Benchmarks
// ============================================================================

fn bench_glyph_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("glyph_cache");

    let font = Font::parse(FONT_DATA).unwrap();

    // Benchmark cache warming
    group.bench_function("warm_ascii_16px", |bencher| {
        bencher.iter(|| {
            let mut cache = GlyphCache::new();
            cache.warm_ascii(&font, 16.0);
            black_box(cache.len())
        })
    });

    group.bench_function("warm_ascii_32px", |bencher| {
        bencher.iter(|| {
            let mut cache = GlyphCache::new();
            cache.warm_ascii(&font, 32.0);
            black_box(cache.len())
        })
    });

    // Benchmark cache hit vs miss
    group.bench_function("cache_hit", |bencher| {
        let mut cache = GlyphCache::new();
        cache.warm_ascii(&font, 16.0);

        bencher.iter(|| {
            black_box(cache.get(&font, 'A', 16.0));
        })
    });

    group.bench_function("cache_miss", |bencher| {
        bencher.iter(|| {
            let mut cache = GlyphCache::new();
            black_box(cache.get(&font, 'A', 16.0));
        })
    });

    group.finish();
}

fn bench_cached_vs_uncached_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_vs_uncached_eval");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let font = Font::parse(FONT_DATA).unwrap();

    let x = Field::sequential(16.0);
    let y = Field::from(16.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    // Uncached glyph evaluation
    let uncached = font.glyph_scaled('A', 32.0).unwrap();

    group.bench_function("uncached_glyph_A_32px", |bencher| {
        bencher.iter(|| black_box(uncached.eval_raw(black_box(x), y, z, w)))
    });

    // Cached glyph evaluation
    let cached = CachedGlyph::new(&uncached, 32);

    group.bench_function("cached_glyph_A_32px", |bencher| {
        bencher.iter(|| black_box(cached.eval_raw(black_box(x), y, z, w)))
    });

    // Complex glyph (@)
    let uncached_at = font.glyph_scaled('@', 32.0).unwrap();
    let cached_at = CachedGlyph::new(&uncached_at, 32);

    group.bench_function("uncached_glyph_@_32px", |bencher| {
        bencher.iter(|| black_box(uncached_at.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("cached_glyph_@_32px", |bencher| {
        bencher.iter(|| black_box(cached_at.eval_raw(black_box(x), y, z, w)))
    });

    // AA (Jet2) evaluation - anti-aliased glyphs
    let x_jet = Jet2::x(x);
    let y_jet = Jet2::y(y);
    let z_jet = Jet2::constant(z);
    let w_jet = Jet2::constant(w);

    group.bench_function("aa_glyph_A_32px", |bencher| {
        bencher.iter(|| black_box(uncached.eval_raw(black_box(x_jet), y_jet, z_jet, w_jet)))
    });

    group.bench_function("aa_glyph_@_32px", |bencher| {
        bencher.iter(|| black_box(uncached_at.eval_raw(black_box(x_jet), y_jet, z_jet, w_jet)))
    });

    group.finish();
}

fn bench_cached_vs_uncached_raster(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_vs_uncached_raster");

    let font = Font::parse(FONT_DATA).unwrap();

    for size in [32, 64, 128].iter() {
        let total_pixels = size * size;
        group.throughput(Throughput::Elements(total_pixels as u64));

        // Uncached rasterization
        group.bench_with_input(
            BenchmarkId::new("uncached_glyph_A", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                let glyph = font.glyph_scaled('A', size as f32).unwrap();
                let colored = Grayscale(glyph);
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&colored, &mut buffer, shape, 1);
                    black_box(&buffer);
                })
            },
        );

        // Cached rasterization (sampling from texture)
        group.bench_with_input(
            BenchmarkId::new("cached_glyph_A", format!("{}x{}", size, size)),
            size,
            |bencher, &size| {
                let glyph = font.glyph_scaled('A', size as f32).unwrap();
                let cached = CachedGlyph::new(&glyph, size);
                let colored = Grayscale(cached);
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); size * size];
                let shape = TensorShape::new(size, size);

                bencher.iter(|| {
                    rasterize(&colored, &mut buffer, shape, 1);
                    black_box(&buffer);
                })
            },
        );
    }

    group.finish();
}

fn bench_cached_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_text");

    let font = Font::parse(FONT_DATA).unwrap();
    let text = "Hello, World!";
    let size = 16.0;

    // Pre-warm cache
    let mut cache = GlyphCache::new();
    cache.warm_ascii(&font, size);

    group.bench_function("cached_text_layout", |bencher| {
        let mut cache_copy = cache.clone();
        bencher.iter(|| {
            let cached_text = CachedText::new(&font, &mut cache_copy, text, size);
            black_box(cached_text)
        })
    });

    // Benchmark rendering
    let width = 200;
    let height = 32;
    let total_pixels = width * height;
    group.throughput(Throughput::Elements(total_pixels as u64));

    group.bench_function("render_cached_text", |bencher| {
        let mut cache_copy = cache.clone();
        let cached_text = CachedText::new(&font, &mut cache_copy, text, size);
        let colored = Grayscale(cached_text);
        let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); width * height];
        let shape = TensorShape::new(width, height);

        bencher.iter(|| {
            rasterize(&colored, &mut buffer, shape, RenderOptions::default());
            rasterize(&colored, &mut buffer, shape, RenderOptions::default());
            black_box(&buffer);
        })
    });

    group.finish();
}

// ============================================================================
// End-to-End Text Rendering Benchmark
// ============================================================================

fn bench_text_rendering(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_rendering");

    let font = Font::parse(FONT_DATA).unwrap();

    group.bench_function("layout_hello_world", |bencher| {
        let text = "Hello, World!";
        bencher.iter(|| {
            let mut cursor = 0.0f32;
            let size = 16.0;
            for c in text.chars() {
                let _glyph = font.glyph_scaled(c, size);
                let _advance = font.advance_scaled(c, size).unwrap_or(0.0);
                cursor += _advance;
            }
            black_box(cursor)
        })
    });

    group.bench_function("render_single_char_64px", |bencher| {
        let glyph = font.glyph_scaled('A', 64.0).unwrap();
        let colored = Grayscale(glyph);
        let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); 64 * 64];
        let shape = TensorShape::new(64, 64);

        bencher.iter(|| {
            rasterize(&colored, &mut buffer, shape, 1);
            black_box(&buffer);
        })
    });

    group.bench_function("compile_and_render_alphabet", |bencher| {
        bencher.iter(|| {
            for c in 'A'..='Z' {
                let glyph = font.glyph_scaled(c, 32.0).unwrap();
                let colored = Grayscale(glyph);
                let mut buffer: Vec<Rgba8> = vec![Rgba8::default(); 32 * 32];
                let shape = TensorShape::new(32, 32);
                rasterize(&colored, &mut buffer, shape, RenderOptions::default());
                black_box(&buffer);
            }
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    font_benches,
    bench_font_parsing,
    bench_cmap_lookup,
    bench_glyph_compilation,
    bench_advance_width,
);

criterion_group!(glyph_eval_benches, bench_glyph_evaluation,);

criterion_group!(
    rasterize_benches,
    bench_rasterize_solid_color,
    bench_rasterize_gradient,
    bench_rasterize_circle,
    bench_rasterize_glyph,
    bench_rasterize_glyph_aa,
);

criterion_group!(
    color_benches,
    bench_color_conversion,
    bench_discrete_pack,
    bench_color_manifold,
);

criterion_group!(shape_benches, bench_shapes,);

criterion_group!(text_benches, bench_text_rendering,);

// ============================================================================
// Scene3D Benchmarks (Chrome Sphere, etc.)
// ============================================================================

fn bench_scene3d(c: &mut Criterion) {
    use pixelflow_core::jet::Jet3;
    use pixelflow_graphics::render::frame::Frame;
    use pixelflow_graphics::scene3d::{
        Checker, ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface,
        PlaneGeometry, Reflect, ScreenToDir, Sky, Surface,
    };

    /// Sphere at given center with radius (local to this benchmark).
    #[derive(Clone, Copy)]
    struct SphereAt {
        center: (f32, f32, f32),
        radius: f32,
    }

    impl Manifold<Jet3> for SphereAt {
        type Output = Jet3;

        #[inline]
        fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
            let cx = Jet3::constant(Field::from(self.center.0));
            let cy = Jet3::constant(Field::from(self.center.1));
            let cz = Jet3::constant(Field::from(self.center.2));

            let d_dot_c = rx * cx + ry * cy + rz * cz;
            let c_sq = cx * cx + cy * cy + cz * cz;
            let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
            let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

            let epsilon_sq = Jet3::constant(Field::from(0.0001));
            d_dot_c - (discriminant + epsilon_sq).sqrt()
        }
    }

    // Helper: grayscale Field -> Discrete RGBA
    #[derive(Copy, Clone)]
    struct GrayToRgba<M> {
        inner: M,
    }

    impl<M: Manifold<Output = Field>> Manifold for GrayToRgba<M> {
        type Output = Discrete;
        fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
            let gray = self.inner.eval_raw(x, y, z, w);
            Discrete::pack(gray, gray, gray, Field::from(1.0))
        }
    }

    // Helper: remap pixel coords to normalized screen coords
    #[derive(Copy, Clone)]
    struct ScreenRemap<M> {
        inner: M,
        width: f32,
        height: f32,
    }

    impl<M: Manifold<Output = Field>> Manifold for ScreenRemap<M> {
        type Output = Field;
        fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
            let scale = 2.0 / self.height;
            let sx = (x - Field::from(self.width * 0.5)) * Field::from(scale);
            let sy = (Field::from(self.height * 0.5) - y) * Field::from(scale);
            At {
                inner: &self.inner,
                x: sx,
                y: sy,
                z,
                w,
            }
            .eval()
        }
    }

    let mut group = c.benchmark_group("scene3d");

    // Chrome sphere: 400x300
    let w = 400usize;
    let h = 300usize;
    group.throughput(Throughput::Elements((w * h) as u64));

    group.bench_function("chrome_sphere_400x300", |bencher| {
        let world = Surface {
            geometry: PlaneGeometry { height: -1.0 },
            material: Checker,
            background: Sky,
        };

        let scene = Surface {
            geometry: SphereAt {
                center: (0.0, 0.0, 4.0),
                radius: 1.0,
            },
            material: Reflect { inner: world },
            background: world,
        };

        let screen = ScreenRemap {
            inner: ScreenToDir { inner: scene },
            width: w as f32,
            height: h as f32,
        };

        let renderable = GrayToRgba { inner: screen };
        let mut frame = Frame::<Rgba8>::new(w as u32, h as u32);
        let shape = TensorShape::new(w, h);

        bencher.iter(|| {
            rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions::default());
            black_box(&frame);
        })
    });

    // Sky only (baseline)
    group.bench_function("sky_only_400x300", |bencher| {
        let scene = Surface {
            geometry: PlaneGeometry { height: -1000.0 }, // Never hits
            material: Checker,
            background: Sky,
        };

        let screen = ScreenRemap {
            inner: ScreenToDir { inner: scene },
            width: w as f32,
            height: h as f32,
        };

        let renderable = GrayToRgba { inner: screen };
        let mut frame = Frame::<Rgba8>::new(w as u32, h as u32);
        let shape = TensorShape::new(w, h);

        bencher.iter(|| {
            rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions::default());
            black_box(&frame);
        })
    });

    // Floor only (no reflection)
    group.bench_function("floor_only_400x300", |bencher| {
        let scene = Surface {
            geometry: PlaneGeometry { height: -1.0 },
            material: Checker,
            background: Sky,
        };

        let screen = ScreenRemap {
            inner: ScreenToDir { inner: scene },
            width: w as f32,
            height: h as f32,
        };

        let renderable = GrayToRgba { inner: screen };
        let mut frame = Frame::<Rgba8>::new(w as u32, h as u32);
        let shape = TensorShape::new(w, h);

        bencher.iter(|| {
            rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions::default());
            rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions::default());
            black_box(&frame);
        })
    });

    // ========================================================================
    // MULLET ARCHITECTURE: Color rendering (geometry once, colors as Discrete)
    // ========================================================================

    // Helper: remap for Discrete output
    #[derive(Copy, Clone)]
    struct ColorScreenRemap<M> {
        inner: M,
        width: f32,
        height: f32,
    }

    impl<M: Manifold<Output = Discrete>> Manifold for ColorScreenRemap<M> {
        type Output = Discrete;
        fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
            let scale = 2.0 / self.height;
            let sx = (x - Field::from(self.width * 0.5)) * Field::from(scale);
            let sy = (Field::from(self.height * 0.5) - y) * Field::from(scale);
            At {
                inner: &self.inner,
                x: sx,
                y: sy,
                z,
                w,
            }
            .eval()
        }
    }

    // Color chrome sphere (mullet architecture) - 1080p
    let w_hd = 1920usize;
    let h_hd = 1080usize;
    group.throughput(Throughput::Elements((w_hd * h_hd) as u64));

    group.bench_function("color_chrome_1920x1080_mullet", |bencher| {
        let world = ColorSurface {
            geometry: PlaneGeometry { height: -1.0 },
            material: ColorChecker,
            background: ColorSky,
        };

        let scene = ColorSurface {
            geometry: SphereAt {
                center: (0.0, 0.0, 4.0),
                radius: 1.0,
            },
            material: ColorReflect { inner: world },
            background: world,
        };

        let renderable = ColorScreenRemap {
            inner: ColorScreenToDir { inner: scene },
            width: w_hd as f32,
            height: h_hd as f32,
        };

        let mut frame = Frame::<Rgba8>::new(w_hd as u32, h_hd as u32);
        let shape = TensorShape::new(w_hd, h_hd);

        bencher.iter(|| {
            rasterize(&renderable, frame.as_slice_mut(), shape, 1);
            black_box(&frame);
        })
    });

    // Color chrome sphere at 400x300 for comparison
    group.throughput(Throughput::Elements((w * h) as u64));

    group.bench_function("color_chrome_400x300_mullet", |bencher| {
        let world = ColorSurface {
            geometry: PlaneGeometry { height: -1.0 },
            material: ColorChecker,
            background: ColorSky,
        };

        let scene = ColorSurface {
            geometry: SphereAt {
                center: (0.0, 0.0, 4.0),
                radius: 1.0,
            },
            material: ColorReflect { inner: world },
            background: world,
        };

        let renderable = ColorScreenRemap {
            inner: ColorScreenToDir { inner: scene },
            width: w as f32,
            height: h as f32,
        };

        let mut frame = Frame::<Rgba8>::new(w as u32, h as u32);
        let shape = TensorShape::new(w, h);

        bencher.iter(|| {
            rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions::default());
            black_box(&frame);
        })
    });

    // ========================================================================
    // PARALLEL RENDERING
    // ========================================================================

    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    group.throughput(Throughput::Elements((w_hd * h_hd) as u64));

    group.bench_function(
        &format!("color_chrome_1920x1080_parallel_{}t", num_threads),
        |bencher| {
            let world = ColorSurface {
                geometry: PlaneGeometry { height: -1.0 },
                material: ColorChecker,
                background: ColorSky,
            };

            let scene = ColorSurface {
                geometry: SphereAt {
                    center: (0.0, 0.0, 4.0),
                    radius: 1.0,
                },
                material: ColorReflect { inner: world },
                background: world,
            };

            let renderable = ColorScreenRemap {
                inner: ColorScreenToDir { inner: scene },
                width: w_hd as f32,
                height: h_hd as f32,
            };

            let mut frame = Frame::<Rgba8>::new(w_hd as u32, h_hd as u32);
            let shape = TensorShape::new(w_hd, h_hd);

            bencher.iter(|| {
                rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions { num_threads });
                black_box(&frame);
            })
        },
    );

    group.finish();
}

criterion_group!(scene3d_benches, bench_scene3d,);

// ============================================================================
// Scheduler Comparison Benchmarks
// ============================================================================

fn bench_scheduler_comparison(c: &mut Criterion) {
    use pixelflow_core::jet::Jet3;
    use pixelflow_graphics::render::frame::Frame;
    use pixelflow_graphics::scene3d::{
        ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, PlaneGeometry,
    };

    /// Sphere at given center with radius (local to this benchmark).
    #[derive(Clone, Copy)]
    struct SphereAt {
        center: (f32, f32, f32),
        radius: f32,
    }

    impl Manifold<Jet3> for SphereAt {
        type Output = Jet3;

        #[inline]
        fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
            let cx = Jet3::constant(Field::from(self.center.0));
            let cy = Jet3::constant(Field::from(self.center.1));
            let cz = Jet3::constant(Field::from(self.center.2));

            let d_dot_c = rx * cx + ry * cy + rz * cz;
            let c_sq = cx * cx + cy * cy + cz * cz;
            let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
            let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

            let epsilon_sq = Jet3::constant(Field::from(0.0001));
            d_dot_c - (discriminant + epsilon_sq).sqrt()
        }
    }

    // Helper: remap for Discrete output
    #[derive(Copy, Clone)]
    struct ColorScreenRemap<M> {
        inner: M,
        width: f32,
        height: f32,
    }

    impl<M: Manifold<Output = Discrete>> Manifold for ColorScreenRemap<M> {
        type Output = Discrete;
        fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
            let scale = 2.0 / self.height;
            let sx = (x - Field::from(self.width * 0.5)) * Field::from(scale);
            let sy = (Field::from(self.height * 0.5) - y) * Field::from(scale);
            At {
                inner: &self.inner,
                x: sx,
                y: sy,
                z,
                w,
            }
            .eval()
        }
    }

    let mut group = c.benchmark_group("scheduler_comparison");

    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    // Test at 1080p - most realistic workload
    let w = 1920usize;
    let h = 1080usize;
    group.throughput(Throughput::Elements((w * h) as u64));

    // Build the scene once
    let world = ColorSurface {
        geometry: PlaneGeometry { height: -1.0 },
        material: ColorChecker,
        background: ColorSky,
    };

    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world },
        background: world,
    };

    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: w as f32,
        height: h as f32,
    };

    let shape = TensorShape::new(w, h);

    // Benchmark 1: Work-stealing with atomic counter (scoped threads)
    group.bench_function(
        &format!("work_stealing_atomic_{}t", num_threads),
        |bencher| {
            let mut frame = Frame::<Rgba8>::new(w as u32, h as u32);
            bencher.iter(|| {
                rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions { num_threads });
                black_box(&frame);
            })
        },
    );

    // Benchmark 2: Single-threaded (baseline)
    group.bench_function("single_threaded", |bencher| {
        let mut frame = Frame::<Rgba8>::new(w as u32, h as u32);
        bencher.iter(|| {
            rasterize(&renderable, frame.as_slice_mut(), shape, RenderOptions::default());
            black_box(&frame);
        })
    });

    group.finish();
}

criterion_group!(scheduler_benches, bench_scheduler_comparison,);

criterion_group!(
    cache_benches,
    bench_glyph_cache,
    bench_cached_vs_uncached_eval,
    bench_cached_vs_uncached_raster,
    bench_cached_text,
);

criterion_main!(
    font_benches,
    glyph_eval_benches,
    rasterize_benches,
    color_benches,
    shape_benches,
    text_benches,
    cache_benches,
    scene3d_benches,
    scheduler_benches
);
