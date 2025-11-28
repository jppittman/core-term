use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_core::{TensorView, TensorViewMut, batch::Batch, execute, ops::{SampleAtlas, Offset, Skew, Max, Over}, dsl::{SurfaceExt, MaskExt}};

// Benchmark constants to avoid magic numbers
const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const STRIDE: usize = 256;
const BUFFER_SIZE: usize = WIDTH * HEIGHT;

fn bench_gather_2d(c: &mut Criterion) {
    let data = vec![0u8; BUFFER_SIZE];
    let view = TensorView::new(&data, WIDTH, HEIGHT, STRIDE);

    let x = Batch::<u32>::new(0, 10, 20, 30);
    let y = Batch::<u32>::splat(10);

    c.bench_function("gather_2d", |b| {
        b.iter(|| unsafe {
            black_box(view.gather_2d(black_box(x), black_box(y)));
        })
    });
}

fn bench_sample_4bit_bilinear(c: &mut Criterion) {
    let data = vec![0u8; BUFFER_SIZE];
    let view = TensorView::new(&data, WIDTH, HEIGHT, STRIDE);

    // 16.16 fixed point coordinates
    let u = Batch::<u32>::new(10 << 16, 20 << 16, 30 << 16, 40 << 16);
    let v = Batch::<u32>::splat(50 << 16);

    c.bench_function("sample_4bit_bilinear", |b| {
        b.iter(|| unsafe {
            black_box(view.sample_4bit_bilinear(black_box(u), black_box(v)));
        })
    });
}

fn bench_pipeline_execution(c: &mut Criterion) {
    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);

    let mut target_data = vec![0u8; BUFFER_SIZE];

    let pipeline = SampleAtlas { atlas: atlas_view, step_x_fp: 65536, step_y_fp: 65536 };

    c.bench_function("pipeline_simple_sample", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            execute(pipeline, &mut target_view);
        })
    });
}

// --- New Benchmarks ---

// 1. Blend Math Micro-benchmark
#[inline(always)]
fn blend_math_impl(fg: Batch<u32>, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
    let inv_alpha = Batch::splat(256) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

fn bench_blend_math(c: &mut Criterion) {
    let fg = Batch::splat(0xFFFFFFFF);
    let bg = Batch::splat(0x00000000);
    let alpha = Batch::splat(128);

    c.bench_function("blend_math", |b| {
        b.iter(|| {
            black_box(blend_math_impl(black_box(fg), black_box(bg), black_box(alpha)));
        })
    });
}

// 2. Operators Benchmarks
fn bench_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators");

    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);
    let source = SampleAtlas { atlas: atlas_view, step_x_fp: 65536, step_y_fp: 65536 };

    let mut target_data = vec![0u8; BUFFER_SIZE];

    // Offset
    group.bench_function("Offset", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = source.offset(10, 10);
            execute(pipe, &mut target_view);
        })
    });

    // Skew
    group.bench_function("Skew", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = source.skew(50);
            execute(pipe, &mut target_view);
        })
    });

    // Max
    group.bench_function("Max", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = Max(source, source.offset(1, 0));
            execute(pipe, &mut target_view);
        })
    });

    group.finish();
}

// 3. Sampling Scaling Benchmarks
fn bench_sampling_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_scaling");

    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);
    let mut target_data = vec![0u8; BUFFER_SIZE];

    // 0.5x Scale (Minification) - Step = 2.0 (131072)
    group.bench_function("0.5x", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
             let pipe = SampleAtlas { atlas: atlas_view, step_x_fp: 131072, step_y_fp: 131072 };
            execute(pipe, &mut target_view);
        })
    });

    // 1.0x Scale - Step = 1.0 (65536)
    group.bench_function("1.0x", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
             let pipe = SampleAtlas { atlas: atlas_view, step_x_fp: 65536, step_y_fp: 65536 };
            execute(pipe, &mut target_view);
        })
    });

    // 2.0x Scale (Magnification) - Step = 0.5 (32768)
    group.bench_function("2.0x", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
             let pipe = SampleAtlas { atlas: atlas_view, step_x_fp: 32768, step_y_fp: 32768 };
            execute(pipe, &mut target_view);
        })
    });

    group.finish();
}

// 4. Text Pipelines Benchmarks
fn bench_text_pipelines(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_pipelines");

    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);
    let source = SampleAtlas { atlas: atlas_view, step_x_fp: 65536, step_y_fp: 65536 };

    let mut target_data = vec![0u32; BUFFER_SIZE]; // u32 for colored output
    let fg = Batch::splat(0xFFFFFFFF);
    let bg = Batch::splat(0xFF000000);

    // Normal: Sampler -> Over
    group.bench_function("Normal", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = source.over(fg, bg);
            execute(pipe, &mut target_view);
        })
    });

    // Bold: Max(Sampler, Sampler.offset) -> Over
    group.bench_function("Bold", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let bold = Max(source, source.offset(1, 0));
            let pipe = bold.over(fg, bg);
            execute(pipe, &mut target_view);
        })
    });

    // Italic: Skew(Sampler) -> Over
    group.bench_function("Italic", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let italic = source.skew(50);
            let pipe = italic.over(fg, bg);
            execute(pipe, &mut target_view);
        })
    });

    // Bold Italic: Max(Italic, Italic.offset) -> Over
    group.bench_function("BoldItalic", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let italic = source.skew(50);
            let bold_italic = Max(italic, italic.offset(1, 0));
            let pipe = bold_italic.over(fg, bg);
            execute(pipe, &mut target_view);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gather_2d,
    bench_sample_4bit_bilinear,
    bench_pipeline_execution,
    bench_blend_math,
    bench_operators,
    bench_sampling_scaling,
    bench_text_pipelines
);
criterion_main!(benches);
